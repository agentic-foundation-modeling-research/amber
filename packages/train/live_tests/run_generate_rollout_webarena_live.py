#!/usr/bin/env python3
"""Run one real WebArena generate_rollout() call against live services.

This runner exercises the custom WebArena rollout entrypoint in
packages.train.generate_webarena. It passes multiple prompt groups through the
Miles data-source contract so generate_rollout() owns VM leasing, rollout
metadata assignment, custom generate calls, reward-model calls, and filtering.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml


THIS_FILE = Path(__file__).resolve()
TRAIN_PKG_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]

for path in (REPO_ROOT, TRAIN_PKG_DIR):
    if path.exists():
        sys.path.insert(0, str(path))

from packages.train.live_tests.run_generate_webarena_live import (  # noqa: E402
    DEFAULT_TASK_CONFIG,
    assert_generate_state_args,
    build_miles_args as build_base_miles_args,
    load_task_config,
    sample_to_json,
    start_sglang,
    stop_process,
    wait_for_http_ok,
)


CUSTOM_GENERATE_PATH = "packages.train.generate_webarena.generate"
CUSTOM_RM_PATH = "packages.train.generate_webarena.reward_model"
CUSTOM_REWARD_POST_PROCESS_PATH = "packages.train.generate_webarena.postprocess_reward"
CUSTOM_ROLLOUT_PATH = "packages.train.generate_webarena.generate_rollout"
DEFAULT_TASK_IDS = [10010, 10054]
DEFAULT_CUSTOM_CONFIG = (
    REPO_ROOT / "packages" / "train" / "training_configs" / "rl" / "append_memory.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call packages.train.generate_webarena.generate_rollout() against live WebArena services.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--custom-config-path",
        type=Path,
        default=DEFAULT_CUSTOM_CONFIG,
        help="Path to YAML config for WebArena custom rollout args.",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=DEFAULT_TASK_CONFIG,
        help=(
            "Path to a WebArena task config file. Supports a JSON list or JSONL rows "
            "with either raw task configs or rows containing a 'metadata' object."
        ),
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=DEFAULT_TASK_IDS,
        help="Distinct WebArena task_id values to feed to generate_rollout().",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=2,
        help="Number of accepted prompt groups generate_rollout() should return.",
    )
    parser.add_argument(
        "--hf-checkpoint",
        required=True,
        help="Hugging Face checkpoint path or model repo id used by SGLang and the local Miles tokenizer.",
    )
    parser.add_argument(
        "--chat-template-path",
        default=None,
        help="Optional chat template path passed to the local Miles tokenizer.",
    )
    parser.add_argument(
        "--apply-chat-template-kwargs",
        type=json.loads,
        default={},
        help=(
            "JSON object with extra kwargs forwarded to tokenizer.apply_chat_template "
            "for local tokenizer setup, e.g. '{\"clear_thinking\": false}'."
        ),
    )
    parser.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=1,
        help="Number of parallel samples per prompt group produced by the live data source.",
    )
    parser.add_argument(
        "--sglang-server-concurrency",
        type=int,
        default=1,
        help="Miles-side maximum concurrent generate_and_rm calls.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens per model generation request sent by the agent loop.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature forwarded to SGLang /generate.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p value forwarded to SGLang /generate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling value forwarded to SGLang /generate; -1 leaves it effectively disabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed value used for deterministic rollout/env initialization where supported.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional path to write returned Sample objects as JSONL for inspection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for this runner, for example DEBUG, INFO, WARNING, or ERROR.",
    )
    parser.add_argument(
        "--start-sglang",
        action="store_true",
        help="Start a local SGLang server subprocess before calling generate_rollout(). If unset, an existing backend is assumed.",
    )
    parser.add_argument(
        "--sglang-host",
        default="127.0.0.1",
        help="Host address for the SGLang server to start or connect to.",
    )
    parser.add_argument(
        "--sglang-port",
        type=int,
        default=30000,
        help="Port for the SGLang server to start or connect to.",
    )
    parser.add_argument(
        "--sglang-log",
        type=Path,
        default=Path("/tmp/webarena_live_sglang.log"),
        help="Log file path for the local SGLang subprocess started by --start-sglang.",
    )
    parser.add_argument(
        "--sglang-startup-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for the local SGLang server health endpoint before failing.",
    )
    parser.add_argument(
        "--sglang-extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument token to append to 'python -m sglang.launch_server'. "
            "Repeat for each flag/value; use --sglang-extra-arg=--flag for flag names."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for the local SGLang subprocess.",
    )
    return parser.parse_args()


def load_custom_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"--custom-config-path does not exist: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def config_value(
    args: argparse.Namespace,
    name: str,
    default: Any = None,
) -> Any:
    return args.custom_config.get(name, default)


def resolve_runner_args(args: argparse.Namespace) -> None:
    args.custom_config = load_custom_config(args.custom_config_path)

    args.env_server_url = config_value(args, "webarena_env_server_url")
    args.webarena_max_steps = config_value(args, "webarena_max_steps")
    args.webarena_env_timeout_s = config_value(args, "webarena_env_timeout_s")
    args.webarena_debug_loss_mask = config_value(args, "webarena_debug_loss_mask", False)


def validate_args(args: argparse.Namespace) -> None:
    if not args.env_server_url:
        raise ValueError("Provide webarena_env_server_url in --custom-config-path")
    configured_vms = config_value(args, "webarena_vms") or []
    if not configured_vms:
        raise ValueError("Provide non-empty webarena_vms in --custom-config-path")
    if args.rollout_batch_size < 1:
        raise ValueError("--rollout-batch-size must be >= 1")
    if args.n_samples_per_prompt < 1:
        raise ValueError("--n-samples-per-prompt must be >= 1")
    site_port_stride = config_value(args, "webarena_site_port_stride", 1)
    if site_port_stride < 1:
        raise ValueError("webarena_site_port_stride in --custom-config-path must be >= 1")
    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError(f"--task-ids must be distinct, got {args.task_ids}")
    if len(args.task_ids) < args.rollout_batch_size:
        raise ValueError(
            f"--rollout-batch-size={args.rollout_batch_size} requires at least that many distinct --task-ids"
        )
    port_bases = config_value(args, "webarena_port_bases") or []
    if not port_bases:
        raise ValueError("Provide non-empty webarena_port_bases in --custom-config-path")


def build_miles_args(args: argparse.Namespace, first_task_config: dict[str, Any]) -> Namespace:
    miles_args = build_base_miles_args(args, first_task_config)
    for key, value in args.custom_config.items():
        if not hasattr(miles_args, key):
            setattr(miles_args, key, value)

    miles_args.rollout_function_path = CUSTOM_ROLLOUT_PATH
    miles_args.custom_generate_function_path = CUSTOM_GENERATE_PATH
    miles_args.custom_rm_path = CUSTOM_RM_PATH
    miles_args.custom_reward_post_process_path = CUSTOM_REWARD_POST_PROCESS_PATH
    miles_args.group_rm = False
    miles_args.partial_rollout = False
    miles_args.mask_offpolicy_in_partial_rollout = False
    miles_args.sglang_router_policy = "round_robin"
    miles_args.reward_key = None
    miles_args.rm_type = None
    miles_args.ci_test = False
    miles_args.n_samples_per_prompt = args.n_samples_per_prompt
    miles_args.sglang_server_concurrency = args.sglang_server_concurrency

    miles_args.rollout_global_dataset = True
    miles_args.rollout_batch_size = args.rollout_batch_size
    miles_args.dynamic_sampling_filter_path = None
    miles_args.rollout_sample_filter_path = None
    miles_args.rollout_all_samples_process_path = None

    miles_args.webarena_vms = list(config_value(args, "webarena_vms") or [])
    miles_args.webarena_port_bases = list(config_value(args, "webarena_port_bases") or [])

    # generate_rollout() calls dumper_utils.configure_sglang(); keep dumping
    # disabled while still providing the attributes that utility expects.
    miles_args.dumper_enable = False
    miles_args.dumper_inference = []
    miles_args.dumper_dir = "/tmp/webarena_live_dumper"
    miles_args.dumper_source_patcher_config_inference = None
    return miles_args


class LiveTaskDataSource:
    def __init__(self, args: argparse.Namespace, task_configs: list[dict[str, Any]], sample_cls: type) -> None:
        self.args = args
        self.task_configs = [copy.deepcopy(task_config) for task_config in task_configs]
        self.sample_cls = sample_cls
        self.offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.returned_aborted_groups: list[list[Any]] = []

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        groups = []
        for _ in range(num_samples):
            base_task_config = self.task_configs[self.offset % len(self.task_configs)]
            self.offset += 1

            group = []
            for _sample_offset in range(self.args.n_samples_per_prompt):
                task_config = copy.deepcopy(base_task_config)
                task_config.pop("rollout_metadata", None)
                group.append(
                    self.sample_cls(
                        index=self.sample_index,
                        group_index=self.sample_group_index,
                        prompt=task_config.get("intent", ""),
                        metadata=task_config,
                    )
                )
                self.sample_index += 1
            self.sample_group_index += 1
            self._validate_and_log_group(group)
            groups.append(group)
        return groups

    def add_samples(self, samples: list[list[Any]]) -> None:
        self.returned_aborted_groups.extend(samples)

    def _validate_and_log_group(self, group: list[Any]) -> None:
        if not group:
            raise ValueError("Live WebArena data source produced an empty prompt group")

        task_ids = [sample.metadata.get("task_id") for sample in group]
        missing_indices = [sample.index for sample, task_id in zip(group, task_ids, strict=True) if task_id is None]
        if missing_indices:
            raise ValueError(f"Live WebArena samples missing metadata['task_id']: sample_indices={missing_indices}")

        unique_task_ids = set(task_ids)
        if len(unique_task_ids) != 1:
            sample_task_ids = {
                sample.index: task_id
                for sample, task_id in zip(group, task_ids, strict=True)
            }
            raise ValueError(
                "Live WebArena prompt group must contain exactly one task_id, "
                f"got sample_task_ids={sample_task_ids}"
            )

        logging.info(
            "Prepared live WebArena prompt group: group_index=%s task_id=%s sample_indices=%s sites=%s",
            group[0].group_index,
            task_ids[0],
            [sample.index for sample in group],
            group[0].metadata.get("sites") or [],
        )


def iter_leaf_samples(result_group: list[Any]):
    for item in result_group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def print_result_groups(result_groups: list[list[Any]]) -> None:
    print(f"generate_rollout() returned {len(result_groups)} accepted group(s)")
    for group_index, group in enumerate(result_groups):
        print(f"group[{group_index}] len={len(group)}")
        for sample_index, item in enumerate(group):
            samples = item if isinstance(item, list) else [item]
            for nested_index, sample in enumerate(samples):
                prefix = f"group[{group_index}][{sample_index}.{nested_index}]" if isinstance(item, list) else (
                    f"group[{group_index}][{sample_index}]"
                )
                rollout_metadata = sample.metadata.get("rollout_metadata") or {}
                print(
                    f"{prefix} task_id={sample.metadata.get('task_id')} "
                    f"status={sample.status.value} "
                    f"reward={sample.reward!r} "
                    f"episode_reward={sample.metadata.get('episode_reward')!r} "
                    f"response_length={sample.response_length} "
                    f"tokens={len(sample.tokens)} "
                    f"loss_mask={len(sample.loss_mask or [])} "
                    f"rollout_log_probs={len(sample.rollout_log_probs or [])} "
                    f"site_ports={json.dumps(rollout_metadata.get('site_ports', {}), sort_keys=True)}"
                )


def run_generate_rollout(args: argparse.Namespace) -> None:
    from packages.train.generate_webarena import generate_rollout
    from miles.utils.http_utils import init_http_client
    from miles.utils.types import Sample

    task_configs = [dict(load_task_config(args.task_config, task_id)) for task_id in args.task_ids]
    rollout_task_configs = task_configs[: args.rollout_batch_size]
    data_source = LiveTaskDataSource(args, rollout_task_configs, Sample)
    miles_args = build_miles_args(args, rollout_task_configs[0])

    print(
        "Running live WebArena generate_rollout() for "
        f"task_ids={[task_config['task_id'] for task_config in rollout_task_configs]} "
        f"rollout_batch_size={args.rollout_batch_size} "
        f"n_samples_per_prompt={args.n_samples_per_prompt} "
        f"max_steps={miles_args.webarena_max_steps}"
    )
    print(f"rollout_function_path={miles_args.rollout_function_path}")
    print(f"custom_generate_function_path={miles_args.custom_generate_function_path}")
    print(f"custom_rm_path={miles_args.custom_rm_path}")
    print(f"custom_reward_post_process_path={miles_args.custom_reward_post_process_path}")
    print(
        "webarena_vm_config="
        + json.dumps(
            {
                "vms": miles_args.webarena_vms,
                "port_bases": miles_args.webarena_port_bases,
                "site_port_stride": miles_args.webarena_site_port_stride,
            },
            sort_keys=True,
        )
    )

    assert_generate_state_args(miles_args)
    init_http_client(miles_args)
    rollout_started_at = time.monotonic()
    output = generate_rollout(miles_args, rollout_id=0, data_source=data_source, evaluation=False)
    print_result_groups(output.samples)
    rollout_elapsed_minutes = (time.monotonic() - rollout_started_at) / 60
    print(f"generate_rollout elapsed_minutes={rollout_elapsed_minutes:.2f}")
    print(f"metrics={json.dumps(output.metrics, sort_keys=True, default=str)}")
    if data_source.returned_aborted_groups:
        print(f"aborted_groups_returned_to_data_source={len(data_source.returned_aborted_groups)}")

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as f:
            for group in output.samples:
                for returned in iter_leaf_samples(group):
                    f.write(json.dumps(sample_to_json(returned), default=str) + "\n")
        print(f"Wrote returned leaf samples to {args.output_jsonl}")


def main() -> None:
    args = parse_args()
    resolve_runner_args(args)
    validate_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sglang_process = None
    try:
        if args.start_sglang:
            sglang_process = start_sglang(args)
        else:
            wait_for_http_ok(
                f"http://{args.sglang_host}:{args.sglang_port}/health_generate",
                process=None,
                timeout=args.sglang_startup_timeout,
                log_path=args.sglang_log,
                name="existing SGLang",
            )
        run_generate_rollout(args)
    finally:
        stop_process(sglang_process)


if __name__ == "__main__":
    main()
