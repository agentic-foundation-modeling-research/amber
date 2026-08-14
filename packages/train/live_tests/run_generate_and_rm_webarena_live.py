#!/usr/bin/env python3
"""Run one real WebArena generate_and_rm_group() call against live services.

This runner intentionally follows the Miles rollout hierarchy around
generate_and_rm: one prompt group is passed to generate_and_rm_group(), which
loads the configured custom generate and reward functions by path.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
TRAIN_PKG_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]

for path in (REPO_ROOT, TRAIN_PKG_DIR):
    sys.path.insert(0, str(path))

from packages.train.live_tests.run_generate_webarena_live import (  # noqa: E402
    DEFAULT_CUSTOM_CONFIG,
    assert_generate_state_args,
    build_miles_args as build_base_miles_args,
    build_rollout_metadata,
    build_sampling_params,
    load_task_config,
    resolve_runner_args,
    sample_to_json,
    start_sglang,
    stop_process,
    validate_runner_args,
    wait_for_http_ok,
)


CUSTOM_GENERATE_PATH = "packages.train.generate_webarena.generate"
CUSTOM_RM_PATH = "packages.train.generate_webarena.reward_model"
CUSTOM_REWARD_POST_PROCESS_PATH = "packages.train.generate_webarena.postprocess_reward"
DEFAULT_TASK_CONFIG = REPO_ROOT / "task_configs" / "webarena_train.json"
ROLLOUT_METADATA_KEYS = (
    "site_ports",
    "site_urls",
    "setup_server_url",
    "homepage_url",
    "calculator_url",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Miles generate_and_rm_group() once with WebArena custom generate/RM paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--custom-config-path",
        type=Path,
        default=DEFAULT_CUSTOM_CONFIG,
        help="Path to YAML config for WebArena custom live-runner args.",
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
        "--task-id",
        type=int,
        required=True,
        help="Specific WebArena task_id to load from --task-config.",
    )
    parser.add_argument(
        "--env-server-url",
        default=None,
        help="Base URL for the running context_scythe env_server, for example http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--website-host",
        default=None,
        help="Host/IP where the WebArena setup server, homepage, and task site ports are reachable.",
    )
    parser.add_argument(
        "--setup-server-port",
        type=int,
        default=None,
        help="Port for the WebArena rollout setup server that handles per-task site reset/teardown.",
    )
    parser.add_argument(
        "--homepage-port",
        type=int,
        default=None,
        help="Port for the WebArena homepage/calculator service.",
    )
    parser.add_argument(
        "--port-bases",
        type=int,
        nargs=2,
        default=None,
        metavar="PORT",
        help=(
            "Two base ports used to allocate mutable task sites. Each site gets its own base and parallel "
            "samples use base + sample_index * --site-port-stride."
        ),
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
        "--webarena-max-steps",
        type=int,
        default=None,
        help="Maximum number of browser-agent steps to execute inside generate(). Keep this low for live validation.",
    )
    parser.add_argument(
        "--webarena-env-timeout-s",
        type=float,
        default=None,
        help="Seconds to wait for each env_server HTTP request before failing.",
    )
    parser.add_argument(
        "--webarena-debug-loss-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print decoded WebArena loss-mask spans for action output, inserted compression prompt, and memory output.",
    )
    parser.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=1,
        help="Number of parallel samples to place in the prompt group passed to generate_and_rm_group().",
    )
    parser.add_argument(
        "--sglang-server-concurrency",
        type=int,
        default=1,
        help=(
            "Miles-side maximum concurrent generate_and_rm calls. Increase this "
            "with --n-samples-per-prompt to allow true parallel rollouts against "
            "an existing or managed SGLang server."
        ),
    )
    parser.add_argument(
        "--site-port-stride",
        type=int,
        default=None,
        help="Port-base offset between parallel WebArena samples so each rollout gets distinct mutable site ports.",
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
        help="Start a local SGLang server subprocess before calling generate_and_rm_group(). If unset, an existing backend is assumed.",
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


def build_miles_args(args: argparse.Namespace, task_config: dict[str, Any]) -> Namespace:
    miles_args = build_base_miles_args(args, task_config)
    for key, value in getattr(args, "custom_config", {}).items():
        if not hasattr(miles_args, key):
            setattr(miles_args, key, value)

    miles_args.rollout_function_path = "miles.rollout.sglang_rollout.generate_rollout"
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
    return miles_args


def assign_rollout_metadata(
    args: argparse.Namespace,
    task_config: dict[str, Any],
    *,
    site_url_templates: dict[str, str],
    format_site_url: Any,
    site_ports: dict[str, int],
) -> None:
    task_config["rollout_metadata"] = build_rollout_metadata(
        task_config,
        website_host=args.website_host,
        setup_server_port=args.setup_server_port,
        homepage_port=args.homepage_port,
        port_base=args.port_bases[0],
        site_url_templates=site_url_templates,
        format_site_url=format_site_url,
    )
    task_config["rollout_metadata"]["site_ports"] = site_ports
    task_config["rollout_metadata"]["site_urls"] = {
        site: format_site_url(site_url_templates[site], args.website_host, port)
        for site, port in site_ports.items()
    }

    missing = [key for key in ROLLOUT_METADATA_KEYS if key not in task_config["rollout_metadata"]]
    if missing:
        raise RuntimeError(f"Internal runner bug: missing rollout_metadata keys: {missing}")


def port_bases_from_args(args: argparse.Namespace) -> list[int]:
    port_bases = args.port_bases
    if len(port_bases) != 2:
        raise ValueError("--port-bases requires exactly two per-site base ports")
    return port_bases


def site_ports_for_sample(
    *,
    sites: list[str],
    port_bases: list[int],
    sample_index: int,
    site_port_stride: int,
) -> dict[str, int]:
    if len(sites) > len(port_bases):
        raise ValueError(
            f"--port-bases provided {len(port_bases)} per-site base ports, but selected task has {len(sites)} sites"
        )
    return {
        site: port_bases[site_index] + sample_index * site_port_stride
        for site_index, site in enumerate(sites)
    }


def build_sample_group(
    args: argparse.Namespace,
    base_task_config: dict[str, Any],
    *,
    sample_cls: type,
    site_url_templates: dict[str, str],
    format_site_url: Any,
) -> list[Any]:
    if args.n_samples_per_prompt < 1:
        raise ValueError("--n-samples-per-prompt must be >= 1")
    if args.site_port_stride < 1:
        raise ValueError("--site-port-stride must be >= 1")
    port_bases = port_bases_from_args(args)
    sites = list(base_task_config.get("sites") or [])

    group = []
    for sample_index in range(args.n_samples_per_prompt):
        task_config = copy.deepcopy(base_task_config)
        site_ports = site_ports_for_sample(
            sites=sites,
            port_bases=port_bases,
            sample_index=sample_index,
            site_port_stride=args.site_port_stride,
        )
        assign_rollout_metadata(
            args,
            task_config,
            site_url_templates=site_url_templates,
            format_site_url=format_site_url,
            site_ports=site_ports,
        )
        group.append(
            sample_cls(
                index=sample_index,
                group_index=0,
                prompt=task_config.get("intent", ""),
                metadata=task_config,
            )
        )
    return group


def iter_leaf_samples(result_group: list[Any]):
    for item in result_group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def print_result_group(result_group: list[Any]) -> None:
    print(f"generate_and_rm_group() returned type={type(result_group).__name__} len={len(result_group)}")
    for index, item in enumerate(result_group):
        print(f"[{index}] type={type(item).__name__}")
        samples = item if isinstance(item, list) else [item]
        if isinstance(item, list):
            print(f"[{index}] nested_samples={len(samples)}")
        for nested_index, sample in enumerate(samples):
            prefix = f"[{index}.{nested_index}]" if isinstance(item, list) else f"[{index}]"
            print(
                f"{prefix} status={sample.status.value} "
                f"reward={sample.reward!r} "
                f"episode_reward={sample.metadata.get('episode_reward')!r} "
                f"response_length={sample.response_length} "
                f"tokens={len(sample.tokens)} "
                f"loss_mask={len(sample.loss_mask or [])} "
                f"rollout_log_probs={len(sample.rollout_log_probs or [])}"
            )


async def run_generate_and_rm(args: argparse.Namespace) -> None:
    from packages.train.generate_webarena import SITE_URL_TEMPLATES, format_url
    from miles.rollout.sglang_rollout import generate_and_rm_group
    from miles.utils.http_utils import init_http_client
    from miles.utils.types import Sample

    base_task_config = dict(load_task_config(args.task_config, args.task_id))
    group = build_sample_group(
        args,
        base_task_config,
        sample_cls=Sample,
        site_url_templates=SITE_URL_TEMPLATES,
        format_site_url=format_url,
    )

    miles_args = build_miles_args(args, group[0].metadata)

    print(
        "Running live WebArena generate_and_rm_group() for "
        f"task_id={base_task_config['task_id']} sites={base_task_config.get('sites')} "
        f"max_steps={args.webarena_max_steps} "
        f"n_samples_per_prompt={args.n_samples_per_prompt}"
    )
    print(f"custom_generate_function_path={miles_args.custom_generate_function_path}")
    print(f"custom_rm_path={miles_args.custom_rm_path}")
    print(f"custom_reward_post_process_path={miles_args.custom_reward_post_process_path}")
    for sample in group:
        print(
            f"sample index={sample.index} group_index={sample.group_index} "
            f"rollout_metadata={json.dumps(sample.metadata['rollout_metadata'], sort_keys=True)}"
        )

    assert_generate_state_args(miles_args)
    init_http_client(miles_args)
    rollout_started_at = time.monotonic()
    result_group = await generate_and_rm_group(
        miles_args,
        group,
        sampling_params=build_sampling_params(args),
        evaluation=False,
    )
    print_result_group(result_group)
    rollout_elapsed_minutes = (time.monotonic() - rollout_started_at) / 60
    print(f"group rollout elapsed_minutes={rollout_elapsed_minutes:.2f}")

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as f:
            for returned in iter_leaf_samples(result_group):
                f.write(json.dumps(sample_to_json(returned), default=str) + "\n")
        print(f"Wrote returned leaf samples to {args.output_jsonl}")


async def main() -> None:
    args = parse_args()
    resolve_runner_args(args, default_env_timeout_s=60)
    validate_runner_args(args)
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
        await run_generate_and_rm(args)
    finally:
        stop_process(sglang_process)


if __name__ == "__main__":
    asyncio.run(main())
