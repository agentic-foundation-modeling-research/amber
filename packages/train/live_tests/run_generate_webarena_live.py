#!/usr/bin/env python3
"""Run one real WebArena no-TITO generate() call against live services.

This script does not mock model or environment calls. It can either connect to
an already-running SGLang service, or start a local SGLang subprocess for the
duration of the run. Unlike run_generate_webarena_live.py, it does not use the
Miles session server because generate_webarena.py calls SGLang
directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import requests
import yaml


THIS_FILE = Path(__file__).resolve()
TRAIN_PKG_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]

for path in (REPO_ROOT, TRAIN_PKG_DIR):
    sys.path.insert(0, str(path))


DEFAULT_TASK_CONFIG = REPO_ROOT / "task_configs" / "webarena_train.json"
DEFAULT_CUSTOM_CONFIG = (
    REPO_ROOT / "packages" / "train" / "training_configs" / "rl" / "append_memory.yaml"
)
SETUP_SERVER_SERVICES = {"shopping", "shopping_admin", "gitlab", "reddit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call examples/webarena/generate_webarena.py::generate() once with live servers.",
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
        "--port-base",
        type=int,
        default=None,
        help=(
            "Legacy base port used to allocate mutable task sites; each selected site uses base + 1, "
            "base + 2, etc. Ignored when --port-bases or webarena_port_bases is provided."
        ),
    )
    parser.add_argument(
        "--port-bases",
        type=int,
        nargs="+",
        default=None,
        metavar="PORT",
        help="Per-site mutable task port bases. Defaults to webarena_port_bases from --custom-config-path.",
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
        help="Start a local SGLang server subprocess before calling generate(). If unset, an existing backend is assumed.",
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


def _first_configured_vm(args: argparse.Namespace) -> str | None:
    vms = config_value(args, "webarena_vms") or []
    return vms[0] if vms else None


def _arg_or_config(args: argparse.Namespace, attr: str, config_name: str, default: Any = None) -> Any:
    value = getattr(args, attr, None)
    if value is not None:
        return value
    configured = config_value(args, config_name, None)
    return configured if configured is not None else default


def resolve_runner_args(args: argparse.Namespace, *, default_env_timeout_s: float = 30) -> None:
    args.custom_config = load_custom_config(args.custom_config_path)

    args.env_server_url = _arg_or_config(args, "env_server_url", "webarena_env_server_url")
    args.website_host = getattr(args, "website_host", None) or _first_configured_vm(args)
    args.setup_server_port = int(_arg_or_config(args, "setup_server_port", "webarena_setup_server_port", 7565))
    args.homepage_port = int(_arg_or_config(args, "homepage_port", "webarena_homepage_port", 7564))

    if hasattr(args, "port_bases") and getattr(args, "port_bases", None) is None:
        configured_port_bases = config_value(args, "webarena_port_bases", None)
        if configured_port_bases is not None and getattr(args, "port_base", None) is None:
            args.port_bases = list(configured_port_bases)
        elif not hasattr(args, "port_base"):
            args.port_bases = [8081, 9081]
    if hasattr(args, "site_port_stride"):
        args.site_port_stride = int(_arg_or_config(args, "site_port_stride", "webarena_site_port_stride", 1))
    if hasattr(args, "port_base"):
        args.port_base = _arg_or_config(args, "port_base", "webarena_legacy_port_base", None)
        if args.port_base is None and not getattr(args, "port_bases", None):
            args.port_base = 8080

    args.webarena_max_steps = int(_arg_or_config(args, "webarena_max_steps", "webarena_max_steps", 1))
    args.webarena_env_timeout_s = float(
        _arg_or_config(args, "webarena_env_timeout_s", "webarena_env_timeout_s", default_env_timeout_s)
    )
    args.webarena_debug_loss_mask = bool(
        _arg_or_config(args, "webarena_debug_loss_mask", "webarena_debug_loss_mask", False)
    )


def validate_runner_args(args: argparse.Namespace) -> None:
    if not args.env_server_url:
        raise ValueError("Provide --env-server-url or webarena_env_server_url in --custom-config-path")
    if not args.website_host:
        raise ValueError("Provide --website-host or non-empty webarena_vms in --custom-config-path")
    if hasattr(args, "port_bases") and getattr(args, "port_bases", None):
        if any(port_base < 1 for port_base in args.port_bases):
            raise ValueError("--port-bases / webarena_port_bases values must be >= 1")
    if hasattr(args, "site_port_stride") and args.site_port_stride < 1:
        raise ValueError("--site-port-stride / webarena_site_port_stride must be >= 1")


def load_task_config(path: Path, task_id: int) -> dict[str, Any]:
    with path.open() as f:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]

    task_configs = [row.get("metadata", row) for row in rows]
    for task_config in task_configs:
        if task_config.get("task_id") == task_id:
            return task_config
    raise ValueError(f"Task id {task_id} not found in {path}")


def build_rollout_metadata(
    task_config: dict[str, Any],
    *,
    website_host: str,
    setup_server_port: int,
    homepage_port: int,
    port_base: int,
    site_url_templates: dict[str, str],
    format_site_url: Any,
) -> dict[str, Any]:
    sites = task_config.get("sites") or []
    unsupported = sorted(set(sites) - SETUP_SERVER_SERVICES)
    if unsupported:
        raise ValueError(
            "This live runner only supports setup-server-managed sites "
            f"{sorted(SETUP_SERVER_SERVICES)}. Selected task uses {unsupported}."
        )

    site_ports = {site: port_base + index + 1 for index, site in enumerate(sites)}
    site_urls = {
        site: format_site_url(site_url_templates[site], website_host, site_ports[site])
        for site in sites
    }
    return {
        "site_ports": site_ports,
        "site_urls": site_urls,
        "setup_server_url": f"http://{website_host}:{setup_server_port}",
        "homepage_url": format_site_url(site_url_templates["homepage"], website_host, homepage_port),
        "calculator_url": format_site_url(site_url_templates["calculator"], website_host, homepage_port),
    }


def build_miles_args(args: argparse.Namespace, task_config: dict[str, Any]) -> Namespace:
    miles_args = Namespace(
        env_server_url=args.env_server_url,
        task_id=task_config["task_id"],
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
        webarena_max_steps=args.webarena_max_steps,
        webarena_env_timeout_s=args.webarena_env_timeout_s,
        webarena_debug_loss_mask=args.webarena_debug_loss_mask,
        sglang_router_ip=args.sglang_host,
        sglang_router_port=args.sglang_port,
        sglang_server_concurrency=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=None,
        sglang_enable_deterministic_inference=False,
        rollout_seed=args.seed,
        rollout_temperature=args.temperature,
        rollout_top_p=args.top_p,
        rollout_top_k=args.top_k,
        rollout_max_response_len=args.max_new_tokens,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        n_samples_per_prompt=1,
        use_distributed_post=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        num_layers=0,
        moe_router_topk=0,
    )
    for key, value in getattr(args, "custom_config", {}).items():
        if not hasattr(miles_args, key):
            setattr(miles_args, key, value)
    return miles_args


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }


def assert_generate_state_args(args: Namespace) -> None:
    required = [
        "hf_checkpoint",
        "chat_template_path",
        "sglang_router_ip",
        "sglang_router_port",
        "sglang_server_concurrency",
        "rollout_num_gpus",
        "rollout_num_gpus_per_engine",
        "rollout_temperature",
        "rollout_top_p",
        "rollout_top_k",
        "rollout_max_response_len",
        "rollout_stop",
        "rollout_stop_token_ids",
        "rollout_skip_special_tokens",
        "sglang_dp_size",
        "use_distributed_post",
    ]
    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise RuntimeError(
            "Internal runner bug: Miles GenerateState args are incomplete. "
            f"Missing: {missing}. Present args: {sorted(vars(args))}"
        )


def start_sglang(args: argparse.Namespace) -> subprocess.Popen:
    args.sglang_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.sglang_log.open("w")
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.hf_checkpoint,
        "--host",
        args.sglang_host,
        "--port",
        str(args.sglang_port),
        "--trust-remote-code",
        *args.sglang_extra_arg,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    logging.info("Starting SGLang: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    process._webarena_log_file = log_file  # type: ignore[attr-defined]
    wait_for_http_ok(
        f"http://{args.sglang_host}:{args.sglang_port}/health_generate",
        process=process,
        timeout=args.sglang_startup_timeout,
        log_path=args.sglang_log,
        name="SGLang",
    )
    return process


def wait_for_http_ok(
    url: str,
    *,
    process: subprocess.Popen | None,
    timeout: float,
    log_path: Path,
    name: str,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{name} exited early with code {process.returncode}. Log tail:\n{tail(log_path)}")
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logging.info("%s ready at %s", name, url)
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {name} at {url}. Last error: {last_error}. Log tail:\n{tail(log_path)}")


def tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    logging.info("Stopping SGLang server")
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    log_file = getattr(process, "_webarena_log_file", None)
    if log_file is not None:
        log_file.close()


def sample_to_json(sample: Any) -> dict[str, Any]:
    data = sample.to_dict()
    data["status"] = sample.status.value
    return data


async def run_generate(args: argparse.Namespace) -> None:
    from packages.train.generate_webarena import SITE_URL_TEMPLATES, format_url, generate
    from miles.utils.http_utils import init_http_client
    from miles.utils.types import Sample

    task_config = dict(load_task_config(args.task_config, args.task_id))
    task_config["rollout_metadata"] = build_rollout_metadata(
        task_config,
        website_host=args.website_host,
        setup_server_port=args.setup_server_port,
        homepage_port=args.homepage_port,
        port_base=args.port_base if args.port_base is not None else args.port_bases[0],
        site_url_templates=SITE_URL_TEMPLATES,
        format_site_url=format_url,
    )
    if args.port_bases and args.port_base is None:
        sites = task_config.get("sites") or []
        if len(sites) > len(args.port_bases):
            raise ValueError(
                f"--port-bases provided {len(args.port_bases)} per-site base ports, "
                f"but selected task has {len(sites)} sites"
            )
        site_ports = {
            site: args.port_bases[site_index]
            for site_index, site in enumerate(sites)
        }
        task_config["rollout_metadata"]["site_ports"] = site_ports
        task_config["rollout_metadata"]["site_urls"] = {
            site: format_url(SITE_URL_TEMPLATES[site], args.website_host, port)
            for site, port in site_ports.items()
        }

    sample = Sample(index=0, group_index=0, prompt=task_config.get("intent", ""), metadata=task_config)
    miles_args = build_miles_args(args, task_config)

    print(
        "Running live no-TITO generate() for "
        f"task_id={task_config['task_id']} sites={task_config.get('sites')} "
        f"max_steps={args.webarena_max_steps}"
    )
    print(f"rollout_metadata={json.dumps(task_config['rollout_metadata'], sort_keys=True)}")

    assert_generate_state_args(miles_args)
    init_http_client(miles_args)
    samples = await generate(miles_args, sample, build_sampling_params(args))
    print(f"generate() returned {len(samples)} sample(s)")
    for index, returned in enumerate(samples):
        print(
            f"[{index}] status={returned.status.value} "
            f"response_length={returned.response_length} "
            f"tokens={len(returned.tokens)} "
            f"loss_mask={len(returned.loss_mask or [])} "
            f"rollout_log_probs={len(returned.rollout_log_probs or [])} "
            # f"metadata={json.dumps(returned.metadata, sort_keys=True, default=str)}"
        )

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as f:
            for returned in samples:
                f.write(json.dumps(sample_to_json(returned), default=str) + "\n")
        print(f"Wrote returned samples to {args.output_jsonl}")


async def main() -> None:
    args = parse_args()
    resolve_runner_args(args)
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
        await run_generate(args)
    finally:
        stop_process(sglang_process)


if __name__ == "__main__":
    asyncio.run(main())
