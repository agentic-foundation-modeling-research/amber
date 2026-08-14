import logging
import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from argparse import Namespace
from collections.abc import Callable
from typing import Any, Tuple
from tqdm import tqdm
from copy import deepcopy
import torch

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput, MetricGatherer, call_dynamic_filter
from miles.rollout.sglang_rollout import GenerateState, eval_rollout, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.http_utils import post
from miles.utils.types import Sample
from miles.utils import dumper_utils
from miles.utils.misc import load_function

from browsergym.core.action.highlevel import HighLevelActionSet
from context_scythe.environment import (
    AsyncRemoteRolloutEnv,
    EnvConfig,
    setup_env,
    teardown_env,
)
from context_scythe.environment.const import SITE_URL_TEMPLATES
from context_scythe.agents import (
    TrajectoryData,
    StepData,
    Observation,
    Response,
    SingleTurnPromptBuilder,
    ReasoningParseError,
    ActionParseError,
)
from tito import Qwen3TITO

logger = logging.getLogger(__name__)

FIXED_WEBARENA_SITES = frozenset({"homepage", "calculator", "map", "wikipedia"})


def format_url(url: str, host, port):
    """Format a WebArena URL template with the given VM host and site port."""
    return url.format(host=host, port=port)


@dataclass(frozen=True)
class WebArenaVM:
    host: str
    setup_server_port: int
    homepage_port: int
    calculator_port: int
    map_port: int
    wikipedia_port: int
    mutable_port_bases: list[int]
    site_port_stride: int = 1

    def fixed_port_for_site(self, site: str) -> int:
        """Return the configured static port for a fixed WebArena site."""
        if site == "homepage":
            return self.homepage_port
        if site == "calculator":
            return self.calculator_port
        if site == "map":
            return self.map_port
        if site == "wikipedia":
            return self.wikipedia_port
        raise ValueError(f"WebArena site {site!r} is not fixed")

    def mutable_port_for_site(self, site: str, sample_offset: int, mutable_site_index: int) -> int:
        """Return the per-sample mutable site port for a leased VM."""
        if mutable_site_index >= len(self.mutable_port_bases):
            raise ValueError(
                f"Task needs mutable port base index {mutable_site_index} for site {site!r}, "
                f"but VM {self.host} only has {len(self.mutable_port_bases)} mutable port base(s)"
            )
        return self.mutable_port_bases[mutable_site_index] + sample_offset * self.site_port_stride


class WebArenaVMPool:
    def __init__(self, vms: list[WebArenaVM]):
        """Create a FIFO pool from available WebArena VM descriptors."""
        if not vms:
            raise ValueError("WebArena rollout requires at least one VM")
        self._queue = asyncio.Queue()
        for vm in vms:
            self._queue.put_nowait(vm)

    async def acquire(self) -> WebArenaVM:
        """Wait for and return the next available WebArena VM."""
        return await self._queue.get()

    def release(self, vm: WebArenaVM) -> None:
        """Return a previously acquired WebArena VM to the pool."""
        self._queue.put_nowait(vm)

    @property
    def available_count(self) -> int:
        """Return the number of VMs currently available for lease."""
        return self._queue.qsize()

    @asynccontextmanager
    async def lease(self):
        """Lease a VM for an async context and always release it afterward."""
        vm = await self.acquire()
        try:
            yield vm
        finally:
            self.release(vm)


def _get_arg(args: Namespace, name: str, default: Any = None) -> Any:
    """Read an argparse namespace value with a fallback default."""
    return getattr(args, name, default)


def _shared_mutable_port_bases(args: Namespace) -> list[int]:
    """Load and validate shared mutable WebArena site port bases from args."""
    port_bases = _get_arg(args, "webarena_port_bases", None)
    if port_bases is None:
        raise ValueError("WebArena rollout requires top-level webarena_port_bases")
    if len(port_bases) < 1:
        raise ValueError("WebArena rollout requires at least one webarena_port_bases value")
    return [int(port_base) for port_base in port_bases]


def _vm_from_host(args: Namespace, host: str, mutable_port_bases: list[int]) -> WebArenaVM:
    """Build a WebArena VM descriptor for one host using rollout arguments."""
    if not host:
        raise ValueError("WebArena VM hosts must be non-empty strings")

    homepage_port = _get_arg(args, "webarena_homepage_port", 7564)

    return WebArenaVM(
        host=str(host),
        setup_server_port=int(_get_arg(args, "webarena_setup_server_port", 7565)),
        homepage_port=int(homepage_port),
        calculator_port=int(_get_arg(args, "webarena_calculator_port", homepage_port)),
        map_port=int(_get_arg(args, "webarena_map_port", 443)),
        wikipedia_port=int(_get_arg(args, "webarena_wikipedia_port", 444)),
        mutable_port_bases=mutable_port_bases,
        site_port_stride=int(_get_arg(args, "webarena_site_port_stride", 1)),
    )


def build_webarena_vm_pool(args: Namespace) -> WebArenaVMPool:
    """Construct the rollout VM pool from WebArena host and port settings."""
    hosts = _get_arg(args, "webarena_vms", None)
    if not hosts:
        raise ValueError("WebArena rollout requires top-level webarena_vms")
    mutable_port_bases = _shared_mutable_port_bases(args)
    return WebArenaVMPool([_vm_from_host(args, host, mutable_port_bases) for host in hosts])


def _sites_for_task(task_config: dict[str, Any]) -> list[str]:
    """Return the WebArena site names declared by a task config."""
    sites = task_config.get("sites") or []
    return [str(site) for site in sites]


def validate_webarena_group_task_ids(group: list[Sample]) -> int:
    """Validate that a prompt group contains samples for exactly one task ID."""
    if not group:
        raise ValueError("WebArena prompt group must contain at least one sample")

    task_ids = [sample.metadata.get("task_id") for sample in group]
    missing_indices = [sample.index for sample, task_id in zip(group, task_ids, strict=True) if task_id is None]
    if missing_indices:
        raise ValueError(f"WebArena samples missing metadata['task_id']: sample_indices={missing_indices}")

    unique_task_ids = set(task_ids)
    if len(unique_task_ids) != 1:
        sample_task_ids = {
            sample.index: task_id
            for sample, task_id in zip(group, task_ids, strict=True)
        }
        raise ValueError(
            "WebArena prompt group must contain exactly one task_id, "
            f"got sample_task_ids={sample_task_ids}"
        )

    return task_ids[0]


def assign_webarena_rollout_metadata(group: list[Sample], vm: WebArenaVM) -> None:
    """Attach VM-specific site URLs and ports to each sample in a prompt group."""
    for sample_offset, sample in enumerate(group):
        task_config = sample.metadata
        sites = _sites_for_task(task_config)
        site_ports = {}
        mutable_site_index = 0
        for site in sites:
            if site not in SITE_URL_TEMPLATES:
                raise ValueError(f"Unsupported WebArena site {site!r}")
            if site in FIXED_WEBARENA_SITES:
                site_ports[site] = vm.fixed_port_for_site(site)
            else:
                site_ports[site] = vm.mutable_port_for_site(site, sample_offset, mutable_site_index)
                mutable_site_index += 1
        site_urls = {
            site: format_url(SITE_URL_TEMPLATES[site], vm.host, port)
            for site, port in site_ports.items()
        }
        mutable_site_ports = {
            site: port
            for site, port in site_ports.items()
            if site not in FIXED_WEBARENA_SITES
        }
        mutable_site_urls = {
            site: site_urls[site]
            for site in mutable_site_ports
        }
        rollout_metadata = dict(task_config.get("rollout_metadata") or {})
        rollout_metadata.update({
            "vm_host": vm.host,
            "site_ports": site_ports,
            "site_urls": site_urls,
            "mutable_site_ports": mutable_site_ports,
            "mutable_site_urls": mutable_site_urls,
            "setup_server_url": format_url("http://{host}:{port}", vm.host, vm.setup_server_port),
            "homepage_url": format_url(SITE_URL_TEMPLATES["homepage"], vm.host, vm.homepage_port),
            "calculator_url": format_url(SITE_URL_TEMPLATES["calculator"], vm.host, vm.calculator_port),
        })
        task_config["rollout_metadata"] = rollout_metadata


async def _generate_webarena_group_on_vm(
    args: Namespace,
    state: GenerateState,
    group: list[Sample],
    vm: WebArenaVM,
) -> list[Sample]:
    """Prepare a WebArena group on one VM and run generation plus reward modeling."""
    task_id = validate_webarena_group_task_ids(group)
    assign_webarena_rollout_metadata(group, vm)
    logger.info(
        "Prepared WebArena prompt group: group_index=%s task_id=%s sample_indices=%s vm_host=%s site_ports=%s",
        group[0].group_index,
        task_id,
        [sample.index for sample in group],
        vm.host,
        [sample.metadata["rollout_metadata"]["site_ports"] for sample in group],
    )
    return await generate_and_rm_group(
        args,
        group,
        sampling_params=state.sampling_params.copy(),
        evaluation=False,
    )


def _submit_webarena_group_task(
    args: Namespace,
    state: GenerateState,
    pool: WebArenaVMPool,
    group: list[Sample],
    vm: WebArenaVM,
) -> asyncio.Task:
    """Submit one WebArena group task and release its VM when it finishes."""
    async def _run_with_vm() -> list[Sample]:
        """Run generation for the leased VM and return it to the pool."""
        try:
            return await _generate_webarena_group_on_vm(args, state, group, vm)
        finally:
            pool.release(vm)

    return asyncio.create_task(_run_with_vm())


async def generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Generate a training rollout batch while bounding concurrency by VM leases."""
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    state = GenerateState(args)
    pool = build_webarena_vm_pool(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    target_data_size = args.rollout_batch_size
    data = []
    all_data = []
    pendings = set()
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="WebArena rollout generation")

    try:
        while len(data) < target_data_size:
            # Unlike the default Miles rollout, WebArena capacity is bounded by
            # leased website VMs. Each pending task below owns exactly one VM
            # and exactly one prompt group until the group finishes.
            while len(data) + len(pendings) < target_data_size and pool.available_count > 0:
                vm = await pool.acquire()
                # Pull one prompt group per leased VM. Pulling a larger
                # over-sampling batch would require assigning additional VMs.
                groups = data_source(1)
                if len(groups) != 1:
                    pool.release(vm)
                    raise ValueError(f"Expected data_source(1) to return one WebArena group, got {len(groups)}")
                for group in groups:
                    pendings.add(_submit_webarena_group_task(args, state, pool, group, vm))

            # This can happen only if there is no active rollout and no VM is
            # currently visible as available; yield instead of busy spinning.
            if not pendings:
                await asyncio.sleep(0.01)
                continue

            done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                group: list[Sample] = task.result()

                if do_print:
                    sample = group[0][0] if isinstance(group[0], list) else group[0]
                    logger.info(
                        f"First WebArena rollout sample: {[str(sample.prompt) + sample.response]}, "
                        f"label: {str(sample.label)[:100]}, reward: {sample.reward}",
                    )
                    do_print = False

                assert len(group) == args.n_samples_per_prompt
                all_data.append(group)
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
                if not dynamic_filter_output.keep:
                    metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                    continue

                # Keep the same accepted-group contract as Miles: data contains
                # rollout_batch_size groups, each with n_samples_per_prompt
                # completed samples for one original prompt.
                if len(data) < target_data_size:
                    data.append(group)
                    pbar.update(args.n_samples_per_prompt)
    finally:
        pbar.close()

    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish WebArena rollout: {[str(sample.prompt) + sample.response]}, "
        f"label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # Reset GenerateState just like the default rollout path so later rollout
    # or eval calls do not inherit pending/aborted state.
    state.reset()
    if (x := args.rollout_sample_filter_path) is not None:
        filter_func = load_function(x)
        filter_func(args, data)

    if (x := args.rollout_all_samples_process_path) is not None:
        process_func = load_function(x)
        process_func(args, all_samples, data_source)

    _stamp_group_rewards(args, data)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), []


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Run WebArena rollout generation or delegate to the evaluation rollout path."""
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    output_samples = list(_iter_samples(output.samples))
    aborted_output_samples = list(_iter_samples(aborted_samples))
    sample_token_counts = [
        len(sample.tokens)
        for sample in output_samples + aborted_output_samples
    ]
    total_tokens = sum(sample_token_counts)
    min_tokens = min(sample_token_counts, default=0)
    max_tokens = max(sample_token_counts, default=0)
    mean_tokens = total_tokens / len(sample_token_counts) if sample_token_counts else 0.0
    logger.info(
        "WebArena rollout total tokens: total_tokens=%s min_tokens=%s max_tokens=%s mean_tokens=%.2f sample_count=%s aborted_sample_count=%s",
        total_tokens,
        min_tokens,
        max_tokens,
        mean_tokens,
        len(output_samples) + len(aborted_output_samples),
        len(aborted_output_samples),
    )
    return output


def _iter_samples(value: Any):
    """Yield Sample instances recursively from nested sample containers."""
    if isinstance(value, Sample):
        yield value
        return
    if isinstance(value, list | tuple):
        for item in value:
            yield from _iter_samples(item)


def _episode_key(sample: Sample) -> tuple[int | None, int | None]:
    """Return the key that identifies one WebArena episode."""
    return sample.group_index, sample.index


def _terminal_reward(args: Namespace, sample: Sample) -> float:
    """Resolve the scalar terminal reward stored on a WebArena sample."""
    reward = sample.metadata.get("final_reward")
    if reward is None:
        reward = sample.metadata.get("episode_reward")
    if reward is None:
        reward = sample.get_reward_value(args)
    if reward is None:
        raise ValueError(
            f"Missing WebArena reward for sample index={sample.index}, group_index={sample.group_index}"
        )
    return float(reward)


def _stamp_group_rewards(args: Namespace, data: list[list[Any]]) -> None:
    """Store per-episode reward maps on every sample in accepted rollout groups."""
    for group in data:
        episodes = [list(_iter_samples(episode)) for episode in group]
        if any(not episode for episode in episodes):
            raise ValueError("WebArena rollout group contains an empty episode")

        group_episode_rewards: dict[int | None, float] = {}
        group_final_rewards: dict[int | None, float] = {}
        group_index = episodes[0][-1].group_index
        for episode in episodes:
            terminal_sample = episode[-1]
            terminal_group_index, terminal_index = _episode_key(terminal_sample)
            if terminal_group_index != group_index:
                raise ValueError(
                    f"WebArena rollout group mixes group_index={group_index} "
                    f"and group_index={terminal_group_index}"
                )
            if terminal_index in group_episode_rewards:
                raise ValueError(
                    f"Duplicate WebArena episode index={terminal_index} "
                    f"for group_index={terminal_group_index}"
                )
            group_episode_rewards[terminal_index] = terminal_sample.metadata.get("episode_reward")
            group_final_rewards[terminal_index] = terminal_sample.metadata.get("final_reward")

        if len(group_episode_rewards) != args.n_samples_per_prompt:
            raise ValueError(
                f"Expected {args.n_samples_per_prompt} episodes for group_index={group_index}, "
                f"got {len(group_episode_rewards)}"
            )

        for episode in episodes:
            for sample in episode:
                sample.metadata["group_episode_rewards"] = dict(group_episode_rewards)
                sample.metadata["group_final_rewards"] = dict(group_final_rewards)


def build_tokens_and_mask_from_messages(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int], str, int]:
    """Tokenize chat messages and build a response-only assistant loss mask."""

    if not messages or len(messages) < 2:
        return [], [], "", 0

    # Structure: system, user, assistant, user, assistant
    # First two messages are always 
    prompt_msgs = messages[:2]
    response_msgs = messages[2:]

    prompt_tokens = []
    for msg in prompt_msgs:
        content = msg.get("content", "")
        if content:
            prompt_tokens.extend(tokenizer(content, add_special_tokens=False)["input_ids"])

    response_tokens = []
    loss_mask = []
    response_text_parts = []

    for msg in response_msgs:
        content = msg.get("content", "")
        if not content:
            continue

        tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        token_len = len(tokens)

        response_tokens.extend(tokens)
        response_text_parts.append(content)

        mask_val = 1 if msg.get("role") == "assistant" else 0
        loss_mask.extend([mask_val] * token_len)

    all_tokens = prompt_tokens + response_tokens
    response_text = "".join(response_text_parts)
    response_length = len(response_tokens)

    return all_tokens, loss_mask, response_text, response_length


def _env_server_url(args: Namespace) -> str:
    """Return the configured WebArena environment server URL."""
    url = getattr(args, "env_server_url", None) or getattr(args, "webarena_env_server_url", None)
    if not url:
        raise ValueError("WebArena requires --env-server-url or webarena_env_server_url in --custom-config-path")
    return url


def _restore_prefilled_think_opening(content: str) -> str:
    """Return the logical assistant text when the template prefilled ``<think>``.

    SGLang returns generated assistant content, not prompt-prefill text.  If the
    chat template ends its generation prompt with ``<think>\n``, the returned
    content may start with reasoning and then ``</think>``.  The WebArena parser
    expects complete ``<think>...</think>`` blocks, so restore the opening tag
    for local parsing/history only.  Do not use this normalized text for TITO
    request history; the session server stores SGLang's raw ``message.content``.
    """
    first_close = content.find("</think>")
    if first_close == -1:
        return content

    first_open = content.find("<think>")
    if first_open != -1 and first_open < first_close:
        return content

    return "<think>\n" + content.lstrip("\n")


def _aborted_step_sample(sample: Sample, step_num: int, error: str) -> Sample:
    """Return an aborted copy of a sample annotated with generation failure metadata."""
    aborted_sample = deepcopy(sample)
    aborted_sample.status = Sample.Status.ABORTED
    aborted_sample.metadata.update({
        "step_num": step_num,
        "generation_error": error,
    })
    return aborted_sample


def _sampling_params_for_generate(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Normalize sampling parameters for direct SGLang ``/generate`` calls."""
    params = dict(sampling_params)
    if "max_tokens" in params and "max_new_tokens" not in params:
        params["max_new_tokens"] = params.pop("max_tokens")
    params.pop("model", None)
    params.pop("messages", None)
    return {key: value for key, value in params.items() if value is not None}


def _chat_template_kwargs(args: Namespace, *, prefill_think: bool | None) -> dict[str, Any]:
    """Build chat-template keyword arguments with optional think-prefill override."""
    kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
    if prefill_think is not None:
        kwargs["prefill_think"] = prefill_think
    return kwargs


async def _generate(
    args: Namespace,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    """Call SGLang ``/generate`` and return decoded output plus token logprobs."""
    request_body = {
        "input_ids": prompt_ids,
        "sampling_params": _sampling_params_for_generate(sampling_params),
        "return_logprob": True,
        "return_routed_experts": getattr(args, "use_rollout_routing_replay", False),
        "return_indexer_topk": getattr(args, "use_rollout_indexer_replay", False),
    }
    response = await post(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate", request_body)
    token_logprobs = response.get("meta_info", {}).get("output_token_logprobs")
    if token_logprobs is None:
        raise RuntimeError("SGLang /generate response did not include meta_info.output_token_logprobs")

    return {
        "prompt_ids": prompt_ids, # list[int]
        "content": response["text"], # str, decoded response
        "output_ids": [item[1] for item in token_logprobs], # list[int]
        "output_logprobs": [item[0] for item in token_logprobs], # list[float]
        # id, finish_reason, prompt_tokens(len), weight_version,
        # input_token_logprobs, output_token_logprobs,
        # output_token_logprobs_length, reasning_tokens,
        # completion_tokens, cached_tokens, cached_tokens_details
        "meta_info": response.get("meta_info", {}),
    }


def pretty_print(response):
    """Print the main fields from a direct generation response."""
    args_to_print = ["content", "output_ids", "output_logprobs", "meta_info"]
    print_str = "\n".join([
        f"{k}: {v}" for k, v in response.items() if k in args_to_print
    ])
    print(print_str)


def trunc_print_messages(messages: list[dict[str, Any]], max_chars=300):
    """Pretty-print chat messages by role, truncating long content."""
    for message in messages:
        role = message.get("role", "")
        content = str(message.get("content", ""))
        if max_chars is not None and len(content) > max_chars:
            head_chars = max_chars // 4
            tail_chars = max_chars - head_chars
            content = content[:head_chars] + "..." + content[-tail_chars:]
        print(repr(f"{role}: {content}"))
        print("="*10)


def _build_step_sample(
    sample: Sample,
    state: GenerateState,
    action_turn: dict[str, Any],
    *,
    step_num: int,
    action: str,
    parse_error: str | None,
    generation_error: str | None,
) -> Sample:
    """Build one trainable sample from an action generation."""
    step_sample = deepcopy(sample)

    action_output_ids = action_turn["output_ids"]
    step_sample.tokens = action_turn["prompt_ids"] + action_output_ids
    step_sample.response_length = len(action_output_ids)
    step_sample.loss_mask = [1] * step_sample.response_length
    step_sample.rollout_log_probs = action_turn["output_logprobs"]
    step_sample.response = state.tokenizer.decode(action_output_ids)

    assert len(step_sample.loss_mask) == step_sample.response_length
    assert len(step_sample.rollout_log_probs) == step_sample.response_length
    assert step_sample.tokens[len(action_turn["prompt_ids"]):] == action_output_ids

    finish_reason = action_turn["meta_info"].get("finish_reason", {}).get("type")
    if finish_reason == "abort":
        step_sample.status = Sample.Status.ABORTED
    elif finish_reason == "length":
        step_sample.status = Sample.Status.TRUNCATED
    else:
        step_sample.status = Sample.Status.COMPLETED

    meta_info = action_turn["meta_info"]
    step_sample.prefix_cache_info.add(meta_info)
    if "weight_version" in meta_info:
        step_sample.weight_versions.append(meta_info["weight_version"])

    step_sample.metadata.update({
        "step_num": step_num,
        "action": action,
        "parse_error": parse_error,
        "generation_error": generation_error,
        "loss_mask_segments": {
            "action_output_len": len(action_output_ids),
        },
    })
    return step_sample


def _webarena_eval_log_details(task_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact eval metadata for rollout completion logs."""
    eval_config = task_config.get("eval") or {}
    if not isinstance(eval_config, dict):
        return [{"eval_info": eval_config}]

    eval_types = eval_config.get("eval_types") or []
    if isinstance(eval_types, str):
        eval_types = [eval_types]

    details = []
    if "string_match" in eval_types:
        reference_answers = eval_config.get("reference_answers")
        if isinstance(reference_answers, dict):
            for match_method, reference_ans in reference_answers.items():
                details.append({
                    "eval_type": "string_match",
                    "match_method": match_method,
                    "reference_ans": reference_ans,
                })
        else:
            details.append({
                "eval_type": "string_match",
                "match_method": None,
                "reference_ans": reference_answers,
            })

    if "program_html" in eval_types:
        program_html = eval_config.get("program_html")
        if isinstance(program_html, list):
            for eval_program in program_html:
                details.append({
                    "eval_type": "program_html",
                    "eval_program": eval_program,
                })
        else:
            details.append({
                "eval_type": "program_html",
                "eval_program": program_html,
            })

    if "url_match" in eval_types:
        details.append({
            "eval_type": "url_match",
            "reference_url": eval_config.get("reference_url"),
            "url_note": eval_config.get("url_note"),
        })

    return details


async def agent_loop(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    env: AsyncRemoteRolloutEnv,
    goal: str,
    calculator_url: str,
    site_urls: dict[str, str],
    action_set: HighLevelActionSet,
    max_steps=30,
):
    """Run the browser-agent loop for one WebArena episode and collect step samples."""
    state = GenerateState(args)

    trajectory_data = TrajectoryData(
        goal,
        calculator_url=calculator_url,
        site_urls=site_urls
    )
    prompt_builder = SingleTurnPromptBuilder()

    stepwise_samples: list[Sample] = []
    error_counter = 0

    obs, info = await env.reset(42)
    for step_num in range(max_steps):
        # print("-"*50, f"Step {step_num}", "-"*50)
        logging.info(f"Index {sample.index}, Group Index: {sample.group_index}: step {step_num}")
        observation = Observation(
            axtree=obs["axtree"],
            viewport_state=obs["viewport_state"],
            extra_element_properties=obs["extra_element_properties"],
            open_pages_urls=obs["open_pages_urls"],
            open_pages_titles=obs["open_pages_titles"],
            active_page_index=obs["active_page_index"],
            last_action_error=obs["last_action_error"],
            screenshot=None,
            use_axtree=True,
            filter_viewport=args.webarena_filter_viewport,
            use_screenshot=False,
            use_tabs_info=args.webarena_use_tabs_info,
            axtree_max_tokens=args.webarena_axtree_max_tokens,
        )
        step_data = StepData(step_num, observation)
        response = None
        parse_error = None
        step_format_reward = 0.0
        action_messages = prompt_builder.build_messages(
            step_num,
            trajectory_data,
            action_set,
            current_step_data=step_data,
        )["prompt"]
        action_messages = prompt_builder.flatten_messages(action_messages)

        step_sample = deepcopy(sample)
        action_turn = None
        generation_error = None

        try:
            tito = Qwen3TITO(
                state.tokenizer,
                chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None) or {},
            )
            action_prompt_ids = tito.prepare(
                action_messages,
                template_kwargs={"prefill_think": True},
            )
            action_turn = await _generate(
                args,
                action_prompt_ids,
                sampling_params,
            )
            raw_response = action_turn["content"]
            logical_response = _restore_prefilled_think_opening(raw_response)
            parse_error = ""
            action_parse_error = ""

            # Assign format reward here since we're decoding here anyway
            # format reward: 0.5 for correct <think>
            #                0.5 for correct <action>
            try:
                possible_reasoning, action = Response.parse_action(logical_response, raise_on_action_parse_error=True)
                step_format_reward += 0.5
            except ActionParseError as e:
                possible_reasoning, action = Response.parse_action(logical_response, raise_on_action_parse_error=False)
                action_parse_error = str(e)
                parse_error = action_parse_error
                logger.info("Step %s action parsing failed; using noop(). Error: %s", step_num, action_parse_error)
                action = "noop()"

            try:
                reasoning = Response.parse_reasoning(possible_reasoning, raise_on_reasoning_parse_error=True)
                step_format_reward += 0.5
            except ReasoningParseError as e:
                reasoning_parse_error = str(e)
                parse_error = ", ".join(error for error in [action_parse_error, reasoning_parse_error] if error)
                reasoning = possible_reasoning
                logger.info("Step %s reasoning parsing failed. Error: %s", step_num, reasoning_parse_error)

            if parse_error:
                # If too many errors in parsing, generation is going wrong
                error_counter += 1
            
            response = Response(logical_response, reasoning=reasoning, action=action)
            step_data.response = response
            step_data.observation.last_action_error = parse_error
        except Exception as e:
            generation_error = str(e)
            response = Response("")
            response.action = "noop()"
            step_data.response = response
            step_data.observation.last_action_error = generation_error
            logger.warning("Step %s generation failed; using noop(). Error: %s", step_num, generation_error, exc_info=True)

        if action_turn is not None:
            merged_sample = _build_step_sample(
                step_sample,
                state,
                action_turn,
                step_num=step_num,
                action=response.action,
                parse_error=parse_error,
                generation_error=generation_error,
            )
        else:
            merged_sample = _aborted_step_sample(
                step_sample,
                step_num,
                generation_error or "direct SGLang action generation did not complete",
            )

        trajectory_data.add_step(step_data)

        try:
            obs, reward, terminated, truncated, info = await env.step(response.action)
        except Exception as exc:
            env_step_error = str(exc)
            logger.warning(
                "Step %s environment step failed; truncating trajectory. Error: %s",
                step_num,
                env_step_error,
                exc_info=True,
            )
            reward = 0.0
            terminated = False
            truncated = True
            merged_sample.metadata.update({
                "env_reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "format_reward": step_format_reward,
                "env_step_error": env_step_error,
            })
            stepwise_samples.append(merged_sample)
            trajectory_data.reward = reward
            trajectory_data.terminated = terminated
            trajectory_data.truncated = truncated
            break

        merged_sample.metadata.update({
            "env_reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "format_reward": step_format_reward
        })
        stepwise_samples.append(merged_sample)

        trajectory_data.reward = reward # Store the episode reward
        logging.info(f"Step {step_num} done. Action: {response.action}.")

        if terminated or truncated or error_counter > 2:
            logging.info(
                "Trajectory done. Task ID: %s, Step: %s, Action: %s, Reward: %s, Terminated: %s, Truncated: %s, Eval Info: %s",
                sample.metadata.get("task_id"),
                step_num,
                response.action,
                reward,
                terminated,
                truncated,
                _webarena_eval_log_details(sample.metadata),
            )
            trajectory_data.terminated = terminated
            trajectory_data.truncated = truncated or error_counter > 2
            break

    # Store final episode reward in each stepeise sample for reward model
    for sample in stepwise_samples:
        assert sample.reward is None, "Reward should be assigned by the rm"
        sample.metadata.update({"episode_reward": trajectory_data.reward})

    return stepwise_samples


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> list[Sample]:
    """
    Custom generation function for Web Agent integration.

    Orchestrates the interaction with the external Gym environment:
    1. Sends prompt/metadata to Gym.
    2. Receives execution trace (messages) and rewards.
    3. Formats data for Miles training format.

    Note: We cannot do in-place modification of the sample as we need to return multiple Sample
    instances per generation.
    """
    logging.info(f"Starting generation for Index {sample.index}, Group {sample.group_index}.")

    # Orchestrate the loop
    task_config = sample.metadata # Metadata stores the full task config
    rollout_metadata = task_config["rollout_metadata"]

    # Preassigned in generate_and_rm_group
    site_ports = rollout_metadata["site_ports"] # dict site -> port
    site_urls = rollout_metadata["site_urls"]
    mutable_site_ports = rollout_metadata.get("mutable_site_ports", site_ports)
    mutable_site_urls = rollout_metadata.get("mutable_site_urls", site_urls)
    setup_server_url = rollout_metadata["setup_server_url"]
    homepage_url = rollout_metadata["homepage_url"]
    calculator_url = rollout_metadata["calculator_url"]

    env_configs = []
    for site, url in mutable_site_urls.items():
        port = mutable_site_ports[site]
        env_config = EnvConfig(name=site, port=port, endpoint_url=url)
        env_configs.append(env_config)
    
    env = None
    is_setup = False
    try:
        # TODO: Make setup async native
        is_setup = await asyncio.to_thread(
            setup_env,
            setup_server_url=setup_server_url,
            env_configs=env_configs,
        )
        if not is_setup:
            # TODO: Handle this better.
            # mark generation for abort
            raise RuntimeError("Website server setup failed.")
        
        # Define the action set to be used
        action_set = HighLevelActionSet(["webarena"])
        
        env = AsyncRemoteRolloutEnv(
            server_url=_env_server_url(args),
            task_id=task_config["task_id"],
            task_config=task_config,
            homepage_url=homepage_url,
            site_urls=site_urls,
            seed=42,
            timeout_s=getattr(args, "webarena_env_timeout_s", 90),
        )
        
        stepwise_samples = await agent_loop(
            args,
            sample,
            sampling_params,
            env,
            task_config["intent"],
            calculator_url,
            site_urls,
            action_set,
            max_steps=args.webarena_max_steps,
        )
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                logger.exception(
                    "Ignoring WebArena remote env close failure for index=%s group_index=%s",
                    sample.index,
                    sample.group_index,
                )
        if is_setup:
            # TODO: make teardown async native
            await asyncio.to_thread(
                teardown_env,
                setup_server_url=setup_server_url,
                env_configs=env_configs
            )

    return stepwise_samples


async def reward_model(args: Namespace, stepwise_samples: list[Sample]):
    """Assign the final episode reward to every WebArena step sample."""
    
    # since generate returns a list[Sample], generate_and_rm will call
    # batched_async_rm.
    # If a custom rm is set, batch_async_rm passes samples directly to
    # the rm, i.e., list[Sample] will be passed directly instead of
    # iterating over it.
    # sample here is a list of stepwise_samples

    # episode reward is the same for all stepwise samples
    # episode reward is the outcome reward
    episode_reward = float(stepwise_samples[-1].metadata["episode_reward"])

    for sample in stepwise_samples:
        sample.metadata["final_reward"] = episode_reward

    return [episode_reward] * len(stepwise_samples)


def postprocess_reward(args: Namespace, samples: list[Sample]) -> Tuple[list[float], list[float]]:
    """
    Outcome reward normalization.

    Miles RolloutDataSource assigns the same ``group_index`` to all
    n_samples_per_prompt samples for one prompt, while ``index`` is globally
    unique per sample. WebArena expands one sample into multiple stepwise
    samples, so rewards are first collapsed per ``(group_index, index)``
    episode, normalized within each ``group_index``, and then broadcast back to
    every stepwise sample.
    """
    episode_samples: dict[tuple[int | None, int | None], list[Sample]] = {}
    for sample in samples:
        episode_samples.setdefault(_episode_key(sample), []).append(sample)

    # Multiple samples for one (group_index, index) combination are the
    # stepwise samples for one WebArena episode.
    episode_rewards = {
        key: stepwise_samples[-1].metadata.get("episode_reward")
        for key, stepwise_samples in episode_samples.items()
    }
    raw_rewards = [episode_rewards[(sample.group_index, sample.index)] for sample in samples]

    episode_rewards = {
        key: _terminal_reward(args, stepwise_samples[-1])
        for key, stepwise_samples in episode_samples.items()
    }

    should_normalize = (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
        and getattr(args, "n_samples_per_prompt", 1) > 1
    )
    if not should_normalize:
        return raw_rewards, raw_rewards

    normalized_by_episode: dict[tuple[int | None, int | None], float] = {}

    group_reward_maps: dict[int | None, dict[int | None, float]] = {}
    for (group_index, _), stepwise_samples in episode_samples.items():
        group_final_rewards = stepwise_samples[-1].metadata.get("group_final_rewards")
        if group_final_rewards is None:
            continue
        if not isinstance(group_final_rewards, dict):
            raise ValueError(
                f"Expected stored episode rewards for group_index={group_index} to be a dict, "
                f"got {type(group_final_rewards).__name__}"
            )
        if len(group_final_rewards) != args.n_samples_per_prompt:
            raise ValueError(
                f"Expected {args.n_samples_per_prompt} stored episode rewards for group_index={group_index}, "
                f"got {len(group_final_rewards)}"
            )
        if group_index in group_reward_maps and group_reward_maps[group_index] != group_final_rewards:
            raise ValueError(f"Inconsistent stored episode rewards for group_index={group_index}")
        group_reward_maps[group_index] = group_final_rewards

    groups_with_stored_rewards = set(group_reward_maps)
    for (group_index, index), reward in episode_rewards.items():
        if group_index not in groups_with_stored_rewards:
            group_reward_maps.setdefault(group_index, {})[index] = reward

    for group_index, reward_map in group_reward_maps.items():
        items = sorted(reward_map.items(), key=lambda item: item[0] if item[0] is not None else -1)
        if len(items) != args.n_samples_per_prompt:
            raise ValueError(
                f"Expected {args.n_samples_per_prompt} episodes for group_index={group_index}, got {len(items)}"
            )

        torch_rewards = torch.tensor([reward for _, reward in items], dtype=torch.float)
        torch_rewards = torch_rewards - torch_rewards.mean()

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            torch_rewards = torch_rewards / (torch_rewards.std() + 1e-6)

        for (index, _), normalized_reward in zip(items, torch_rewards.tolist(), strict=True):
            normalized_by_episode[(group_index, index)] = normalized_reward

    rewards = [normalized_by_episode[(sample.group_index, sample.index)] for sample in samples]
    return raw_rewards, rewards
