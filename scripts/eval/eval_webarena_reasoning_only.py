import argparse
import os
import sys
import logging
import json
import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

from browsergym.core.action.highlevel import HighLevelActionSet

from context_scythe.environment import (
    AsyncRemoteRolloutEnv,
    EnvConfig,
    setup_env,
    teardown_env,
)
from context_scythe.agents import (
    TrajectoryData,
    StepData,
    Observation,
    Response,
    SingleTurnPromptBuilder,
    BaseLLM,
    OpenAILLM,
    AnthropicLLM,
    ReasoningParseError,
    ActionParseError,
)
from context_scythe.environment.const import SITE_URL_TEMPLATES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    stream=sys.stdout,
    force=True,
)

DIR = Path(__file__).resolve().parent
DEFAULT_SETUP_SERVICE_PORT = 7565
DEFAULT_HOMEPAGE_SERVICE_PORT = 7564
DEFAULT_PORT_BASES = [8080, 9080]
DEFAULT_TASK_CONFIG_PATH = DIR / ".." / ".." / "task_configs" / "test_webarena_lite.json"
DEFAULT_MODEL = "qwen3.5-webarena-memory-sft"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_BASE_URL = "http://localhost:30000/v1"
DEFAULT_LLM_MAX_TOKENS = 1024
DEFAULT_LLM_TEMPERATURE = 1.0
DEFAULT_MAX_STEPS = 20
DEFAULT_SEED = 42
DEFAULT_TIMEOUT_S = 60
DEFAULT_PARSE_RETRIES = 5

MUTABLE_SITES = {"shopping", "shopping_admin", "reddit", "gitlab"}
FIXED_PORTS = {"map": 443, "wikipedia": 444}


@dataclass(frozen=True)
class RunnerConfig:
    env_server_service_url: str
    websites_vm_host: str
    setup_service_port: int
    homepage_service_port: int
    port_bases: list[int]
    task_id: int | None
    task_ids: list[int] | None
    all_tasks: bool
    parallelism: int
    task_config_path: Path
    group_size: int
    save_dir: Path
    model: str
    llm_provider: str
    llm_base_url: str | None
    llm_extra_body: dict | None
    llm_max_tokens: int
    llm_temperature: float
    max_steps: int
    save_screenshots: bool
    filter_viewport: bool
    use_tabs_info: bool
    axtree_max_tokens: int | None
    seed: int
    timeout_s: int
    parse_retries: int
    think_prefix: str


@dataclass(frozen=True)
class RolloutUrls:
    homepage_url: str
    calculator_url: str
    site_urls: dict[str, str]


def format_url(url: str, host, port):
    return url.format(host=host, port=port)


def parse_args(argv: list[str] | None = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Set up and test a single WebArena task.")
    parser.add_argument(
        "--env_server_service_url",
        required=True,
        help="URL for the env_server service.",
    )
    parser.add_argument(
        "--websites_vm_host",
        required=True,
        help="Host or IP address where the Websites VM services are reachable.",
    )
    parser.add_argument(
        "--setup_service_port",
        type=int,
        default=DEFAULT_SETUP_SERVICE_PORT,
        help="Port for the setup service.",
    )
    parser.add_argument(
        "--homepage_service_port",
        type=int,
        default=DEFAULT_HOMEPAGE_SERVICE_PORT,
        help="Port for the homepage service.",
    )
    parser.add_argument(
        "--port_bases",
        type=int,
        nargs="+",
        default=DEFAULT_PORT_BASES,
        help="Base ports for mutable rollout services. The script adds the rollout slot to each base.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Task id from the task config file.",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,
        help="Task ids from the task config file to run concurrently. Supports spaces or commas.",
    )
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="Run every task in the task config file once.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Maximum number of concurrent tasks in --all_tasks mode. Must be between 1 and 8.",
    )
    parser.add_argument(
        "--task_config_path",
        type=Path,
        default=DEFAULT_TASK_CONFIG_PATH,
        help="Path to the task config JSON file.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=2,
        help="Group size for rollouts.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory where trajectory JSON files will be saved.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to use.",
    )
    parser.add_argument(
        "--llm_provider",
        choices=("openai", "anthropic"),
        default=DEFAULT_LLM_PROVIDER,
        help="LLM API protocol to use.",
    )
    parser.add_argument(
        "--llm_base_url",
        default=None,
        help="LLM API base URL. Defaults to localhost for OpenAI and the provider environment for Anthropic.",
    )
    parser.add_argument(
        "--llm_extra_body",
        type=json.loads,
        default=None,
        help="JSON object forwarded as OpenAI client extra_body, e.g. '{\"chat_template_kwargs\":{\"clear_thinking\":false,\"prefill_think\":true}}'.",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=DEFAULT_LLM_MAX_TOKENS,
        help="Maximum completion tokens for each LLM call.",
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=DEFAULT_LLM_TEMPERATURE,
        help="Sampling temperature for OpenAI-compatible models.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of agent steps per rollout.",
    )
    parser.add_argument(
        "--save_screenshots",
        action="store_true",
        help="Request screenshots from env_server for saving alongside rollouts.",
    )
    parser.add_argument(
        "--filter_viewport",
        action="store_true",
        help="Filter the accessibility tree to the viewport.",
    )
    parser.add_argument(
        "--use_tabs_info",
        action="store_true",
        help="Include tabs information in observations.",
    )
    parser.add_argument(
        "--axtree_max_tokens",
        type=int,
        default=None,
        help="Maximum number of accessibility tree tokens to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for environment reset and rollout creation.",
    )
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Timeout in seconds for the remote rollout environment.",
    )
    parser.add_argument(
        "--parse_retries",
        type=int,
        default=DEFAULT_PARSE_RETRIES,
        help="Maximum attempts to parse a model action response.",
    )
    parser.add_argument(
        "--think_prefix",
        type=str,
        default="",
        help="Prefix for <think>",
    )
    args = parser.parse_args(argv)
    if args.llm_extra_body is not None and not isinstance(args.llm_extra_body, dict):
        parser.error("--llm_extra_body must be a JSON object.")
    args.task_ids = parse_task_ids(args.task_ids)
    selected_modes = sum(
        [
            args.task_id is not None,
            args.task_ids is not None,
            args.all_tasks,
        ]
    )
    if selected_modes != 1:
        parser.error("Specify exactly one of --task_id, --task_ids, or --all_tasks.")
    if args.parallelism < 1 or args.parallelism > 8:
        parser.error("--parallelism must be between 1 and 8.")
    return RunnerConfig(**vars(args))


def parse_task_ids(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None

    task_ids = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                task_ids.append(int(part))
    return task_ids


def load_task_configs(config_path: Path) -> list[dict]:
    with open(config_path, "r") as f:
        return json.load(f)


def load_task_config(config_path: Path, task_id: int) -> dict:
    task_configs = load_task_configs(config_path)

    try:
        return next(c for c in task_configs if c["task_id"] == task_id)
    except StopIteration as error:
        raise ValueError(f"Task id {task_id} not found in {config_path}") from error


def select_task_configs(task_configs: list[dict], task_ids: list[int]) -> list[dict]:
    configs_by_id = {task_config["task_id"]: task_config for task_config in task_configs}
    missing_task_ids = [
        task_id for task_id in task_ids if task_id not in configs_by_id
    ]
    if missing_task_ids:
        raise ValueError(f"Task ids not found: {missing_task_ids}")
    return [configs_by_id[task_id] for task_id in task_ids]


def env_configs_for_task(
    task_config: dict,
    websites_vm_host: str,
    port_bases: list[int],
    rollout_slot_index: int,
) -> list[EnvConfig]:
    sites = task_config["sites"]
    if len(port_bases) < len(sites):
        raise ValueError(
            f"Task needs {len(sites)} port bases for sites {sites}, got {len(port_bases)}"
        )

    env_configs = []
    rollout_slot = rollout_slot_index + 1
    for site, port_base in zip(sites, port_bases):
        if site in MUTABLE_SITES:
            port = port_base + rollout_slot
        else:
            port = FIXED_PORTS[site]
        url = format_url(SITE_URL_TEMPLATES[site], host=websites_vm_host, port=port)
        env_configs.append(EnvConfig(name=site, port=port, endpoint_url=url))

    return env_configs


def build_rollout_urls(config: RunnerConfig, env_configs: list[EnvConfig]) -> RolloutUrls:
    homepage_url = format_url(
        SITE_URL_TEMPLATES["homepage"],
        host=config.websites_vm_host,
        port=config.homepage_service_port,
    )
    site_urls = {
        env_config.name: format_url(
            SITE_URL_TEMPLATES[env_config.name],
            config.websites_vm_host,
            env_config.port,
        )
        for env_config in env_configs
    }
    calculator_url = format_url(
        SITE_URL_TEMPLATES["calculator"],
        host=config.websites_vm_host,
        port=config.homepage_service_port,
    )
    return RolloutUrls(
        homepage_url=homepage_url,
        calculator_url=calculator_url,
        site_urls=site_urls,
    )


def call_llm_and_parse_response_sync(llm: BaseLLM, messages: list, max_attempts: int, think_prefix: str = ""):
    raw_response = None

    for attempt in range(1, max_attempts + 1):
        raw_response, misc = llm(messages)

        # Append the <think> tag as the server prefills it for us
        raw_response = think_prefix + raw_response

        try:
            reasoning = Response.parse_reasoning(raw_response, True)
            _, action = Response.parse_action(raw_response, True)
            return Response(raw_response, reasoning=reasoning, action=action), misc
        except (ReasoningParseError, ActionParseError):
            logging.info(
                "Query failed. Retrying %s/%s.\n[LLM]:\n%s",
                attempt,
                max_attempts,
                raw_response,
                exc_info=True,
            )

    raise ActionParseError(f"Could not parse a valid action. Last response:\n{raw_response}")


async def call_llm_and_parse_response(llm: BaseLLM, messages: list, max_attempts: int, think_prefix: str = ""):
    return await asyncio.to_thread(
        call_llm_and_parse_response_sync,
        llm,
        messages,
        max_attempts,
        think_prefix=think_prefix
    )


async def call_llm(llm: BaseLLM, messages: list):
    return await asyncio.to_thread(llm, messages)


async def agent_loop(
    env: AsyncRemoteRolloutEnv,
    task_config: dict,
    calculator_url: str,
    site_urls: dict[str, str],
    llm: BaseLLM,
    action_set: HighLevelActionSet,
    max_steps: int,
    seed: int,
    parse_retries: int,
    think_prefix="",
    screenshot_dir: Path | None = None,
    **obs_kwargs,
):

    goal = task_config["intent"]

    trajectory_data = TrajectoryData(
        goal,
        calculator_url=calculator_url,
        site_urls=site_urls
    )
    prompt_builder_obj = SingleTurnPromptBuilder()

    obs, _ = await env.reset(seed)
    for step_num in range(max_steps):
        save_observation_screenshot(obs, screenshot_dir=screenshot_dir, step_num=step_num)
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
            use_screenshot=False,
            # obs_kwargs is expected to have:
            # filter_viewport,
            # use_tabs_info,
            # axtree_max_tokens,
            **obs_kwargs
        )
        step_data = StepData(step_num, observation)
        messages = prompt_builder_obj.build_messages(
            step_num,
            trajectory_data,
            action_set,
            current_step_data=step_data
        )["prompt"]

        # Get response from model
        response, _ = await call_llm_and_parse_response(
            llm,
            messages,
            max_attempts=parse_retries,
            think_prefix=think_prefix
        )
        step_data.response = response

        trajectory_data.add_step(step_data)
        
        # HACK: BGym needs literal \\n for successful action parsing
        action_to_step = response.action.replace("\n", "\\n")

        obs, reward, terminated, truncated, _ = await env.step(action_to_step)

        logging.info(f"Step {step_num} done. Action: {action_to_step}.")

        if terminated or truncated:
            save_observation_screenshot(
                obs,
                screenshot_dir=screenshot_dir,
                step_num=step_num + 1,
            )
            logging.info(f"Trajectory done. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            trajectory_data.reward = reward
            trajectory_data.terminated = terminated
            trajectory_data.truncated = truncated
            break

    if trajectory_data.reward is None:
        # Reached max steps without completion
        trajectory_data.reward = 0
        trajectory_data.terminated = False
        trajectory_data.truncated = True

    return trajectory_data


async def run_rollout_with_lifecycle(
    config: RunnerConfig,
    rollout_slot_index: int,
    setup_service_url: str,
    task_config,
    llm: BaseLLM,
    artifact_rollout_index: int | None = None,
):
    
    env_configs = env_configs_for_task(
        task_config,
        config.websites_vm_host,
        config.port_bases,
        rollout_slot_index
    )

    env = None
    is_setup = False
    try:
        # TODO: Make setup async native
        is_setup = await asyncio.to_thread(
            setup_env,
            setup_server_url=setup_service_url,
            env_configs=env_configs,
        )
        if not is_setup:
            raise RuntimeError("Websites VM setup service failed.")

        rollout_urls = build_rollout_urls(config, env_configs)
        action_set = HighLevelActionSet(["webarena"])
        screenshot_dir = prepare_screenshot_dir(
            config,
            task_id=task_config["task_id"],
            rollout_index=(
                rollout_slot_index
                if artifact_rollout_index is None
                else artifact_rollout_index
            ),
        )

        env = AsyncRemoteRolloutEnv(
            server_url=config.env_server_service_url,
            task_id=task_config["task_id"],
            task_config=task_config,
            homepage_url=rollout_urls.homepage_url,
            site_urls=rollout_urls.site_urls,
            seed=config.seed,
            timeout_s=config.timeout_s,
            include_screenshots=config.save_screenshots,
        )

        obs_kwargs = {
            "filter_viewport": config.filter_viewport,
            "use_tabs_info": config.use_tabs_info,
            "axtree_max_tokens": config.axtree_max_tokens,
        }
        trajectory_data = await agent_loop(
            env,
            task_config,
            rollout_urls.calculator_url,
            rollout_urls.site_urls,
            llm,
            action_set,
            max_steps=config.max_steps,
            seed=config.seed,
            parse_retries=config.parse_retries,
            think_prefix=config.think_prefix,
            screenshot_dir=screenshot_dir,
            **obs_kwargs,
        )
    finally:
        if is_setup:
            # TODO: make teardown async native
            await asyncio.to_thread(
                teardown_env,
                setup_server_url=setup_service_url,
                env_configs=env_configs
            )
        if env is not None:
            await env.close()

    return trajectory_data


def screenshot_dir_for_rollout(save_dir: Path, task_id: int, rollout_index: int) -> Path:
    return save_dir / str(task_id) / f"{rollout_index}_screenshots"


def prepare_screenshot_dir(
    config: RunnerConfig,
    task_id: int,
    rollout_index: int,
) -> Path | None:
    if not config.save_screenshots:
        return None

    screenshot_dir = screenshot_dir_for_rollout(
        config.save_dir,
        task_id,
        rollout_index,
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    return screenshot_dir


def save_observation_screenshot(
    obs: dict,
    *,
    screenshot_dir: Path | None,
    step_num: int,
) -> Path | None:
    if screenshot_dir is None:
        return None

    screenshot = obs.get("screenshot")
    if not screenshot:
        return None

    output_path = screenshot_dir / f"step_{step_num:03d}.png"
    output_path.write_bytes(base64.b64decode(screenshot))
    return output_path


def save_trajectory(
    save_dir: Path,
    task_id: int,
    rollout_index: int,
    rollout_trajectory: TrajectoryData,
) -> Path:
    task_save_dir = save_dir / str(task_id)
    task_save_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_save_dir / f"{rollout_index}.json"
    with open(output_path, "w") as f:
        json.dump(rollout_trajectory.to_json(), f, indent=2)
    return output_path


def trajectory_is_complete(output_path: Path) -> bool:
    if not output_path.is_file():
        return False

    try:
        with open(output_path, "r") as f:
            trajectory = json.load(f)
    except (OSError, json.JSONDecodeError):
        logging.warning("Ignoring unreadable trajectory file: %s", output_path)
        return False

    return isinstance(trajectory, dict) and trajectory.get("reward") is not None


def task_is_complete(save_dir: Path, task_id: int, rollout_count: int = 1) -> bool:
    return all(
        trajectory_is_complete(save_dir / str(task_id) / f"{rollout_index}.json")
        for rollout_index in range(rollout_count)
    )


async def run_all_tasks(
    config: RunnerConfig,
    setup_service_url: str,
    task_configs: list[dict],
    llm: BaseLLM,
) -> None:
    task_queue: asyncio.Queue[dict | None] = asyncio.Queue()
    for task_config in task_configs:
        task_queue.put_nowait(task_config)
    for _ in range(config.parallelism):
        task_queue.put_nowait(None)

    successes: list[Path] = []
    failures: list[tuple[int, Exception]] = []

    async def worker(rollout_slot_index: int) -> None:
        while True:
            task_config = await task_queue.get()
            try:
                if task_config is None:
                    return

                task_id = task_config["task_id"]
                logging.info(
                    "Starting task %s on rollout slot %s.",
                    task_id,
                    rollout_slot_index + 1,
                )
                try:
                    rollout_trajectory = await run_rollout_with_lifecycle(
                        config,
                        rollout_slot_index,
                        setup_service_url,
                        task_config,
                        llm,
                        artifact_rollout_index=0,
                    )
                    output_path = save_trajectory(
                        config.save_dir,
                        task_id,
                        0,
                        rollout_trajectory,
                    )
                    successes.append(output_path)
                    logging.info("Saved task %s trajectory to %s.", task_id, output_path)
                except Exception as error:
                    failures.append((task_id, error))
                    logging.exception("Task %s failed.", task_id)
            finally:
                task_queue.task_done()

    await asyncio.gather(
        *(worker(rollout_slot_index) for rollout_slot_index in range(config.parallelism))
    )

    for task_id, failure in failures:
        logging.exception("Task %s failed.", task_id, exc_info=failure)

    logging.info(
        "All-tasks run finished. successes=%s failures=%s save_dir=%s",
        len(successes),
        len(failures),
        config.save_dir,
    )
    if failures:
        failed_task_ids = ", ".join(str(task_id) for task_id, _ in failures)
        raise RuntimeError(f"{len(failures)} task(s) failed: {failed_task_ids}")


async def main_async(argv: list[str] | None = None) -> None:
    config = parse_args(argv)

    setup_service_url = f"http://{config.websites_vm_host}:{config.setup_service_port}"
    llm_base_url = config.llm_base_url
    if llm_base_url is None and config.llm_provider == "openai":
        llm_base_url = DEFAULT_LLM_BASE_URL
    logging.info(
        "Using model: %s (provider=%s, base_url=%s)",
        config.model,
        config.llm_provider,
        llm_base_url or "provider default",
    )
    if config.llm_provider == "anthropic":
        llm = AnthropicLLM(
            config.model,
            base_url=llm_base_url,
            max_tokens=config.llm_max_tokens,
        )
    else:
        llm = OpenAILLM(
            config.model,
            base_url=llm_base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            extra_body=config.llm_extra_body,
        )

    if config.all_tasks or config.task_ids is not None:
        all_task_configs = load_task_configs(config.task_config_path)
        skipped_task_ids = [
            task_config["task_id"]
            for task_config in all_task_configs
            if "gitlab" in task_config.get("sites", [])
        ] if config.all_tasks else []
        task_configs = (
            [
                task_config
                for task_config in all_task_configs
                if "gitlab" not in task_config.get("sites", [])
            ]
            if config.all_tasks
            else select_task_configs(all_task_configs, config.task_ids)
        )
        completed_task_ids = [
            task_config["task_id"]
            for task_config in task_configs
            if task_is_complete(config.save_dir, task_config["task_id"])
        ]
        completed_task_id_set = set(completed_task_ids)
        task_configs = [
            task_config
            for task_config in task_configs
            if task_config["task_id"] not in completed_task_id_set
        ]
        logging.info(
            "Skipping %s completed task(s): %s",
            len(completed_task_ids),
            completed_task_ids,
        )
        logging.info(
            "Running %s incomplete task(s): %s",
            len(task_configs),
            [task_config["task_id"] for task_config in task_configs],
        )
        # return
        try:
            await run_all_tasks(
                config,
                setup_service_url,
                task_configs,
                llm,
            )
        finally:
            if config.all_tasks:
                logging.info(
                    "Skipped %s GitLab-related task(s): %s",
                    len(skipped_task_ids),
                    skipped_task_ids,
                )
        return

    assert config.task_id is not None
    if task_is_complete(config.save_dir, config.task_id, config.group_size):
        logging.info(
            "Skipping completed task %s with %s rollout(s) in %s.",
            config.task_id,
            config.group_size,
            config.save_dir,
        )
        return
    task_config = load_task_config(config.task_config_path, config.task_id)
    rollout_trajectories = await asyncio.gather(
        *(
            run_rollout_with_lifecycle(
                config,
                rollout_slot_index,
                setup_service_url,
                task_config,
                llm,
            )
            for rollout_slot_index in range(config.group_size)
        )
    )

    for rollout_slot_index, rollout_trajectory in enumerate(rollout_trajectories):
        save_trajectory(
            config.save_dir,
            config.task_id,
            rollout_slot_index,
            rollout_trajectory,
        )


def main(argv: list[str] | None = None) -> None:
    asyncio.run(main_async(argv))

if __name__ == "__main__":
    main()
