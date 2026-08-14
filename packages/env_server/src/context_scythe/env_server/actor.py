"""Ray actor implementation for live BrowserGym environment sessions."""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any

import ray
from browsergym.core.action.highlevel import HighLevelActionSet

from context_scythe.environment.webarena_task import create_env_for_task
from context_scythe.environment.viewport import get_viewport_state
from context_scythe.env_server.schemas import SessionInitRequest
from context_scythe.env_server.observations import compact_observation

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class EnvSessionActor:
    """Owns one live BrowserGym environment for a rollout session."""

    def __init__(self, request: SessionInitRequest | dict[str, Any]):
        self.request = _coerce_request(request)
        self.closed = False
        self._atexit_registered = False
        self._atexit_close = None
        logger.info("env_actor_initializing actor=%s", self._actor_label())

        self.action_set = HighLevelActionSet(subsets=["webarena"], multiaction=False)
        task_config = self.request.task_config

        base_url, api_key = _openai_evaluator_config()

        self.env = create_env_for_task(
            task_config=task_config,
            task_id=self.request.task_id,
            homepage_url=self.request.webarena.homepage_url,
            site_urls=self.request.webarena.site_urls,
            base_url=base_url,
            api_key=api_key,
            headless=True,
            action_mapping=self.action_set.to_python_code,
        )
        self._atexit_close = self._close_at_exit
        atexit.register(self._atexit_close)
        self._atexit_registered = True
        self.last_obs: dict[str, Any] | None = None
        logger.info("env_actor_ready actor=%s", self._actor_label())

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._raise_if_closed()
        logger.info("env_actor_reset_started actor=%s", self._actor_label())

        obs, info = self.env.reset(seed=seed)
        
        # Get the viewport state
        viewport_state = get_viewport_state(self.env.unwrapped.page)
        obs["viewport_state"] = viewport_state
        
        self.last_obs = obs
        logger.info("env_actor_reset_finished actor=%s", self._actor_label())
        return {
            "observation": compact_observation(
                obs,
                use_screenshots=self.request.include_screenshots,
            ).model_dump(),
            "info": info,
        }

    def step(self, action: str) -> dict[str, Any]:
        self._raise_if_closed()
        logger.info("env_actor_step_started actor=%s", self._actor_label())

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get the viewport state
        viewport_state = get_viewport_state(self.env.unwrapped.page)
        obs["viewport_state"] = viewport_state
        
        self.last_obs = obs
        logger.info("env_actor_step_finished actor=%s", self._actor_label())
        return {
            "observation": compact_observation(
                obs,
                use_screenshots=self.request.include_screenshots,
            ).model_dump(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }

    def close(self) -> dict[str, bool]:
        return self._close(log=True)

    def exit(self) -> None:
        try:
            self._close(log=False)
        finally:
            ray.actor.exit_actor()

    def _close_at_exit(self) -> None:
        self._close(log=False)

    def _close(self, *, log: bool) -> dict[str, bool]:
        if not self.closed:
            try:
                self.env.close()
            finally:
                self.closed = True
                self._unregister_atexit()
                if log:
                    logger.info("env_actor_closed actor=%s", self._actor_label())
        return {"closed": True}

    def ping(self) -> dict[str, bool]:
        logger.info("env_actor_ping actor=%s", self._actor_label())
        return {"ok": not self.closed}

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("Rollout environment session is closed.")

    def _actor_label(self) -> str:
        return f"{os.getpid()}:{id(self)}"

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_close)
            self._atexit_registered = False


def _coerce_request(request: SessionInitRequest | dict[str, Any]) -> SessionInitRequest:
    if isinstance(request, SessionInitRequest):
        return request
    return SessionInitRequest.model_validate(request)


def _openai_evaluator_config() -> tuple[str, str]:
    missing = [
        env_var
        for env_var in ("OPENAI_BASE_URL", "OPENAI_API_KEY")
        if not os.getenv(env_var)
    ]
    if missing:
        raise RuntimeError(
            "Missing required WebArena evaluator environment variables: "
            + ", ".join(missing)
        )

    return os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
