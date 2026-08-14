"""Gym-style rollout environment clients backed by the remote env_server API."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx
import requests

logger = logging.getLogger(__name__)


class HttpSession(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response: ...

    def close(self) -> None: ...


class AsyncHttpSession(Protocol):
    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response: ...

    async def aclose(self) -> None: ...


class RemoteRolloutEnvError(RuntimeError):
    """Raised when the remote rollout server rejects or fails a request."""

    def __init__(
        self,
        message: str,
        *,
        method: str,
        url: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.url = url
        self.status_code = status_code
        self.response_body = response_body


class _BaseRemoteRolloutEnv:
    def __init__(
        self,
        *,
        server_url: str,
        task_id: int,
        task_config: dict[str, Any],
        homepage_url: str,
        site_urls: dict[str, str],
        seed: int | None = None,
        timeout_s: float = 30,
        include_screenshots: bool = False,
    ) -> None:
        if task_config.get("task_id") != task_id:
            raise ValueError(
                f"task_id={task_id} does not match task_config['task_id']={task_config.get('task_id')}"
            )

        self.server_url = server_url.rstrip("/")
        self.task_id = task_id
        self.task_config = task_config
        self.homepage_url = homepage_url
        self.site_urls = site_urls
        self.seed = seed
        self.timeout_s = timeout_s
        self.include_screenshots = include_screenshots
        self._rollout_id: str | None = None
        self._closed = False

    @property
    def rollout_id(self) -> str | None:
        return self._rollout_id

    def _session_init_payload(self, seed: int | None) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": seed,
            "task_config": self.task_config,
            "webarena": {
                "homepage_url": self.homepage_url,
                "site_urls": self.site_urls,
            },
            "include_screenshots": self.include_screenshots,
        }

    def _reset_payload(self, seed: int | None) -> dict[str, Any]:
        return {"seed": seed}

    def _step_payload(self, action: str) -> dict[str, Any]:
        return {"action": action}

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed.")

    def _reset_seed(self, seed: int | None) -> int | None:
        return self.seed if seed is None else seed

    def _raise_if_options(self, options: dict[str, Any] | None) -> None:
        if options is not None:
            raise ValueError(f"{self.__class__.__name__} does not support reset options yet.")

    def _raise_if_uninitialized_for_step(self) -> None:
        if self._rollout_id is None:
            raise RuntimeError("Cannot step before reset creates a rollout session.")

    def _raise_if_uninitialized_for_status(self) -> None:
        if self._rollout_id is None:
            raise RuntimeError("Cannot fetch status before reset creates a rollout session.")


class RemoteRolloutEnv(_BaseRemoteRolloutEnv):
    """Synchronous Gym-style client for one remote rollout session."""

    def __init__(
        self,
        *,
        server_url: str,
        task_id: int,
        task_config: dict[str, Any],
        homepage_url: str,
        site_urls: dict[str, str],
        seed: int | None = None,
        timeout_s: float = 30,
        include_screenshots: bool = False,
        http_session: HttpSession | None = None,
    ) -> None:
        super().__init__(
            server_url=server_url,
            task_id=task_id,
            task_config=task_config,
            homepage_url=homepage_url,
            site_urls=site_urls,
            seed=seed,
            timeout_s=timeout_s,
            include_screenshots=include_screenshots,
        )
        self._http_session = http_session or requests.Session()
        self._owns_http_session = http_session is None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._raise_if_options(options)
        self._raise_if_closed()

        reset_seed = self._reset_seed(seed)
        if self._rollout_id is None:
            self._init_session(seed=reset_seed)

        response = self._request_json(
            "POST",
            f"/rollout/{self._rollout_id}/reset",
            self._reset_payload(reset_seed),
        )
        return response["observation"], response["info"]

    def step(
        self,
        action: str,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._raise_if_closed()
        self._raise_if_uninitialized_for_step()

        response = self._request_json(
            "POST",
            f"/rollout/{self._rollout_id}/step",
            self._step_payload(action),
        )
        return (
            response["observation"],
            response["reward"],
            response["terminated"],
            response["truncated"],
            response["info"],
        )

    def status(self) -> dict:
        self._raise_if_closed()
        self._raise_if_uninitialized_for_status()

        return self._request_json("GET", f"/rollout/{self._rollout_id}")

    def close(self) -> None:
        if self._closed:
            return

        try:
            if self._rollout_id is not None:
                self._request_json("POST", f"/rollout/{self._rollout_id}/close")
        finally:
            self._closed = True
            self._rollout_id = None
            if self._owns_http_session:
                self._http_session.close()

    def __enter__(self) -> RemoteRolloutEnv:
        self._raise_if_closed()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _init_session(self, *, seed: int | None) -> None:
        response = self._request_json(
            "POST",
            "/session_init",
            self._session_init_payload(seed),
        )
        self._rollout_id = response["rollout_id"]

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        try:
            response = self._http_session.request(
                method,
                url,
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else None
            status_code = exc.response.status_code if exc.response is not None else None
            raise RemoteRolloutEnvError(
                _format_request_error(method, url, status_code, body),
                method=method,
                url=url,
                status_code=status_code,
                response_body=body,
            ) from exc
        except requests.RequestException as exc:
            detail = str(exc) or f"after {self.timeout_s}s"
            raise RemoteRolloutEnvError(
                f"{method} {url} failed with {exc.__class__.__name__}: {detail}",
                method=method,
                url=url,
            ) from exc

        if not response.content:
            return {}
        return response.json()


class AsyncRemoteRolloutEnv(_BaseRemoteRolloutEnv):
    """Async Gym-style client for one remote rollout session."""

    def __init__(
        self,
        *,
        server_url: str,
        task_id: int,
        task_config: dict[str, Any],
        homepage_url: str,
        site_urls: dict[str, str],
        seed: int | None = None,
        timeout_s: float = 30,
        include_screenshots: bool = False,
        http_session: AsyncHttpSession | None = None,
    ) -> None:
        super().__init__(
            server_url=server_url,
            task_id=task_id,
            task_config=task_config,
            homepage_url=homepage_url,
            site_urls=site_urls,
            seed=seed,
            timeout_s=timeout_s,
            include_screenshots=include_screenshots,
        )
        self._http_session = http_session or httpx.AsyncClient()
        self._owns_http_session = http_session is None

    async def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._raise_if_options(options)
        self._raise_if_closed()

        reset_seed = self._reset_seed(seed)
        if self._rollout_id is None:
            await self._init_session(seed=reset_seed)

        logger.info(
            "remote_rollout_reset request server_url=%s rollout_id=%s seed=%s site_urls=%s",
            self.server_url,
            self._rollout_id,
            reset_seed,
            self.site_urls,
        )
        response = await self._request_json(
            "POST",
            f"/rollout/{self._rollout_id}/reset",
            self._reset_payload(reset_seed),
        )
        return response["observation"], response["info"]

    async def step(
        self,
        action: str,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._raise_if_closed()
        self._raise_if_uninitialized_for_step()

        logger.info(
            "remote_rollout_step request server_url=%s rollout_id=%s site_urls=%s",
            self.server_url,
            self._rollout_id,
            self.site_urls,
        )
        response = await self._request_json(
            "POST",
            f"/rollout/{self._rollout_id}/step",
            self._step_payload(action),
        )
        return (
            response["observation"],
            response["reward"],
            response["terminated"],
            response["truncated"],
            response["info"],
        )

    async def status(self) -> dict:
        self._raise_if_closed()
        self._raise_if_uninitialized_for_status()

        return await self._request_json("GET", f"/rollout/{self._rollout_id}")

    async def close(self) -> None:
        if self._closed:
            return

        try:
            if self._rollout_id is not None:
                await self._request_json("POST", f"/rollout/{self._rollout_id}/close")
        finally:
            self._closed = True
            self._rollout_id = None
            if self._owns_http_session:
                await self._http_session.aclose()

    async def __aenter__(self) -> AsyncRemoteRolloutEnv:
        self._raise_if_closed()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def _init_session(self, *, seed: int | None) -> None:
        logger.info(
            "remote_rollout_session_init request server_url=%s task_id=%s seed=%s site_urls=%s",
            self.server_url,
            self.task_id,
            seed,
            self.site_urls,
        )
        response = await self._request_json(
            "POST",
            "/session_init",
            self._session_init_payload(seed),
        )
        self._rollout_id = response["rollout_id"]

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        try:
            response = await self._http_session.request(
                method,
                url,
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            status_code = exc.response.status_code
            raise RemoteRolloutEnvError(
                _format_request_error(method, url, status_code, body),
                method=method,
                url=url,
                status_code=status_code,
                response_body=body,
            ) from exc
        except httpx.HTTPError as exc:
            detail = str(exc) or f"after {self.timeout_s}s"
            raise RemoteRolloutEnvError(
                f"{method} {url} failed with {exc.__class__.__name__}: {detail}",
                method=method,
                url=url,
            ) from exc

        if not response.content:
            return {}
        return response.json()


def _format_request_error(
    method: str,
    url: str,
    status_code: int | None,
    body: str | None,
) -> str:
    if status_code is None:
        return f"{method} {url} failed with an unknown HTTP error."
    if body:
        return f"{method} {url} failed with HTTP {status_code}: {body}"
    return f"{method} {url} failed with HTTP {status_code}."


__all__ = ["AsyncRemoteRolloutEnv", "RemoteRolloutEnv", "RemoteRolloutEnvError"]
