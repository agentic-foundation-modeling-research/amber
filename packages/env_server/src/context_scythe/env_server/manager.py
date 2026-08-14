"""Session lifecycle management for rollout server actors."""

from __future__ import annotations

import time
import uuid
import asyncio
import inspect
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

import ray
from ray.exceptions import GetTimeoutError

from context_scythe.env_server.schemas import (
    CloseResponse,
    CompactObservation,
    ResetRequest,
    ResetResponse,
    RolloutStatusResponse,
    SessionInitRequest,
    SessionInitResponse,
    SessionStatus,
    StepRequest,
    StepResponse,
    WebArenaUrls,
)
from context_scythe.env_server.actor import EnvSessionActor
from context_scythe.env_server.errors import (
    InfrastructureError,
    InvalidActionError,
    MaxSessionsExceededError,
    SessionClosedError,
    SessionFailedError,
    UnknownRolloutError,
)

logger = logging.getLogger(__name__)


@dataclass
class RolloutSession:
    rollout_id: str
    task_id: int
    seed: int | None
    status: SessionStatus
    actor_handle: Any
    webarena_urls: WebArenaUrls
    step_count: int
    created_at: float
    last_activity_at: float


class SessionManager:
    """FastAPI-side lifecycle manager for live rollout actors."""

    def __init__(
        self,
        *,
        max_live_sessions: int = 64,
        idle_timeout_s: float = 30 * 60,
        actor_close_timeout_s: float = 10,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.max_live_sessions = max_live_sessions
        self.idle_timeout_s = idle_timeout_s
        self.actor_close_timeout_s = actor_close_timeout_s
        self.id_factory = id_factory if id_factory is not None else _default_rollout_id
        self.clock = clock
        self.sessions: dict[str, RolloutSession] = {}
        self.closed_rollout_ids: set[str] = set()
        self._pending_rollout_ids: set[str] = set()
        self._lock = threading.RLock()

    def init_session(self, request: SessionInitRequest) -> SessionInitResponse:
        self.cleanup_idle_sessions()
        with self._lock:
            if self._reserved_session_count() >= self.max_live_sessions:
                raise MaxSessionsExceededError

            rollout_id = self._next_rollout_id()
            self._pending_rollout_ids.add(rollout_id)
            now = self.clock()
        actor_handle = None
        try:
            actor_handle = self._create_actor(request)
            self._call_actor(actor_handle, "ping")
        except Exception as exc:
            logger.exception(
                "env_session_init_failed actor_id=%s",
                _actor_id(actor_handle) if actor_handle is not None else None,
            )
            if actor_handle is not None:
                self._discard_actor(actor_handle)
            with self._lock:
                self._pending_rollout_ids.discard(rollout_id)
            raise InfrastructureError("Failed to initialize rollout actor.") from exc

        with self._lock:
            self._pending_rollout_ids.discard(rollout_id)
            self.sessions[rollout_id] = RolloutSession(
                rollout_id=rollout_id,
                task_id=request.task_id,
                seed=request.seed,
                status="ready",
                actor_handle=actor_handle,
                webarena_urls=request.webarena,
                step_count=0,
                created_at=now,
                last_activity_at=now,
            )
        logger.info("env_session_ready actor_id=%s", _actor_id(actor_handle))
        return SessionInitResponse(rollout_id=rollout_id, status="ready")

    async def init_session_async(self, request: SessionInitRequest) -> SessionInitResponse:
        await self.cleanup_idle_sessions_async()
        with self._lock:
            if self._reserved_session_count() >= self.max_live_sessions:
                raise MaxSessionsExceededError

            rollout_id = self._next_rollout_id()
            self._pending_rollout_ids.add(rollout_id)
            now = self.clock()
        actor_handle = None
        try:
            actor_handle = self._create_actor(request)
            await self._call_actor_async(actor_handle, "ping")
        except Exception as exc:
            logger.exception(
                "env_session_init_failed actor_id=%s",
                _actor_id(actor_handle) if actor_handle is not None else None,
            )
            if actor_handle is not None:
                await self._discard_actor_async(actor_handle)
            with self._lock:
                self._pending_rollout_ids.discard(rollout_id)
            raise InfrastructureError("Failed to initialize rollout actor.") from exc

        with self._lock:
            self._pending_rollout_ids.discard(rollout_id)
            self.sessions[rollout_id] = RolloutSession(
                rollout_id=rollout_id,
                task_id=request.task_id,
                seed=request.seed,
                status="ready",
                actor_handle=actor_handle,
                webarena_urls=request.webarena,
                step_count=0,
                created_at=now,
                last_activity_at=now,
            )
        logger.info("env_session_ready actor_id=%s", _actor_id(actor_handle))
        return SessionInitResponse(rollout_id=rollout_id, status="ready")

    def reset_session(self, rollout_id: str, request: ResetRequest) -> ResetResponse:
        session = self._get_active_session(rollout_id)
        self._ensure_usable(session)

        try:
            result = self._call_actor(session.actor_handle, "reset", request.seed)
        except Exception as exc:
            session.status = "failed"
            session.last_activity_at = self.clock()
            logger.exception(
                "env_session_reset_failed actor_id=%s",
                _actor_id(session.actor_handle),
            )
            raise InfrastructureError("Failed to reset rollout actor.") from exc

        session.status = "running"
        session.step_count = 0
        session.last_activity_at = self.clock()
        logger.info("env_session_reset actor_id=%s", _actor_id(session.actor_handle))
        return ResetResponse(
            rollout_id=rollout_id,
            status=session.status,
            observation=CompactObservation.model_validate(result["observation"]),
            info=result.get("info", {}),
        )

    async def reset_session_async(self, rollout_id: str, request: ResetRequest) -> ResetResponse:
        session = self._get_active_session(rollout_id)
        self._ensure_usable(session)

        try:
            result = await self._call_actor_async(session.actor_handle, "reset", request.seed)
        except Exception as exc:
            session.status = "failed"
            session.last_activity_at = self.clock()
            logger.exception(
                "env_session_reset_failed actor_id=%s",
                _actor_id(session.actor_handle),
            )
            raise InfrastructureError("Failed to reset rollout actor.") from exc

        session.status = "running"
        session.step_count = 0
        session.last_activity_at = self.clock()
        logger.info("env_session_reset actor_id=%s", _actor_id(session.actor_handle))
        return ResetResponse(
            rollout_id=rollout_id,
            status=session.status,
            observation=CompactObservation.model_validate(result["observation"]),
            info=result.get("info", {}),
        )

    def step_session(self, rollout_id: str, request: StepRequest) -> StepResponse:
        session = self._get_active_session(rollout_id)
        self._ensure_usable(session)

        try:
            result = self._call_actor(session.actor_handle, "step", request.action)
        except ValueError as exc:
            logger.info(
                "env_session_invalid_action actor_id=%s",
                _actor_id(session.actor_handle),
            )
            raise InvalidActionError("Invalid BrowserGym action.") from exc
        except Exception as exc:
            session.status = "failed"
            session.last_activity_at = self.clock()
            logger.exception(
                "env_session_step_failed actor_id=%s",
                _actor_id(session.actor_handle),
            )
            raise InfrastructureError("Failed to step rollout actor.") from exc

        session.step_count += 1
        terminated = bool(result["terminated"])
        truncated = bool(result["truncated"])
        session.status = "terminated" if terminated or truncated else "running"
        session.last_activity_at = self.clock()
        logger.info("env_session_step actor_id=%s", _actor_id(session.actor_handle))
        return StepResponse(
            rollout_id=rollout_id,
            status=session.status,
            step=session.step_count,
            observation=CompactObservation.model_validate(result["observation"]),
            reward=float(result["reward"]),
            terminated=terminated,
            truncated=truncated,
            info=result.get("info", {}),
        )

    async def step_session_async(self, rollout_id: str, request: StepRequest) -> StepResponse:
        session = self._get_active_session(rollout_id)
        self._ensure_usable(session)

        try:
            result = await self._call_actor_async(session.actor_handle, "step", request.action)
        except ValueError as exc:
            logger.info(
                "env_session_invalid_action actor_id=%s",
                _actor_id(session.actor_handle),
            )
            raise InvalidActionError("Invalid BrowserGym action.") from exc
        except Exception as exc:
            session.status = "failed"
            session.last_activity_at = self.clock()
            logger.exception(
                "env_session_step_failed actor_id=%s",
                _actor_id(session.actor_handle),
            )
            raise InfrastructureError("Failed to step rollout actor.") from exc

        session.step_count += 1
        terminated = bool(result["terminated"])
        truncated = bool(result["truncated"])
        session.status = "terminated" if terminated or truncated else "running"
        session.last_activity_at = self.clock()
        logger.info("env_session_step actor_id=%s", _actor_id(session.actor_handle))
        return StepResponse(
            rollout_id=rollout_id,
            status=session.status,
            step=session.step_count,
            observation=CompactObservation.model_validate(result["observation"]),
            reward=float(result["reward"]),
            terminated=terminated,
            truncated=truncated,
            info=result.get("info", {}),
        )

    def close_session(self, rollout_id: str) -> CloseResponse:
        with self._lock:
            session = self.sessions.pop(rollout_id, None)
            if session is None and rollout_id in self.closed_rollout_ids:
                return CloseResponse(rollout_id=rollout_id, status="closed")
        if session is None:
            raise UnknownRolloutError

        logger.info(
            "env_session_close_requested rollout_id=%s actor_id=%s",
            rollout_id,
            _actor_id(session.actor_handle),
        )
        session.status = "closed"

        try:
            self._terminate_actor(session.actor_handle)
        finally:
            with self._lock:
                self.closed_rollout_ids.add(rollout_id)

        logger.info(
            "env_session_closed rollout_id=%s actor_id=%s",
            rollout_id,
            _actor_id(session.actor_handle),
        )
        return CloseResponse(rollout_id=rollout_id, status="closed")

    async def close_session_async(self, rollout_id: str) -> CloseResponse:
        with self._lock:
            session = self.sessions.pop(rollout_id, None)
            if session is None and rollout_id in self.closed_rollout_ids:
                return CloseResponse(rollout_id=rollout_id, status="closed")
        if session is None:
            raise UnknownRolloutError

        logger.info(
            "env_session_close_requested rollout_id=%s actor_id=%s",
            rollout_id,
            _actor_id(session.actor_handle),
        )
        session.status = "closed"

        try:
            await self._terminate_actor_async(session.actor_handle)
        finally:
            with self._lock:
                self.closed_rollout_ids.add(rollout_id)

        logger.info(
            "env_session_closed rollout_id=%s actor_id=%s",
            rollout_id,
            _actor_id(session.actor_handle),
        )
        return CloseResponse(rollout_id=rollout_id, status="closed")

    def get_status(self, rollout_id: str) -> RolloutStatusResponse:
        session = self._get_active_session(rollout_id)
        actor_reachable = None
        if session.status not in {"closed", "failed"}:
            try:
                self._call_actor(session.actor_handle, "ping")
                actor_reachable = True
            except Exception:
                session.status = "failed"
                actor_reachable = False

        logger.info("env_session_status actor_id=%s", _actor_id(session.actor_handle))
        return RolloutStatusResponse(
            rollout_id=session.rollout_id,
            status=session.status,
            task_id=session.task_id,
            seed=session.seed,
            step_count=session.step_count,
            created_at=session.created_at,
            last_activity_at=session.last_activity_at,
            actor_id=_actor_id(session.actor_handle),
            actor_reachable=actor_reachable,
        )

    async def get_status_async(self, rollout_id: str) -> RolloutStatusResponse:
        session = self._get_active_session(rollout_id)
        actor_reachable = None
        if session.status not in {"closed", "failed"}:
            try:
                await self._call_actor_async(session.actor_handle, "ping")
                actor_reachable = True
            except Exception:
                session.status = "failed"
                actor_reachable = False

        logger.info("env_session_status actor_id=%s", _actor_id(session.actor_handle))
        return RolloutStatusResponse(
            rollout_id=session.rollout_id,
            status=session.status,
            task_id=session.task_id,
            seed=session.seed,
            step_count=session.step_count,
            created_at=session.created_at,
            last_activity_at=session.last_activity_at,
            actor_id=_actor_id(session.actor_handle),
            actor_reachable=actor_reachable,
        )

    def cleanup_idle_sessions(self) -> list[str]:
        now = self.clock()
        with self._lock:
            sessions = list(self.sessions.items())
        expired = []
        for rollout_id, session in sessions:
            if now - session.last_activity_at > self.idle_timeout_s:
                expired.append(rollout_id)
        for rollout_id in expired:
            try:
                session = self._get_active_session(rollout_id)
            except UnknownRolloutError:
                continue
            logger.info(
                "env_session_idle_timeout actor_id=%s",
                _actor_id(session.actor_handle),
            )
            try:
                self.close_session(rollout_id)
            except UnknownRolloutError:
                continue
        return expired

    async def cleanup_idle_sessions_async(self) -> list[str]:
        now = self.clock()
        with self._lock:
            sessions = list(self.sessions.items())
        expired = []
        for rollout_id, session in sessions:
            if now - session.last_activity_at > self.idle_timeout_s:
                expired.append(rollout_id)
        for rollout_id in expired:
            try:
                session = self._get_active_session(rollout_id)
            except UnknownRolloutError:
                continue
            logger.info(
                "env_session_idle_timeout actor_id=%s",
                _actor_id(session.actor_handle),
            )
            try:
                await self.close_session_async(rollout_id)
            except UnknownRolloutError:
                continue
        return expired

    def live_session_count(self) -> int:
        with self._lock:
            return len(self.sessions)

    def _create_actor(self, request: SessionInitRequest) -> Any:
        payload = request.model_dump()
        return EnvSessionActor.remote(payload)

    def _call_actor(
        self,
        actor_handle: Any,
        method_name: str,
        *args: Any,
        timeout_s: float | None = None,
    ) -> Any:
        method = getattr(actor_handle, method_name)
        result_ref = method.remote(*args)
        if timeout_s is None:
            return ray.get(result_ref)
        return ray.get(result_ref, timeout=timeout_s)

    async def _call_actor_async(
        self,
        actor_handle: Any,
        method_name: str,
        *args: Any,
        timeout_s: float | None = None,
    ) -> Any:
        method = getattr(actor_handle, method_name)
        result_ref = method.remote(*args)
        if inspect.isawaitable(result_ref):
            if timeout_s is not None:
                return await asyncio.wait_for(result_ref, timeout=timeout_s)
            return await result_ref
        if timeout_s is None:
            return await asyncio.to_thread(ray.get, result_ref)
        return await asyncio.to_thread(ray.get, result_ref, timeout=timeout_s)

    def _discard_actor(self, actor_handle: Any) -> None:
        try:
            self._terminate_actor(actor_handle)
        except Exception:
            logger.exception(
                "env_actor_discard_failed actor_id=%s",
                _actor_id(actor_handle),
            )

    async def _discard_actor_async(self, actor_handle: Any) -> None:
        try:
            await self._terminate_actor_async(actor_handle)
        except Exception:
            logger.exception(
                "env_actor_discard_failed actor_id=%s",
                _actor_id(actor_handle),
            )

    def _terminate_actor(self, actor_handle: Any) -> None:
        close_error: Exception | None = None
        try:
            self._call_actor(
                actor_handle,
                "close",
                timeout_s=self.actor_close_timeout_s,
            )
        except (GetTimeoutError, TimeoutError):
            logger.warning(
                "env_actor_close_timeout actor_id=%s timeout_s=%s",
                _actor_id(actor_handle),
                self.actor_close_timeout_s,
            )
            self._kill_actor(actor_handle)
            return
        except Exception as exc:
            close_error = exc
            logger.exception(
                "env_actor_close_failed actor_id=%s",
                _actor_id(actor_handle),
            )

        try:
            self._queue_actor_termination(actor_handle)
        except Exception:
            logger.exception(
                "env_actor_graceful_terminate_failed actor_id=%s",
                _actor_id(actor_handle),
            )
            self._kill_actor(actor_handle)

        if close_error is not None:
            raise close_error

    async def _terminate_actor_async(self, actor_handle: Any) -> None:
        close_error: Exception | None = None
        try:
            await self._call_actor_async(
                actor_handle,
                "close",
                timeout_s=self.actor_close_timeout_s,
            )
        except (GetTimeoutError, TimeoutError):
            logger.warning(
                "env_actor_close_timeout actor_id=%s timeout_s=%s",
                _actor_id(actor_handle),
                self.actor_close_timeout_s,
            )
            await self._kill_actor_async(actor_handle)
            return
        except Exception as exc:
            close_error = exc
            logger.exception(
                "env_actor_close_failed actor_id=%s",
                _actor_id(actor_handle),
            )

        try:
            self._queue_actor_termination(actor_handle)
        except Exception:
            logger.exception(
                "env_actor_graceful_terminate_failed actor_id=%s",
                _actor_id(actor_handle),
            )
            await self._kill_actor_async(actor_handle)

        if close_error is not None:
            raise close_error

    def _queue_actor_termination(self, actor_handle: Any) -> None:
        actor_handle.exit.remote()

    def _kill_actor(self, actor_handle: Any) -> None:
        ray.kill(actor_handle, no_restart=True)

    async def _kill_actor_async(self, actor_handle: Any) -> None:
        await asyncio.to_thread(ray.kill, actor_handle, no_restart=True)

    def _get_active_session(self, rollout_id: str) -> RolloutSession:
        with self._lock:
            try:
                return self.sessions[rollout_id]
            except KeyError as exc:
                raise UnknownRolloutError from exc

    def _ensure_usable(self, session: RolloutSession) -> None:
        if session.status == "closed":
            raise SessionClosedError
        if session.status == "failed":
            raise SessionFailedError

    def _next_rollout_id(self) -> str:
        while True:
            rollout_id = self.id_factory()
            if (
                rollout_id not in self.sessions
                and rollout_id not in self.closed_rollout_ids
                and rollout_id not in self._pending_rollout_ids
            ):
                return rollout_id

    def _reserved_session_count(self) -> int:
        return len(self.sessions) + len(self._pending_rollout_ids)


def _default_rollout_id() -> str:
    return f"rollout-{uuid.uuid4().hex}"


def _actor_id(actor_handle: Any) -> str | None:
    actor_id = getattr(actor_handle, "_actor_id", None)
    return str(actor_id) if actor_id is not None else None
