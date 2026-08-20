"""FastAPI application entrypoint for the rollout server."""

from __future__ import annotations

import logging
import os
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import ray
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from context_scythe.env_server.schemas import (
    CloseResponse,
    ResetRequest,
    ResetResponse,
    RolloutStatusResponse,
    SessionInitRequest,
    SessionInitResponse,
    StepRequest,
    StepResponse,
)
from context_scythe.env_server.errors import (
    InfrastructureError,
    InvalidActionError,
    MaxSessionsExceededError,
    RolloutServerError,
    SessionClosedError,
    SessionFailedError,
    UnknownRolloutError,
)
from context_scythe.env_server.manager import SessionManager


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutServerSettings:
    max_live_sessions: int = 64
    idle_timeout_s: float = 1800
    actor_close_timeout_s: float = 10


def create_app(
    manager: SessionManager | None = None,
    settings: RolloutServerSettings | None = None,
) -> FastAPI:
    """Build the rollout server FastAPI app."""
    settings = settings or RolloutServerSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if manager is None:
            _initialize_ray()
            app.state.session_manager = SessionManager(
                max_live_sessions=settings.max_live_sessions,
                idle_timeout_s=settings.idle_timeout_s,
                actor_close_timeout_s=settings.actor_close_timeout_s,
            )
            logger.info("env_server_started")
        else:
            app.state.session_manager = manager
            logger.info("env_server_started_with_injected_manager")
        yield

    fastapi_app = FastAPI(title="Amber Rollout Server", lifespan=lifespan)
    _register_exception_handlers(fastapi_app)
    _register_routes(fastapi_app)
    return fastapi_app


def main() -> None:
    import uvicorn

    args = parse_args()
    settings = settings_from_args(args)
    configured_app = create_app(settings=settings)
    uvicorn.run(
        configured_app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
        reload=False,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Amber rollout server.")
    parser.add_argument(
        "--host",
        default=os.getenv("ROLLOUT_SERVER_HOST", "0.0.0.0"),
        help="Host interface to bind. Defaults to ROLLOUT_SERVER_HOST or 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ROLLOUT_SERVER_PORT", "8000")),
        help="Port to bind. Defaults to ROLLOUT_SERVER_PORT or 8000.",
    )
    parser.add_argument(
        "--max-live-sessions",
        type=int,
        default=int(os.getenv("ROLLOUT_SERVER_MAX_SESSIONS", "64")),
        help="Maximum number of concurrent live rollout sessions.",
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=float,
        default=float(os.getenv("ROLLOUT_SERVER_IDLE_TIMEOUT_S", "1800")),
        help="Seconds before an idle rollout session is closed.",
    )
    parser.add_argument(
        "--actor-close-timeout-s",
        type=float,
        default=float(os.getenv("ROLLOUT_SERVER_ACTOR_CLOSE_TIMEOUT_S", "10")),
        help="Seconds to wait for graceful Ray actor close before force killing it.",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=90,
        help="HTTP connection keep alive time.",
    )
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> RolloutServerSettings:
    return RolloutServerSettings(
        max_live_sessions=args.max_live_sessions,
        idle_timeout_s=args.idle_timeout_s,
        actor_close_timeout_s=args.actor_close_timeout_s,
    )


def get_manager(fastapi_app: FastAPI) -> SessionManager:
    return fastapi_app.state.session_manager


def _register_routes(fastapi_app: FastAPI) -> None:
    @fastapi_app.get("/health")
    async def health() -> dict[str, Any]:
        manager = get_manager(fastapi_app)
        live_sessions = (
            manager.live_session_count()
            if hasattr(manager, "live_session_count")
            else len(manager.sessions)
        )
        return {
            "status": "ok",
            "live_sessions": live_sessions,
            "max_live_sessions": manager.max_live_sessions,
        }

    @fastapi_app.post("/session_init", response_model=SessionInitResponse)
    async def session_init(request: SessionInitRequest) -> SessionInitResponse:
        return await _call_manager_method(get_manager(fastapi_app), "init_session", request)

    @fastapi_app.post("/rollout/{rollout_id}/reset", response_model=ResetResponse)
    async def reset_rollout(rollout_id: str, request: ResetRequest) -> ResetResponse:
        return await _call_manager_method(
            get_manager(fastapi_app),
            "reset_session",
            rollout_id,
            request,
        )

    @fastapi_app.post("/rollout/{rollout_id}/step", response_model=StepResponse)
    async def step_rollout(rollout_id: str, request: StepRequest) -> StepResponse:
        return await _call_manager_method(
            get_manager(fastapi_app),
            "step_session",
            rollout_id,
            request,
        )

    @fastapi_app.post("/rollout/{rollout_id}/close", response_model=CloseResponse)
    async def close_rollout(rollout_id: str) -> CloseResponse:
        return await _call_manager_method(
            get_manager(fastapi_app),
            "close_session",
            rollout_id,
        )

    @fastapi_app.get("/rollout/{rollout_id}", response_model=RolloutStatusResponse)
    async def rollout_status(rollout_id: str) -> RolloutStatusResponse:
        return await _call_manager_method(
            get_manager(fastapi_app),
            "get_status",
            rollout_id,
        )


async def _call_manager_method(manager: Any, method_name: str, *args: Any) -> Any:
    async_method = getattr(manager, f"{method_name}_async", None)
    if async_method is not None:
        return await async_method(*args)

    method = getattr(manager, method_name)
    return await asyncio.to_thread(method, *args)


def _register_exception_handlers(fastapi_app: FastAPI) -> None:
    @fastapi_app.exception_handler(UnknownRolloutError)
    async def unknown_rollout_handler(_, exc: UnknownRolloutError):
        return _error_response(404, exc)

    @fastapi_app.exception_handler(InvalidActionError)
    async def invalid_action_handler(_, exc: InvalidActionError):
        return _error_response(400, exc)

    @fastapi_app.exception_handler(MaxSessionsExceededError)
    async def max_sessions_handler(_, exc: MaxSessionsExceededError):
        return _error_response(429, exc)

    @fastapi_app.exception_handler(SessionClosedError)
    async def session_closed_handler(_, exc: SessionClosedError):
        return _error_response(409, exc)

    @fastapi_app.exception_handler(SessionFailedError)
    async def session_failed_handler(_, exc: SessionFailedError):
        return _error_response(409, exc)

    @fastapi_app.exception_handler(InfrastructureError)
    async def infrastructure_handler(_, exc: InfrastructureError):
        logger.exception("Rollout infrastructure failure")
        return _error_response(500, exc)


def _error_response(status_code: int, exc: RolloutServerError) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "error": exc.__class__.__name__,
                "message": exc.public_message,
            },
        },
    )


def _initialize_ray() -> None:
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env=_ray_runtime_env(),
        )
        logger.info("ray_initialized")
    else:
        logger.info("ray_already_initialized")


def _ray_runtime_env() -> dict[str, dict[str, str]] | None:
    env_vars: dict[str, str] = {}

    active_venv = os.getenv("VIRTUAL_ENV")
    if active_venv:
        env_vars.update(
            {
                "VIRTUAL_ENV": active_venv,
                "UV_PROJECT_ENVIRONMENT": active_venv,
            }
        )

    for env_var in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        value = os.getenv(env_var)
        if value:
            env_vars[env_var] = value

    if not env_vars:
        return None

    return {"env_vars": env_vars}


app = create_app()
