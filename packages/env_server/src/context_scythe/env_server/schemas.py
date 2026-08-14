"""Pydantic schemas for the rollout server HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SessionStatus = Literal[
    "initializing",
    "ready",
    "running",
    "terminated",
    "closed",
    "failed",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebArenaUrls(StrictBaseModel):
    homepage_url: str = Field(min_length=1)
    site_urls: dict[str, str] = Field(min_length=1)


class CompactObservation(StrictBaseModel):
    axtree: str
    viewport_state: dict[str, Any]
    extra_element_properties: dict[str, Any]
    active_page_index: list[int]
    open_pages_titles: list[str]
    open_pages_urls: list[str]
    last_action_error: str
    screenshot: str | None = None


class SessionInitRequest(StrictBaseModel):
    task_id: int
    seed: int | None = None
    task_config: dict[str, Any] = Field(min_length=1)
    webarena: WebArenaUrls
    include_screenshots: bool = False


class SessionInitResponse(StrictBaseModel):
    rollout_id: str
    status: SessionStatus


class ResetRequest(StrictBaseModel):
    seed: int | None = None


class ResetResponse(StrictBaseModel):
    rollout_id: str
    status: SessionStatus
    observation: CompactObservation
    info: dict[str, Any]


class StepRequest(StrictBaseModel):
    action: str = Field(min_length=1)


class StepResponse(StrictBaseModel):
    rollout_id: str
    status: SessionStatus
    step: int
    observation: CompactObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class CloseResponse(StrictBaseModel):
    rollout_id: str
    status: Literal["closed"]


class RolloutStatusResponse(StrictBaseModel):
    rollout_id: str
    status: SessionStatus
    task_id: int
    seed: int | None
    step_count: int
    created_at: float
    last_activity_at: float
    actor_id: str | None = None
    actor_reachable: bool | None = None
