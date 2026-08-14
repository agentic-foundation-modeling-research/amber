"""Service-level rollout server exceptions."""

from __future__ import annotations


class RolloutServerError(Exception):
    """Base class for service-level rollout server failures."""

    public_message = "Rollout server error."


class UnknownRolloutError(RolloutServerError):
    public_message = "Unknown rollout ID."


class MaxSessionsExceededError(RolloutServerError):
    public_message = "Maximum number of live sessions exceeded."


class SessionClosedError(RolloutServerError):
    public_message = "Rollout session is closed."


class SessionFailedError(RolloutServerError):
    public_message = "Rollout session has failed."


class InvalidActionError(RolloutServerError):
    public_message = "Invalid BrowserGym action."


class InfrastructureError(RolloutServerError):
    public_message = "Rollout infrastructure failure."
