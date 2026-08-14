"""Parsing and validation helpers for state-memory synthetic labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

from context_scythe.agents.trajectory_data import Memory


_STATE_MEMORY_RESPONSE = re.compile(
    r"\s*<think>\s*(?P<reasoning>.*?)\s*</think>\s*"
    r"<memory>\s*(?P<memory>.*?)\s*</memory>\s*"
    r"<state>\s*(?P<state>.*?)\s*</state>\s*",
    re.DOTALL,
)


class StateMemoryLabelError(ValueError):
    """Raised when a generated state-memory label violates its contract."""


def parse_state_memory_response(raw_response: str) -> Memory:
    """Parse one raw ``think/memory/state`` response with strict tag ordering."""
    if not isinstance(raw_response, str):
        raise StateMemoryLabelError("State-memory response must be a string.")

    match = _STATE_MEMORY_RESPONSE.fullmatch(raw_response)
    if match is None:
        raise StateMemoryLabelError(
            "State-memory response must contain exactly <think>, <memory>, and "
            "<state> blocks in that order, with no text outside them."
        )

    parts = {name: match.group(name).strip() for name in ("reasoning", "memory", "state")}
    for name, value in parts.items():
        if not value:
            raise StateMemoryLabelError(f"<{name}> must not be empty.")

    return Memory(
        model_full_response=raw_response,
        reasoning=parts["reasoning"],
        memory=parts["memory"],
        state=parts["state"],
    )


def load_state_memory_jsonl(
    path: str | Path,
    expected_steps: list[int] | None = None,
) -> dict[int, Memory]:
    """Load and validate raw state-memory labels keyed by trajectory step."""
    path = Path(path)
    labels: dict[int, Memory] = {}
    observed_steps: list[int] = []

    with path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                raise StateMemoryLabelError(f"{path}:{line_num}: blank lines are not allowed.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateMemoryLabelError(f"{path}:{line_num}: invalid JSON: {exc}") from exc

            if not isinstance(record, dict) or set(record) != {"step", "memory"}:
                raise StateMemoryLabelError(
                    f"{path}:{line_num}: each line must contain exactly 'step' and 'memory'."
                )
            step = record["step"]
            if isinstance(step, bool) or not isinstance(step, int):
                raise StateMemoryLabelError(f"{path}:{line_num}: 'step' must be an integer.")
            if step in labels:
                raise StateMemoryLabelError(f"{path}:{line_num}: duplicate step {step}.")

            try:
                labels[step] = parse_state_memory_response(record["memory"])
            except StateMemoryLabelError as exc:
                raise StateMemoryLabelError(f"{path}:{line_num}: {exc}") from exc
            observed_steps.append(step)

    if observed_steps != sorted(observed_steps):
        raise StateMemoryLabelError(f"{path}: steps must be in ascending order.")
    if expected_steps is not None and observed_steps != expected_steps:
        raise StateMemoryLabelError(
            f"{path}: expected steps {expected_steps}, found {observed_steps}."
        )
    return labels
