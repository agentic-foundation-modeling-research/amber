"""Unit tests for trajectory classes (no BrowserGym env required)."""

import pytest

from context_scythe.agents.trajectory_data import (
    Observation,
    ActionParseError,
    ReasoningParseError,
    Response,
    StepData,
    StepDataWithMemory,
)
from trajectory_primitives import (
    DUMMY_SUMMARY_TEXT,
    build_two_step_single_turn,
    build_two_step_single_turn_w_mem,
    make_observation,
    make_response,
)


class TestObservation:
    def test_compose_axtree_only(self):
        """AXTree-only observation produces a single text part with the tree."""
        obs = make_observation(use_tabs_info=False)
        parts = obs.content()
        
        assert len(parts) == 1
        assert "Accessibility Tree" in parts[0]["text"]
        assert "Welcome" in parts[0]["text"]

    def test_compose_tabs_info(self):
        """Tabs info is included in the composed message when enabled."""
        obs = make_observation(
            use_tabs_info=True,
            open_pages_urls=["https://example.com"],
            open_pages_titles=["Example"],
            active_page_index=[0],
        )
        
        parts = obs.content()
        # Should have axtree + tabs
        texts = [p["text"] for p in parts]
        tabs_text = next(t for t in texts if "Currently open tabs" in t)
        assert "Tab 0 (active tab)" in tabs_text
        assert "https://example.com" in tabs_text

    def test_axtree_truncation(self):
        """AXTree is truncated when max_tokens is set low."""
        obs = make_observation(axtree_max_tokens=2, use_tabs_info=False)  # ~8 chars
        parts = obs.content()
        print(parts)
        text = parts[0]["text"]
        assert "... (truncated)" in text

    def test_json_roundtrip(self):
        """TurnObservation survives JSON serialization round-trip."""
        obs = make_observation(
            use_tabs_info=True,
            open_pages_urls=["https://a.com"],
            open_pages_titles=["A"],
            active_page_index=0,
            axtree_max_tokens=100,
        )
        restored = Observation.from_json(obs.to_json())
        assert restored.axtree == obs.axtree
        assert restored.use_axtree == obs.use_axtree
        assert restored.open_pages_urls == obs.open_pages_urls
        assert restored.axtree_max_tokens == obs.axtree_max_tokens


# ===========================================================================
# TurnResponse tests
# ===========================================================================


class TestTurnResponse:
    def test_parse_action(self):
        """Action is extracted from <action> block."""
        _, action = Response.parse_action(make_response().model_full_response)
        assert action == 'click("3")'

    def test_parse_action_returns_prefix_and_action(self):
        """Action parsing returns text before the final action block and action."""
        prefix, action = Response.parse_action("blabla<action>some_action</action>")
        assert prefix == "blabla"
        assert action == "some_action"

    def test_parse_action_uses_last_action_block(self):
        """Only the last action block is extracted from the returned prefix."""
        prefix, action = Response.parse_action(
            "a<action>one</action>b<action>two</action>c"
        )
        assert prefix == "a<action>one</action>b"
        assert action == "two"

    def test_parse_reasoning(self):
        """Reasoning is extracted from <think> block."""
        reasoning = Response.parse_reasoning(make_response().model_full_response)
        assert reasoning == "I see the page. I will click the link."

    def test_parse_reasoning_uses_last_think_block(self):
        """Reasoning parsing returns content from the last think block."""
        reasoning = Response.parse_reasoning(
            "a<think>one</think>b<think>two</think>c"
        )
        assert reasoning == "two"

    def test_no_think_block(self):
        """Response constructor leaves parsed fields unset."""
        resp = Response(model_full_response='<action>\nclick("1")\n</action>')
        assert resp.reasoning is None
        assert resp.action is None

    def test_no_action_block(self):
        """Missing <action> block yields None action."""
        resp = Response(model_full_response="just some raw text")
        assert resp.action is None

    def test_parse_reasoning_raise_on_parse_error(self):
        """Malformed reasoning text raises ReasoningParseError when enabled."""
        with pytest.raises(ReasoningParseError):
            Response.parse_reasoning("just some raw text", True)

    def test_parse_action_raise_on_parse_error(self):
        """Malformed action text raises ActionParseError when enabled."""
        with pytest.raises(ActionParseError):
            Response.parse_action("just some raw text", True)

    def test_json_roundtrip(self):
        """TurnResponse with token IDs and log probs survives round-trip."""
        resp = make_response()
        resp.response_token_ids = [10, 20, 30]
        resp.response_log_probs = [-0.1, -0.2, -0.3]
        data = resp.to_json()
        restored = Response.from_json(data)
        assert restored.model_full_response == resp.model_full_response
        assert restored.response_token_ids == [10, 20, 30]
        assert restored.response_log_probs == [-0.1, -0.2, -0.3]

    def test_compose_message(self):
        """Composed message includes the full response text."""
        resp = make_response()
        parts = resp.content()
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert "<think>" in parts[0]["text"]


class TestStepData:
    def test_no_response(self):
        """Response-seeking message contains observation and action instruction."""
        step_data = StepData(step_num=0, observation=make_observation())
        assert step_data.response is None    

    def test_observation_content(self):
        """Full turn message has user observation and assistant response."""
        obs = make_observation()
        step_data = StepData(step_num=0, observation=obs)
        assert step_data.observation_message_content() == obs.content()

    def test_with_response(self):
        obs = make_observation()
        response = make_response()
        step_data = StepData(step_num=0, observation=obs, response=response)
        assert step_data.response_message_content() == response.content()

    def test_json_with_response(self):
        obs = make_observation()
        response = make_response()
        step_data = StepData(step_num=0, observation=obs, response=response)

        data = step_data.to_json()

        assert data["type"] == "StepData"
        assert data["step_num"] == 0
        assert data["observation"] == obs.to_json()
        assert data["response"] == response.to_json()

    def test_json_without_response(self):
        obs = make_observation()
        step_data = StepData(step_num=1, observation=obs)

        data = step_data.to_json()

        assert data["type"] == "StepData"
        assert data["step_num"] == 1
        assert data["observation"] == obs.to_json()
        assert "response" not in data


class TestStepDataWithMemory:
    def test_memory(self):
        obs = make_observation()
        response = make_response()
        step_data = StepDataWithMemory(step_num=0, observation=obs, response=response, memory=DUMMY_SUMMARY_TEXT)
        assert step_data.memory == DUMMY_SUMMARY_TEXT

    def test_json_includes_memory(self):
        obs = make_observation()
        response = make_response()
        step_data = StepDataWithMemory(step_num=0, observation=obs, response=response, memory=DUMMY_SUMMARY_TEXT)

        data = step_data.to_json()

        assert data["type"] == "StepDataWithMemory"
        assert data["step_num"] == 0
        assert data["observation"] == obs.to_json()
        assert data["response"] == response.to_json()
        assert data["memory"] == DUMMY_SUMMARY_TEXT


class TestTrajectory:
    def test_num_turns(self):
        trajectory = build_two_step_single_turn()
        assert len(trajectory.steps) == 2

    def test_update_reward(self):
        import math
        trajectory = build_two_step_single_turn()
        trajectory.update_reward(1.0)
        assert math.fabs(trajectory.reward - 1.0) < 1e-8

    def test_json_includes_metadata_and_steps(self):
        trajectory = build_two_step_single_turn()
        trajectory.reward = 1.0
        trajectory.terminated = True
        trajectory.truncated = False

        data = trajectory.to_json()

        assert data["type"] == "TrajectoryData"
        assert data["goal"] == trajectory.goal
        assert data["calculator_url"] == trajectory.calculator_url
        assert data["site_urls"] == trajectory.site_urls
        assert data["reward"] == 1.0
        assert data["terminated"] is True
        assert data["truncated"] is False
        assert data["steps"] == [step.to_json() for step in trajectory.steps]

    def test_json_includes_none_status_fields(self):
        trajectory = build_two_step_single_turn()

        data = trajectory.to_json()

        assert "reward" in data
        assert "terminated" in data
        assert "truncated" in data
        assert data["reward"] is None
        assert data["terminated"] is None
        assert data["truncated"] is None


class TestTrajectoryDataWithMemory:
    def test_json_includes_memory_type_and_steps(self):
        trajectory = build_two_step_single_turn_w_mem()

        data = trajectory.to_json()

        assert data["type"] == "TrajectoryDataWithMemory"
        assert data["goal"] == trajectory.goal
        assert data["calculator_url"] == trajectory.calculator_url
        assert data["site_urls"] == trajectory.site_urls
        assert data["steps"] == [step.to_json() for step in trajectory.steps]
