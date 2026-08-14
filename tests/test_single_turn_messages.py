import pytest

from browsergym.core.action.highlevel import HighLevelActionSet

from context_scythe.agents.prompt_builders.single_turn import SingleTurnPromptBuilder
from context_scythe.agents.trajectory_data import (
    StepData,
    TrajectoryData,
)
from trajectory_primitives import (
    build_two_step_single_turn,
    make_action_set,
    make_observation,
)


@pytest.fixture
def builder():
    return SingleTurnPromptBuilder()


@pytest.fixture
def action_set():
    return make_action_set()


def content_texts(message):
    return [part["text"] for part in message["content"]]


def build_unanswered_trajectory(goal="Submit the form"):
    trajectory = TrajectoryData(
        goal=goal,
        calculator_url="http://dummy.html",
        site_urls={"dummy": "http://dummy_site.com"}
    )
    trajectory.add_step(
        StepData(
            step_num=0,
            observation=make_observation(use_tabs_info=False),
        )
    )
    return trajectory


class TestSingleTurnPromptBuilder:
    def test_first_step_prompt_structure_without_response(self, builder: SingleTurnPromptBuilder, action_set: HighLevelActionSet):
        trajectory = build_unanswered_trajectory()

        output = builder.build_messages(0, trajectory, action_set)

        # Builder should return only the prompt payload when no response is present.
        assert set(output) == {"prompt"}
        # Prompt should contain the expected system and user messages in order.
        assert [message["role"] for message in output["prompt"]] == ["system", "user"]

        system_message, user_message = output["prompt"]
        # System message content should be represented as text.
        assert system_message["content"][0]["type"] == "text"
        # System message should include the instructions section.
        assert "# Instructions" in system_message["content"][0]["text"]

        texts = content_texts(user_message)
        # Every user message content part should be text.
        assert [part["type"] for part in user_message["content"]] == ["text"] * len(texts)
        # User prompt should start with the requested goal.
        assert texts[0].startswith("# Goal\n\nSubmit the form")
        # Current page accessibility tree should be included.
        assert "# Current page Accessibility Tree" in texts[1]
        # Action space section should follow the page context.
        assert texts[2].startswith("# Action Space")
        # Allowed website content
        assert texts[3].startswith("# Allowed list of websites")
        # Final instruction should ask for the next action.
        assert texts[4].startswith("# Next action")
        # First-step prompt should not include past action history.
        assert not any("# History of past actions" in text for text in texts)

    def test_response_structure_when_step_has_response(self, builder: SingleTurnPromptBuilder, action_set: HighLevelActionSet):
        trajectory = build_two_step_single_turn()

        output = builder.build_messages(0, trajectory, action_set)

        # Response payload should include exactly one assistant response message.
        assert len(output["response"]) == 1
        assistant_message = output["response"][0]
        # Response message should have the assistant role.
        assert assistant_message["role"] == "assistant"
        # Assistant content should match the trajectory step response content.
        assert assistant_message["content"] == trajectory.steps[0].response_message_content()
        # Assistant text should preserve the full model response.
        assert assistant_message["content"][0]["text"] == trajectory.steps[0].response.model_full_response

    def test_later_step_includes_past_action_history_before_final_instruction(
        self, builder: SingleTurnPromptBuilder, action_set: HighLevelActionSet
    ):
        trajectory = build_two_step_single_turn()

        output = builder.build_messages(1, trajectory, action_set)
        texts = content_texts(output["prompt"][1])

        history_text = texts[-3]
        # past action history should be before allowed websites and final instruction
        assert history_text.startswith("# History of past actions")
        # Prior assistant reasoning should be included in history.
        assert "I see the page. I will click the link." in history_text
        # Prior action for the earlier step should be included in history.
        assert 'click("3")' in history_text
        # Current step action should be excluded from past action history.
        assert 'click("7")' not in history_text
        # Final user prompt section should ask for the next action.
        assert texts[-1].startswith("# Next action")
