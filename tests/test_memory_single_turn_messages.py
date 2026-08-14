import pytest

from browsergym.core.action.highlevel import HighLevelActionSet

from context_scythe.agents.prompt_builders.memory_single_turn import (
    SingleTurnWithMemoryPromptBuilder,
)
from context_scythe.agents.trajectory_data import (
    StepDataWithMemory,
    TrajectoryDataWithMemory,
)
from trajectory_primitives import (
    build_two_step_single_turn_w_mem,
    make_action_set,
    make_observation,
)


@pytest.fixture
def builder():
    return SingleTurnWithMemoryPromptBuilder()


@pytest.fixture
def action_set():
    return make_action_set()


def content_texts(message):
    return [part["text"] for part in message["content"]]


def build_unanswered_trajectory(goal="Submit the form"):
    trajectory = TrajectoryDataWithMemory(
        goal=goal,
        calculator_url="http://dummy.html",
        site_urls={"dummy": "http://dummy_site.com"}
    )
    trajectory.add_step(
        StepDataWithMemory(
            step_num=0,
            observation=make_observation(use_tabs_info=False, filter_viewport=True),
        )
    )
    return trajectory


class TestSingleTurnWithMemoryPromptBuilder:
    def test_first_step_action_prompt_structure_without_response(
        self,
        builder: SingleTurnWithMemoryPromptBuilder,
        action_set: HighLevelActionSet,
    ):
        trajectory = build_unanswered_trajectory()

        output = builder.build_messages(0, "action", trajectory, action_set)

        # Builder should return only the prompt payload when no response or memory is present.
        assert set(output) == {"prompt"}
        # Prompt should contain the expected system and user messages in order.
        assert [message["role"] for message in output["prompt"]] == ["system", "user"]

        system_message, user_message = output["prompt"]
        # System message content should be represented as text.
        assert system_message["content"][0]["type"] == "text"
        # System message should include the instructions section from the base builder.
        assert "# Instructions" in system_message["content"][0]["text"]

        texts = content_texts(user_message)
        # Every user message content part should be text.
        assert [part["type"] for part in user_message["content"]] == ["text"] * len(texts)
        # User prompt should start with the requested goal.
        assert texts[0].startswith("# Goal\n\nSubmit the form")
        # Current page accessibility tree should be included.
        assert "# Current page Accessibility Tree" in texts[1]
        # Viewport context is included with the page context.
        assert texts[2].startswith("# Current Viewport")
        # Action space section should follow the page context.
        assert texts[3].startswith("# Action Space")
        # Allowed website content
        assert texts[4].startswith("# Allowed list of websites")
        # Final instruction should ask for the next action.
        assert texts[5].startswith("# Next action")
        # First-step prompt should not include past action or memory history.
        assert not any("# History of past actions" in text for text in texts)
        assert not any("# History of past memories" in text for text in texts)

    def test_compression_prompt_includes_action_response_and_memory_instruction(
        self,
        builder: SingleTurnWithMemoryPromptBuilder,
        action_set: HighLevelActionSet,
    ):
        trajectory = build_two_step_single_turn_w_mem()

        output = builder.build_messages(0, "compression", trajectory, action_set)

        # Compression mode should append the action response and a memory-generation prompt.
        assert [message["role"] for message in output["prompt"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

        action_response = output["prompt"][2]
        # Assistant action response should match the trajectory step response content.
        assert action_response["content"] == trajectory.steps[0].response_message_content()
        assert 'click("3")' in action_response["content"][0]["text"]

        compression_instruction = output["prompt"][3]
        instruction_texts = content_texts(compression_instruction)
        # Compression instruction should ask the model to create memory.
        assert instruction_texts[0].startswith("# Memory")
        assert "The memory block is the only thing carried over between steps" in instruction_texts[0]

        # Existing memory should be exposed as the compression response payload.
        assert output["compression_response"]["role"] == "assistant"
        assert output["compression_response"]["content"] == [
            {"type": "text", "text": "Step 0 Memory."}
        ]

    def test_later_step_includes_past_action_and_memory_history_before_final_instruction(
        self,
        builder: SingleTurnWithMemoryPromptBuilder,
        action_set: HighLevelActionSet,
    ):
        trajectory = build_two_step_single_turn_w_mem()

        output = builder.build_messages(1, "action", trajectory, action_set)
        texts = content_texts(output["prompt"][1])

        history_text = texts[-3]
        # past action history should be before allowed websites and final instruction
        assert history_text.startswith("# History of past actions")
        assert "# History of past memories" in history_text
        # Prior action should be included in history.
        assert 'click("3")' in history_text
        # Prior memory should be included and labeled by step.
        assert "## Step 0\n\nStep 0 Memory." in history_text
        # Current step action and memory should be excluded from past history.
        assert 'click("7")' not in history_text
        assert "Step 1 Memory." not in history_text
        # Final user prompt section should ask for the next action.
        assert texts[-1].startswith("# Next action")

        # Current step memory should still be exposed as the compression response payload.
        assert output["compression_response"]["content"] == [
            {"type": "text", "text": "Step 1 Memory."}
        ]
