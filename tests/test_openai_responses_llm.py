from types import SimpleNamespace
from unittest.mock import MagicMock

from context_scythe.agents.llm import OpenAIResponsesLLM


def test_normalize_messages_flattens_legacy_text_blocks_without_mutating_input():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ],
        }
    ]

    normalized = OpenAIResponsesLLM._normalize_messages(messages)

    assert normalized == [{"role": "user", "content": "First\nSecond"}]
    assert messages[0]["content"][0]["type"] == "text"


def test_normalize_messages_preserves_responses_native_content():
    messages = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Already valid"}],
        }
    ]

    normalized = OpenAIResponsesLLM._normalize_messages(messages)

    assert normalized == messages


def test_chat_sends_flattened_messages_to_responses_api():
    llm = OpenAIResponsesLLM.__new__(OpenAIResponsesLLM)
    llm.model_name = "test-model"
    llm.max_tokens = 128
    llm.extra_body = None
    llm.client = MagicMock()
    llm.client.responses.create.return_value = SimpleNamespace(
        output_text="ok",
        usage=None,
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Instructions"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Question"}],
        },
    ]

    response, misc = llm.chat(messages)

    assert response == "ok"
    assert misc == {"usage": {}}
    llm.client.responses.create.assert_called_once_with(
        model="test-model",
        input=[
            {"role": "system", "content": "Instructions"},
            {"role": "user", "content": "Question"},
        ],
        max_output_tokens=128,
    )
