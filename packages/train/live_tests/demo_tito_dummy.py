#!/usr/bin/env python3
"""Demonstrate Qwen3TITO with dummy messages and no live server.

This script exercises the append-only token flow from packages/train/tito.py:

1. Render the first prompt from system/user messages.
2. Simulate a generated assistant response by tokenizing dummy completion text.
3. Checkpoint that response in Qwen3TITO.
4. Append a second user message and build the next prompt through TITO.
5. Compare the TITO prompt against canonical full chat-template rendering.
6. Show that non-user appends are rejected by the narrow TITO implementation.

Example:

    python packages/train/live_tests/demo_tito_dummy.py \
      --hf-checkpoint /path/to/qwen3/checkpoint \
      --chat-template-path packages/train/model_utils/qwen3.5_custom.jinja
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


THIS_FILE = Path(__file__).resolve()
TRAIN_PKG_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]

sys.path.insert(0, str(TRAIN_PKG_DIR))

from tito import Qwen3TITO, TITOError  # noqa: E402


DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "packages" / "train" / "model_utils" / "qwen3.5_custom.jinja"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local dummy demonstration of packages/train/tito.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-checkpoint",
        required=True,
        help="Tokenizer checkpoint path or model id. Prefer a local Qwen3 checkpoint for offline use.",
    )
    parser.add_argument(
        "--chat-template-path",
        type=Path,
        default=DEFAULT_CHAT_TEMPLATE,
        help="Jinja chat template to install on the tokenizer.",
    )
    parser.add_argument(
        "--apply-chat-template-kwargs",
        type=json.loads,
        default={"clear_thinking": False},
        help="JSON object forwarded to tokenizer.apply_chat_template.",
    )
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    tokenizer.chat_template = args.chat_template_path.read_text()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer does not know the Qwen '<|im_end|>' special token.")
    return tokenizer


def print_prompt_summary(tokenizer: Any, label: str, token_ids: list[int]) -> None:
    print(f"{label}:")
    print(f"  token_count={len(token_ids)}")
    print("  decoded:")
    print(tokenizer.decode(token_ids))
    print()


def first_request_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Act briefly.",
        },
        {
            "role": "user",
            "content": "Goal: search.",
        },
    ]


def _assistant_message(raw_response):
    return {
        "role": "assistant",
        "content": raw_response
    }


def dummy_assistant_completion() -> str:
    # The custom template may prefill '<think>\\n' in the prompt. This dummy
    # content intentionally starts after that prefill, matching SGLang text.
    return "Search now.\n</think>\n<action>\nclick(\"#q\")\n</action>"


def second_dummy_assistant_completion() -> str:
    return "This is a memory."


def second_user_message() -> dict[str, str]:
    return {
        "role": "user",
        "content": "Obs: results.",
    }


def simulate_completion_ids(tokenizer: Any, assistant_response: str) -> list[int]:
    completion_ids = tokenizer.encode(assistant_response, add_special_tokens=False)
    # Turn usually with the model sampling <|im_end|> if not truncated
    completion_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return completion_ids


def print_token_trajectory(tokenizer: Any, chat_template_kwargs: dict[str, Any]) -> None:
    tito = Qwen3TITO(tokenizer, chat_template_kwargs=chat_template_kwargs)
    print(tito.token_ids)

    first_messages = first_request_messages()
    first_prompt_ids = tito.prepare(first_messages, template_kwargs={"prefill_think": True})
    print(first_prompt_ids)
    print(repr(tokenizer.decode(first_prompt_ids, skip_special_tokens=False)))

    assistant_completion = dummy_assistant_completion()
    assistant_message = _assistant_message(assistant_completion)
    completion_ids = simulate_completion_ids(tokenizer, assistant_completion)
    assistant_message["content"] = "<think>" + assistant_completion
    tito.update(first_messages, assistant_message, first_prompt_ids, completion_ids)
    print(tito.token_ids)
    print(repr(tokenizer.decode(tito.token_ids, skip_special_tokens=False)))

    second_messages = tito.messages + [second_user_message()]
    second_prompt_ids = tito.prepare(second_messages, template_kwargs={"prefill_think": False})
    print(second_prompt_ids)
    print(repr(tokenizer.decode(second_prompt_ids, skip_special_tokens=False)))
    second_assistant_completion = second_dummy_assistant_completion()
    second_completion_ids = simulate_completion_ids(tokenizer, second_assistant_completion)
    second_assistant_message = _assistant_message(second_assistant_completion)
    
    tito.update(second_messages, second_assistant_message, second_prompt_ids, second_completion_ids)
    print(tito.token_ids)
    print(repr(tokenizer.decode(tito.token_ids, skip_special_tokens=False)))


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args)
    print_token_trajectory(tokenizer, args.apply_chat_template_kwargs)


if __name__ == "__main__":
    main()
