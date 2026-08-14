"""Minimal standalone TITO support for append-only chat trajectories.

TITO means "token-in/token-out": instead of re-rendering and re-tokenizing an
entire multi-turn chat on every model call, we keep the exact token sequence
that has already gone into and come out of the model, then append only the
tokens needed for the next non-assistant turn and generation prompt.

The intended session shape is:

1. First request:
   Render the full request messages with the chat template and
   ``add_generation_prompt=True``. These prompt token IDs are sent to the model.

2. Model response:
   Collect the model's generated token IDs from the runtime response. Store a
   checkpoint equal to ``prompt_token_ids + completion_token_ids`` along with
   the corresponding messages, including the assistant message.

3. Later request:
   Validate that the new request is an append-only extension of the stored
   messages, allowing only configured non-assistant roles such as ``tool`` and,
   if explicitly enabled, ``user`` or ``system``. Compute token IDs only for
   those appended messages plus the next assistant generation prompt. Merge
   those incremental IDs onto the stored checkpoint and send the result as the
   next request's prompt token IDs.

The important invariant is that the accumulated TITO token sequence should
match the token sequence produced by canonical full chat-template rendering,
except for any model-specific boundary tokens we deliberately normalize. Some
chat templates have boundary quirks: a model may stop before a trailing newline
that the template would render, or it may emit a token that is both an assistant
stop marker and the next message's start marker. A practical implementation
therefore needs a small, model-specific merge hook around the checkpoint /
incremental-token boundary.

This module is intentionally independent of Miles session-server internals. The
target integration is direct SGLang ``/generate`` usage, where callers already
own prompt construction, send ``input_ids``, and receive generated token IDs
from ``meta_info.output_token_logprobs``.

The first implementation should be narrow:

- Append surface: only a new ``user`` message after an assistant response.
- Model family: Qwen3-style chat templates, where assistant messages render as
  ``<|im_start|>assistant ... <|im_end|>\n`` but generation may stop after
  ``<|im_end|>`` without the trailing newline.
- Boundary fix: when the stored checkpoint ends in the Qwen3 ``<|im_end|>``
  token, insert the tokenizer's newline token before appending the next user
  turn and assistant generation prompt.

For this minimal implementation, the core pieces are:

- A trajectory object that stores ``messages`` and the latest accumulated
  ``token_ids`` checkpoint.
- An append-only validator for request messages.
- A way to render and tokenize the first full prompt.
- A way to tokenize appended non-assistant messages in a synthetic context that
  preserves role boundary tokens.
- A merge step that joins the previous checkpoint with incremental suffix
  tokens and applies any model-specific boundary fix.
- A checkpoint update after each successful assistant response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TITOError(ValueError):
    """Raised when a request cannot be represented as a TITO append."""


_DUMMY_SYSTEM = {"role": "system", "content": "dummy system"}


@dataclass
class Qwen3TITO:
    """Minimal Qwen3 TITO accumulator for appending user turns."""

    tokenizer: Any
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise TITOError(f"expected newline to be one token, got {newline_ids}")
        self._newline_id = newline_ids[0]
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def prepare(
        self,
        request_messages: list[dict[str, Any]],
        *,
        template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Return prompt token IDs for ``request_messages``."""
        if not self.token_ids:
            return self.render_messages(request_messages, add_generation_prompt=True, template_kwargs=template_kwargs)

        self._assert_user_append(request_messages)
        return self._merge_tokens(request_messages, template_kwargs=template_kwargs)

    def update(
        self,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
    ) -> None:
        """Checkpoint a completed assistant turn."""
        all_token_ids = list(prompt_token_ids) + list(completion_token_ids)
        if self.token_ids and all_token_ids[: len(self.token_ids)] != self.token_ids:
            raise TITOError("new checkpoint is not prefixed by the previous checkpoint")

        self.messages = list(request_messages) + [assistant_message]
        self.token_ids = all_token_ids

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        text = self._render_text(messages, add_generation_prompt=add_generation_prompt, template_kwargs=template_kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _merge_tokens(
        self,
        request_messages: list[dict[str, Any]],
        *,
        template_kwargs: dict[str, Any] | None,
    ) -> list[int]:
        prefix = list(self.token_ids)
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
        return prefix + self._tokenize_user_append(request_messages, template_kwargs=template_kwargs)

    def _tokenize_user_append(
        self,
        request_messages: list[dict[str, Any]],
        *,
        template_kwargs: dict[str, Any] | None,
    ) -> list[int]:
        appended_messages = request_messages[len(self.messages) :]
        incremental: list[int] = []
        for message in appended_messages:
            incremental.extend(
                self._tokenize_rendered_suffix(
                    [_DUMMY_SYSTEM],
                    [message],
                    add_generation_prompt=False,
                    template_kwargs=template_kwargs,
                )
            )
        incremental.extend(
            self._tokenize_rendered_suffix(
                request_messages,
                [],
                add_generation_prompt=True,
                template_kwargs=template_kwargs,
            )
        )
        return incremental

    def _tokenize_rendered_suffix(
        self,
        base_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        template_kwargs: dict[str, Any] | None,
    ) -> list[int]:
        text_without = self._render_text(
            base_messages,
            add_generation_prompt=False,
            template_kwargs=template_kwargs,
        )
        text_with = self._render_text(
            base_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            template_kwargs=template_kwargs,
        )
        if not text_with.startswith(text_without):
            roles = [message.get("role") for message in appended_messages] or ["generation_prompt"]
            raise TITOError(f"chat template is not append-only for {roles}")
        return self.tokenizer.encode(text_with[len(text_without) :], add_special_tokens=False)

    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        template_kwargs: dict[str, Any] | None,
    ) -> str:
        kwargs = {**self.chat_template_kwargs, **(template_kwargs or {})}
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def _assert_user_append(self, request_messages: list[dict[str, Any]]) -> None:
        if request_messages[: len(self.messages)] != self.messages:
            raise TITOError("request messages are not an append-only extension of stored messages")

        appended_messages = request_messages[len(self.messages) :]
        if not appended_messages:
            return
        bad_roles = [message.get("role") for message in appended_messages if message.get("role") != "user"]
        if bad_roles:
            raise TITOError(f"only user appends are supported, got {bad_roles}")
