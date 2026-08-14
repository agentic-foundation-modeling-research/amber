"""WebArena SFT rollout builder for Miles.

This rollout function expects Miles' dataset loader to pass raw OpenAI-style
message lists from parquet (`--input-key prompt`) and therefore should be used
without `--apply-chat-template`. Chat formatting is owned here instead:

- `--chat-template-path` is still honored when the tokenizer is loaded, so the
  same custom Qwen3.5 template used for WebArena RL can be used for SFT.
- `--apply-chat-template-kwargs` is still honored by this builder when it calls
  `tokenizer.apply_chat_template`.
- `clear_thinking` defaults to false for stable Qwen3.5 prefix slicing across
  the action turn and the later memory turn.

The explicit rendering here is intentional. We need exact token spans for:
action output, inserted memory prompt, and memory output, with loss only on the
two assistant outputs.
"""

import logging
from dataclasses import dataclass
from typing import Any

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

__all__ = ["generate_sft", "generate_bc"]

logger = logging.getLogger(__name__)


TOKENIZER = None
SFT_BUILDER = None
BC_BUILDER = None
SAMPLE_PRINTED = False


EXPECTED_ROLES = ("system", "user", "assistant", "user", "assistant")
EXPECTED_BC_ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class AssistantSpans:
    action_prompt_ids: list[int]
    action_output_ids: list[int]
    inserted_memory_prompt_ids: list[int]
    memory_output_ids: list[int]

    @property
    def tokens(self) -> list[int]:
        return (
            self.action_prompt_ids
            + self.action_output_ids
            + self.inserted_memory_prompt_ids
            + self.memory_output_ids
        )

    @property
    def response_ids(self) -> list[int]:
        return self.action_output_ids + self.inserted_memory_prompt_ids + self.memory_output_ids


@dataclass(frozen=True)
class BCSpans:
    prompt_ids: list[int]
    output_ids: list[int]

    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def response_ids(self) -> list[int]:
        return self.output_ids


@dataclass(frozen=True)
class SFTBuildResult:
    tokens: list[int]
    response_length: int
    loss_mask: list[int]
    response_ids: list[int]
    spans: Any


class QwenAssistantMaskPolicy:
    """Mask template-owned assistant prefill tokens from supervised spans."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Qwen3.5 can insert the first assistant turn's thinking opener through
        # the generation prompt. We do not want to teach the model to emit that
        # template-owned opener, but we still train on the reasoning that follows.
        self.think_open_ids = tokenizer.encode("<think>\n", add_special_tokens=False)

    def build_loss_mask(self, output_ids: list[int]) -> list[int]:
        loss_mask = [1] * len(output_ids)
        prefix = self.think_open_ids
        if prefix and output_ids[: len(prefix)] == prefix:
            loss_mask[: len(prefix)] = [0] * len(prefix)
        return loss_mask


class WebArenaSFTBuilder:
    """Build explicit SFT tokens/masks for WebArena action plus memory turns."""

    def __init__(self, tokenizer, *, template_kwargs: dict[str, Any], validate: bool):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs)
        self.validate = validate
        self.mask_policy = QwenAssistantMaskPolicy(tokenizer)

    def build(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> SFTBuildResult:
        self._validate_messages(messages)

        # Build token spans by rendering canonical chat prefixes and taking
        # prefix differences. This keeps the supervised boundary aligned with
        # the tokenizer's chat template instead of guessing where role markers
        # and generation prompts end.
        action_prompt_ids = self._render_ids(messages[:2], add_generation_prompt=True, tools=tools)
        action_full_ids = self._render_ids(messages[:3], add_generation_prompt=False, tools=tools)
        self._require_prefix(action_full_ids, action_prompt_ids, "action response")
        action_output_ids = action_full_ids[len(action_prompt_ids) :]

        memory_prompt_ids = self._render_ids(messages[:4], add_generation_prompt=True, tools=tools)
        self._require_prefix(memory_prompt_ids, action_full_ids, "memory prompt")
        inserted_memory_prompt_ids = memory_prompt_ids[len(action_full_ids) :]

        full_ids = self._render_ids(messages[:5], add_generation_prompt=False, tools=tools)
        self._require_prefix(full_ids, memory_prompt_ids, "memory response")
        memory_output_ids = full_ids[len(memory_prompt_ids) :]

        spans = AssistantSpans(
            action_prompt_ids=action_prompt_ids,
            action_output_ids=action_output_ids,
            inserted_memory_prompt_ids=inserted_memory_prompt_ids,
            memory_output_ids=memory_output_ids,
        )
        # The action and memory assistant outputs are supervised. The inserted
        # memory prompt is context needed to reach the memory response and must
        # remain in the response suffix with zero loss.
        loss_mask = (
            self.mask_policy.build_loss_mask(action_output_ids)
            + [0] * len(inserted_memory_prompt_ids)
            + self.mask_policy.build_loss_mask(memory_output_ids)
        )
        result = SFTBuildResult(
            tokens=spans.tokens,
            response_length=len(spans.response_ids),
            loss_mask=loss_mask,
            response_ids=spans.response_ids,
            spans=spans,
        )

        self._validate_result(result)
        return result

    def describe_supervised_spans(self, result: SFTBuildResult, *, max_chars: int = 500) -> str:
        rows = []
        for name, mask_value, ids in (
            ("action_output", 1, result.spans.action_output_ids),
            ("inserted_memory_prompt", 0, result.spans.inserted_memory_prompt_ids),
            ("memory_output", 1, result.spans.memory_output_ids),
        ):
            decoded = self.tokenizer.decode(ids)
            if len(decoded) > max_chars:
                head_chars = max_chars // 4
                tail_chars = max_chars - head_chars
                decoded = decoded[:head_chars] + "..." + decoded[-tail_chars:]
            rows.append(f"{name}: mask={mask_value} tokens={len(ids)} decoded={decoded!r}")
        return "\n".join(rows)

    def _render_ids(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.template_kwargs,
        )

    def _validate_messages(self, messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list):
            raise TypeError(f"WebArena SFT prompt must be a message list, got {type(messages)}")
        if len(messages) != len(EXPECTED_ROLES):
            raise ValueError(f"WebArena SFT prompt must contain {len(EXPECTED_ROLES)} messages, got {len(messages)}")

        roles = tuple(message.get("role") for message in messages)
        if roles != EXPECTED_ROLES:
            raise ValueError(f"WebArena SFT roles must be {EXPECTED_ROLES}, got {roles}")

        for index, message in enumerate(messages):
            if not isinstance(message.get("content"), str):
                raise TypeError(f"WebArena SFT message {index} content must be a string")

    def _validate_result(self, result: SFTBuildResult) -> None:
        if result.tokens[-result.response_length :] != result.response_ids:
            raise ValueError("response_ids are not the suffix of the training token sequence")
        if len(result.loss_mask) != result.response_length:
            raise ValueError(
                f"loss mask length {len(result.loss_mask)} != response length {result.response_length}"
            )
        if sum(result.loss_mask) == 0:
            raise ValueError("loss mask contains no supervised tokens")

        if not self.validate:
            return

        action_mask = result.loss_mask[: len(result.spans.action_output_ids)]
        memory_start = len(result.spans.action_output_ids) + len(result.spans.inserted_memory_prompt_ids)
        memory_mask = result.loss_mask[memory_start:]
        if sum(action_mask) == 0:
            raise ValueError("action output contains no supervised tokens")
        if sum(memory_mask) == 0:
            raise ValueError("memory output contains no supervised tokens")

    @staticmethod
    def _require_prefix(full_ids: list[int], prefix_ids: list[int], label: str) -> None:
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(f"chat template prefix mismatch while building {label}")


class WebArenaBCBuilder:
    """Build explicit BC tokens/masks for single-turn WebArena action examples."""

    def __init__(self, tokenizer, *, template_kwargs: dict[str, Any], validate: bool):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs)
        self.validate = validate
        self.mask_policy = QwenAssistantMaskPolicy(tokenizer)

    def build(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> SFTBuildResult:
        self._validate_messages(messages)

        prompt_ids = self._render_ids(messages[:2], add_generation_prompt=True, tools=tools)
        full_ids = self._render_ids(messages[:3], add_generation_prompt=False, tools=tools)
        self._require_prefix(full_ids, prompt_ids, "bc response")
        output_ids = full_ids[len(prompt_ids) :]

        spans = BCSpans(prompt_ids=prompt_ids, output_ids=output_ids)
        loss_mask = self.mask_policy.build_loss_mask(output_ids)
        result = SFTBuildResult(
            tokens=spans.tokens,
            response_length=len(spans.response_ids),
            loss_mask=loss_mask,
            response_ids=spans.response_ids,
            spans=spans,
        )

        self._validate_result(result)
        return result

    def describe_supervised_spans(self, result: SFTBuildResult, *, max_chars: int = 500) -> str:
        decoded = self.tokenizer.decode(result.spans.output_ids)
        if len(decoded) > max_chars:
            head_chars = max_chars // 4
            tail_chars = max_chars - head_chars
            decoded = decoded[:head_chars] + "..." + decoded[-tail_chars:]
        return f"output: mask=1 tokens={len(result.spans.output_ids)} decoded={decoded!r}"

    def _render_ids(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.template_kwargs,
        )

    def _validate_messages(self, messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list):
            raise TypeError(f"WebArena BC prompt must be a message list, got {type(messages)}")
        if len(messages) != len(EXPECTED_BC_ROLES):
            raise ValueError(f"WebArena BC prompt must contain {len(EXPECTED_BC_ROLES)} messages, got {len(messages)}")

        roles = tuple(message.get("role") for message in messages)
        if roles != EXPECTED_BC_ROLES:
            raise ValueError(f"WebArena BC roles must be {EXPECTED_BC_ROLES}, got {roles}")

        for index, message in enumerate(messages):
            if not isinstance(message.get("content"), str):
                raise TypeError(f"WebArena BC message {index} content must be a string")

    def _validate_result(self, result: SFTBuildResult) -> None:
        if result.tokens[-result.response_length :] != result.response_ids:
            raise ValueError("response_ids are not the suffix of the training token sequence")
        if len(result.loss_mask) != result.response_length:
            raise ValueError(
                f"loss mask length {len(result.loss_mask)} != response length {result.response_length}"
            )
        if sum(result.loss_mask) == 0:
            raise ValueError("loss mask contains no supervised tokens")

        if not self.validate:
            return

        if sum(self.mask_policy.build_loss_mask(result.spans.output_ids)) == 0:
            raise ValueError("BC output contains no supervised tokens")

    @staticmethod
    def _require_prefix(full_ids: list[int], prefix_ids: list[int], label: str) -> None:
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(f"chat template prefix mismatch while building {label}")


def _template_kwargs(args) -> dict[str, Any]:
    kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
    # Qwen3.5's template can remove earlier thinking blocks when later user
    # turns exist. Prefix slicing requires stable rendering across message
    # prefixes.
    kwargs.setdefault("clear_thinking", False)
    return kwargs


def _get_tokenizer(args):
    global TOKENIZER
    if TOKENIZER is None:
        # `load_tokenizer` applies `args.chat_template_path` to the tokenizer.
        # The builder later passes `args.apply_chat_template_kwargs` on every
        # render, keeping custom template selection and template kwargs separate.
        TOKENIZER = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=args.chat_template_path,
            trust_remote_code=True,
        )
    return TOKENIZER


def _get_sft_builder(args) -> WebArenaSFTBuilder:
    global SFT_BUILDER
    if SFT_BUILDER is None:
        SFT_BUILDER = WebArenaSFTBuilder(
            _get_tokenizer(args),
            template_kwargs=_template_kwargs(args),
            validate=bool(getattr(args, "webarena_sft_validate", False)),
        )
    return SFT_BUILDER


def _get_bc_builder(args) -> WebArenaBCBuilder:
    global BC_BUILDER
    if BC_BUILDER is None:
        BC_BUILDER = WebArenaBCBuilder(
            _get_tokenizer(args),
            template_kwargs=_template_kwargs(args),
            validate=bool(getattr(args, "webarena_bc_validate", False)),
        )
    return BC_BUILDER


def _generate_rollout(args, rollout_id, data_buffer, builder, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset
    if args.n_samples_per_prompt != 1:
        raise ValueError("WebArena supervised generation requires --n-samples-per-prompt 1")

    global SAMPLE_PRINTED
    sample_groups = data_buffer.get_samples(args.rollout_batch_size)

    for group in sample_groups:
        if len(group) != 1:
            raise ValueError(f"WebArena supervised generation expected one sample per prompt group, got {len(group)}")
        sample = group[0]
        tools = sample.metadata.get("tools", None)
        result = builder.build(sample.prompt, tools=tools)

        sample.tokens = result.tokens
        sample.response_length = result.response_length
        sample.reward = 0
        sample.loss_mask = result.loss_mask
        sample.response = builder.tokenizer.decode(result.response_ids)
        # Supervised examples are fully materialized offline, so all accepted samples
        # are completed training examples rather than rollout completions.
        sample.status = Sample.Status.COMPLETED

        if not SAMPLE_PRINTED:
            logger.info(
                "webarena_supervised::generate_rollout example: "
                "total_tokens=%s response_length=%s supervised_tokens=%s",
                len(sample.tokens),
                sample.response_length,
                sum(sample.loss_mask),
            )
            if builder.validate:
                logger.info("webarena_supervised loss-mask spans:\n%s", builder.describe_supervised_spans(result))
            SAMPLE_PRINTED = True

    return sample_groups


def generate_sft(args, rollout_id, data_buffer, evaluation=False):
    return _generate_rollout(args, rollout_id, data_buffer, _get_sft_builder(args), evaluation=evaluation)


def generate_bc(args, rollout_id, data_buffer, evaluation=False):
    return _generate_rollout(args, rollout_id, data_buffer, _get_bc_builder(args), evaluation=evaluation)
