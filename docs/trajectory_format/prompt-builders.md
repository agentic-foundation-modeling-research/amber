# Prompt Builders

A prompt builder answers one question: given a trajectory and a step number,
what messages does the model see? Nothing else in the agent loop changes between
experiments, so the builder is where each method's context-management strategy
lives.

Builders are in
`packages/core/src/context_scythe/agents/prompt_builders/` and are exported from
`context_scythe.agents`.

## The family

| Builder | Context at step `n` | Method it implements |
| --- | --- | --- |
| `SingleTurnPromptBuilder` | one user turn: observation plus a text history of all past `<think>`/`<action>` pairs | reasoning-only (WebAgent-R1 baseline) |
| `SingleTurnWithMemoryPromptBuilder` | observation, action history, plus memory — all past blocks (`append`) or only the previous one (`overwrite`) | append memory (ours) and overwrite memory (MEM1 baseline) |
| `SingleTurnWithStateMemoryPromptBuilder` | observation, action history, one cumulative memory block, plus the previous step's `state` | long+short term memory |
| `MultiTurnPromptBuilder` | genuine multi-turn conversation, one user/assistant pair per step, optionally limited to the last `window_size` steps | full context |
| `APIModelPromptBuilder` | same as append/overwrite memory, with stricter one-action-per-response instructions | API models (Claude, GPT) |

All of them subclass `BasePromptBuilder`, which supplies `flatten_messages()` for
collapsing content-part lists into plain strings when a consumer needs simple
`{role, content}` messages.

One detail worth knowing when comparing them: the reasoning-only builder replays
past steps as `<think>` plus `<action>`, while the memory builders replay past
steps as `<action>` alone and rely on the memory block to carry anything the
reasoning contained.

## Single-turn versus multi-turn

The distinction the names refer to is how history is represented, not how many
steps have happened:

- **Single-turn** builders emit exactly `[system, user]` on every step. Past
  steps appear as text sections (`# History of past actions`, `# History of past
  memories`) inside that one user message. The prompt is rebuilt from scratch
  each step.
- **Multi-turn** builders emit a real conversation that grows with the
  trajectory: `[system, user, assistant, user, assistant, ...]`.

`MultiTurnPromptBuilder` with `window_size=None` keeps everything and is the full
context configuration; a finite `window_size` truncates to the most recent steps.
It also hoists the action space and allowed-website text into the system message,
since those are constant across steps.

## The action/compression protocol

Reasoning-only and full-context builders take one model call per step. Memory
builders take two, and `build_messages(...)` therefore requires a `mode`:

- `mode="action"` returns the prompt that asks for `<think>` and `<action>`.
- `mode="compression"` returns that same prompt plus the assistant's action
  response plus a memory instruction, asking the model to write the memory block
  that carries forward.

That second call is what `compression_seeking_instruction()` supplies, and it is
the only place the memory contract is stated to the model. Subclasses override it
to change the contract: the state builder's version demands three tags in a fixed
order, and the API builder's version explicitly allows free-form memory.

Memory builders return a dict rather than a list:

```python
out = builder.build_messages(step_num=n, mode="compression", ...)
out["prompt"]                # messages to send
out["action_response"]       # assistant action turn, when already recorded
out["compression_response"]  # assistant memory turn, when already recorded
```

The single-turn and multi-turn builders return `{"prompt": ..., "response": ...}`
and a plain message list respectively. Because the recorded responses are
returned alongside the prompt, the same call serves both inference (send
`prompt`, then fill in the response) and dataset construction (concatenate prompt
and response into a training row).

## What every builder puts in the prompt

Beyond history, the shared sections come from `SingleTurnPromptBuilder` methods
and are reused by subclasses:

| Method | Section |
| --- | --- |
| `system_message_content` | UI-assistant role and formatting contract |
| `goal_message_content` | `# Goal` — restated every step |
| `action_space_message` | BrowserGym action set description plus chain-of-thought examples |
| `allowed_website_message` | calculator URL and the rollout's `site_urls`, with an instruction not to leave them |
| `final_instruction_message` | the `<think>`/`<action>` output contract |
| `privileged_information_message` | optional solution sketch, only when `use_privileged_information=True` |

The observation section itself is rendered by the step's `Observation`, not by
the builder. See [trajectory-data.md](trajectory-data.md).

Because the goal, action space, and allowed websites are re-emitted every step,
prompt growth across a trajectory comes almost entirely from history and memory —
which is the quantity the append/overwrite comparison measures.

## Which builder each entrypoint uses

| Entrypoint | Builder |
| --- | --- |
| `scripts/eval/eval_webarena.py --prompt_builder standard` | `SingleTurnWithMemoryPromptBuilder`, `--memory_format append\|overwrite` |
| `scripts/eval/eval_webarena.py --prompt_builder api` | `APIModelPromptBuilder` |
| `scripts/eval/eval_webarena_reasoning_only.py` | `SingleTurnPromptBuilder` |
| `scripts/eval/eval_webarena_multi_turn.py --window_size N` | `MultiTurnPromptBuilder` |
| `scripts/eval/eval_long_short_memory.py` | `SingleTurnWithStateMemoryPromptBuilder` |
| RL rollout generation (`packages/train/generate_webarena.py`) | `SingleTurnWithMemoryPromptBuilder`, format from `webarena_memory_format` in the train config |
| RL rollout generation (`packages/train/generate_reasoning.py`) | `SingleTurnPromptBuilder` |
| RL rollout generation (`packages/train/generate_webarena_state.py`) | `SingleTurnWithStateMemoryPromptBuilder` |
| SFT dataset creation (`scripts/sft/create_memory_sft_data.py`) | `SingleTurnWithMemoryPromptBuilder` in `compression` mode |
| SFT dataset creation (`scripts/sft/create_bc_sft_data.py`) | `SingleTurnPromptBuilder` |
| SFT dataset creation (`scripts/sft/create_refined_memory_sft_data.py`) | `SingleTurnWithStateMemoryPromptBuilder` |

## Adding a builder

Subclass the closest existing builder and override only what changes — most
methods are small and independently overridable. For a new memory contract that
is usually just `compression_seeking_instruction()` plus the history formatting
(`format_history`, `format_memory`, `build_history_messages`). Export it from
`prompt_builders/__init__.py`, then add a case wherever the entrypoint selects a
builder.

If the new builder needs a different per-step record, pair it with a trajectory
and step subclass as described in [trajectory-data.md](trajectory-data.md).

## Source of Truth

- Builders: `packages/core/src/context_scythe/agents/prompt_builders/`
- Base helpers: `prompt_builders/base.py`
- Message-shape tests: `tests/test_memory_single_turn_messages.py`,
  `tests/test_single_turn_messages.py`
- Chat template used at serving time:
  `packages/train/model_utils/qwen3.5_custom.jinja`
