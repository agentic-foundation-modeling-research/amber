# Trajectory Data Format

One rollout is one `TrajectoryData` object holding a list of per-step records.
The same structure is used during eval, during RL rollout generation, and as the
input to SFT dataset creation, so a trajectory saved by an eval run can be
replayed through a prompt builder without conversion.

All classes below live in
`packages/core/src/context_scythe/agents/trajectory_data.py` and are re-exported
from `context_scythe.agents`.

## Shape

```text
TrajectoryData
  goal, calculator_url, site_urls     # task-level context, constant per rollout
  reward, terminated, truncated       # outcome, filled in at the end
  steps: [StepData, ...]              # one per agent step

StepData
  step_num
  observation: Observation            # what the page looked like
  response: Response                  # what the model produced
```

Two subclasses add memory, and each pairs with its own step type:

| Trajectory class | Step class | `memory` field | Used by |
| --- | --- | --- | --- |
| `TrajectoryData` | `StepData` | none | reasoning-only, full-context |
| `TrajectoryDataWithMemory` | `StepDataWithMemory` | `str` | append and overwrite memory |
| `TrajectoryDataWithStateMemory` | `StepDataWithStateMemory` | `Memory` object | long+short term (state) memory |

`add_step()` asserts that `step_num == len(steps)`, so steps are append-only and
densely numbered from zero.

## Observation

`Observation` wraps the compacted BrowserGym observation (accessibility tree,
tab info, viewport state, last action error) and owns its own rendering. Calling
`content()` returns the `{"type": "text", ...}` parts that a prompt builder drops
straight into a user message.

The flags that control what the model sees live on the observation, not on the
builder: `use_axtree`, `use_tabs_info`, `filter_viewport`, `axtree_max_tokens`.
This is why `--filter_viewport` and `--use_tabs_info` are eval-time flags — they
are recorded per step and survive serialization.

Viewport filtering uses `extra_element_properties` to keep only on-screen
elements; see `context_scythe.environment.viewport`. Screenshots are stored as
files alongside the trajectory rather than composed into messages.

## Tag protocol

Model output is free text; the parsed fields come from tags. Parsing always
takes the **last** matching block, so a model that thinks about the format before
emitting it still parses.

| Tag | Parsed into | Parser |
| --- | --- | --- |
| `<think>...</think>` | `Response.reasoning` | `Response.parse_reasoning` |
| `<action>...</action>` | `Response.action` | `Response.parse_action` |
| `<memory>...</memory>` | `Memory.memory` | `Memory.parse_memory` |
| `<state>...</state>` | `Memory.state` | `Memory.parse_state` |

`Response` keeps `model_full_response` unmodified next to the parsed fields, and
optionally carries `response_token_ids` and `response_log_probs` during RL
rollout. The `raise_on_*_parse_error` flags decide whether a missing tag is a
`None` field or an exception (`ReasoningParseError`, `ActionParseError`,
`MemoryParseError`, `StateParseError`).

For append and overwrite memory, `StepDataWithMemory.memory` is a plain string.
For state memory, `StepDataWithStateMemory.memory` is a `Memory` object that
splits the compression response into reasoning, the cumulative `memory`, and the
per-step `state`; `Memory.parse_response()` requires all three tags.

## On-disk format

Every class implements `to_json()` / `from_json()`, and the dict carries a
`"type"` discriminator (`"TrajectoryData"`, `"TrajectoryDataWithMemory"`,
`"StepDataWithMemory"`, …) so the right class can be selected on load.

Eval runs write one JSON file per rollout:

```text
<save_dir>/<task_id>/<rollout_index>.json
<save_dir>/<task_id>/<rollout_index>_screenshots/step_000.png
```

Reading a saved rollout back:

```python
from context_scythe.agents import TrajectoryDataWithMemory

trajectory = TrajectoryDataWithMemory.from_json(json.loads(path.read_text()))
trajectory.reward            # 1.0 on success, 0.0 otherwise
trajectory.steps[0].memory   # memory produced after step 0
```

Save and load code: `scripts/eval/eval_webarena.py`.

## SFT dataset schema

SFT datasets are Hugging Face datasets with one row per **step**, not per
trajectory. Each row's `prompt` is the fully rendered message list for that
step, with the supervised assistant turn last:

| Column | Type | Meaning |
| --- | --- | --- |
| `traj_id` | string | source trajectory file stem |
| `step_num` | int32 | step index within that trajectory |
| `prompt` | list of `{role, content}` | rendered conversation, flattened to plain strings |

For memory training, `prompt` holds five messages: system, user (observation and
history), assistant (action), user (memory instruction), assistant (memory). For
reasoning-only (`bc`) training it holds three: system, user, assistant action.

Loss is applied only to the assistant spans. The training-side builder renders
chat-template prefixes and takes prefix differences to find the exact token
boundaries, then masks the inserted memory instruction to zero so both assistant
outputs are supervised inside one sequence. See `WebArenaSFTBuilder` and
`WebArenaBCBuilder` in `packages/train/sft_generate.py`.

Dataset creation: `scripts/sft/create_memory_sft_data.py` (and the `bc` and
`refined_memory` variants in the same directory).

## Source of Truth

- Dataclasses and parsers:
  `packages/core/src/context_scythe/agents/trajectory_data.py`
- BrowserGym observation adapter:
  `packages/core/src/context_scythe/agents/observation.py`
- Viewport filtering:
  `packages/core/src/context_scythe/environment/viewport.py`
- Eval save/load:
  `scripts/eval/eval_webarena.py`
- SFT tokenization and loss masks:
  `packages/train/sft_generate.py`
- Tests covering message construction:
  `tests/test_memory_single_turn_messages.py`,
  `tests/test_single_turn_messages.py`, `tests/test_trajectory.py`

## Next Reading

- [prompt-builders.md](prompt-builders.md): how these records become prompts.
