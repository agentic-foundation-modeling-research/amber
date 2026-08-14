# AGENTS.md

Orientation for coding agents. This file is intentionally short: it says what the project
is and where to read the details.

**Read [`docs/README.md`](docs/README.md) before answering anything about setup,
infrastructure, data preparation, training, evaluation, or checkpoints.** It is the
documentation index and the source of truth; this file does not repeat it.
Repo setup, package layout, and runnable rollout examples live in [`README.md`](README.md);
test and contribution conventions in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## What this project is

Training long-context web agents (WebArena) that carry an explicit **memory** across steps,
instead of relying on reasoning history alone. Every method is trained in two stages: SFT
bootstrap, then GRPO RL from the SFT checkpoint (RL from base is unstable). The claim is
about *which memory strategy works* under controlled comparison, not about beating frontier
models.

The methods, and the baseline each one stands in for:

| Method | Baseline it reflects | Memory behavior |
| --- | --- | --- |
| Append memory (**ours**) | — | Memory grows linearly across the trajectory |
| Overwrite memory | MEM1 | Memory rewritten each step, roughly constant size |
| Reasoning only | WebAgent-R1 | No memory turn; reasoning is the only carried state |
| Long + short memory | — | Overwritten long-term state plus a short-term scratchpad |

All four bootstrap from the same Go-Browse WebArena trajectories; Go-Browse is the data
source, not a baseline for any one method.

## Repository map

```text
packages/core/       context-scythe-core: agents, prompt builders, trajectory utils, env helpers, datagen
packages/env_server/ context-scythe-env-server: FastAPI/Ray server hosting BrowserGym sessions
packages/train/      Miles rollout functions (sft_generate, generate_webarena, ...), training_configs/,
                     and tests/ for rollout and reward post-processing
scripts/             Entry points: rollout examples, run_webarena_*.sh (SFT/RL)
scripts/eval/        Eval entrypoints (eval_webarena*.py, eval_long_short_memory.py) plus one shell
                     wrapper per method; flags pinned, environment read from exported vars
scripts/sft/         SFT data generation and dataset builders
task_configs/        WebArena task configs (webarena_train.jsonl is the RL task set)
datasets/            SFT datasets (git-lfs; see README prerequisites)
vm_utils/            Provisioning the websites VM and env server VM
environment_setup/   WebArena container setup on the websites VM; env server management
tests/               Unit tests for prompt building, trajectories, viewport filtering, LLM client
docs/                All documentation — start at docs/README.md
```

`vm_utils/` and `environment_setup/` hold the rollout infrastructure that
[`docs/setup/`](docs/setup/) describes; nothing in them is imported by the packages.

## Message / prompt formats

Prompt formats are code, not prose — read the builder:

- Append and overwrite memory: [`memory_single_turn.py`](packages/core/src/context_scythe/agents/prompt_builders/memory_single_turn.py)
  (`memory_format=append` / `overwrite`)
- Reasoning only: [`single_turn.py`](packages/core/src/context_scythe/agents/prompt_builders/single_turn.py)
- Long + short memory: [`state_memory_single_turn.py`](packages/core/src/context_scythe/agents/prompt_builders/state_memory_single_turn.py)

## Working in this repo

- Dependencies and the Python workspace are managed with `uv`; `uv sync --all-packages`.
  The workspace members are `packages/core` and `packages/env_server` only — `packages/train`
  is not installed, and the training scripts reach it by putting it on `PYTHONPATH`.
- Tests: `uv run pytest tests packages/train/tests`. There are no tests inside
  `packages/core` or `packages/env_server`, so scope by test directory, not by package.
- The packaged source of truth is under `packages/`.
- Training and evaluation scripts carry no per-method defaults: behavior comes from the YAML
  config, the dataset path, and exported environment variables. Configure through those
  rather than editing the scripts.
