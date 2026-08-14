# Training

Every method in this project is trained the same way: an **SFT bootstrap** on trajectory
data, followed by **GRPO RL** starting from the SFT checkpoint. Starting RL from the base
model is unstable, so the SFT stage is not optional.

Both stages run on Miles (Megatron under the hood) through two shared entry-point scripts:

| Stage | Script | Config directory |
| --- | --- | --- |
| SFT | `scripts/run_webarena_qwen3.5-9b-sft.sh` | [`packages/train/training_configs/sft`](../../packages/train/training_configs/sft) |
| RL | `scripts/run_webarena_qwen3.5-9b.sh` | [`packages/train/training_configs/rl`](../../packages/train/training_configs/rl) |

The scripts carry no per-method defaults. What method you train is decided entirely by
two environment variables: `MILES_SCRIPT_CUSTOM_CONFIG_PATH` (which YAML config, i.e.
which rollout function and hyperparameters) and `MILES_SCRIPT_PROMPT_DATA` (which SFT
dataset). Every variable the scripts read is mandatory and the script exits if any is
unset:

| Variable | SFT | RL |
| --- | --- | --- |
| `MILES_SCRIPT_CUSTOM_CONFIG_PATH` | required | required |
| `MILES_SCRIPT_PROMPT_DATA` | required | defaults to `task_configs/webarena_train.jsonl` |
| `MILES_SCRIPT_HF_CHECKPOINT` | required | required |
| `MILES_SCRIPT_MEGATRON_CHECKPOINT` | required | required |
| `MILES_SCRIPT_RUN_NAME` | required | required |
| `MILES_SCRIPT_WANDB_TEAM` | required | required |
| `MILES_SCRIPT_WANDB_PROJECT` | required | required |

## Per-method guides

Each guide is end-to-end: SFT config, SFT run, RL config, RL run, and evaluation of the
resulting checkpoint.

- [append.md](append.md) — **append memory** (our method). Memory grows linearly across
  the trajectory.
- [overwrite.md](overwrite.md) — **overwrite memory** (MEM1 baseline). Memory is rewritten
  from scratch each step, so it stays roughly constant in size.
- [reasoning-only.md](reasoning-only.md) — **reasoning only** (WebAgentR1 baseline). No
  memory turn; reasoning is the only state carried forward. Uses different rollout
  functions (`sft_generate.generate_bc`, `generate_reasoning`).
- [long_short_memory.md](long_short_memory.md) — **long + short term memory**. An
  overwritten long-term state plus a short-term scratchpad, via
  `generate_webarena_state`.

## Evaluation

Each method has an eval script under [../../scripts/eval](../../scripts/eval) that pins the
flags for that method — the entrypoint, and `--memory_format` / `--memory_type` where it
applies:

| Method | Script |
| --- | --- |
| Append memory | [append.sh](../../scripts/eval/append.sh) |
| Overwrite memory | [overwrite.sh](../../scripts/eval/overwrite.sh) |
| Reasoning only | [reasoning_only.sh](../../scripts/eval/reasoning_only.sh) |
| Long + short term memory | [long_short_memory.sh](../../scripts/eval/long_short_memory.sh) |

Everything environment-specific is read from exported variables, so the scripts never need
editing:

```sh
# Environment. See ../setup/services.md for the env server url.
export ENV_SERVER_SERVICE_URL=http://<ENV_SERVER_IP>:8082
export WEBSITES_VM_HOST=<WEBSITE_VM_IP_ADDR>

# The sglang server hosting the checkpoint, reached through the SSH tunnel
# (see ../eval.md).
export LLM_BASE_URL=http://localhost:30000/v1

# Served model name, i.e. --served-model-name of that sglang server.
# See ../datasets_and_models.md for the checkpoint names.
export MODEL=qwen3.5-append-memory-sft+rl

# Output goes to outputs/evals_30_steps/$RUN_NAME, so keep RUN_NAME unique per run.
export RUN_NAME=qwen3.5-append-memory-sft+rl-seed100
export SEED=100

bash scripts/eval/append.sh
```

Each script evaluates one seed per invocation; re-run it per seed, changing `SEED` and
`RUN_NAME` together. The per-method guide above has the details for its method, and
[../eval.md](../eval.md) covers the untrained baselines (full context, API models).

## Launching the cluster

- [cluster-launch.md](cluster-launch.md) — submitting the single-node 8×H200 job and
  running the training scripts in `tmux`.

## Getting started: base checkpoint

Every per-method guide trains from the 9B base checkpoint, so that is what the commands
below download and convert. For 4B, substitute the size in all three places (`MODEL_NAME`,
the `source` line, and the training script you run — `scripts/run_webarena_qwen3.5-4b-sft.sh`
and `scripts/run_webarena_qwen3.5-4b.sh`). For 27B only the SFT stage has a script
(`scripts/run_webarena_qwen3.5-27b-sft.sh`); there is no 27B RL script.

Download the pretrained checkpoint:
```sh
STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
MODEL_NAME=Qwen3.5-9B   # or Qwen3.5-4B / Qwen3.5-27B
hf download Qwen/$MODEL_NAME --local-dir $STORAGE_ROOT/models/$MODEL_NAME
```

Then convert from HF to Megatron:
```sh
cd /root/miles
source scripts/models/qwen3.5-9B.sh   # or qwen3.5-4B.sh / qwen3.5-27B.sh
# STORAGE_ROOT and MODEL_NAME as set above
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $STORAGE_ROOT/models/$MODEL_NAME \
   --save          $STORAGE_ROOT/models/${MODEL_NAME}_torch_dist
```

The `models/<name>` and `models/<name>_torch_dist` layout is what every guide's
`MILES_SCRIPT_HF_CHECKPOINT` and `MILES_SCRIPT_MEGATRON_CHECKPOINT` point at. The scripts
have no built-in checkpoint paths, so both variables must be exported before you run them.

For converting other models, refer to [https://miles.radixark.com/docs/models/qwen/qwen3-5](https://miles.radixark.com/docs/models/qwen/qwen3-5).

## Editing configs

Config keys use Miles argument names with underscores (`num_epoch`, not `--num-epoch`).
`lr` needs a dot in the mantissa (`1.0e-5`, not `1e-5`) or YAML parses it as a string.
Miles silently ignores keys it does not recognize, so check the job log for
`will override with` lines to confirm a new key took effect.

## Related docs

- [../data_preparation.md](../data_preparation.md) — building or downloading the SFT
  datasets. The RL stage instead reads the task configs in
  [`task_configs/webarena_train.jsonl`](../../task_configs/webarena_train.jsonl).
- [../datasets_and_models.md](../datasets_and_models.md) — dataset and checkpoint
  inventory.
- [../checkpoint_conversion.md](../checkpoint_conversion.md) — converting checkpoints
  between HF and Megatron, needed between the SFT and RL stages.
- [../trajectory_format/](../trajectory_format/README.md) — the trajectory format and
  prompt builders shared by eval, SFT, and RL.
- [../eval.md](../eval.md) — evaluating the untrained baselines (full context, API models),
  plus the sglang hosting and SSH tunnel steps the per-method eval sections reuse.
- [../setup/gcp-resources.md](../setup/gcp-resources.md) and
  [../setup/vm-setup.md](../setup/vm-setup.md) — bringing up the WebArena rollout
  environment that RL training talks to.
