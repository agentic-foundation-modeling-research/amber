# Long + Short Term Memory Training

End-to-end training for the long+short term memory format: first SFT bootstrap, then GRPO
RL starting from the SFT checkpoint. The memory here is split into a long term state that
is overwritten each step and a short term scratchpad, so the rollout code path differs
from append/overwrite memory (`generate_webarena_state` instead of `generate_webarena`).

## SFT Bootstrap

### Training Config

`scripts/run_webarena_qwen3.5-9b-sft.sh` is shared by every SFT method. The method and
its hyperparameters come from a YAML config under
[packages/train/training_configs/sft](../../packages/train/training_configs/sft) —
long+short term memory uses
[long_short_memory.yaml](../../packages/train/training_configs/sft/long_short_memory.yaml),
which sets the rollout function (`sft_generate.generate_sft`) along with `num_epoch`,
`rollout_batch_size`, `global_batch_size`, `max_tokens_per_gpu`, `save_interval`, and `lr`.
Edit that file to change hyperparameters.

The script has no per-method defaults of its own, so all of
`MILES_SCRIPT_PROMPT_DATA`, `MILES_SCRIPT_HF_CHECKPOINT`,
`MILES_SCRIPT_MEGATRON_CHECKPOINT`, `MILES_SCRIPT_CUSTOM_CONFIG_PATH`,
`MILES_SCRIPT_RUN_NAME`, `MILES_SCRIPT_WANDB_TEAM`, and `MILES_SCRIPT_WANDB_PROJECT`
are mandatory — it exits if any is unset.

The RL script (`scripts/run_webarena_qwen3.5-9b.sh`) requires the same set except
`MILES_SCRIPT_PROMPT_DATA`, which defaults to `task_configs/webarena_train.jsonl`.

Keys use Miles argument names with underscores (`num_epoch`, not `--num-epoch`). Note that
`lr` needs the dot in the mantissa (`5.0e-6`, not `5e-6`) or YAML parses the value as a
string. Miles silently ignores keys it does not recognize, so check the job log for
`will override with` lines to confirm a new key took effect.

The SFT data is the refined memory dataset
(`webarena_overwrite_memory_state_sft`); see
[data_preparation.md](../data_preparation.md) for how it is built or downloaded.

### Run

SSH into the cluster and train the SFT model
```sh
cd ~/workdir/amber
tmux new -s train

# ----- CUSTOM VARIABLES ------
# Change these to your custom paths
STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
CHECKPOINT_NAME=qwen3.5-long-short-memory-sft

# Long+short term memory training data
export MILES_SCRIPT_PROMPT_DATA=$STORAGE_ROOT/datasets/webarena_overwrite_memory_state_sft/data/train-00000-of-00001.parquet

# Rollout function and training hyperparameters
export MILES_SCRIPT_CUSTOM_CONFIG_PATH="packages/train/training_configs/sft/long_short_memory.yaml"

# Logging
export MILES_SCRIPT_RUN_NAME=$CHECKPOINT_NAME
export MILES_SCRIPT_WANDB_TEAM=<WANDB_TEAM>
export MILES_SCRIPT_WANDB_PROJECT="amber-webarena"

# ----

export MILES_SCRIPT_OUTPUT_DIR="$STORAGE_ROOT/workspace/${RL_USER}/${CHECKPOINT_NAME}"
export MILES_SCRIPT_HF_CHECKPOINT="$STORAGE_ROOT/models/Qwen3.5-9B" # Predownloaded
export MILES_SCRIPT_MEGATRON_CHECKPOINT="$STORAGE_ROOT/models/Qwen3.5-9B_torch_dist" # Converted from HF checkpoint

# Run the training script. This will save the checkpoints to $MILES_SCRIPT_OUTPUT_DIR
bash scripts/run_webarena_qwen3.5-9b-sft.sh
```

## RL Training

The script uses the SFT checkpoint instead of starting from the base model. This is
crucial to stabilize the training.

### Training Config
The rollout config can be changed through custom YAML files. Long+short term memory
rollouts use
[long_short_memory.yaml](../../packages/train/training_configs/rl/long_short_memory.yaml),
which points the rollout, generate, reward model, and reward post-process hooks at
`generate_webarena_state` and sets `webarena_memory_additional_tokens` (the extra decode
budget for the memory turn) instead of the `webarena_memory_format` key used by the
append/overwrite configs.

Ensure that IP addresses for the environment server and website servers are set to the
correct values.

```sh
cd ~/workdir/amber
tmux new -s train

STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
CHECKPOINT_NAME=qwen3.5-long-short-memory-sft+rl # Replace
export MILES_SCRIPT_OUTPUT_DIR="$STORAGE_ROOT/workspace/${RL_USER}/${CHECKPOINT_NAME}"
export MILES_SCRIPT_HF_CHECKPOINT="$STORAGE_ROOT/workspace/${RL_USER}/hf_checkpoints/qwen3.5-long-short-memory-sft/" # Model trained using SFT
export MILES_SCRIPT_MEGATRON_CHECKPOINT="$STORAGE_ROOT/workspace/${RL_USER}/torch_dist_checkpoints/qwen3.5-long-short-memory-sft/" # Converted from HF

# Config that governs the rollout code
export MILES_SCRIPT_CUSTOM_CONFIG_PATH="packages/train/training_configs/rl/long_short_memory.yaml"

# Logging
export MILES_SCRIPT_RUN_NAME=$CHECKPOINT_NAME
export MILES_SCRIPT_WANDB_TEAM=<WANDB_TEAM>
export MILES_SCRIPT_WANDB_PROJECT="amber-webarena"

# Run the training script. This will save the checkpoints to $MILES_SCRIPT_OUTPUT_DIR
bash scripts/run_webarena_qwen3.5-9b.sh
```

## Evaluation

Convert the trained checkpoint to HF format
([checkpoint_conversion.md](../checkpoint_conversion.md)), then host it with sglang and
open the SSH tunnel as described in
[eval.md](../eval.md#hosting-trained-checkpoints-with-sglang).

Evaluation runs through
[scripts/eval/long_short_memory.sh](../../scripts/eval/long_short_memory.sh), which uses a
separate entrypoint (`scripts/eval/eval_long_short_memory.py`) and selects the schema with
`--memory_type state` rather than `--memory_format`. All of the environment-specific values
are read from exported variables, so the script itself never needs editing.

```sh
# Environment. See docs/setup/services.md for the env server url.
export ENV_SERVER_SERVICE_URL=http://<ENV_SERVER_IP>:8082
export WEBSITES_VM_HOST=<WEBSITE_VM_IP_ADDR>

# The sglang server hosting the checkpoint, reached through the SSH tunnel.
# Point this at whichever port this checkpoint's server is tunneled to.
export LLM_BASE_URL=http://localhost:30001/v1

# Served model name, i.e. --served-model-name of that sglang server.
# See ../datasets_and_models.md for the checkpoint names.
export MODEL=qwen3.5-long-short-memory-sft+rl

# Output goes to outputs/evals_30_steps/$RUN_NAME, so keep RUN_NAME unique per run.
export RUN_NAME=qwen3.5-long-short-memory-sft+rl-seed100
export SEED=100

bash scripts/eval/long_short_memory.sh
```

The script evaluates one seed per invocation. For the 3-seed numbers we report, re-run it
with each seed, changing `SEED` and `RUN_NAME` together.

See [eval.md](../eval.md) for evaluating untrained baselines (full context, API models).
