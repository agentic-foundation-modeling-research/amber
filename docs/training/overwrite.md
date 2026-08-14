# Overwrite Memory Training

End-to-end training for the overwrite memory format (the MEM1 baseline): first SFT
bootstrap, then GRPO RL starting from the SFT checkpoint. The memory is rewritten from
scratch at every step, so it stays roughly constant in size instead of growing across the
trajectory.

## SFT Bootstrap

### Training Config

`scripts/run_webarena_qwen3.5-9b-sft.sh` is shared by every SFT method. The method and
its hyperparameters come from a YAML config under
[packages/train/training_configs/sft](../../packages/train/training_configs/sft) —
overwrite memory uses
[overwrite_memory.yaml](../../packages/train/training_configs/sft/overwrite_memory.yaml),
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
`lr` needs the dot in the mantissa (`1.0e-5`, not `1e-5`) or YAML parses the value as a
string. Miles silently ignores keys it does not recognize, so check the job log for
`will override with` lines to confirm a new key took effect.

The SFT data is the overwrite memory dataset (`webarena_overwrite_memory_sft`); see
[data_preparation.md](../data_preparation.md) for how it is built or downloaded — it is
`scripts/sft/create_memory_sft_data.py` run with `--memory_format overwrite`.

### Run

SSH into the cluster and train the SFT model
```sh
cd ~/workdir/context-scythe
tmux new -s train

# ----- CUSTOM VARIABLES ------
# Change these to your custom paths
STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
CHECKPOINT_NAME=qwen3.5-overwrite-memory-sft

# Overwrite memory training data
export MILES_SCRIPT_PROMPT_DATA=$STORAGE_ROOT/datasets/webarena_overwrite_memory_sft/data/train-00000-of-00001.parquet

# Rollout function and training hyperparameters
export MILES_SCRIPT_CUSTOM_CONFIG_PATH="packages/train/training_configs/sft/overwrite_memory.yaml"

# Logging
export MILES_SCRIPT_RUN_NAME=$CHECKPOINT_NAME
export MILES_SCRIPT_WANDB_TEAM=<WANDB_TEAM>
export MILES_SCRIPT_WANDB_PROJECT="context-scythe-webarena"

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
The rollout config can be changed through custom YAML files. Overwrite memory rollouts use
[overwrite_memory.yaml](../../packages/train/training_configs/rl/overwrite_memory.yaml).
It is identical to the append memory RL config except for
`webarena_memory_format: overwrite`, which is the key that selects the memory contract in
`SingleTurnWithMemoryPromptBuilder`.

Ensure that IP addresses for the environment server and website servers are set to the
correct values.

```sh
cd ~/workdir/context-scythe
tmux new -s train

STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
CHECKPOINT_NAME=qwen3.5-overwrite-memory-sft+rl # Replace
export MILES_SCRIPT_OUTPUT_DIR="$STORAGE_ROOT/workspace/${RL_USER}/${CHECKPOINT_NAME}"
export MILES_SCRIPT_HF_CHECKPOINT="$STORAGE_ROOT/workspace/${RL_USER}/hf_checkpoints/qwen3.5-overwrite-memory-sft/" # Model trained using SFT
export MILES_SCRIPT_MEGATRON_CHECKPOINT="$STORAGE_ROOT/workspace/${RL_USER}/torch_dist_checkpoints/qwen3.5-overwrite-memory-sft/" # Converted from HF

# Config that governs the rollout code
export MILES_SCRIPT_CUSTOM_CONFIG_PATH="packages/train/training_configs/rl/overwrite_memory.yaml"

# Logging
export MILES_SCRIPT_RUN_NAME=$CHECKPOINT_NAME
export MILES_SCRIPT_WANDB_TEAM=<WANDB_TEAM>
export MILES_SCRIPT_WANDB_PROJECT="context-scythe-webarena"

# Run the training script. This will save the checkpoints to $MILES_SCRIPT_OUTPUT_DIR
bash scripts/run_webarena_qwen3.5-9b.sh
```

## Evaluation

Convert the trained checkpoint to HF format
([checkpoint_conversion.md](../checkpoint_conversion.md)), then host it with sglang and
open the SSH tunnel as described in
[eval.md](../eval.md#hosting-trained-checkpoints-with-sglang).

Evaluation runs through [scripts/eval/overwrite.sh](../../scripts/eval/overwrite.sh),
which is identical to the append memory script except for `--memory_format overwrite` —
the same key that selects the memory contract during training. All of the
environment-specific values are read from exported variables, so the script itself never
needs editing.

```sh
# Environment. See docs/setup/services.md for the env server url.
export ENV_SERVER_SERVICE_URL=http://<ENV_SERVER_IP>:8082
export WEBSITES_VM_HOST=<WEBSITE_VM_IP_ADDR>

# The sglang server hosting the checkpoint, reached through the SSH tunnel.
export LLM_BASE_URL=http://localhost:30000/v1

# Served model name, i.e. --served-model-name of that sglang server.
# See ../datasets_and_models.md for the checkpoint names.
export MODEL=qwen3.5-overwrite-memory-sft+rl

# Output goes to outputs/evals_30_steps/$RUN_NAME, so keep RUN_NAME unique per run.
export RUN_NAME=qwen3.5-overwrite-memory-sft+rl-seed100
export SEED=100

bash scripts/eval/overwrite.sh
```

The script evaluates one seed per invocation. For the 3-seed numbers we report, re-run it
with each seed, changing `SEED` and `RUN_NAME` together.

See [eval.md](../eval.md) for evaluating untrained baselines (full context, API models).
