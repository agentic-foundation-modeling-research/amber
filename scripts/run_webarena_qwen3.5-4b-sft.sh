#!/bin/bash

# WebArena SFT training through Miles async train.py with Megatron.
#
# The training config (packages/train/training_configs/sft/) selects the rollout
# function and the hyperparameters, so this script covers every SFT method.
# Expects a parquet dataset whose `prompt` column contains raw message lists:
#   memory formats: system, user, assistant action, user memory prompt, assistant memory
#   reasoning only: system, user, assistant action

# Fail before touching any running processes if required config is missing.
for _required_var in \
   MILES_SCRIPT_PROMPT_DATA \
   MILES_SCRIPT_HF_CHECKPOINT \
   MILES_SCRIPT_MEGATRON_CHECKPOINT \
   MILES_SCRIPT_CUSTOM_CONFIG_PATH \
   MILES_SCRIPT_RUN_NAME \
   MILES_SCRIPT_WANDB_TEAM \
   MILES_SCRIPT_WANDB_PROJECT; do
   if [ -z "${!_required_var}" ]; then
      echo "Error: ${_required_var} must be set." >&2
      exit 1
   fi
done
unset _required_var

pkill -9 sglang
sleep 3
if [ -z "$MILES_SCRIPT_EXTERNAL_RAY" ] || [ "$MILES_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 python
sleep 3
if [ -z "$MILES_SCRIPT_EXTERNAL_RAY" ] || [ "$MILES_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 python
pkill -9 miles

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HF_CHECKPOINT="${MILES_SCRIPT_HF_CHECKPOINT}"
MEGATRON_CHECKPOINT="${MILES_SCRIPT_MEGATRON_CHECKPOINT}"
PROMPT_DATA="${MILES_SCRIPT_PROMPT_DATA}"

NUM_GPUS=${MILES_SCRIPT_NUM_GPUS:-8}
ACTOR_NUM_GPUS=${MILES_SCRIPT_ACTOR_NUM_GPUS:-${NUM_GPUS}}
# Training hyperparameters (num_epoch, rollout_batch_size, global_batch_size,
# max_tokens_per_gpu, save_interval, lr) come from a per-experiment YAML config
# under packages/train/training_configs/sft/, which Miles applies on top of the
# CLI arguments below.
CUSTOM_CONFIG_PATH="${MILES_SCRIPT_CUSTOM_CONFIG_PATH}"

if [ ! -f "${REPO_ROOT}/${CUSTOM_CONFIG_PATH}" ] && [ ! -f "${CUSTOM_CONFIG_PATH}" ]; then
   echo "Training config not found: ${CUSTOM_CONFIG_PATH}" >&2
   exit 1
fi

RUN_NAME="${MILES_SCRIPT_RUN_NAME}"
WANDB_TEAM="${MILES_SCRIPT_WANDB_TEAM}"
WANDB_PROJECT="${MILES_SCRIPT_WANDB_PROJECT}"

if [ -n "$MILES_SCRIPT_OUTPUT_DIR" ]; then
   OUTPUT_DIR="${MILES_SCRIPT_OUTPUT_DIR}"
elif [ -n "$RL_USER" ]; then
   OUTPUT_DIR="${STORAGE_ROOT:-/mnt/shared-storage}/workspace/${RL_USER}/${RUN_NAME}"
else
   OUTPUT_DIR="${REPO_ROOT}/outputs/${RUN_NAME}"
fi

MEGATRON_PATH=${MILES_SCRIPT_MEGATRON_PATH:-"/root/Megatron-LM/"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

mkdir -p "${OUTPUT_DIR}"

source "${REPO_ROOT}/packages/train/model_utils/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${MEGATRON_CHECKPOINT}"
   --load "${OUTPUT_DIR}"
   --save "${OUTPUT_DIR}"
)

SFT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --chat-template-path packages/train/model_utils/qwen3.5_custom.jinja
   --apply-chat-template-kwargs '{"clear_thinking": false}'
   --rollout-shuffle
   # Placeholders: --rollout-batch-size is a required flag and --num-epoch is
   # asserted during argument validation, both before Miles applies
   # --custom-config-path. The config file supplies the real values.
   --num-epoch 1
   --rollout-batch-size 1
   --n-samples-per-prompt 1
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

SFT_ARGS+=(--custom-config-path "${CUSTOM_CONFIG_PATH}")

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-team "${WANDB_TEAM}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group $RUN_NAME
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
   )
fi

PERF_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # Placeholder: asserted non-None under --use-dynamic-batch-size before Miles
   # applies --custom-config-path. The config file supplies the real value.
   --max-tokens-per-gpu 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # Automatically convert HF -> Megatron if needed.
   # --megatron-to-hf-mode bridge
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}"
)

if [ -z "$MILES_SCRIPT_EXTERNAL_RAY" ] || [ "$MILES_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:packages/train\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir "${REPO_ROOT}" \
   -- python3 scripts/train_async.py \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"
