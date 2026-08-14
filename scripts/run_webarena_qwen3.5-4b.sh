#!/bin/bash

# WebArena RL training through Miles sync train.py.
#
# The training config (packages/train/training_configs/rl/) selects the rollout
# function and the hyperparameters, so this script covers every RL method. Point
# MILES_SCRIPT_CUSTOM_CONFIG_PATH at the config for the method you are training
# (append_memory.yaml, overwrite_memory.yaml, reasoning_only.yaml, or
# long_short_memory.yaml), and edit that file before running to set
# webarena_env_server_url and webarena_vms.

# Fail before touching any running processes if required config is missing.
for _required_var in \
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

NUM_GPUS=${MILES_SCRIPT_NUM_GPUS:-8} # 2 actor GPUs + 6 rollout GPUs
ACTOR_NUM_GPUS=${MILES_SCRIPT_ACTOR_NUM_GPUS:-2}
ROLLOUT_NUM_GPUS=${MILES_SCRIPT_ROLLOUT_NUM_GPUS:-6}
CUSTOM_CONFIG_PATH="${MILES_SCRIPT_CUSTOM_CONFIG_PATH}"
PROMPT_DATA=${MILES_SCRIPT_PROMPT_DATA:-"task_configs/webarena_train.jsonl"}

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

# Source the model args (TODO: maybe not needed?)
source "${REPO_ROOT}/packages/train/model_utils/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${MEGATRON_CHECKPOINT}"
   --load "${OUTPUT_DIR}"
   --save "${OUTPUT_DIR}"
   --save-interval 1
   --no-load-optim
   --no-load-rng
   --finetune
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --metadata-key metadata
   --apply-chat-template
   --chat-template-path packages/train/model_utils/qwen3.5_custom.jinja
   --apply-chat-template-kwargs '{"clear_thinking": false}'
   --custom-config-path "${CUSTOM_CONFIG_PATH}"
   --rollout-shuffle
   --num-epoch 15
   --rollout-batch-size 32
   --n-samples-per-prompt 6
   --rollout-max-response-len 1024
   --rollout-temperature 0.6
   # Miles derives train_iters as:
   # num_rollout * rollout_batch_size * n_samples_per_prompt // global_batch_size
   # Keep this product >= global_batch_size so Megatron lr_decay_steps stays > 0.
   --micro-batch-size 1
   --use-rollout-logprobs
   --use-dynamic-global-batch-size
   # --global-batch-size 32
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
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

SGLANG_ARGS=(
   # num_engines = rollout_num_gpus / rollout_num_gpus_per_engine
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)

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
   --max-tokens-per-gpu 2048
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 128
   # Automatically convert HF -> Megatron
   # --megatron-to-hf-mode bridge
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}"
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
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
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir "${REPO_ROOT}" \
   -- python3 scripts/train_async.py \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
