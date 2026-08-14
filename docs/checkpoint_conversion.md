# Convert training checkpoints to hostable model checkpoints

To convert the checkpoint saved during training to transformers format
```sh
STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
CHECKPOINT_NAME=<NAME> # CHange this to your checkpoint name
CKPT_ROOT=$STORAGE_ROOT/workspace/$RL_USER/$CHECKPOINT_NAME # Change this to your checkpoint path
HF_CHECKPOINT_SAVE_PATH=$STORAGE_ROOT/workspace/$RL_USER/hf_checkpoints/$CHECKPOINT_NAME
ORIGIN_HF_CHECKPOINT=$STORAGE_ROOT/models/Qwen3.5-9B # IMPORTANT: this is required for conversion. Change to your base model.

cd /root/miles
ITER=$(cat "${CKPT_ROOT}/latest_checkpointed_iteration.txt")
ITER_DIR=$(printf "%s/iter_%07d" "${CKPT_ROOT}" "${ITER}")

PYTHONPATH=/root/Megatron-LM python \
    tools/convert_torch_dist_to_hf.py \
    --input-dir "${ITER_DIR}" \
    --output-dir "${HF_CHECKPOINT_SAVE_PATH}" \
    --origin-hf-dir "${ORIGIN_HF_CHECKPOINT}" \
    --model-name qwen3_5 \
    --vocab-size 248320 --force
```

To use this model as a starting checkpoint for continued training, we need to convert the checkpoint to torch dist again. Using the checkpoints saved by the training scripts does not work (results in garbage generation).
```sh
TORCH_DIST_CHECKPOINT_SAVE_PATH=$STORAGE_ROOT/workspace/$RL_USER/torch_dist_checkpoints/$CHECKPOINT_NAME
cd /root/miles

source scripts/models/qwen3.5-9B.sh # sources MODEL_ARGS
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $HF_CHECKPOINT_SAVE_PATH \
   --save          $TORCH_DIST_CHECKPOINT_SAVE_PATH
```
