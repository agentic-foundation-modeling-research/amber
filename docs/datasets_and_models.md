# Datasets and model checkpoints

Which dataset each experiment trains on, and the checkpoint name it produces. Use
the checkpoint name as `CHECKPOINT_NAME` when hosting a trained model for
evaluation (see [eval.md](eval.md)).

| Exp                   | Checkpoint name                    | Dataset                              |
| --------------------- | ---------------------------------- | ------------------------------------ |
| Append Memory SFT     | qwen3.5-append-memory-sft          | webarena_append_memory_sft           |
| Append Memory RL      | qwen3.5-append-memory-sft+rl       | task_configs/webarena_train.jsonl    |
| Long+Short Memory SFT | qwen3.5-long-short-memory-sft      | webarena_overwrite_memory_state_sft  |
| Long+Short Memory RL  | qwen3.5-long-short-memory-sft+rl   | task_configs/webarena_train.jsonl    |
| Overwrite Memory SFT  | qwen3.5-overwrite-memory-sft       | webarena_overwrite_memory_sft        |
| Overwrite Memory RL   | qwen3.5-overwrite-memory-sft+rl    | task_configs/webarena_train.jsonl    |
| Reasoning Only SFT    | qwen3.5-reasoning-only-sft         | webarena_reasoning_only_sft          |
| Reasoning Only RL     | qwen3.5-reasoning-only-sft+rl      | task_configs/webarena_train.jsonl    |

The SFT datasets ship with this repository under [`datasets/`](../datasets); see
[data_preparation.md](data_preparation.md). The RL stage reads the task configs in
[`task_configs/webarena_train.jsonl`](../task_configs/webarena_train.jsonl)
instead.
