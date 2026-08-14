# Evaluate API based and untrained models on WebArena Lite

This page covers the baselines that have no training stage: the full context agent and API
based models. It also documents the hosting and SSH tunnel steps that the trained-model
evals reuse.

To evaluate a **trained** checkpoint, see the eval section of its training guide:

- [training/append.md](training/append.md#evaluation): append memory (our method).
- [training/overwrite.md](training/overwrite.md#evaluation): overwrite memory (MEM1).
- [training/reasoning-only.md](training/reasoning-only.md#evaluation): reasoning only
  (WebAgentR1).
- [training/long_short_memory.md](training/long_short_memory.md#evaluation): long + short
  term memory.

See [datasets_and_models.md](datasets_and_models.md) for the checkpoint name of each
experiment (and the dataset it was trained on). Use that name as `CHECKPOINT_NAME` below.

## Hosting trained checkpoints with SGLang
```sh
STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
CHECKPOINT_NAME="Qwen3.5-9B" # Replace with your checkpoint
HF_CHECKPOINT_PATH=$STORAGE_ROOT/models/$CHECKPOINT_NAME
python -m sglang.launch_server \
  --model-path $HF_CHECKPOINT_PATH \
  --served-model-name $CHECKPOINT_NAME \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code \
  --chat-template ~/workdir/context-scythe/packages/train/model_utils/qwen3.5_custom.jinja \
  --mem-fraction-static 0.7 \
  --chunked-prefill-size 4096 \
  --max-running-requests 4 \
  --log-level info
```
On your laptop, use an SSH tunnel
```sh
# If hosted on a remote cluster, tunnel through SSH on your laptop
ssh -N -L 30000:localhost:30000 <cluster-name>
```

## Evaluation

There is one script per baseline under [scripts/eval](../scripts/eval). Each one pins the
flags for its baseline and reads everything environment-specific from exported variables,
so the scripts themselves never need editing:

| Baseline | Script |
| --- | --- |
| Full context | [full_context.sh](../scripts/eval/full_context.sh) |
| API models | [api_model.sh](../scripts/eval/api_model.sh) |

Every script requires these variables:

```sh
# Environment. See docs/setup/services.md for the env server url.
export ENV_SERVER_SERVICE_URL=http://<ENV_SERVER_IP>:8082
export WEBSITES_VM_HOST=<WEBSITE_VM_IP_ADDR>

# Model endpoint and served model name.
export LLM_BASE_URL=http://localhost:30000/v1
export MODEL=<SERVED_MODEL_NAME>

# Output goes to outputs/evals_30_steps/$RUN_NAME, so keep RUN_NAME unique per run.
export RUN_NAME=<RUN_NAME>
export SEED=100
```

Each script evaluates one seed per invocation. For the 3-seed numbers we report, re-run it
with each seed, changing `SEED` and `RUN_NAME` together.

### Full context

No memory turn and no compaction: the whole interaction history stays in the context, so
this uses the multi-turn entrypoint (`scripts/eval/eval_webarena_multi_turn.py`) with
`--window_size 30`.

```sh
export MODEL=Qwen3.5-9B
export RUN_NAME=qwen3.5-full-context-seed100
export SEED=100

bash scripts/eval/full_context.sh
```

### API models

API based models need an explicit instruction to emit only one action per step, so this
script uses `--prompt_builder api` instead of the trained prompt builders. It also needs
`LLM_PROVIDER`, since the provider varies per model.

```sh
# Anthropic
export LLM_PROVIDER=anthropic
export MODEL=anthropic:claude-sonnet-4-6
export LLM_BASE_URL=<BASE_URL>

# For OpenAI models:
# export LLM_PROVIDER=openai-responses-api
# export MODEL=openai:gpt-5.5
# export LLM_BASE_URL=<BASE_URL>

export RUN_NAME=claude-sonnet-4-6-seed100
export SEED=100

bash scripts/eval/api_model.sh
```
