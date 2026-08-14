#!/usr/bin/env bash
# Evaluate an API based model on WebArena Lite.
#
# API models need an explicit instruction to emit only one action per step, so
# this uses --prompt_builder api instead of the trained prompt builders.
#
# Export the required variables before running; see docs/eval.md:
#   ENV_SERVER_SERVICE_URL  env_server URL (docs/setup/services.md)
#   WEBSITES_VM_HOST        Websites VM IP address
#   LLM_PROVIDER            anthropic | openai | openai-responses-api
#   LLM_BASE_URL            provider/proxy base URL
#   MODEL                   e.g. anthropic:claude-sonnet-4-6, openai:gpt-5.5
#   RUN_NAME                output directory name under outputs/evals_30_steps
#   SEED                    evaluation seed, e.g. 100
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${ENV_SERVER_SERVICE_URL:?export it first, see docs/eval.md}"
: "${WEBSITES_VM_HOST:?export it first, see docs/eval.md}"
: "${LLM_PROVIDER:?export it first, see docs/eval.md}"
: "${LLM_BASE_URL:?export it first, see docs/eval.md}"
: "${MODEL:?export it first, see docs/eval.md}"
: "${RUN_NAME:?export it first, see docs/eval.md}"
: "${SEED:?export it first, see docs/eval.md}"

COMMON_ARGS=(
    --env_server_service_url "$ENV_SERVER_SERVICE_URL"
    --websites_vm_host "$WEBSITES_VM_HOST"
    --task_config_path task_configs/test_webarena_lite.json
    --max_steps 30
    --timeout_s 300
)

EXTRA_ARGS=(
    --prompt_builder api
    --filter_viewport
    --use_tabs_info
    --llm_max_tokens 8192
    --llm_provider "$LLM_PROVIDER"
    --model "$MODEL"
    --llm_base_url "$LLM_BASE_URL"
)

save_dir=outputs/evals_30_steps/$RUN_NAME
echo "==> api model $MODEL | seed $SEED | $save_dir"
uv run --env-file .env python scripts/eval/eval_webarena.py \
    "${COMMON_ARGS[@]}" \
    --parallelism 4 \
    --group_size 1 \
    --save_screenshots \
    --all_tasks \
    --save_dir "$save_dir" --seed "$SEED" \
    "${EXTRA_ARGS[@]}"
