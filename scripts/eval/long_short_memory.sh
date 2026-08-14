#!/usr/bin/env bash
# Evaluate a long + short term memory checkpoint on WebArena Lite.
#
# This method uses its own entrypoint (scripts/eval/eval_long_short_memory.py) and
# selects the schema with --memory_type state rather than --memory_format.
#
# Export the required variables before running; see docs/training/long_short_memory.md:
#   ENV_SERVER_SERVICE_URL  env_server URL (docs/setup/services.md)
#   WEBSITES_VM_HOST        Websites VM IP address
#   LLM_BASE_URL            sglang server, e.g. http://localhost:30001/v1
#   MODEL                   served model name (docs/datasets_and_models.md)
#   RUN_NAME                output directory name under outputs/evals_30_steps
#   SEED                    evaluation seed, e.g. 100
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${ENV_SERVER_SERVICE_URL:?export it first, see docs/training/long_short_memory.md}"
: "${WEBSITES_VM_HOST:?export it first, see docs/training/long_short_memory.md}"
: "${LLM_BASE_URL:?export it first, see docs/training/long_short_memory.md}"
: "${MODEL:?export it first, see docs/training/long_short_memory.md}"
: "${RUN_NAME:?export it first, see docs/training/long_short_memory.md}"
: "${SEED:?export it first, see docs/training/long_short_memory.md}"

COMMON_ARGS=(
    --env_server_service_url "$ENV_SERVER_SERVICE_URL"
    --websites_vm_host "$WEBSITES_VM_HOST"
    --task_config_path task_configs/test_webarena_lite.json
    --max_steps 30
    --timeout_s 300
)

EXTRA_ARGS=(
    --model "$MODEL"
    --llm_extra_body '{"chat_template_kwargs":{"clear_thinking":false,"prefill_think":true}}'
    --llm_temperature 0.6
    --think_prefix "<think>\n"
    --filter_viewport
    --use_tabs_info
    --llm_provider openai
    --llm_base_url "$LLM_BASE_URL"
    --memory_type state
)

save_dir=outputs/evals_30_steps/$RUN_NAME
echo "==> long + short term memory | seed $SEED | $save_dir"
uv run --env-file .env python scripts/eval/eval_long_short_memory.py \
    "${COMMON_ARGS[@]}" \
    --parallelism 4 \
    --group_size 1 \
    --save_screenshots \
    --all_tasks \
    --save_dir "$save_dir" --seed "$SEED" \
    "${EXTRA_ARGS[@]}"
