#!/usr/bin/env bash
# Evaluate the full context agent on WebArena Lite.
#
# No memory turn and no compaction: the whole interaction history stays in the
# context, so this uses the multi-turn entrypoint and --window_size.
#
# Export the required variables before running; see docs/eval.md:
#   ENV_SERVER_SERVICE_URL  env_server URL (docs/setup/services.md)
#   WEBSITES_VM_HOST        Websites VM IP address
#   LLM_BASE_URL            sglang server, e.g. http://localhost:30000/v1
#   MODEL                   served model name (docs/datasets_and_models.md)
#   RUN_NAME                output directory name under outputs/evals_30_steps
#   SEED                    evaluation seed, e.g. 100
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${ENV_SERVER_SERVICE_URL:?export it first, see docs/eval.md}"
: "${WEBSITES_VM_HOST:?export it first, see docs/eval.md}"
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
    --model "$MODEL"
    --llm_extra_body '{"chat_template_kwargs":{"clear_thinking":false,"prefill_think":true}}'
    --llm_temperature 0.8
    --llm_max_tokens 4096
    --think_prefix "<think>\n"
    --filter_viewport
    --use_tabs_info
    --llm_provider openai
    --llm_base_url "$LLM_BASE_URL"
    --window_size 30
)

save_dir=outputs/evals_30_steps/$RUN_NAME
echo "==> full context | seed $SEED | $save_dir"
uv run --env-file .env python scripts/eval/eval_webarena_multi_turn.py \
    "${COMMON_ARGS[@]}" \
    --parallelism 4 \
    --group_size 1 \
    --save_screenshots \
    --all_tasks \
    --save_dir "$save_dir" --seed "$SEED" \
    "${EXTRA_ARGS[@]}"
