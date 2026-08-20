# WebArena Live Runners

`run_generate_webarena_live.py` calls
`../generate_webarena.py::generate(args, sample, sampling_params)` once.
`run_generate_and_rm_webarena_live.py` calls the Miles
`generate_and_rm_group()` integration path once. `run_generate_rollout_webarena_live.py`
calls the full WebArena `generate_rollout()` entrypoint. They all use
WebArena, the rollout env server, and SGLang.

The runners support two model-server modes:

- **Existing services**: connect to an already-running SGLang server.
- **Managed local model services**: start local SGLang
  subprocesses for this run, then stop them when the script exits.

## Rollout Flow

The Miles/WebArena rollout path starts at the training entrypoint and narrows
down to the model request. Each live runner covers one layer of that path:

```text
[tested by run_generate_rollout_webarena_live.py]
packages.train.generate_webarena.generate_rollout
  -> generate_rollout_async
    [tested by run_generate_and_rm_webarena_live.py]
    -> generate_and_rm_group
      -> generate_and_rm
        -> custom_generate_function_path=packages.train.generate_webarena.generate
          [tested by run_generate_webarena_live.py]
          -> SGLang /generate
        -> custom_rm_path=packages.train.generate_webarena.reward_model
```

`generate_and_rm_group()` runs one `generate_and_rm()` call per sample in the
prompt group. For true concurrent WebArena generation, set the Miles-side
`--sglang-server-concurrency` at least as high as the desired rollout
concurrency; an independently concurrent SGLang server is not enough if the
runner-side Miles semaphore is still `1`.

## Prerequisites

Run these commands from the repository root:

```bash
cd amber
```

These instructions assume the `context_scythe` core package and `miles` are
installed in the Python environment used to run the test. The external runtime
dependencies still need to exist:

- `sglang`
- `browsergym`
- `ray`
- `fastapi` / `uvicorn`
- `context_scythe`
- `miles`
- Docker access for WebArena service reset scripts
- A GPU when starting SGLang locally

All three live runners read WebArena runtime configuration from
`packages/train/training_configs/rl/append_memory.yaml` by default. Use
`--custom-config-path` to point at a different YAML file — for example
`training_configs/rl/overwrite_memory.yaml`, or one of the `generate_reasoning`
/ `generate_webarena_state` configs when testing those rollout functions.
Before running a live test, edit that config; each WebArena runtime field is
documented inline above the YAML key.

Pick a model checkpoint path and keep it consistent between SGLang and the live
runner:

```bash
export HF_CHECKPOINT=/mnt/shared-storage/models/Qwen3.5-9B
```

To enable Qwen3-style reasoning prefill for action generation, use the WebArena
custom chat template and keep the renderer from clearing previous assistant
thinking. The WebArena generator sends `prefill_think=true` for action requests
and `prefill_think=false` for memory-compression requests, so memory generation
remains plain text.

```bash
export CHAT_TEMPLATE=packages/train/model_utils/qwen3.5_custom.jinja
export CHAT_TEMPLATE_KWARGS='{"clear_thinking": false}'
```

## Step 1: Run With Existing SGLang server

Use this mode when an SGLang server is already running. Easier for quick
debugging.

Start SGLang first, using the same checkpoint that the runner uses:

```bash
export SGLANG_HOST=127.0.0.1
export SGLANG_PORT=30000

python -m sglang.launch_server \
  --model-path "${HF_CHECKPOINT}" \
  --host "${SGLANG_HOST}" \
  --port "${SGLANG_PORT}" \
  --trust-remote-code
```

If you need to pin SGLang to a specific GPU, set `CUDA_VISIBLE_DEVICES` before
starting the server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "${HF_CHECKPOINT}" \
  --host "${SGLANG_HOST}" \
  --port "${SGLANG_PORT}" \
  --trust-remote-code
```

## Step 2: Run the `generate` Live Runner

Use this mode to validate `generate_webarena.py`. The generator renders the
same WebArena messages locally with the Miles tokenizer, sends pre-tokenized
`input_ids` to SGLang `/generate`, and expects
`meta_info.output_token_logprobs` in the response.

To connect to an already-running SGLang server:

```bash
python packages/train/live_tests/run_generate_webarena_live.py \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --chat-template-path "${CHAT_TEMPLATE}" \
  --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
  --sglang-host 127.0.0.1 \
  --sglang-port 30000 \
  --task-id 10010 \
  --output-jsonl ./webarena_generate_live_existing.jsonl
```

To have the runner start local SGLang:

```bash
python packages/train/live_tests/run_generate_webarena_live.py \
  --start-sglang \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --chat-template-path "${CHAT_TEMPLATE}" \
  --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
  --sglang-host 127.0.0.1 \
  --sglang-port 30000 \
  --task-id 10010 \
  --output-jsonl ./webarena_generate_live_managed.jsonl
```

Expected result:

- The script prints `Running live generate()...`.
- WebArena setup starts the selected task site.
- The env server creates a browser rollout session.
- The model is called through SGLang `/generate`.
- The script prints `generate() returned N sample(s)`.
- Each printed sample includes `loss_mask=` and `rollout_log_probs=` lengths;
  these should match `response_length` for completed samples.
- JSONL output is written if `--output-jsonl` is set.

## Step 3: Run the `generate_and_rm` Live Runner

Use this mode to validate the next Miles rollout hierarchy around
`generate_webarena.py`. This runner intentionally uses the real Miles
custom-function loading path.

To connect to an already-running SGLang server:

```bash
python packages/train/live_tests/run_generate_and_rm_webarena_live.py \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --chat-template-path "${CHAT_TEMPLATE}" \
  --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
  --sglang-host 127.0.0.1 \
  --sglang-port 30000 \
  --n-samples-per-prompt 8 \
  --sglang-server-concurrency 8 \
  --task-id 10010 \
  --output-jsonl ./webarena_generate_and_rm_live_existing.jsonl
```

To have the runner start local SGLang:

```bash
python packages/train/live_tests/run_generate_and_rm_webarena_live.py \
  --start-sglang \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --chat-template-path "${CHAT_TEMPLATE}" \
  --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
  --sglang-host 127.0.0.1 \
  --sglang-port 30000 \
  --n-samples-per-prompt 2 \
  --sglang-server-concurrency 2 \
  --task-id 10010 \
  --output-jsonl ./webarena_generate_and_rm_live_managed.jsonl
```

Expected result:

- The script prints `Running live WebArena generate_and_rm_group()...`.
- The script prints the configured `custom_generate_function_path`,
  `custom_rm_path`, and `custom_reward_post_process_path`.
- The script builds one prompt group with `--n-samples-per-prompt` samples and
  gives each sample distinct mutable WebArena site ports by offsetting the two
  per-site port bases from `packages/train/training_configs/rl/append_memory.yaml`.
- WebArena setup starts the selected task site.
- The env server creates a browser rollout session.
- The model is called through SGLang `/generate`.
- If the call completes, the script prints the returned group shape. A
  WebArena custom generate call may return nested stepwise `Sample` objects, so
  entries can look like `[0.0]`, `[0.1]`, and so on.
- JSONL output is written if `--output-jsonl` is set.

This runner is intentionally faithful to current Miles behavior, even if that
surfaces a failure. In particular, if `generate_webarena.generate()` returns
`list[Sample]`, Miles may call the configured custom RM as a batch function. If
`reward_model` only accepts a single `Sample`, this runner should fail at that
integration boundary. Treat that as a useful result: `generate_and_rm`
successfully reached the custom-generate/custom-RM contract.

To select a GPU for SGLang:

```bash
--cuda-visible-devices 0
```

To pass extra SGLang launch arguments, repeat `--sglang-extra-arg`. Use the
`--sglang-extra-arg=...` form for values that start with `--`:

```bash
--sglang-extra-arg=--tp --sglang-extra-arg=1 \
--sglang-extra-arg=--mem-fraction-static --sglang-extra-arg=0.6
```

## Step 4: Run the `generate_rollout` Live Runner

Use this mode to validate the full custom WebArena rollout entrypoint used by
Miles training. The same `packages/train/training_configs/rl/append_memory.yaml`
setup described above is used here; pass `--custom-config-path` to use a
different YAML file.

To connect to an already-running SGLang server:

```bash
python packages/train/live_tests/run_generate_rollout_webarena_live.py \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --chat-template-path "${CHAT_TEMPLATE}" \
  --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
  --sglang-host 127.0.0.1 \
  --sglang-port 30000 \
  --task-ids 10010 10054 \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 4 \
  --sglang-server-concurrency 16 \
  --output-jsonl ./webarena_generate_rollout_live_existing.jsonl
```

To have the runner start local SGLang:

```bash
python packages/train/live_tests/run_generate_rollout_webarena_live.py \
  --start-sglang \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --chat-template-path "${CHAT_TEMPLATE}" \
  --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
  --sglang-host 127.0.0.1 \
  --sglang-port 30000 \
  --task-ids 10010 10054 \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 1 \
  --sglang-server-concurrency 1 \
  --output-jsonl ./webarena_generate_rollout_live_managed.jsonl
```

Expected result:

- The script prints `Running live WebArena generate_rollout()...`.
- It loads distinct tasks from `--task-ids`; by default it uses `10010` and
  `10054`, so `--rollout-batch-size` defaults to `2`.
- `generate_rollout()` leases configured WebArena VMs from
  `training_configs/rl/append_memory.yaml` and assigns rollout metadata for each
  prompt group.
- The model is called through SGLang `/generate`.
- The script prints the accepted rollout groups, per-sample status/reward
  fields, token counts, loss-mask lengths, rollout-log-prob lengths, and
  assigned WebArena site ports.
- JSONL output is written if `--output-jsonl` is set.

## Selecting a Task

The `generate()` and `generate_and_rm_group()` runners require a WebArena task
id:

```bash
--task-id 10010
```

By default, they read tasks from:

```text
task_configs/webarena_train.json
```

The `generate_rollout()` runner uses multiple distinct task IDs. By default it
uses:

```bash
--task-ids 10010 10054
```

For that runner, `--rollout-batch-size` must be less than or equal to the
number of distinct task IDs passed.

Use a JSONL task file:

```bash
--task-config task_configs/webarena_train.jsonl
```

The runner intentionally rejects tasks whose `sites` are not managed by the
rollout setup server. The supported mutable sites are:

```text
shopping
shopping_admin
gitlab
reddit
```

## Useful Options

`--n-samples-per-prompt`

Number of parallel samples to place in the prompt group passed to
`generate_and_rm_group()`. Defaults to `1`; use `4` as the recommended starting
point for WebArena parallel rollout validation. Use `2` for a conservative
smoke test and only increase to `8` after the env server, WebArena setup, and
SGLang are stable under load.

This controls the group size, but not the Miles semaphore by itself. For actual
parallel execution, pass `--sglang-server-concurrency` with the same value, for
example `4`.

`--sglang-server-concurrency`

Miles-side maximum concurrent `generate_and_rm` calls. This controls the
runner-side semaphore and HTTP client concurrency; it is separate from the
SGLang server launch command. Use `4` with `--n-samples-per-prompt 4` as the
recommended starting point.

`--max-new-tokens`

Maximum tokens per model generation request. The agent loop makes two model
requests per step: one for action and one for memory compression.

`--custom-config-path`

Path to the WebArena runtime YAML. Defaults to
`packages/train/training_configs/rl/append_memory.yaml` for all three live runners.

`--output-jsonl`

Writes returned Miles `Sample` objects as JSONL. This is the easiest artifact to
inspect after a run.

`--chat-template-path`

Path to a Jinja chat template loaded by the local Miles tokenizer. For the
WebArena Qwen3 reasoning-prefill template, use
`packages/train/model_utils/qwen3.5_custom.jinja`.
This template only emits the assistant `<think>` prefill when the request sets
`prefill_think=true`.

`--apply-chat-template-kwargs`

JSON object forwarded to `tokenizer.apply_chat_template`. 
Use `{"clear_thinking": false}` with the WebArena Qwen3 custom template.

`--log-level DEBUG`

Enables more runner-side logging.

## Troubleshooting

If SGLang does not start, inspect:

```bash
tail -n 80 /tmp/webarena_live_sglang.log
```

If WebArena setup fails, inspect:

```bash
tail -n 80 environment_setup/webarena/rollout_env/setup_server.log
```

If homepage/calculator is unreachable, inspect:

```bash
tail -n 80 environment_setup/webarena/rollout_env/homepage.log
```

If the env server fails during browser setup, check the env-server terminal and
Ray logs. BrowserGym/Playwright issues usually surface there.

If ports are stale from a previous run:

```bash
bash environment_setup/webarena/rollout_env/stop_rollout_servers.sh
bash environment_setup/webarena/rollout_env/start_rollout_servers.sh
```

The live runners tear down WebArena task sites after generation returns, but
they do not stop the WebArena setup/homepage services or the env server.
