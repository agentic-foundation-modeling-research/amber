# Data Preparation for SFT

The first step is to train the raw instruct model using SFT for stable RL rollouts. The SFT
datasets have been pre-created for quick use and ship with this repository under
[`datasets/`](../datasets), one directory per method:

| Method | Directory |
| --- | --- |
| Append Memory | `datasets/webarena_append_memory_sft` |
| Overwrite Memory | `datasets/webarena_overwrite_memory_sft` |
| Long+Short Term Memory | `datasets/webarena_overwrite_memory_state_sft` |
| Reasoning only | `datasets/webarena_reasoning_only_sft` |

These are the names the training guides under [training/](training/README.md) and the
inventory in [datasets_and_models.md](datasets_and_models.md) expect, so keep the directory
names as they are.

Each one is a Hugging Face `Dataset` written with `save_to_disk` — an Arrow file plus
`state.json` / `dataset_info.json` — with columns `traj_id`, `step_num`, and `prompt`.

## Copying to the training cluster

Miles reads `--prompt-data` from `.jsonl` or `.parquet` only, so it cannot load the Arrow
directories directly. Export them to Parquet on the persistent storage your training job
mounts, in the layout the training guides expect:

```sh
STORAGE_ROOT=/mnt/shared-storage # Your shared filesystem mount
DATA_ROOT=$STORAGE_ROOT/datasets

uv run python - "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path

from datasets import load_from_disk

data_root = Path(sys.argv[1])
for src in sorted(Path("datasets").glob("webarena_*_sft")):
    dst = data_root / src.name / "data" / "train-00000-of-00001.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    load_from_disk(str(src)).to_parquet(str(dst))
    print(f"{src} -> {dst}")
PY
```

This produces exactly the path the guides build `MILES_SCRIPT_PROMPT_DATA` from,
`$STORAGE_ROOT/datasets/<dataset_name>/data/train-00000-of-00001.parquet`, so no further
edits are needed.

If you curate your own data instead (see below), the `scripts/sft/create_*_sft_data.py`
scripts also write with `save_to_disk`, so run the same export over their
`--dataset_save_dir` output before training on it.

# Curating your own dataset

If you want to curate your own data, from GoBrowse trajectories, first run this command to process the Go-Browse data and convert it to our trajectory format. Excludes GitLab and gives 2328 trajectories.

```sh
uv run scripts/sft/prepare_go_browse_data.py \
    --traj_save_dir data/sft_memory_data/trajectories \
    --metadata_file_path data/sft_memory_data/metadata.jsonl \
    --num_duplicate_traj 1 \
    --write_to_jsonl
```
In the first run, this will download the raw data from [https://huggingface.co/datasets/apurvaga/go-browse-wa-raw](https://huggingface.co/datasets/apurvaga/go-browse-wa-raw) and write to a jsonl format. Ensure there's at least 50GB free space on your device.

Create the memory for each trajectory. Uses Pi Coding Agent with GPT 5.5 to analyze the trajectory and reverse engineer the memory.

This step shells out to the `pi` CLI, so install it first from
[https://pi.dev](https://pi.dev). The exact invocation is in
[`sft_analyzer.py`](../packages/core/src/context_scythe/datagen/sft_analyzer.py)
(`pi -p --provider openai --model gpt-5.5 --thinking medium`); edit it there to use a
different provider or model.

```sh
uv run scripts/sft/generate_memory.py \
    data/sft_memory_data/trajectories \
    --save_dir data/sft_memory_data/memories \
    --few_shot_dir data/sft_memory_data/few_shot_memory_trajectories/ \
    [--task_start $TASK_START_IDX] [--task_end $END_TASK_IDX]
```

Create HF dataset and push to hub. Replace HF_TOKEN and HF_REPO with the appropriate values. HF_REPO should be in format `<username>/<dataset_name>`.

This script will also fill the generated memories into the trajectories in `data/sft_memory_data/trajectories`
```sh
HF_TOKEN=$HF_TOKEN uv run scripts/sft/create_memory_sft_data.py \
    --trajectories_dir data/sft_memory_data/trajectories \
    --memories_dir data/sft_memory_data/memories \
    --dataset_save_dir data/sft_memory_data/webarena_append_memory_sft \
    [--memory_format overwrite] \
    [--push-to-hub] [--hub-path $HF_REPO] [--hub-private]
```

# Reasoning only (behavior cloning) dataset

This is the dataset for the reasoning-only baseline
([training/reasoning-only.md](training/reasoning-only.md)) — `webarena_reasoning_only_sft`
in the inventory. There is no memory turn, so the pipeline is shorter than the memory one:
run `prepare_go_browse_data.py` as above, then go straight to the dataset step. The
`generate_memory.py` pass is not needed, because `prepare_go_browse_data.py` already writes
each step's response as `<think>...</think><action>...</action>`, and behavior cloning
supervises exactly that action turn.

```sh
HF_TOKEN=$HF_TOKEN uv run scripts/sft/create_bc_sft_data.py \
    --trajectories_dir data/sft_memory_data/trajectories \
    --dataset_save_dir data/sft_memory_data/webarena_reasoning_only_sft \
    [--push-to-hub] [--hub-path $HF_REPO] [--hub-private]
```

Note there is no `--memories_dir` and no `--memory_format`: the script renders every step
through `SingleTurnPromptBuilder`, which is the same builder the reasoning-only eval and RL
paths use. Rows carry the same `traj_id` / `step_num` / `prompt` columns as the memory
datasets, but each `prompt` holds three messages (system, user, assistant action) instead
of five. See
[trajectory_format/trajectory-data.md](trajectory_format/trajectory-data.md#sft-dataset-schema).

Because it reads only the trajectories, this script can be run on the same
`data/sft_memory_data/trajectories` directory either before or after memory generation —
the memories that `create_memory_sft_data.py` fills in are ignored here.

# Refining synthetic memory data

## Generating refined memory data 
We use an generation-verification system to refine the synthetic memory data created above to generate
higher quality long term memory data. The data has been pre-created at
[`datasets/webarena_overwrite_memory_state_sft`](../datasets/webarena_overwrite_memory_state_sft).
To create your own data, run
```sh
uv run scripts/sft/generate_refined_memory.py \
    --trajectories-dir data/sft_memory_data/trajectories \
    --states-dir data/sft_memory_data/memories \
    --output-dir data/sft_memory_data/refined_memories \
    --audit-dir data/sft_memory_data/refined_memory_audits \
    [--max-concurrency 4] \
    [--provider openai] [--model-name ] [--max-tokens 16000] \
    [--judge-provider openai] [--judge-model-name ] [--judge-max-tokens 4096] \
    [--max-revisions 8]
```

## Converting to SFT format
```sh
uv run scripts/sft/create_refined_memory_sft_data.py \
    --trajectories_dir data/sft_memory_data/trajectories \
    --memories_dir data/sft_memory_data/refined_memories \
    --dataset_save_dir data/sft_memory_data/webarena_overwrite_memory_state_sft \
    [--push-to-hub] [--hub-path $HF_REPO] [--hub-private]
```