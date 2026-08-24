# Launching the training cluster

Training runs on a single node with 8×H200 GPUs. How you get that node depends on
your scheduler — submit the job with whatever your cluster uses (Slurm, Kubernetes,
your cloud provider's batch/job API, or an interactive reservation).

## Container image

Run the job in the Miles image, pinned by digest:

```text
docker:radixark/miles@sha256:44fe869bf5e7f9e90442e74c9864a1020b5a0530fa308c5d52eebbe5c019225f
```

It ships Megatron-LM at `/root/Megatron-LM` and Miles at `/root/miles`, which is what
`MILES_SCRIPT_MEGATRON_PATH` and the checkpoint conversion commands in
[../checkpoint_conversion.md](../checkpoint_conversion.md) assume. Pin the digest rather
than a tag — Miles and Megatron are version-coupled, and a moving tag will break
checkpoint conversion.

## Environment

Pass these environment variables through to the job:

- `HF_TOKEN` — for downloading base checkpoints and datasets from Hugging Face
- `WANDB_API_KEY` — for run logging
- `RL_USER` — scopes checkpoint and artifact paths under
  `$STORAGE_ROOT/workspace/$RL_USER`, so concurrent runs by different users do not
  collide. `$(whoami)` is a reasonable value.

The job also needs the shared storage volume mounted at `$STORAGE_ROOT`, holding the
base checkpoints under `models/` and the SFT datasets under `datasets/` (see
[../data_preparation.md](../data_preparation.md)).

Express that as a job spec in whatever format your launcher takes: the image above, 8
GPUs on one node, the shared storage volume mounted at `$STORAGE_ROOT`, and the three
environment variables above. Multi-node runs need the same spec with the node count
raised and `MASTER_ADDR` pointing at the rank-0 node.

## Exclude `datasets/` when uploading the repo

If your launcher ships a copy of the working directory to the job, exclude `datasets/`
from that upload. It holds the git-lfs SFT datasets (~700 MB of Arrow files) and the job
never reads them: training loads Parquet from `$STORAGE_ROOT/datasets/` on the shared
volume, exported once beforehand per
[../data_preparation.md](../data_preparation.md#copying-to-the-training-cluster).
Uploading it just makes every job submission slow.

Use whatever exclude mechanism your launcher provides — an ignore file, a `--exclude`
flag, or a build-context ignore. Worth excluding alongside it: `outputs/`, `data/`,
`checkpoints/`, `.venv/`, and `__pycache__/`.

## Installing the package on the node

Once the node is up, SSH into it and install the core package into the image's Python
environment. Training imports `context_scythe` — rollout functions are referenced by
dotted path from the training configs — and the image does not ship it:

```sh
cd /path/to/amber
python -m pip install -e ./packages/core/
```

Use `pip` here rather than `uv sync`: the image already has the Miles and Megatron
dependency set installed, and an editable install adds the repo to it without rebuilding
a separate virtualenv.

Then launch the training scripts inside a `tmux` session so the run survives a dropped
connection.
