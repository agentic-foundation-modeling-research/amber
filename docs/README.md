# Rollout Environment Docs

These docs explain how to understand, create, operate, and debug the remote
WebArena rollout environment used for RL training.

## Start Here

New readers should start with the conceptual path before running infrastructure:

1. [concepts/mental-model.md](concepts/mental-model.md): the problem this system
   solves, the main moving pieces, and the vocabulary used across the docs.
2. [concepts/architecture.md](concepts/architecture.md): VM boundaries, service
   ownership, ports, and rollout isolation.
3. [concepts/rollout-lifecycle.md](concepts/rollout-lifecycle.md): how setup,
   browser execution, and teardown fit together for one rollout.

## Reading Paths

### I want to understand the system

- [concepts/mental-model.md](concepts/mental-model.md): the shortest conceptual
  overview.
- [concepts/architecture.md](concepts/architecture.md): the `Websites VM`,
  `env_server VM`, rollout client, ports, and ownership boundaries.
- [concepts/rollout-lifecycle.md](concepts/rollout-lifecycle.md): the per-rollout
  sequence from mutable site setup through cleanup.
- [concepts/browsergym-webarena.md](concepts/browsergym-webarena.md): how a
  WebArena task config becomes a BrowserGym `BrowserEnv`.
- [concepts/environment-server.md](concepts/environment-server.md): how
  env_server hosts and coordinates live BrowserGym sessions.

### I want to run it

- [setup/gcp-resources.md](setup/gcp-resources.md): the Google Cloud side —
  network tags, firewall rules, VM creation, repo sync, and SSH. Start here if
  you are deploying on GCP.
- [setup/vm-setup.md](setup/vm-setup.md): provider-agnostic VM setup — install
  dependencies, download WebArena data, start the required services, and verify
  the environment.
- [setup/services.md](setup/services.md): start, stop, health checks, and logs
  for the setup service and env_server service.
- [reference/troubleshooting.md](reference/troubleshooting.md): common symptoms,
  likely causes, and fixes.
- [firewall.md](firewall.md): restricting the HTTP firewall rules to an explicit
  allowlist instead of `0.0.0.0/0`.
- [setup/building_webarena_images.md](setup/building_webarena_images.md):
  building the WebArena site images yourself instead of using prebuilt ones, and
  setting up without an artifact registry or bucket.

### I want to integrate or debug APIs

- [concepts/rollout-lifecycle.md](concepts/rollout-lifecycle.md): when setup,
  reset, step, close, and teardown happen.
- [reference/api-reference.md](reference/api-reference.md): raw HTTP APIs for
  setup/status/teardown and env_server endpoints.
- [reference/troubleshooting.md](reference/troubleshooting.md): health checks,
  logs, reachability checks, and stuck port recovery.

## Trajectory
- [trajectory_format/](trajectory_format/README.md) explains the trajectory data
  format shared by eval, RL, and SFT, and the prompt builders that render it.

## Data
- [data_preparation.md](data_preparation.md) contains the details for data preparation for the SFT phase.
- The data used for RL phase is located in the [task_configs](../task_configs/webarena_train.jsonl)
- [datasets_and_models.md](datasets_and_models.md) maps each experiment to the
  dataset it trains on and the checkpoint name it produces.

## Training

Every method is trained in 2 stages: SFT bootstrap followed by GRPO RL from the SFT
checkpoint. Both stages run through shared entry-point scripts; the method is selected by
a YAML config and a dataset path.

- [training/](training/README.md) is the entry point: the shared scripts, the required
  environment variables, and the base-checkpoint download and conversion.

Per-method, end-to-end guides (SFT config, SFT run, RL config, RL run, evaluation):

- [training/append.md](training/append.md): append memory (our method).
- [training/overwrite.md](training/overwrite.md): overwrite memory (MEM1 baseline).
- [training/reasoning-only.md](training/reasoning-only.md): reasoning only
  (WebAgentR1 baseline).
- [training/long_short_memory.md](training/long_short_memory.md): long + short term
  memory.
- [training/cluster-launch.md](training/cluster-launch.md): submitting the training job
  and running the scripts in `tmux`.

## Evaluation

There is one eval script per method under [scripts/eval](../scripts/eval). Each pins the
flags for its method and reads the environment (`ENV_SERVER_SERVICE_URL`,
`WEBSITES_VM_HOST`, `LLM_BASE_URL`, `MODEL`, `RUN_NAME`, `SEED`) from exported variables.

- [eval.md](eval.md): the sglang hosting and SSH tunnel steps, the variables every script
  needs, and the untrained baselines (full context, API models).
- For a trained checkpoint, see the evaluation section of its training guide under
  [training/](training/README.md#evaluation).

## Checkpoint conversion

Miles uses Megatron for training, so we have to convert checkpoints from HF to Megatron
and back again. See [checkpoint_conversion.md](checkpoint_conversion.md) for the exact
commands — training checkpoint to transformers format, and back to torch dist for
continued training (needed between the SFT and RL stages; reusing the checkpoints saved by
the training scripts directly produces garbage generations).
