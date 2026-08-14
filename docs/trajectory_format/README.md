# Trajectory Data and Prompt Builders

These docs explain the two data structures every agent loop in this repository
shares: the trajectory record that accumulates one rollout, and the prompt
builders that turn that record into chat messages.

The split is deliberate:

- **Trajectory data** is storage. It holds observations, model responses, and
  memory for each step, and knows how to serialize itself to JSON.
- **Prompt builders** are presentation. They read a trajectory and decide what
  the model actually sees at step `n`: full history, a memory block, or a
  sliding window.

Every experiment in this repository uses the same trajectory record and differs
only in which prompt builder reads it. That is the mechanism behind the
append/overwrite/reasoning-only comparison.

```text
env observation ──> StepData.observation
model response  ──> StepData.response   ──┐
memory response ──> StepData.memory     ──┤
                                          ├──> PromptBuilder.build_messages()
goal, site_urls ──> TrajectoryData      ──┘         │
                                                    v
                                          chat messages for step n+1
```

## Contents

- [trajectory-data.md](trajectory-data.md): the step and trajectory dataclasses,
  the tag protocol used to parse model output, the on-disk eval JSON, and the
  SFT dataset schema.
- [prompt-builders.md](prompt-builders.md): the builder family, which experiment
  uses which builder, and the two-call action/compression protocol that memory
  builders rely on.

## Related Docs

- [prompt-builders.md](prompt-builders.md#which-builder-each-entrypoint-uses): which eval
  entrypoint and flags select each builder.
- [../eval.md](../eval.md): hosting a checkpoint and running the untrained baselines.
- [../data_preparation.md](../data_preparation.md): how trajectories plus
  generated memories become SFT datasets.
- [../concepts/browsergym-webarena.md](../concepts/browsergym-webarena.md):
  where the raw BrowserGym observation comes from.
