# BrowserGym and WebArena Task Model

This guide explains how Context Scythe turns a WebArena task config into a
BrowserGym environment. It is a conceptual map for reading the code; for VM
setup, service operations, and teardown, use the linked operational docs.

## Mental Model

Context Scythe keeps four concerns separate:

| Layer | Main code | Responsibility |
| --- | --- | --- |
| WebArena task config | WebArena task JSON passed through rollout code | Declares the task intent, required sites, start URLs, geolocation, and evaluator rules. |
| BrowserGym adapter | `packages/core/src/context_scythe/environment/webarena_task.py` | Wraps one WebArena config as a BrowserGym `AbstractBrowserTask`. |
| Remote BrowserGym owner | `packages/env_server/src/context_scythe/env_server/actor.py` | Creates and owns one live BrowserGym `BrowserEnv` inside env_server. |
| Remote rollout client | `packages/core/src/context_scythe/environment/gym.py` | Provides sync and async Gym-style HTTP clients for rollout code. |

The key idea is that BrowserGym owns the browser loop, while
`ConfigTaskWrapper` owns the WebArena-specific task setup and validation hooks.
`RemoteRolloutEnv` and `AsyncRemoteRolloutEnv` are not BrowserGym classes. They
are HTTP clients that talk to env_server, whose actor creates the actual
BrowserGym environment.

Conceptually:

```text
rollout code
  -> RemoteRolloutEnv / AsyncRemoteRolloutEnv
  -> env_server API
  -> Ray actor
  -> create_env_for_task(...)
  -> BrowserGym BrowserEnv
  -> ConfigTaskWrapper
  -> WebArena websites
```

## From Config to `BrowserEnv`

`create_env_for_task(...)` is the entrypoint that builds a BrowserGym
`BrowserEnv` for one task. It freezes the WebArena inputs into a BrowserGym task
entrypoint and passes through BrowserGym environment options such as headless
mode and action mapping.

Code pointer:
`packages/core/src/context_scythe/environment/webarena_task.py`

Conceptually:

```text
task_config + task_id + homepage_url + site_urls
  -> ConfigTaskWrapper(...)
  -> BrowserGym BrowserEnv(...)
  -> reset()/step()/close()
```

The task config is passed directly instead of relying on BrowserGym or WebArena
global task registration. This matters for rollout isolation: each rollout can
receive different `site_urls`, usually pointing to rollout-specific mutable
WebArena containers.

This is the BrowserGym interface. It is about constructing and running a Python
`BrowserEnv` object. It does not know about env_server rollout IDs, HTTP
requests, setup-service ports, or remote session cleanup.

## `ConfigTaskWrapper`

`ConfigTaskWrapper` adapts a WebArena config to BrowserGym's
`AbstractBrowserTask` interface.

During construction, it:

- Stores BrowserGym task properties such as viewport, timeout, locale, and
  timezone.
- Verifies that every site listed by the task config exists in `site_urls`.
- Builds a `CustomWebArenaInstance` from `homepage_url` and `site_urls`.
- Replaces WebArena URL placeholders such as `__SHOPPING__`, `__REDDIT__`, and
  `__GITLAB__` with the rollout-specific URLs.
- Stores evaluator configuration for later setup.

The placeholder replacement is the bridge between static WebArena task configs
and per-rollout infrastructure. The task can say "shopping" conceptually, while
rollout orchestration decides which concrete host and port that shopping site
uses.

## What Happens on `reset()`

BrowserGym calls the task's `setup(page)` hook during environment reset. In this
repository, `ConfigTaskWrapper.setup(...)` performs the WebArena-specific browser
initialization:

1. Warm up every configured WebArena site through `CustomWebArenaInstance`.
2. Build the WebArena evaluator using the task config.
3. Optionally apply extra Playwright HTTP headers from `PW_EXTRA_HEADERS`.
4. Log in to each required site declared by `config["sites"]`.
5. Apply task geolocation when present.
6. Navigate to the configured start URL or URLs.
7. Return the task intent as the BrowserGym goal.

Code pointers:

- `packages/core/src/context_scythe/environment/webarena_task.py`
- `packages/core/src/context_scythe/environment/webarena_instance.py`

Multiple start URLs are split on WebArena's ` |AND| ` separator. Each additional
start URL opens in a new browser page, matching WebArena's multi-tab task model.

## What Happens on `step()`

Inside BrowserGym, `BrowserEnv.step(action)` maps the high-level action into
Playwright operations, applies it to the active browser page, updates the
observation, and asks the WebArena task wrapper whether the rollout has
terminated.

When the environment is used remotely, rollout code does not call
`BrowserEnv.step(...)` directly. It calls `RemoteRolloutEnv.step(...)` or
`AsyncRemoteRolloutEnv.step(...)`; env_server forwards that request to the Ray
actor that owns the live BrowserGym environment, and the actor calls
`BrowserEnv.step(...)`.

The rollout client only sees the Gym-style API shape:

```text
reset(seed) -> observation, info
step(action) -> observation, reward, terminated, truncated, info
close()
```

For the remote HTTP lifecycle around these calls, see
[rollout-lifecycle.md](rollout-lifecycle.md) and
[../reference/api-reference.md](../reference/api-reference.md).

## Remote Rollout Client

The remote rollout clients live in
`packages/core/src/context_scythe/environment/gym.py`:

- `RemoteRolloutEnv`: synchronous `requests` client.
- `AsyncRemoteRolloutEnv`: async `httpx.AsyncClient` client.

These classes deliberately expose a Gym-like shape to rollout code, but they do
not implement BrowserGym's task interface and they do not create local browser
objects. Their job is to translate method calls into env_server API requests:

```text
reset(seed)
  -> first reset: POST /session_init, then POST /rollout/{rollout_id}/reset
  -> later resets: POST /rollout/{rollout_id}/reset

step(action)
  -> POST /rollout/{rollout_id}/step

status()
  -> GET /rollout/{rollout_id}

close()
  -> POST /rollout/{rollout_id}/close
```

The logical connection is the payload. The remote client sends `task_config`,
`task_id`, `homepage_url`, and `site_urls` during session initialization.
env_server passes those values to its actor, and the actor uses them to build
the BrowserGym `BrowserEnv` through `create_env_for_task(...)`.

Closing a remote rollout client closes only the env_server browser session. It
does not tear down WebArena containers or release setup-service rollout slots;
that remains the responsibility of rollout orchestration code.

## Validation, Reward, and Termination

BrowserGym calls `ConfigTaskWrapper.validate(...)` after actions to decide
whether the rollout is done and what reward to return.

Validation has three important behaviors:

- It rejects navigation outside the configured WebArena hosts and homepage host.
  Unauthorized navigation terminates the task with zero reward.
- It treats the latest assistant message as the WebArena stop answer. An
  `infeasible` message becomes the answer `N/A`.
- It calls the WebArena evaluator only after a stop answer exists.

The evaluator receives a minimal fake WebArena trajectory because the WebArena
evaluation code only needs the final answer for these tasks. If the evaluator
returns a positive score, the rollout terminates successfully. If the agent
stops with an incorrect answer, the rollout also terminates, but with zero
reward. Otherwise, the rollout remains active.

## Where This Fits in the Remote Rollout System

The BrowserGym/WebArena adapter does not allocate ports, start containers, or
tear down WebArena services. Those responsibilities belong to the setup service
and rollout orchestration code.

Use this split when reading or changing the system:

- `webarena_task.py`: BrowserGym adapter; turns one WebArena task config into a
  BrowserGym task and `BrowserEnv`.
- `webarena_instance.py`: site reachability checks and UI login behavior.
- `env_server/actor.py`: remote owner of one live BrowserGym environment.
- `environment/gym.py`: sync and async HTTP clients used by rollout code; not
  the BrowserGym environment implementation.
- [rollout-lifecycle.md](rollout-lifecycle.md): how setup service, env_server,
  browser sessions, and teardown fit together operationally.
- [architecture.md](architecture.md): VM boundaries, ownership, ports, and
  isolation model.
