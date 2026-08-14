# Environment Server Model

This guide explains how the env_server service works internally. It is a
conceptual companion to
[../reference/api-reference.md](../reference/api-reference.md); use that file
for endpoint shapes and raw HTTP examples.

## Purpose

The env_server service is the remote owner of live BrowserGym browser sessions.
Training or rollout code should not create browsers directly. Instead, rollout
code talks to `RemoteRolloutEnv` or `AsyncRemoteRolloutEnv`, which call the
env_server HTTP API.

The service exists to make concurrent browser execution practical. It uses Ray
actors so each rollout session gets its own isolated BrowserGym environment,
while the FastAPI service and session manager provide one API for creation,
reset, step, status, close, limits, and idle cleanup.

The env_server is responsible for:

- Creating one live BrowserGym environment per rollout session.
- Running multiple live browser sessions concurrently through Ray actors.
- Applying actions to that environment.
- Returning compact JSON-safe observations.
- Tracking session status, step count, and activity timestamps.
- Closing idle or explicitly closed browser sessions.

It is not responsible for starting or tearing down WebArena website containers.
That belongs to the setup service on the `Websites VM`.

## Main Components

| Component | Main code | Responsibility |
| --- | --- | --- |
| FastAPI app | `packages/env_server/src/context_scythe/env_server/app.py` | Defines HTTP routes, initializes Ray, installs error handlers, and forwards requests to the session manager. |
| Session manager | `packages/env_server/src/context_scythe/env_server/manager.py` | Owns rollout IDs, session state, concurrency limits, idle cleanup, and actor lifecycle. |
| Ray actor | `packages/env_server/src/context_scythe/env_server/actor.py` | Owns one live BrowserGym `BrowserEnv` and executes reset/step/close. |
| Schemas | `packages/env_server/src/context_scythe/env_server/schemas.py` | Defines request and response models shared by routes, clients, and tests. |
| Observation compaction | `packages/env_server/src/context_scythe/env_server/observations.py` | Converts BrowserGym observations into smaller API responses. |
| Remote clients | `packages/core/src/context_scythe/environment/gym.py` | Provide sync and async Gym-style clients over the env_server API. |

The useful boundary is: FastAPI handles transport, `SessionManager` handles
coordination and concurrency limits, and each Ray `EnvSessionActor` handles one
browser execution session.

## Request Flow

At a high level, one rollout session moves through this path:

```text
RemoteRolloutEnv / AsyncRemoteRolloutEnv
  -> env_server FastAPI route
  -> SessionManager
  -> Ray EnvSessionActor
  -> BrowserGym BrowserEnv
  -> WebArena websites
```

The browser runs on the `env_server VM`. The WebArena sites run on the
`Websites VM`. The browser therefore needs network access from the env_server VM
to the concrete URLs in `homepage_url` and `site_urls`.

## Startup

When the FastAPI app starts, it initializes Ray unless a test has injected a
custom session manager. It then stores a `SessionManager` on `app.state`.

The key settings are:

- `ROLLOUT_SERVER_HOST`: bind host, defaulting to `0.0.0.0`.
- `ROLLOUT_SERVER_PORT`: bind port, defaulting to `8000` in code and usually
  set to `8082` by service startup scripts.
- `ROLLOUT_SERVER_MAX_SESSIONS`: maximum live plus pending sessions, defaulting
  to `64`.
- `ROLLOUT_SERVER_IDLE_TIMEOUT_S`: idle timeout in seconds, defaulting to
  `1800`.

Ray inherits the active virtual environment through the runtime environment when
`VIRTUAL_ENV` is present. This keeps actor workers on the same Python
environment as the server process.

## Session Creation

Session creation starts at `POST /session_init`.

Conceptually, the session manager:

1. Cleans up idle sessions.
2. Checks the live plus pending session count against `max_live_sessions`.
3. Reserves a new rollout ID.
4. Creates one Ray `EnvSessionActor`.
5. Pings the actor to verify it is reachable.
6. Stores a `RolloutSession` record with status `ready`.

The actor builds the actual BrowserGym environment. It creates a WebArena
high-level action set, reads the evaluator configuration from
`OPENAI_BASE_URL` and `OPENAI_API_KEY`, and calls `create_env_for_task(...)`.

The created session is ready but not yet reset. The browser task setup happens
when the client calls reset.

## Reset

Reset starts at `POST /rollout/{rollout_id}/reset`.

The session manager looks up the active session, verifies it is usable, and
calls `reset(seed)` on the actor. The actor delegates to `BrowserEnv.reset(...)`.
That invokes the WebArena task setup described in
[browsergym-webarena.md](browsergym-webarena.md): site warmup, evaluator setup,
login, geolocation, start URL navigation, and goal recovery.

After reset succeeds, the manager:

- Sets session status to `running`.
- Resets `step_count` to `0`.
- Updates `last_activity_at`.
- Returns a compact observation and BrowserGym info.

## Step

Step starts at `POST /rollout/{rollout_id}/step`.

The session manager forwards the action string to the actor. The actor calls
`BrowserEnv.step(action)`, which applies the BrowserGym action mapping and runs
WebArena validation.

After a successful step, the manager:

- Increments `step_count`.
- Sets status to `terminated` when BrowserGym returns `terminated` or
  `truncated`.
- Keeps status as `running` otherwise.
- Updates `last_activity_at`.
- Returns the compact observation, reward, termination flags, and info.

Invalid BrowserGym actions are reported as `InvalidActionError`. Unexpected
actor or BrowserGym failures mark the session as `failed` and are reported as
infrastructure errors.

## Observation Shape

BrowserGym observations can contain data that is too large or not JSON-safe for
the rollout API. The actor compacts observations before returning them to the
manager.

The compact observation currently includes:

- `axtree`: flattened accessibility tree text.
- `active_page_index`: active browser page index or indexes.
- `open_pages_titles`: titles for all open pages.
- `open_pages_urls`: URLs for all open pages.
- `last_action_error`: BrowserGym's last action error, when present.

Screenshots are off by default and are opt-in per session: `POST /session_init`
takes `include_screenshots`, which the actor forwards to observation compaction
so each response also carries `screenshot`, a base64-encoded PNG.

This exists only for debugging and inspecting eval rollouts — it is never
enabled during training. The eval entrypoints turn it on through
`--save_screenshots`, which writes the PNGs to disk next to the trajectory JSON;
they are never composed into prompts, so they cost nothing at training time and
do not affect what the model sees. RL rollout generation leaves the flag at its
default of `false`, since encoding a PNG per step would add latency and response
size to every step of every rollout for data nothing reads.

## Status and Health

`GET /health` reports whether the FastAPI service is alive and how many live
sessions the manager currently tracks.

`GET /rollout/{rollout_id}` returns session-level status. For active sessions,
the manager also pings the actor. If the ping fails, the manager marks the
session `failed` and reports `actor_reachable=false`.

Session status values are:

- `ready`: actor exists, but reset has not run yet.
- `running`: reset has run and the task is active.
- `terminated`: BrowserGym returned `terminated` or `truncated`.
- `closed`: the session has been explicitly closed.
- `failed`: actor, BrowserGym, or infrastructure failure.
- `initializing`: defined in the schema for completeness, but current manager
  code keeps initializing sessions in a pending set rather than exposing this as
  a stored session status.

## Close and Cleanup

Explicit close starts at `POST /rollout/{rollout_id}/close`.

The manager removes the session from the live session map, marks it closed,
calls `close()` on the actor, queues actor exit, and records the rollout ID in
`closed_rollout_ids`. Repeated close calls for the same rollout ID are
idempotent after the first close.

Idle cleanup runs before session creation. Any session whose
`last_activity_at` is older than `idle_timeout_s` is closed using the same close
path. This protects the env_server from leaked browser sessions, but rollout
code should still call `close()` explicitly in a `finally` block.

Closing an env_server session closes only the browser-side BrowserGym session.
It does not stop WebArena website containers or release setup-service rollout
ports. That teardown is separate and is covered in
[rollout-lifecycle.md](rollout-lifecycle.md).

## Error Boundary

The service translates internal failures into stable API errors:

- Unknown rollout ID -> `404`.
- Invalid BrowserGym action -> `400`.
- Maximum live sessions exceeded -> `429`.
- Closed or failed session reuse -> `409`.
- Actor, Ray, BrowserGym, or other infrastructure failure -> `500`.

The public response body uses the service-level error class and public message,
while server logs keep the exception details.

## Reading Order

For conceptual reading:

1. [mental-model.md](mental-model.md): the shortest conceptual overview.
2. [architecture.md](architecture.md): VM and ownership boundaries.
3. [browsergym-webarena.md](browsergym-webarena.md): how a WebArena task becomes
   a BrowserGym env.
4. [environment-server.md](environment-server.md): how env_server hosts and
   coordinates live sessions.
5. [rollout-lifecycle.md](rollout-lifecycle.md): how setup, browser execution,
   and teardown fit together operationally.
6. [../reference/api-reference.md](../reference/api-reference.md): raw endpoint
   reference.
