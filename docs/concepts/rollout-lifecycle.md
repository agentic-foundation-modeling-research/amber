# Rollout Lifecycle

This guide explains what happens for each remote rollout after the VMs and
services are already running.

For VM boundaries and port ownership, read [architecture.md](architecture.md)
first. For the setup path, use
[../setup/gcp-resources.md](../setup/gcp-resources.md) (GCP provisioning) and
[../setup/vm-setup.md](../setup/vm-setup.md) (install and start).

## End-to-End Flow

For each rollout:

1. Choose a free rollout slot on the `Websites VM`.
2. Build one `EnvConfig` per mutable service needed by the task.
3. Call `setup_env(...)`, or send `POST /setup` to
   `http://<WEBSITES_VM_HOST>:7565`, for each `(service, port)` pair.
4. Poll `GET /status?port=<PORT>` until every mutable port reports `occupied`.
5. Build `homepage_url` and `site_urls` with URLs that are reachable from the
   `env_server VM`.
6. Create a `RemoteRolloutEnv` or `AsyncRemoteRolloutEnv` with
   `server_url="http://<ENV_SERVER_VM_HOST>:8082"`.
7. Run the rollout through the env_server service. env_server uses Ray actors to
   support concurrent browser sessions, with one BrowserGym environment per
   rollout session.
8. Close the remote rollout env session. This closes the browser session only.
9. Call `teardown_env(...)`, or send `POST /teardown`, for each mutable
   `(service, port)` pair.
10. Poll `GET /status?port=<PORT>` until every mutable port reports `idle`.
11. Reuse a rollout slot only after all of its mutable ports return to `idle`.

Cleanup should happen in a `finally` block. `RemoteRolloutEnv.close()` and
`AsyncRemoteRolloutEnv.close()` do not remove WebArena containers; setup service
teardown is a separate required step.

## Example Ports

For a task that needs `shopping` and `shopping_admin` with group size 2:

```text
rollout 0:
  shopping       -> http://<WEBSITES_VM_HOST>:8081
  shopping_admin -> http://<WEBSITES_VM_HOST>:9081/admin

rollout 1:
  shopping       -> http://<WEBSITES_VM_HOST>:8082
  shopping_admin -> http://<WEBSITES_VM_HOST>:9082/admin
```

The corresponding `site_urls` for rollout 0:

```python
site_urls = {
    "shopping": "http://<WEBSITES_VM_HOST>:8081",
    "shopping_admin": "http://<WEBSITES_VM_HOST>:9081/admin",
}
```

## Remote Rollout Client

Rollout code should use the remote rollout clients from
`packages/core/src/context_scythe/environment/gym.py`. Use
`AsyncRemoteRolloutEnv` for async rollout loops:

```python
env = AsyncRemoteRolloutEnv(
    server_url="http://<ENV_SERVER_VM_HOST>:8082",
    task_id=task_config["task_id"],
    task_config=task_config,
    homepage_url=homepage_url,
    site_urls=site_urls,
)

try:
    obs, info = await env.reset(seed=42)
    obs, reward, terminated, truncated, info = await env.step(action)
finally:
    await env.close()
```

`RemoteRolloutEnv` is the synchronous `requests` client.
`AsyncRemoteRolloutEnv` is the async-native `httpx.AsyncClient` client. Both are
thin HTTP clients over the env_server service.

Important client behavior:

- Session creation is lazy. Constructing either client has no server-side effect.
  The first `reset()` calls `/session_init`, stores the returned `rollout_id`,
  then calls `/rollout/{rollout_id}/reset`.
- Subsequent `reset()` calls reuse the same `rollout_id` and call only
  `/rollout/{rollout_id}/reset`.
- `step(action)` requires a prior reset and returns
  `(observation, reward, terminated, truncated, info)`.
- `close()` calls `/rollout/{rollout_id}/close` when a session exists and is
  idempotent locally.
- The constructor requires `task_id` separately from `task_config` and validates
  that `task_config["task_id"]` matches it.
- WebArena setup service lifecycle is intentionally outside this wrapper. Port
  allocation, setup polling, and teardown require separate coordination.

The clients accept optional HTTP session objects for tests and advanced callers.
By default `RemoteRolloutEnv` creates and owns a `requests.Session`, while
`AsyncRemoteRolloutEnv` creates and owns an `httpx.AsyncClient`. If a caller
passes a session explicitly, the caller owns that session; `close()` closes the
remote rollout session but does not close the shared HTTP session/client.

## Next Reading

- [browsergym-webarena.md](browsergym-webarena.md): what happens inside
  BrowserGym during reset, step, validation, and termination.
- [environment-server.md](environment-server.md): how env_server manages live
  remote browser sessions.
- [../reference/api-reference.md](../reference/api-reference.md): raw endpoint
  shapes for setup service and env_server.
