# Rollout API Reference

Most training code should use `setup_env(...)`, `teardown_env(...)`,
`RemoteRolloutEnv`, and `AsyncRemoteRolloutEnv`. Use this reference when
debugging services directly or writing a new orchestrator.

For the order in which these APIs are used during a rollout, read
[../concepts/rollout-lifecycle.md](../concepts/rollout-lifecycle.md) first.

## setup Service

The setup service runs on the `Websites VM`:

```sh
SETUP_SERVICE_URL=http://<WEBSITES_VM_HOST>:7565
```

Allowed mutable services:

- `shopping`
- `shopping_admin`
- `gitlab`
- `reddit`

Allowed rollout ports:

- `8081` through `8088`
- `9081` through `9088`

Use each port for only one rollout container at a time. A port can be reused
after teardown completes and `/status` reports `idle`.

Port states:

```text
idle -> setting_up -> occupied -> tearing_down -> idle
                  \-> error -------------------/
```

## `POST /setup`

Starts setup for one rollout service container.

```sh
curl -X POST "$SETUP_SERVICE_URL/setup" \
  -H "Content-Type: application/json" \
  -d '{"service": "shopping", "port": 8081}'
```

Request body:

```json
{
  "service": "shopping",
  "port": 8081
}
```

Successful response:

```json
{
  "message": "Reset initiated",
  "port": 8081,
  "service": "shopping"
}
```

Setup is asynchronous. Poll `GET /status?port=<PORT>` until the port reports
`occupied`.

## `GET /status`

Returns the current state for one rollout port.

```sh
curl "$SETUP_SERVICE_URL/status?port=8081"
```

Successful response:

```json
{
  "status": "occupied",
  "port": 8081,
  "service": "shopping"
}
```

Possible `status` values:

- `idle`: no rollout container is running on the port.
- `setting_up`: setup was accepted and is still running.
- `occupied`: setup completed and the service owns the port.
- `tearing_down`: teardown was accepted and is still running.
- `error`: the last setup or teardown operation failed.

If the port is in `error`, the server returns `500` and includes the failing
command output:

```json
{
  "message": "Operation failed.",
  "port": 8081,
  "error": "<stderr>"
}
```

## `POST /teardown`

Starts teardown for one rollout service container.

```sh
curl -X POST "$SETUP_SERVICE_URL/teardown" \
  -H "Content-Type: application/json" \
  -d '{"service": "shopping", "port": 8081}'
```

Request body:

```json
{
  "service": "shopping",
  "port": 8081
}
```

Successful response:

```json
{
  "message": "Teardown initiated",
  "port": 8081,
  "service": "shopping"
}
```

Teardown is asynchronous. Poll `GET /status?port=<PORT>` until the port reports
`idle`.

Teardown is accepted only when:

- The port is `occupied` or `error`.
- The request uses the same service that owns the port.

Static shared services such as map and wikipedia are not in the allowed service
list at all, so they can never be torn down through this API. Any `service` value
outside the four allowed names is rejected by request validation with `422`
before the handler runs.

## Polling Examples

Wait for setup to finish:

```sh
while true; do
  status="$(curl -s "$SETUP_SERVICE_URL/status?port=8081")"
  echo "$status"
  echo "$status" | grep -q '"status":"occupied"' && break
  sleep 5
done
```

Wait for teardown to finish:

```sh
while true; do
  status="$(curl -s "$SETUP_SERVICE_URL/status?port=8081")"
  echo "$status"
  echo "$status" | grep -q '"status":"idle"' && break
  sleep 5
done
```

## Error Responses

Invalid port:

```json
{
  "message": "Port 8080 not allowed."
}
```

Setup requested for an occupied port:

```json
{
  "message": "Port Occupied",
  "port": 8081,
  "service": "shopping"
}
```

Setup requested while another setup is already running on the port:

```json
{
  "message": "Setup already in progress",
  "port": 8081,
  "service": "shopping"
}
```

Teardown requested while another teardown is already running on the port:

```json
{
  "message": "Teardown already in progress",
  "port": 8081,
  "service": "shopping"
}
```

Teardown requested for a port owned by another service:

```json
{
  "message": "Port occupied by different service",
  "port": 8081,
  "service": "shopping"
}
```

Teardown requested for an idle port:

```json
{
  "message": "Port not occupied",
  "port": 8081,
  "service": null
}
```

## env_server API

The env_server service runs on the `env_server VM`:

```sh
http://<ENV_SERVER_VM_HOST>:8082
```

Rollout code should use `RemoteRolloutEnv` or `AsyncRemoteRolloutEnv` instead
of calling env_server endpoints directly.

The API is backed by Ray-managed BrowserGym sessions. Each successful
`POST /session_init` reserves one rollout session that can run concurrently with
other live sessions until it is closed or cleaned up as idle.

Useful endpoints:

- `GET /health`: server health and live session count.
- `POST /session_init`: create a remote rollout session.
- `POST /rollout/{rollout_id}/reset`: reset a rollout session.
- `POST /rollout/{rollout_id}/step`: apply one BrowserGym action.
- `GET /rollout/{rollout_id}`: fetch rollout session status.
- `POST /rollout/{rollout_id}/close`: close a rollout session.

### Screenshots

`POST /session_init` takes an optional `include_screenshots` (default `false`). When
`true`, every `reset` and `step` observation also carries `screenshot`, a base64-encoded
PNG.

This is a debugging aid for inspecting eval rollouts, not a training input. The eval
entrypoints set it from `--save_screenshots` and write the PNGs to disk beside the
trajectory JSON; nothing puts them into a prompt. RL rollout generation leaves it at
`false`, since encoding a PNG on every step would add latency and response size for data
nothing reads.

Request field:

```json
{
  "task_id": 0,
  "task_config": {"task_id": 0, "sites": ["shopping"]},
  "webarena": {
    "homepage_url": "http://<WEBSITES_VM_HOST>:7564",
    "site_urls": {"shopping": "http://<WEBSITES_VM_HOST>:8081"}
  },
  "include_screenshots": true
}
```

Both `task_config` and `site_urls` must be non-empty, and `task_config["task_id"]` must
match the top-level `task_id`.

## Stress Testing

Use [scripts/stresstest_setup.py](../../scripts/stresstest_setup.py) to stress
test setup, readiness polling, endpoint reachability, and teardown across the
rollout port range.

The setup service must already be running:

```sh
bash environment_setup/webarena/rollout_env/start_rollout_servers.sh
```

Run the stress test from the repository root:

```sh
python scripts/stresstest_setup.py <WEBSITES_VM_HOST>
```

The script targets `http://<WEBSITES_VM_HOST>:7565`, starts `shopping` on
`8081-8088`, starts `shopping_admin` on `9081-9088`, checks readiness, checks
service endpoints, sends teardown requests, and waits for ports to return to
`idle`.

Run this only against a VM whose rollout ports are free.
