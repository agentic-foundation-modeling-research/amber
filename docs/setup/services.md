# Rollout Services

Use this guide to start, stop, verify, and find logs for the services that power
remote rollouts.

For the first-time setup sequence, use [gcp-resources.md](gcp-resources.md) to
provision on GCP, then [vm-setup.md](vm-setup.md) to install and start. For the
per-rollout flow that uses these services, see
[../concepts/rollout-lifecycle.md](../concepts/rollout-lifecycle.md).

## Service Summary

| Service | Runs on | Port | Purpose |
| --- | --- | --- | --- |
| setup service | `Websites VM` | `7565` | Sets up, reports status for, and tears down rollout-scoped mutable WebArena containers. |
| homepage service | `Websites VM` | `7564` | Hosts the WebArena homepage utilities, including calculator and scratchpad access. |
| env_server service | `env_server VM` | `8082` | Owns Ray-backed concurrent BrowserGym sessions and exposes remote rollout APIs. |

## Start the setup Service

Run on the `Websites VM`:

```sh
cd ~/context-scythe
bash environment_setup/webarena/rollout_env/start_rollout_servers.sh
```

This starts the setup service and homepage service. The command prints URLs and
log paths similar to:

```log
Setup service running on http://<WEBSITES_VM_HOST>:7565
Homepage running on http://<WEBSITES_VM_HOST>:7564
```

## Verify the setup Service

Run on your laptop:

```sh
curl http://<WEBSITES_VM_IP_ADDRESS>:7565/health
```

Expected response:

```json
{"ok":true}
```

## Stop the setup Service

Run on the `Websites VM`:

```sh
cd ~/context-scythe
bash environment_setup/webarena/rollout_env/stop_rollout_servers.sh
```

## setup Service Logs

On the `Websites VM`:

```text
~/context-scythe/environment_setup/webarena/rollout_env/setup_server.log
~/context-scythe/environment_setup/webarena/rollout_env/homepage.log
```

## Start env_server

Run on the `env_server VM`:

```sh
cd ~/context-scythe
bash environment_setup/environment_server/manage-env-server.sh start
```

The management script loads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from
`~/context-scythe/.env` by default before starting FastAPI and Ray. To use a
different secrets file, set `ENV_SERVER_ENV_FILE`:

Ray is part of the service startup because env_server uses one Ray actor per
live BrowserGym session. This is what allows multiple rollout browser sessions
to run concurrently behind the same HTTP API.

```sh
ENV_SERVER_ENV_FILE=/path/to/env-server.env \
bash environment_setup/environment_server/manage-env-server.sh start
```

The command prints a URL similar to:

```log
env_server running on port http://<ENV_SERVER_VM_IP_ADDRESS>:8082
```

## Verify env_server

Run on your laptop:

```sh
curl http://<ENV_SERVER_VM_IP_ADDRESS>:8082/health
```

Expected response:

```json
{"status":"ok","live_sessions":0,"max_live_sessions":64}
```

## Stop env_server

Run on the `env_server VM`:

```sh
cd ~/context-scythe
bash environment_setup/environment_server/manage-env-server.sh stop
```

## env_server Logs

On the `env_server VM`:

```text
~/context-scythe/environment_setup/environment_server/env_server.log
```

## Direct env_server Startup

The management script is recommended. Advanced callers can start the underlying
CLI directly:

```sh
ROLLOUT_SERVER_HOST=0.0.0.0 \
ROLLOUT_SERVER_PORT=8082 \
ROLLOUT_SERVER_MAX_SESSIONS=64 \
ROLLOUT_SERVER_IDLE_TIMEOUT_S=1800 \
OPENAI_BASE_URL=https://proxy.example/v1 \
OPENAI_API_KEY=... \
context-scythe-env-server
```

## Next Reading

- [../reference/troubleshooting.md](../reference/troubleshooting.md): common
  startup, reachability, and cleanup failures.
- [../reference/api-reference.md](../reference/api-reference.md): raw setup
  service and env_server endpoints.
