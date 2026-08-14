# Troubleshooting Rollout Environments

Use this guide when setup, service startup, rollout execution, or teardown does
not behave as expected.

For service commands and log locations, see
[../setup/services.md](../setup/services.md). For endpoint shapes, see
[api-reference.md](api-reference.md).

## Common Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `/setup` returns `409 Port Occupied` | Previous rollout did not teardown. | Call `/teardown` with the owning service and wait for `idle`. |
| `/setup` returns `409 Setup already in progress` | Another setup is running on that port. | Wait for `/status` to become `occupied` or `error`. |
| `/status` returns `error` | Reset or teardown script failed. | Check the returned stderr and setup service logs. Then teardown the same service/port if possible. |
| Browser cannot load a WebArena site | `site_urls` host or port is wrong, or the `env_server VM` cannot reach the `Websites VM`. | Verify the URL from the `env_server VM` and check firewall rules. |
| env_server returns `429` | Maximum live sessions reached. | Close unused rollout sessions or increase `ROLLOUT_SERVER_MAX_SESSIONS`. |
| Containers remain after an exception | Browser session was closed but setup service teardown did not run. | Put `teardown_env(...)` in a `finally` block separate from `env.close()`. |

## Health Checks

Run from your laptop:

```sh
curl http://<WEBSITES_VM_IP_ADDRESS>:7565/health
curl http://<ENV_SERVER_VM_IP_ADDRESS>:8082/health
```

Expected responses:

```json
{"ok":true}
```

```json
{"status":"ok","live_sessions":0,"max_live_sessions":64}
```

## Logs

On the `Websites VM`:

```text
~/context-scythe/environment_setup/webarena/rollout_env/setup_server.log
~/context-scythe/environment_setup/webarena/rollout_env/homepage.log
```

On the `env_server VM`:

```text
~/context-scythe/environment_setup/environment_server/env_server.log
```

## Check Reachability from env_server

The browser runs on the `env_server VM`, so WebArena URLs must be reachable from
that VM. SSH into the `env_server VM` and check the relevant `Websites VM` URLs:

```sh
curl http://<WEBSITES_VM_HOST>:7564
curl http://<WEBSITES_VM_HOST>:8081
curl http://<WEBSITES_VM_HOST>:9081
```

If these fail from the `env_server VM` but work elsewhere, check network tags,
firewall rules, and the hostnames used in `site_urls`.

## Recover a Stuck Port

Check the port status:

```sh
curl "$SETUP_SERVICE_URL/status?port=8081"
```

If the port is `occupied` or `error`, teardown using the same service that owns
the port:

```sh
curl -X POST "$SETUP_SERVICE_URL/teardown" \
  -H "Content-Type: application/json" \
  -d '{"service": "shopping", "port": 8081}'
```

Then poll until the port is `idle`:

```sh
curl "$SETUP_SERVICE_URL/status?port=8081"
```

## Recovering from failed evaluation or training
If the evaluation or training fails, we need to reset the env_server and setup servers.

On the websites VM
```sh
cd ~/context-scythe

# Remove dangling containers that missed teardown in the failed run
docker rm -f $(docker ps -aq)

# Start the setup server again
bash environment_setup/webarena/rollout_env/start_rollout_servers.sh
```

On the env_server VM
```sh
# Start the env server again to close existing rollout sessions
# that never got a close request. The start command will automatically
# stop the existing process and create a new one.
bash environment_setup/environment_server/manage-env-server.sh start
```

## Related Docs

- [../concepts/rollout-lifecycle.md](../concepts/rollout-lifecycle.md): where
  setup, close, and teardown belong in rollout code.
- [../setup/gcp-resources.md](../setup/gcp-resources.md): network tags and
  firewall rules.
- [../setup/vm-setup.md](../setup/vm-setup.md): dependency installation and
  service startup on the VMs.
