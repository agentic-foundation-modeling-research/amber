# Rollout Environment Architecture

The remote WebArena rollout environment is split across two VMs and a rollout
client. The split keeps mutable WebArena state isolated from browser execution
and lets training jobs orchestrate rollouts without running WebArena containers
or browsers locally. The `env_server VM` exists so concurrent browser sessions
can be hosted through Ray instead of being created inside each training process.

If you are new to the system, read [mental-model.md](mental-model.md) first for
the high-level flow and glossary.

## Components

| Component | Runs on | Owns | Does not own |
| --- | --- | --- | --- |
| `Websites VM` | Google Cloud VM | WebArena websites, shared static services, rollout-scoped mutable containers, homepage service, setup service | BrowserGym sessions |
| setup service | `Websites VM` | FastAPI setup/status/teardown API on port `7565`; lifecycle management for mutable rollout containers | Browser sessions or env_server sessions |
| env_server service | `env_server VM` (Google Cloud VM) | FastAPI API, Ray-backed concurrent BrowserGym sessions, browser interactions | WebArena containers or rollout port allocation |
| Rollout client | Training cluster | Training loop orchestration; calls to setup service and env_server service | Local WebArena containers or local browsers |

```text
Training process
  |-- setup_env() / teardown_env() ------> Websites VM :7565
  |                                         |-- mutable containers on 8081-8088
  |                                         |-- mutable containers on 9081-9088
  |                                         |-- shared homepage/map/wikipedia
  |
  |-- RemoteRolloutEnv / AsyncRemoteRolloutEnv --> env_server VM :8082
                                                   |-- FastAPI
                                                   |-- Ray actors
                                                   |-- BrowserGym BrowserEnv sessions
                                                   |-- browser loads URLs from Websites VM
```

## Key Terms

- `rollout`: one agent attempt on one WebArena task.
- `group size G`: the number of parallel rollouts for the same batch item.
- `mutable service`: a WebArena site whose database can change during a rollout,
  such as `shopping`, `shopping_admin`, `gitlab`, or `reddit`.
- `static service`: a shared WebArena support site that does not mutate during a
  rollout, such as map, wikipedia, homepage, calculator, or scratchpad.
- `Websites VM`: the VM that hosts WebArena websites, shared static services,
  rollout containers, the homepage service, and the setup service.
- `setup service`: the FastAPI service on the `Websites VM` that manages setup,
  status, and teardown for rollout-scoped mutable WebArena containers.
- `env_server VM`: the VM that runs concurrent browser sessions through the
  env_server service.
- `env_server service`: the FastAPI service on the `env_server VM`; it uses Ray
  actors so many rollout sessions can each own an isolated BrowserGym
  environment.
- `rollout slot`: one reusable port position on a `Websites VM`. Slot 1 uses
  ports ending in `1`, slot 2 uses ports ending in `2`, and so on.
- `homepage_url`: the homepage utility URL passed to BrowserGym.
- `site_urls`: the per-task WebArena URLs passed to BrowserGym. Mutable service
  entries point at rollout-specific ports.

## Isolation Model

Some WebArena tasks mutate service databases. To keep parallel rollouts
isolated, each batch item gets a dedicated `Websites VM`, and each rollout for that
item gets dedicated mutable service containers and ports on that VM.

This means a training batch with `N` items needs `N` separate `Websites VM`
instances. Rollouts for the same batch item can share that item's `Websites VM`
by using different rollout slots.

For GRPO with group size `G`, the intended layout is:

- One `Websites VM` per batch item.
- Up to `G` rollout slots on that `Websites VM`.
- One `(service, port)` pair per mutable service in each rollout slot.
- Shared static services on the `Websites VM`, because these services do not
  mutate during rollouts.
- A setup service on the `Websites VM` that manages rollout container lifecycle.
- A separate `env_server VM` that owns Ray-backed BrowserGym sessions and talks
  to the `Websites VM` over HTTP.

## Ports and URL Rules

One `Websites VM` currently supports 8 rollout slots. WebArena tasks use at most 2
mutable services, so the setup service exposes two 8-port ranges:

- `8081` through `8088`
- `9081` through `9088`

For a two-site mutable task, the first mutable service uses the `808x` range and
the second mutable service uses the `908x` range. The port suffix identifies the
rollout slot:

| Rollout slot | First mutable service | Second mutable service |
| --- | --- | --- |
| 1 | `8081` | `9081` |
| 2 | `8082` | `9082` |
| 3 | `8083` | `9083` |
| ... | ... | ... |
| 8 | `8088` | `9088` |

The `Websites VM` and `env_server VM` can both use port number `8082` because they
are different machines. `http://<WEBSITES_VM_HOST>:8082` is a mutable WebArena
service URL, while `http://<ENV_SERVER_VM_HOST>:8082` is the env_server service
API URL.

Fixed `Websites VM` ports:

| Service | Port | Mutable during rollout? | Notes |
| --- | --- | --- | --- |
| map | `443` | No | Shared map service. |
| wikipedia | `444` | No | Shared wikipedia service. |
| homepage | `7564` | No | Includes homepage, calculator, and scratchpad access. |
| setup service | `7565` | N/A | Control-plane API for setup, status, and teardown. |

Required network reachability:

- Training cluster to `Websites VM` setup service: `7565`.
- Training cluster to `env_server VM`: `8082`.
- `env_server VM` to `Websites VM` static service ports: `443`, `444`, `7564`.
- `env_server VM` to `Websites VM` mutable rollout ports: `8081-8088`,
  `9081-9088`.

## Source of Truth

- setup service API:
  `environment_setup/webarena/rollout_env/setup_server.py`
- Setup and teardown helpers:
  `packages/core/src/context_scythe/environment/setup_utils.py`
- Remote rollout clients:
  `packages/core/src/context_scythe/environment/gym.py`
- env_server FastAPI app:
  `packages/env_server/src/context_scythe/env_server/app.py`
- env_server schemas:
  `packages/env_server/src/context_scythe/env_server/schemas.py`
- Firewall ports and network tags:
  `vm_utils/create_network_tags.sh`
- Example remote rollout script:
  `scripts/agent_loop_env_server.py`

## Next Reading

- [rollout-lifecycle.md](rollout-lifecycle.md): how setup, browser execution,
  and teardown happen for one rollout.
- [browsergym-webarena.md](browsergym-webarena.md): how task configs become
  BrowserGym environments.
- [../setup/gcp-resources.md](../setup/gcp-resources.md): creating the VMs and
  firewall rules on GCP.
- [../setup/vm-setup.md](../setup/vm-setup.md): installing, starting, and
  verifying the services on those VMs.
