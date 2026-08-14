# Rollout Environment Mental Model

Use this guide first if you are new to the remote WebArena rollout system. It
explains the problem the system solves and the vocabulary used by the rest of
the docs.

## Problem

WebArena tasks are browser tasks over websites such as shopping, GitLab, and
Reddit. Some of those websites are mutable: an agent can change their backing
database while completing a task.

That mutation matters during RL training. If multiple rollouts share the same
mutable website state, one rollout can affect another rollout's outcome. The
remote rollout environment prevents that by giving each rollout isolated mutable
site containers while sharing static support services.

## Core Idea

The system separates website state from browser execution:

- The `Websites VM` owns WebArena websites and mutable rollout containers.
- The setup service on the `Websites VM` starts, reports, and tears down mutable
  service containers for individual rollout slots.
- The `env_server VM` owns browser sessions through BrowserGym, Playwright, and
  Ray.
- The rollout client coordinates both services from training code.

The env_server is needed because training can require many live browser sessions
at the same time. It uses Ray actors so each rollout session can own an isolated
BrowserGym environment while the service manages concurrency, session state, and
cleanup behind one HTTP API.

```text
training loop
  -> setup service starts mutable WebArena containers
  -> env_server creates a BrowserGym session
  -> browser visits WebArena URLs on the Websites VM
  -> rollout client sends reset() and step(action)
  -> env_server closes the browser session
  -> setup service tears down mutable containers
```

The important boundary is that closing an env_server session closes only the
browser. It does not tear down mutable WebArena containers. Rollout code must do
both cleanup steps.

## Main Pieces

| Piece | Runs on | Responsibility |
| --- | --- | --- |
| `Websites VM` | Google Cloud VM | Hosts WebArena websites, shared static services, rollout-scoped mutable containers, homepage service, and setup service. |
| setup service | `Websites VM` | Starts, checks, and tears down mutable service containers on rollout ports. |
| `env_server VM` | Google Cloud VM | Runs env_server and concurrent browser sessions. |
| env_server service | `env_server VM` | Exposes remote reset/step/close APIs backed by Ray-managed BrowserGym sessions. |
| rollout client | Training process | Allocates rollout slots, calls setup service, drives env_server, and performs teardown. |

## Key Terms

- `rollout`: one agent attempt on one WebArena task.
- `group size G`: the number of parallel rollouts for the same batch item.
- `mutable service`: a WebArena site whose database can change during a rollout,
  such as `shopping`, `shopping_admin`, `gitlab`, or `reddit`.
- `static service`: a shared support site that does not mutate during a rollout,
  such as map, wikipedia, homepage, calculator, or scratchpad.
- `rollout slot`: one reusable port position on a `Websites VM`. Slot 1 uses
  ports ending in `1`, slot 2 uses ports ending in `2`, and so on.
- `homepage_url`: the homepage utility URL passed to BrowserGym.
- `site_urls`: the per-task WebArena URLs passed to BrowserGym. Mutable service
  entries point at rollout-specific ports.

## Request Flow

One rollout follows this sequence:

1. The rollout client chooses a free rollout slot.
2. The rollout client asks the setup service to start each mutable service
   needed by the task.
3. The setup service reports each mutable port as `occupied` when setup is done.
4. The rollout client builds concrete `homepage_url` and `site_urls` values.
5. The rollout client creates a remote env_server session.
6. env_server creates a BrowserGym `BrowserEnv`.
7. BrowserGym logs in to WebArena sites, opens the task start URL, and applies
   agent actions.
8. The rollout client closes the env_server session.
9. The rollout client asks the setup service to tear down mutable services.
10. The rollout slot can be reused after all mutable ports return to `idle`.

## Next Reading

- [architecture.md](architecture.md): VM boundaries, ports, and isolation rules.
- [rollout-lifecycle.md](rollout-lifecycle.md): the operational sequence in
  more detail.
- [../setup/gcp-resources.md](../setup/gcp-resources.md): creating the VMs and
  firewall rules on GCP.
- [../setup/vm-setup.md](../setup/vm-setup.md): installing, starting, and
  verifying the services on those VMs.
