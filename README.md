# Amber

**Append-only Memory for Browser Evidence Retention**

Amber trains long-context web agents to retain browser evidence in an explicit, append-only memory across steps.

## Table of Contents
- [Amber](#amber)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
  - [Package Layout](#package-layout)
  - [Local Setup](#local-setup)
  - [Infrastructure Setup Quickstart](#infrastructure-setup-quickstart)
  - [Contributing](#contributing)
  - [License](#license)

## Prerequisites

**Clone the repo without the datasets** — skips the SFT datasets.

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/agentic-foundation-modeling-research/amber.git
```

**Clone the repo with the datasets** — required to run SFT. Install and initialize git-lfs *before* cloning. The datasets are ~586MB.

```sh
brew install git-lfs   # macOS; on Debian/Ubuntu: apt-get install git-lfs
git lfs install        # registers the clean/smudge filters; only needed once per machine
git clone https://github.com/agentic-foundation-modeling-research/amber.git
```

If you already cloned without git-lfs, there's no need to re-clone:

```sh
git lfs install
git lfs pull
```

Note that a clone without git-lfs leaves the datasets as three-line pointer text files rather than
Arrow data, so `datasets.load_from_disk("datasets/webarena_append_memory_sft")` fails with an Arrow
parse error that gives no hint that LFS is the cause.

## Getting Started
- Check the [Prerequisites](#prerequisites) before cloning
- Complete the [Local Setup](#local-setup)
- Complete the [Infrastructure Setup](#infrastructure-setup-quickstart)
- Go through the [docs](docs/README.md)

## Package Layout
This repository is split into three Python distributions:

- `amber`: a top-level meta package with no Python modules. Installing it installs both runtime packages.
- `amber-core`: the core package under `packages/core`, containing agents, trajectory utilities, WebArena environment helpers, training helpers, and templates.
- `amber-env-server`: the rollout environment server under `packages/env_server`, containing the FastAPI/Ray server and CLI entrypoint.

Both runtime packages retain the shared `context_scythe` import namespace for compatibility:

```text
context_scythe.agents        # amber-core (includes trajectory_data and prompt_builders)
context_scythe.datagen       # amber-core
context_scythe.environment   # amber-core
context_scythe.env_server    # amber-env-server
```

## Local Setup

We use `uv` to manage dependencies and the local Python workspace.

```sh
uv venv # Create the venv
source .venv/bin/activate

# Install both workspace packages without dev dependencies.
uv sync --all-packages
# Install chromium for playwright
uv run playwright install chromium
```

For development across all packages:

```sh
uv sync --all-packages [--extra dev]
uv run playwright install chromium
```

For core-only development:

```sh
uv sync --package amber-core [--extra dev]
uv run playwright install chromium
```

For env-server development:

```sh
uv sync --package amber-env-server [--extra dev]
uv run playwright install chromium
```

## Infrastructure Setup Quickstart
Instructions for remote rollout VM creation and service startup are split in two:

- [docs/setup/gcp-resources.md](docs/setup/gcp-resources.md) — creating the
  firewall rules, network tags, and VMs on Google Cloud, up to SSH access.
- [docs/setup/vm-setup.md](docs/setup/vm-setup.md) — installing dependencies,
  downloading the WebArena data, and starting the services on those VMs. No
  GCP-specific steps, so it applies to any provider.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull request guidelines. Participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Amber is licensed under the [MIT License](LICENSE).
