# VM Setup

This guide installs dependencies, downloads the WebArena data, and starts the
services on two VMs. It is provider-agnostic: nothing here is specific to Google
Cloud. If you are provisioning on GCP, create the VMs first with
[gcp-resources.md](gcp-resources.md), then come back here.

For the mental model, see
[../concepts/mental-model.md](../concepts/mental-model.md). For day-to-day
service operations, see [services.md](services.md).

## The Two VMs

| VM | Purpose | Suggested size | Main port |
| --- | --- | --- | --- |
| `Websites VM` | Hosts WebArena websites, shared services, rollout containers, the homepage service, and the setup service. | 24 vCPU / 32 GB RAM | setup service: `7565` |
| `env_server VM` | Hosts the env_server service and Ray-backed concurrent browser sessions. | 44 vCPU / 128 GB RAM | env_server service: `8082` |

Give the `Websites VM` enough disk for the WebArena container images and
archives (hundreds of GB).

## Prerequisites

On both VMs:

- Ubuntu with `apt`, `sudo`, and outbound internet access.
- SSH access for your user.
- This repository present at `~/amber`. Copy it with `rsync`/`scp` from
  your laptop, or clone it on the VM. On GCP, use
  `bash vm_utils/sync_vm.sh <instance>`.

Inbound ports that must be reachable from wherever the rollout client runs:

- `Websites VM`: `443`, `444`, `7564`, `7565`, `8081-8088`, `9081-9088`
- `env_server VM`: `8082`

On the `Websites VM` only:

- Credentials for the container registry holding the prebuilt WebArena images,
  and read access to the bucket holding the OpenStreetMap and Wikipedia
  archives. Set `WEBARENA_IMAGE_REGISTRY`, `WEBARENA_IMAGE_TAG`, and
  `WEBARENA_ASSETS_BUCKET` in the repo `.env` (see `.env.example`). If you have
  no registry or bucket, skip this and follow
  [building_webarena_images.md](building_webarena_images.md) at step 2 instead.

On the `env_server VM` only:

- A `.env` file at `~/amber/.env` providing `OPENAI_BASE_URL` and
  `OPENAI_API_KEY`.
- `~/.config/uv/uv.toml`, if your `uv` index requires a token. Ray actors use
  `uv` to build their execution environments and will fail to start without the
  index configuration they need.

> One step here assumes Google Cloud tooling: asset downloads use
> `gcloud storage cp` in `environment_setup/webarena/setup.sh`. On a non-GCP VM,
> substitute your own download for that step, or fetch the archives by hand as
> described in [building_webarena_images.md](building_webarena_images.md).
> Registry authentication is a manual step in
> [step 1](#1-install-websites-vm-dependencies) and is not GCP-specific.

## 1. Install `Websites VM` Dependencies

Run on the `Websites VM`:

```sh
cd ~/amber
bash vm_utils/install_website_server_requirements.sh
```

This installs Docker and adds your user to the `docker` group.

The `docker` group membership only takes effect in a new login shell. If you do
not want to reconnect, run:

```sh
newgrp docker
```

`newgrp` opens a subshell, so run the remaining commands inside it.

Then authenticate Docker against the registry holding the prebuilt WebArena
images, so that step 2 can pull them. The script does not do this for you,
because the command depends on where your images live. For Google Artifact
Registry, pass the registry host — the part of `WEBARENA_IMAGE_REGISTRY` before
the first `/`:

```sh
# e.g. WEBARENA_IMAGE_REGISTRY=us-central1-docker.pkg.dev/my-project/webarena-prebuilt-images
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
```

For any other registry, use its own login instead, such as
`docker login <registry-host>`. Skip this entirely if you are
[building the images locally](building_webarena_images.md).

Then create the Python environment:

```sh
bash vm_utils/setup_env_server.sh
source "$HOME/.local/bin/env"
```

`setup_env_server.sh` installs `uv`, syncs the Python environment, and installs
Playwright with Chromium. Despite the name, both VMs need it: the `Websites VM`
runs it here and the `env_server VM` runs the same script in
[step 3](#3-install-env_server-dependencies).

## 2. Download WebArena Images and Data

Run on the `Websites VM`:

```sh
cd ~/amber
screen -S setup
bash environment_setup/webarena/setup.sh
```

This pulls the WebArena site images and downloads the OpenStreetMap website
archive and the Wikipedia `.zim` file. It can take a long time, so use `screen`
to keep it running if your SSH session disconnects.

`setup.sh` assumes a container registry to pull images from and a GCS bucket to
download archives from. If you have neither, follow
[building_webarena_images.md](building_webarena_images.md) instead of the command
above — it covers building the site images locally and preparing the maps and
Wikipedia environments from local archives. Then continue with
[step 3](#3-install-env_server-dependencies).

## 3. Install env_server Dependencies

Run on the `env_server VM`:

```sh
cd ~/amber
bash vm_utils/setup_env_server.sh
source "$HOME/.local/bin/env"
```

This installs `uv`, syncs the Python environment, and installs Playwright with
Chromium. It is the same script the `Websites VM` runs in
[step 1](#1-install-websites-vm-dependencies). The `source` line is only needed
if you do not want to reconnect through a new shell to pick up `uv` on your
`PATH`.

## 4. Start the Setup Service

Run on the `Websites VM`:

```sh
cd ~/amber
bash environment_setup/webarena/rollout_env/start_rollout_servers.sh
```

This starts the setup service on port `7565` and the homepage service on port
`7564`.

## 5. Start the env_server Service

Confirm `~/amber/.env` and, if your index needs it,
`~/.config/uv/uv.toml` are in place on the `env_server VM` (see
[Prerequisites](#prerequisites)), then run on the `env_server VM`:

```sh
cd ~/amber
bash environment_setup/environment_server/manage-env-server.sh start
```

This starts env_server on port `8082`.

## 6. Verify the Environment

Run from your laptop:

```sh
curl http://<WEBSITES_VM_IP_ADDRESS>:7565/health
```

Expected response:

```json
{"ok":true}
```

```sh
curl http://<ENV_SERVER_VM_IP_ADDRESS>:8082/health
```

Expected response:

```json
{"status":"ok","live_sessions":0,"max_live_sessions":64}
```

The setup is ready when:

- The setup service health check returns `{"ok":true}`.
- The env_server health check returns `{"status":"ok", ...}`.
- Both VMs have the repository at `~/amber`.

## Adding More `Websites VM` Instances

Rollouts that need more isolated batch items use more `Websites VM` instances.
Each new instance repeats steps 1, 2, and 4, and must expose the same
`Websites VM` ports.

## Next Reading

- [services.md](services.md): start, stop, health checks, and logs.
- [../concepts/architecture.md](../concepts/architecture.md): how the rollout
  environment fits together, including port ownership.
- [../concepts/rollout-lifecycle.md](../concepts/rollout-lifecycle.md): how
  setup, browser sessions, and teardown work per rollout.
- [../reference/api-reference.md](../reference/api-reference.md): raw setup
  service and env_server APIs.
- [../reference/troubleshooting.md](../reference/troubleshooting.md): common
  failures and fixes.
