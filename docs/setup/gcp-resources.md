# Creating GCP Resources

This guide covers only the Google Cloud side of bringing up the remote WebArena
rollout environment: firewall rules, network tags, the two VMs, repo sync, and
SSH access. It stops once you can SSH into both VMs.

Once you can SSH in, continue with [vm-setup.md](vm-setup.md), which installs
dependencies and starts the services and contains no GCP-specific steps.

For the conceptual reason this system uses two VM roles, see
[../concepts/mental-model.md](../concepts/mental-model.md).

## What You Are Creating

| VM | Purpose | Default machine type | Main port |
| --- | --- | --- | --- |
| `Websites VM` | Hosts WebArena websites, shared services, rollout containers, and the setup service. | `e2-custom-24-32768` | setup service: `7565` |
| `env_server VM` | Hosts the env_server service and Ray-backed concurrent browser sessions. | `n2-custom-44-131072` | env_server service: `8082` |

## Prerequisites

- Local checkout of this repository.
- `gcloud` and `rsync` installed locally, and `gcloud` authenticated.
- Access to the Google Cloud project you are deploying into, and enough quota for
  the two VMs (see [Quota](#quota)).
- Permission to create Compute Engine VMs, network tags, and firewall rules.
- Permission to SSH through IAP, unless you set `GCP_USE_IAP=false`.
- Access to a container registry holding the WebArena images, if you are pulling
  them rather than [building them yourself](building_webarena_images.md).

## Shared Values

Every `vm_utils` script reads its GCP configuration from the repo `.env` file
(see `.env.example`). Values already exported in your environment take
precedence over the file, and `CONTEXT_SCYTHE_ENV_FILE` overrides which file is
read. Set these before running anything in this guide:

```sh
# in .env
GCP_PROJECT_ID=<GCP_PROJECT_ID>   # required; scripts exit if it is unset
GCP_ZONE=us-central1-f
GCP_SUBNET=default
GCP_USE_IAP=true
```

`GCP_SUBNET` is the subnet attached to each VM's network interface. `default` is
correct for a standalone project, which gets an auto-mode `default` VPC with a
subnet in every region; a shared-VPC setup needs its own subnet name here.

`GCP_USE_IAP` controls whether repo sync reaches the VMs through an IAP tunnel.
Leave it `true` when the VMs have no external IP. Set it to `false` if your VMs
have an external IP (the default for a VM created without an org policy
restricting them) and you have not configured IAP.

The machine types and disk sizes are not read from `.env`; they are literals at
the top of `create_website_server_vm.sh` and `create_env_server_vm.sh`. Edit
them there if you need different sizes — see [Quota](#quota) below.

The instance names are yours to pick, and the commands below assume:

```sh
WEBSITES_VM=<WEBSITES_VM_INSTANCE_NAME>
ENV_SERVER_VM=<ENV_SERVER_VM_INSTANCE_NAME>
PROJECT="$GCP_PROJECT_ID"
ZONE="$GCP_ZONE"
```

## Quota

The two default machine types together request **68 vCPUs** (24 + 44) and
**800 GB of SSD** (700 + 100) in a single zone. Both exceed the default
per-region quota on a new Google Cloud project, which is commonly 32 vCPUs and
500 GB of SSD, and lower still on a trial account. Check before you create
anything:

```sh
gcloud compute regions describe "${ZONE%-*}" \
  --project "$PROJECT" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)"
```

Look at the `CPUS`, `SSD_TOTAL_GB`, and `IN_USE_ADDRESSES` rows. If you are over
the limit, either request an increase from **IAM & Admin → Quotas** or lower the
machine types and disk sizes in the two `create_*.sh` scripts. The 700 GB on the
`Websites VM` is sized for the full set of WebArena container images and
archives, so there is not much room to trim it.

## 1. Network Tags and Firewall Rules

Run on your laptop:

```sh
bash vm_utils/create_network_tags.sh
```

The default tags and exposed ports are:

| Tag | Exposed ports | Reason | Attached to |
| --- | --- | --- | --- |
| `webarena-rollout-ssh-iap` | `22` from `35.235.240.0/20` | Allows SSH through Google Cloud IAP. | `Websites VM` and `env_server VM` |
| `webarena-rollout-http` | `443`, `444`, `7564`, `7565`, `8081-8088`, `9081-9088` from `0.0.0.0/0` | Exposes WebArena shared services, setup service, and mutable rollout service ports. | `Websites VM` |
| `webarena-env-server-http` | `8082` from `0.0.0.0/0` | Exposes the env_server service API and health check used by remote rollouts. | `env_server VM` |

Attach these tags when prompted by the VM creation scripts:

- `Websites VM`: `webarena-rollout-ssh-iap`, `webarena-rollout-http`
- `env_server VM`: `webarena-rollout-ssh-iap`, `webarena-env-server-http`

If you choose custom firewall rule names when creating tags, use those custom
names when the VM creation scripts ask which tags to attach.

Both HTTP rules are created with source range `0.0.0.0/0`. To replace that with
an explicit allowlist, see [../firewall.md](../firewall.md).

## 2. Create the `Websites VM`

The `Websites VM` hosts WebArena websites, shared static services, rollout
containers, the homepage service, and the setup service.

Default machine type: `e2-custom-24-32768`.

Run on your laptop:

```sh
bash vm_utils/create_website_server_vm.sh "$WEBSITES_VM"
```

The script creates the VM and installs `rsync`.

## 3. Create the `env_server VM`

The `env_server VM` runs the env_server service and owns browser sessions used
by rollouts. It is separate from the `Websites VM` so browser execution can
scale independently from WebArena container state. env_server uses Ray actors to
run multiple isolated BrowserGym sessions concurrently, so rollout clients can
drive many browser sessions without creating browsers inside the training
process.

Default machine type: `n2-custom-44-131072`.

Run on your laptop:

```sh
bash vm_utils/create_env_server_vm.sh "$ENV_SERVER_VM"
```

The script creates the VM and installs `rsync`.

## 4. Sync the Repo

Run on your laptop:

```sh
bash vm_utils/sync_vm.sh "$WEBSITES_VM"
bash vm_utils/sync_vm.sh "$ENV_SERVER_VM"
```

This rsyncs the repository to `~/context-scythe` on both VMs, over an IAP tunnel
unless `GCP_USE_IAP=false`.

The `env_server VM` also needs the `.env` file itself, which supplies
`OPENAI_BASE_URL` and `OPENAI_API_KEY` to the service:

```sh
bash vm_utils/sync_env_file.sh "$ENV_SERVER_VM"
```

If your `uv` package index needs configuration beyond the defaults — a private
index requiring a token, for example — also copy your `~/.config/uv/uv.toml` to
`~/.config/uv/uv.toml` on the `env_server VM`. Ray actors use `uv` to create
their execution environments and cannot start without the index configuration
they need. Installing from the public PyPI index needs nothing here.

```sh
gcloud compute ssh "$ENV_SERVER_VM" --project "$PROJECT" --zone "$ZONE" \
  --tunnel-through-iap --command="mkdir -p ~/.config/uv"

gcloud compute scp --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap \
  ~/.config/uv/uv.toml "$ENV_SERVER_VM:~/.config/uv/uv.toml"
```

## 5. SSH Into the VMs

```sh
gcloud compute ssh --zone "$ZONE" "$WEBSITES_VM" --project "$PROJECT" --tunnel-through-iap
gcloud compute ssh --zone "$ZONE" "$ENV_SERVER_VM" --project "$PROJECT" --tunnel-through-iap
```

Drop `--tunnel-through-iap` if your VMs have external IPs and you set
`GCP_USE_IAP=false`.

Every command in [vm-setup.md](vm-setup.md) marked "run on the VM" is run inside
one of these sessions.

## Creating More `Websites VM` Instances

Create multiple `Websites VM` instances when rollouts need more isolated batch
items. All `Websites VM` instances can share the same `Websites VM` tags:

- `webarena-rollout-ssh-iap`
- `webarena-rollout-http`

Each new instance repeats steps 2, 4, and 5, then the `Websites VM` half of
[vm-setup.md](vm-setup.md).

## Next Reading

- [vm-setup.md](vm-setup.md): install dependencies, download WebArena data, and
  start the services on the VMs you just created.
- [../firewall.md](../firewall.md): restricting the HTTP rules to an explicit
  allowlist.
- [../concepts/architecture.md](../concepts/architecture.md): port ownership and
  required network reachability.
