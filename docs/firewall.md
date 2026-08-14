# Restricting Public Access With Network Tags

`vm_utils/create_network_tags.sh` creates both HTTP firewall rules with
`--source-ranges=0.0.0.0/0`, which leaves the WebArena sites and the env_server
API open to the internet. This guide replaces those open ranges with an explicit
allowlist.

The SSH rule (`webarena-rollout-ssh-iap`) is already restricted to Google Cloud's
IAP range `35.235.240.0/20` and needs no change.

For the tags, ports, and which VM carries which tag, see
[setup/gcp-resources.md](setup/gcp-resources.md#1-network-tags-and-firewall-rules).

The commands below use `$PROJECT` and `$ZONE`. Set them from the same values as
your repo `.env` (see `.env.example`):

```sh
PROJECT=<GCP_PROJECT_ID>
ZONE=<GCP_ZONE>
```

## Who Needs To Reach What

Traffic is one-directional: the browser runs on the `env_server VM` and loads
WebArena sites from the `Websites VM`, never the reverse. That asymmetry is why
the two rules get different allowlists.

| Rule | Protects | Ports | Must allow |
| --- | --- | --- | --- |
| `webarena-rollout-http` | `Websites VM` | `443`, `444`, `7564`, `7565`, `8081-8088`, `9081-9088` | your laptop, the GPU training cluster, and the `env_server VM` public IP |
| `webarena-env-server-http` | `env_server VM` | `8082` | your laptop and the GPU training cluster |

Why each entry is needed:

- **Laptop**: runs evaluations, health checks, and setup/teardown calls against
  both services.
- **GPU training cluster**: runs the RL training loop, which drives the setup
  service on the `Websites VM` and the env_server API.
- **`env_server VM`**: its browser loads every WebArena URL from the
  `Websites VM`. Omitting it is the most common mistake — services come up
  healthy and every rollout still fails to load a page.

The `Websites VM` does not need access to the `env_server VM`, so it is absent
from the env_server allowlist.

## Collect The Source Ranges

Your laptop's public IP:

```sh
curl -4 https://api.ipify.org
```

Use it as a `/32`, for example `203.0.113.7/32`.

The egress range of your GPU training cluster. Ask your cluster provider for it,
or read it from a node:

```sh
curl -4 https://api.ipify.org
```

Use the CIDR block that covers every node the training job can land on (a `/24`
is typical), not just the one node you happened to query.

The `env_server VM` public IP:

```sh
gcloud compute instances describe "$ENV_SERVER_VM" \
  --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Use it as a `/32`.

## Apply The Allowlists

`firewall-rules update` **replaces** the source range list rather than appending
to it, so pass every entry the rule needs in one command.

Restrict the `Websites VM` rule:

```sh
LAPTOP_IP=<LAPTOP_PUBLIC_IP>/32
ENV_SERVER_IP=<ENV_SERVER_VM_PUBLIC_IP>/32
CLUSTER_RANGE=<TRAINING_CLUSTER_CIDR>

gcloud compute firewall-rules update webarena-rollout-http \
  --project "$PROJECT" \
  --source-ranges="$LAPTOP_IP,$CLUSTER_RANGE,$ENV_SERVER_IP"
```

Restrict the `env_server VM` rule:

```sh
gcloud compute firewall-rules update webarena-env-server-http \
  --project "$PROJECT" \
  --source-ranges="$LAPTOP_IP,$CLUSTER_RANGE"
```

If you chose custom rule names when running
`vm_utils/create_network_tags.sh`, substitute those names.

## Verify

Check what each rule currently allows:

```sh
gcloud compute firewall-rules describe webarena-rollout-http \
  --project "$PROJECT" \
  --format='get(sourceRanges,allowed)'
```

Then re-run the health checks from your laptop:

```sh
curl http://<WEBSITES_VM_IP_ADDRESS>:7565/health
curl http://<ENV_SERVER_VM_IP_ADDRESS>:8082/health
```

And confirm the browser's own path from the `env_server VM`, which the laptop
checks do not exercise:

```sh
curl http://<WEBSITES_VM_HOST>:7564
curl http://<WEBSITES_VM_HOST>:8081
```

## Keeping The Allowlist Current

These entries drift, and the symptom is always a connection that times out
rather than an explicit permission error:

- Laptop IPs change with networks, VPNs, and DHCP leases. Re-run the `ipify`
  check and update the rule when evaluation stops connecting.
- Recreating a VM, or stopping and starting one without a reserved static
  address, changes its public IP. Re-read the `env_server VM` IP and update
  `webarena-rollout-http`.
- Each additional `Websites VM` shares the same tag, so no rule change is needed
  when you scale out. Each additional `env_server VM` needs its public IP added
  to `webarena-rollout-http`.

## Related Docs

- [setup/gcp-resources.md](setup/gcp-resources.md): tags, ports, and VM creation.
- [setup/vm-setup.md](setup/vm-setup.md): the inbound ports each VM role needs
  reachable.
- [concepts/architecture.md](concepts/architecture.md#ports-and-url-rules): the
  required network reachability list these rules implement.
- [reference/troubleshooting.md](reference/troubleshooting.md): what a blocked
  path looks like at rollout time.
