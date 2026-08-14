#!/bin/bash

set -e

INSTANCE_TYPE="n2-custom-44-131072"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load the GCP configuration from the repo .env (see .env.example).
# Values already present in the environment take precedence.
ENV_FILE="${CONTEXT_SCYTHE_ENV_FILE:-$REPO_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [ -z "${GCP_PROJECT_ID:-}" ]; then
  echo "GCP_PROJECT_ID is not set. Set it in $ENV_FILE (see .env.example) or export it." >&2
  exit 1
fi
: "${GCP_ZONE:=us-central1-f}"
: "${GCP_SUBNET:=default}"

usage() {
    echo "Usage: $0 <vm-instance-name>"
}

confirm() {
    read -r -p "Continue? [Y/n] " response
    case "$response" in
        [nN]*) echo "Aborted."; exit 1 ;;
    esac
}

if [ "$#" -ne 1 ]; then
    usage
    echo "A VM instance name argument is required."
    exit 1
fi

INSTANCE_NAME="$1"

read -r -p "Enter a name for the SSH firewall rule (default: webarena-rollout-ssh-iap): " SSH_FW_NAME
SSH_FW_NAME="${SSH_FW_NAME:-webarena-rollout-ssh-iap}"

read -r -p "Enter a name for the HTTP firewall rule for env server (default: webarena-env-server-http): " ENV_SERVER_HTTP_FW_NAME
ENV_SERVER_HTTP_FW_NAME="${ENV_SERVER_HTTP_FW_NAME:-webarena-env-server-http}"

echo "Creating instance '$INSTANCE_NAME' ($INSTANCE_TYPE, 100GB SSD, ubuntu-2204-lts, tags: $SSH_FW_NAME,$ENV_SERVER_HTTP_FW_NAME)"
echo "Startup script will install rsync on first boot."
confirm

STARTUP_SCRIPT="$(cat <<'EOF'
#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y rsync
EOF
)"

gcloud compute instances create "$INSTANCE_NAME" \
    --project="$GCP_PROJECT_ID" \
    --zone="$GCP_ZONE" \
    --machine-type=$INSTANCE_TYPE \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet="$GCP_SUBNET" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags="$SSH_FW_NAME","$ENV_SERVER_HTTP_FW_NAME" \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata=startup-script="$STARTUP_SCRIPT"
