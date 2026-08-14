#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load the GCP configuration from the repo .env (see .env.example).
# Values already present in the environment take precedence. Firewall rules are
# a global resource, so no zone is needed here.
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

confirm() {
    read -r -p "Continue? [Y/n] " response
    case "$response" in
        [nN]*) echo "Aborted."; exit 1 ;;
    esac
}

read -r -p "Enter a name for the SSH firewall rule (default: webarena-rollout-ssh-iap): " SSH_FW_NAME
SSH_FW_NAME="${SSH_FW_NAME:-webarena-rollout-ssh-iap}"

read -r -p "Enter a name for the HTTP firewall rule (default: webarena-rollout-http): " HTTP_FW_NAME
HTTP_FW_NAME="${HTTP_FW_NAME:-webarena-rollout-http}"

read -r -p "Enter a name for the HTTP firewall rule for env server (default: webarena-env-server-http): " ENV_SERVER_HTTP_FW_NAME
ENV_SERVER_HTTP_FW_NAME="${ENV_SERVER_HTTP_FW_NAME:-webarena-env-server-http}"

echo "Creating SSH firewall rule '$SSH_FW_NAME' (allow tcp:22 from IAP range 35.235.240.0/20)"
if gcloud compute firewall-rules describe "$SSH_FW_NAME" --project="$GCP_PROJECT_ID" >/dev/null 2>&1; then
    echo "Firewall rule '$SSH_FW_NAME' already exists; skipping."
else
    confirm
    gcloud compute \
        firewall-rules create "$SSH_FW_NAME" \
        --project="$GCP_PROJECT_ID" \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:22 \
        --source-ranges=35.235.240.0/20 \
        --target-tags="$SSH_FW_NAME"
fi

# Fixed ports for non-mutable services
MAP_PORT=443
WIKIPEDIA_PORT=444
HOMEPAGE_PORT=7564
RESET_SERVER_PORT=7565

# Max 16 containers per VM - these ports will be assigned to whatever
# service is active in the rollout
OPEN_PORTS=(8081 8082 8083 8084 8085 8086 8087 8088 9081 9082 9083 9084 9085 9086 9087 9088)
OPEN_PORT_RULES=$(printf ",tcp:%s" "${OPEN_PORTS[@]}")
OPEN_PORT_RULES="${OPEN_PORT_RULES#,}"

echo "Creating HTTP firewall rule '$HTTP_FW_NAME' (allow tcp:${RESET_SERVER_PORT},${WIKIPEDIA_PORT},${MAP_PORT},${HOMEPAGE_PORT},${OPEN_PORTS[*]} from 0.0.0.0/0)"
if gcloud compute firewall-rules describe "$HTTP_FW_NAME" --project="$GCP_PROJECT_ID" >/dev/null 2>&1; then
    echo "Firewall rule '$HTTP_FW_NAME' already exists; skipping."
else
    confirm
    gcloud compute \
        firewall-rules create "$HTTP_FW_NAME" \
        --project="$GCP_PROJECT_ID" \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:${RESET_SERVER_PORT},tcp:${WIKIPEDIA_PORT},tcp:${MAP_PORT},tcp:${HOMEPAGE_PORT},${OPEN_PORT_RULES} \
        --source-ranges=0.0.0.0/0 \
        --target-tags="$HTTP_FW_NAME"
fi


# Assign a single port to the env server
ENV_SERVER_PORT=8082

echo "Creating HTTP firewall rule '$ENV_SERVER_HTTP_FW_NAME' (allow tcp:${ENV_SERVER_PORT} from 0.0.0.0/0)"
if gcloud compute firewall-rules describe "$ENV_SERVER_HTTP_FW_NAME" --project="$GCP_PROJECT_ID" >/dev/null 2>&1; then
    echo "Firewall rule '$ENV_SERVER_HTTP_FW_NAME' already exists; skipping."
else
    confirm
    gcloud compute \
        firewall-rules create "$ENV_SERVER_HTTP_FW_NAME" \
        --project="$GCP_PROJECT_ID" \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:${ENV_SERVER_PORT} \
        --source-ranges=0.0.0.0/0 \
        --target-tags="$ENV_SERVER_HTTP_FW_NAME"
fi
