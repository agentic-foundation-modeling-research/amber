#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_FILE="$REPO_ROOT/.env"

# Load the GCP configuration from the repo .env (see .env.example). This is the
# same file that gets copied to the VM below. Values already present in the
# environment take precedence.
ENV_FILE="${CONTEXT_SCYTHE_ENV_FILE:-$REPO_ROOT/.env}"
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
: "${GCP_USE_IAP:=true}"

DEST_DIR="~/context-scythe"

usage() {
    echo "Usage: $0 <vm-instance-name>"
}

if [ "$#" -ne 1 ]; then
    usage
    echo "A VM instance name argument is required."
    exit 1
fi

INSTANCE_NAME="$1"

command -v gcloud >/dev/null || { echo "gcloud is required."; exit 1; }

if [ ! -f "$SOURCE_FILE" ]; then
    echo "Repo .env file not found: $SOURCE_FILE"
    exit 1
fi

# Set GCP_USE_IAP=false when the VM has an external IP and IAP is not configured.
IAP_ARGS=()
if [ "$GCP_USE_IAP" = "true" ]; then
    IAP_ARGS=(--tunnel-through-iap)
fi

GCLOUD_INSTANCE_ARGS=(
    "$INSTANCE_NAME"
    --project="$GCP_PROJECT_ID"
    --zone="$GCP_ZONE"
    ${IAP_ARGS[@]+"${IAP_ARGS[@]}"}
)

echo "Creating ${INSTANCE_NAME}:${DEST_DIR}"
gcloud compute ssh "${GCLOUD_INSTANCE_ARGS[@]}" --command="mkdir -p ~/context-scythe"

echo "Copying $SOURCE_FILE to ${INSTANCE_NAME}:${DEST_DIR}/.env"
gcloud compute scp \
    --project="$GCP_PROJECT_ID" \
    --zone="$GCP_ZONE" \
    ${IAP_ARGS[@]+"${IAP_ARGS[@]}"} \
    "$SOURCE_FILE" \
    "${INSTANCE_NAME}:${DEST_DIR}/.env"

echo "Copied .env to ${INSTANCE_NAME}:${DEST_DIR}/.env"
