#!/bin/bash

# Exit immediately on command failure, treat unset variables as errors, and
# propagate failures from any command in a pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load the GCP configuration from the repo .env (see .env.example).
# Values already present in the environment take precedence.
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

DEST_DIR="context-scythe"
DISPLAY_DEST_DIR="~/context-scythe"

usage() {
    echo "Usage: $0 <vm-instance-name>"
}

# Ask for confirmation before doing the network copy.
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

# Fail early with a clear message if required local tools are missing.
command -v gcloud >/dev/null || { echo "gcloud is required."; exit 1; }
command -v rsync >/dev/null || { echo "rsync is required."; exit 1; }

# Arguments shared by the gcloud SSH dry-run command. IAP is used by default
# because VMs without an external IP can only be reached through Google's
# Identity-Aware Proxy tunnel. Set GCP_USE_IAP=false when the VM has an external
# IP and IAP is not configured.
GCLOUD_SSH_ARGS=(
    --project="$GCP_PROJECT_ID"
    --zone="$GCP_ZONE"
)
if [ "$GCP_USE_IAP" = "true" ]; then
    GCLOUD_SSH_ARGS+=(--tunnel-through-iap)
fi

echo "Syncing files from $REPO_ROOT"
echo "rsync will copy changed, unignored files using .gitignore filters, excluding .git metadata."
echo "Copying to ${INSTANCE_NAME}:${DISPLAY_DEST_DIR}"
confirm

# Ask gcloud to print the raw ssh command it would run. rsync needs to use the
# same SSH options so it gets the correct key, username, host, project, zone, and
# optional IAP tunnel behavior.
SSH_DRY_RUN="$(gcloud compute ssh "$INSTANCE_NAME" "${GCLOUD_SSH_ARGS[@]}" --dry-run)"

# gcloud's dry-run output is a complete ssh command ending with the resolved
# remote target, for example:
#   ssh ... user@host
# rsync needs the ssh command in -e, and the user@host as the destination host.
RSYNC_TARGET_HOST="${SSH_DRY_RUN##* }"
RSYNC_RSH="${SSH_DRY_RUN% *}"

# gcloud may include -t for TTY allocation. rsync is non-interactive, so remove
# that flag from the SSH command passed through rsync -e.
RSYNC_RSH="$(printf "%s" "$RSYNC_RSH" | sed -E 's/(^| )-t( |$)/ /g; s/  +/ /g; s/^ //; s/ $//')"

(
    cd "$REPO_ROOT"

    # rsync options used here:
    #   -a: archive mode, preserving normal file metadata and copying dirs
    #   -z: compress file data during transfer
    #   --include: preserve placeholder directories that only contain .gitignore
    #   --filter: merge per-directory .gitignore files as rsync excludes
    #   --exclude: never copy Git repository metadata to the VM
    #   --rsync-path: create ~/context-scythe before remote rsync starts
    #   -e: use gcloud's generated SSH command for auth and routing
    #
    # rsync's .gitignore support is exclude-oriented, so Git-style negation
    # keepalive rules such as "!.gitignore" are not enough by themselves. The
    # early include rules keep known placeholder .gitignore files and their
    # directories transferable while the later per-directory .gitignore filters
    # still exclude generated/cache contents.
    rsync -az \
        --include='/environment_setup/webarena/archive/' \
        --include='/environment_setup/webarena/archive/.gitignore' \
        --include='/outputs/.gitignore' \
        --filter=':- .gitignore' \
        --exclude='.git/' \
        --exclude='data/' \
        --exclude='datasets/' \
        --rsync-path="mkdir -p ~/${DEST_DIR} && rsync" \
        -e "$RSYNC_RSH" \
        ./ "${RSYNC_TARGET_HOST}:~/${DEST_DIR}/"
)

echo "Copied repository contents to ${INSTANCE_NAME}:${DISPLAY_DEST_DIR}"
