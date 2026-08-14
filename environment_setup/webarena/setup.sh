#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARCHIVE_DIR=$SCRIPT_DIR/archive

# Load the registry / bucket configuration from the repo .env (see .env.example).
# Values already present in the environment take precedence.
ENV_FILE="${WEBARENA_SETUP_ENV_FILE:-$REPO_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

: "${WEBARENA_IMAGE_TAG:=latest}"

for var in WEBARENA_IMAGE_REGISTRY WEBARENA_ASSETS_BUCKET; do
  if [ -z "${!var}" ]; then
    echo "$var is not set. Set it in $ENV_FILE (see .env.example) or export it." >&2
    exit 1
  fi
done

# Pull the images and tag them with the local names the compose files expect
pull_and_tag() {
  IMAGE_NAME="$1"
  REMOTE_IMAGE="$WEBARENA_IMAGE_REGISTRY/$IMAGE_NAME:$WEBARENA_IMAGE_TAG"
  docker pull "$REMOTE_IMAGE"
  docker tag "$REMOTE_IMAGE" "$IMAGE_NAME:latest"
}

pull_and_tag shopping
pull_and_tag shopping_admin
pull_and_tag gitlab
pull_and_tag reddit
# OpenStreetMaps
pull_and_tag openstreetmap-website-db
pull_and_tag openstreetmap-website-web

# Wikipedia is hosted using kiwix-serve from a .zim file
docker pull ghcr.io/kiwix/kiwix-serve:3.3.0

# Download the artifacts
# The OpenStreetMaps website is provided by https://github.com/gasse/webarena-setup/tree/main/webarena
OPENSTREETMAP_WEBSITE_ARCHIVE="$ARCHIVE_DIR/openstreetmap-website.tar.gz"
if [ ! -f "$OPENSTREETMAP_WEBSITE_ARCHIVE" ]; then
  gcloud storage cp "$WEBARENA_ASSETS_BUCKET/openstreetmap-website.tar.gz" $ARCHIVE_DIR/
else
  echo "$OPENSTREETMAP_WEBSITE_ARCHIVE already exists; skipping download."
fi

# Setup OpenStreetMaps. This will untar the website archive, and patch the required files
bash "$SCRIPT_DIR/maps/setup.sh"

# Wikipedia
WIKIPEDIA_ARCHIVE="$ARCHIVE_DIR/wikipedia_en_all_maxi_2022-05.zim"
if [ ! -f "$WIKIPEDIA_ARCHIVE" ]; then
  gcloud storage cp "$WEBARENA_ASSETS_BUCKET/wikipedia_en_all_maxi_2022-05.zim" $ARCHIVE_DIR
else
  echo "$WIKIPEDIA_ARCHIVE already exists; skipping download."
fi

bash "$SCRIPT_DIR/wikipedia/setup.sh"
