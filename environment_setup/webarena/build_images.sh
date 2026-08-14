#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_DIR=$SCRIPT_DIR/archive

# Shopping
docker load --input $ARCHIVE_DIR/shopping_final_0712.tar
docker tag shopping_final_0712:latest shopping:latest

# Shoppng Admin
docker load --input $ARCHIVE_DIR/shopping_admin_final_0719.tar
docker tag shopping_admin_final_0719:latest shopping_admin:latest

# Reddit
docker load --input $ARCHIVE_DIR/postmill-populated-exposed-withimg.tar
docker tag postmill-populated-exposed-withimg:latest reddit:latest

# GitLab
docker load --input $ARCHIVE_DIR/gitlab-populated-final-port8023.tar
docker tag gitlab-populated-final-port8023:latest gitlab:latest

# OpenStreetMap
docker load --input $ARCHIVE_DIR/openstreetmap-website-db.tar.gz

docker load --input $ARCHIVE_DIR/openstreetmap-website-web.tar.gz


# Wikipedia
KIWIX_IMAGE="ghcr.io/kiwix/kiwix-serve:3.3.0"
if ! docker image inspect "$KIWIX_IMAGE" >/dev/null 2>&1; then
    echo "Pulling ${KIWIX_IMAGE}..."
    docker pull "$KIWIX_IMAGE"
else
    echo "Kiwix-serve image ${KIWIX_IMAGE} already present."
fi
