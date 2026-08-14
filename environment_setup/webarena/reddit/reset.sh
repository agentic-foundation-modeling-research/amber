#!/bin/bash

REDDIT_PORT="${1:-8080}"
CONTAINER_NAME="${2:-forum}"

# Stop and remove existing containers
bash "$(dirname "$0")/stop_and_remove.sh" "$CONTAINER_NAME"

set -e

# Create the container
docker create --name "$CONTAINER_NAME" -p $REDDIT_PORT:80 reddit:latest

# Start the container
docker start "$CONTAINER_NAME"
echo -n -e "Waiting 15 seconds for all services to start..."
sleep 60
echo -n -e " done\n"

# Patch the container with php-fpm tuning to make the server more responsive
docker exec "$CONTAINER_NAME" sed -i \
  -e 's/^pm.max_children = .*/pm.max_children = 32/' \
  -e 's/^pm.start_servers = .*/pm.start_servers = 10/' \
  -e 's/^pm.min_spare_servers = .*/pm.min_spare_servers = 5/' \
  -e 's/^pm.max_spare_servers = .*/pm.max_spare_servers = 20/' \
  -e 's/^;pm.max_requests = .*/pm.max_requests = 500/' \
  /usr/local/etc/php-fpm.d/www.conf
docker exec "$CONTAINER_NAME" supervisorctl restart php-fpm
