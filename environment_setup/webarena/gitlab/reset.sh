#!/bin/bash

PUBLIC_HOSTNAME=$(curl -s ifconfig.me)

HOST_GITLAB_PORT="${1:-9001}"
CONTAINER_NAME="${2:-gitlab}"
CONTAINER_GITLAB_PORT=9001

GITLAB_URL="http://${PUBLIC_HOSTNAME}:${HOST_GITLAB_PORT}"

# Stop and remove existing containers
bash "$(dirname "$0")/stop_and_remove.sh" "$CONTAINER_NAME"

set -e

# Create the container
docker create --name "$CONTAINER_NAME" -p "${HOST_GITLAB_PORT}:${CONTAINER_GITLAB_PORT}" gitlab:latest /opt/gitlab/embedded/bin/runsvdir-start

# Start the container
docker start "$CONTAINER_NAME"
echo -n -e "Waiting 60 seconds for all services to start..."
sleep 60
echo -n -e " done\n"

# Patch the container with the required config
docker exec "$CONTAINER_NAME" sed -i "s|^external_url.*|external_url '$GITLAB_URL'|" /etc/gitlab/gitlab.rb
docker exec "$CONTAINER_NAME" bash -c "printf '\nnginx[\"listen_port\"] = ${CONTAINER_GITLAB_PORT}' >> /etc/gitlab/gitlab.rb"
docker exec "$CONTAINER_NAME" bash -c "printf '\n\npuma[\"worker_processes\"] = 4' >> /etc/gitlab/gitlab.rb"  # bugfix https://github.com/ServiceNow/BrowserGym/issues/285
docker exec "$CONTAINER_NAME" gitlab-ctl reconfigure
