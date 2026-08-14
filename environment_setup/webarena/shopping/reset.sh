#!/bin/bash

PUBLIC_HOSTNAME=$(curl -s ifconfig.me)

SHOPPING_PORT="${1:-8082}"
CONTAINER_NAME="${2:-shopping}"

SHOPPING_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"

# Stop and remove existing containers
bash "$(dirname "$0")/stop_and_remove.sh" "$CONTAINER_NAME"

set -e

# Create the container
docker create --name "$CONTAINER_NAME" -p $SHOPPING_PORT:80 shopping:latest

# Start the container
docker start "$CONTAINER_NAME"
echo -n -e "Waiting 60 seconds for all services to start..."
sleep 60
echo -n -e " done\n"

# Patch the container with the required config
docker exec "$CONTAINER_NAME" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$PUBLIC_HOSTNAME:$SHOPPING_PORT" # no trailing /
docker exec "$CONTAINER_NAME" mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='http://$PUBLIC_HOSTNAME:$SHOPPING_PORT/' WHERE path = 'web/secure/base_url';"
docker exec "$CONTAINER_NAME" /var/www/magento2/bin/magento cache:flush
