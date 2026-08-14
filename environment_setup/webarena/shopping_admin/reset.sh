#!/bin/bash

PUBLIC_HOSTNAME=$(curl -s ifconfig.me)

SHOPPING_ADMIN_PORT="${1:-8083}"
CONTAINER_NAME="${2:-shopping_admin}"

# Stop and remove existing containers
SHOPPING_ADMIN_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"

bash "$(dirname "$0")/stop_and_remove.sh" "$CONTAINER_NAME"

set -e

# Create the container
docker create --name "$CONTAINER_NAME" -p $SHOPPING_ADMIN_PORT:80 shopping_admin:latest

# Start the container
docker start "$CONTAINER_NAME"
echo -n -e "Waiting 60 seconds for all services to start..."
sleep 60
echo -n -e " done\n"

# Patch the container with the required config
docker exec "$CONTAINER_NAME" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$PUBLIC_HOSTNAME:$SHOPPING_ADMIN_PORT" # no trailing /
docker exec "$CONTAINER_NAME" mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='http://$PUBLIC_HOSTNAME:$SHOPPING_ADMIN_PORT/' WHERE path = 'web/secure/base_url';"
# remove the requirement to reset password
docker exec "$CONTAINER_NAME" php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec "$CONTAINER_NAME" php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec "$CONTAINER_NAME" /var/www/magento2/bin/magento cache:flush
