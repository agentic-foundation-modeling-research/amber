#!/bin/bash

PUBLIC_HOSTNAME=$(curl -s ifconfig.me)

MAP_PORT="${1:-443}"

MAP_URL="http://localhost:${MAP_PORT}" # No need for public http since hosted on this VM

# Stop and remove existing containers
bash "$(dirname "$0")/stop_and_remove.sh"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/openstreetmap-website"

if [ ! -d "$COMPOSE_DIR" ]; then
    echo "Error: $COMPOSE_DIR not found. Run setup.sh first."
    exit 1
fi

# Start the containers
export MAP_PORT
(cd "$COMPOSE_DIR" && docker compose up -d)

echo -n -e "Waiting 30 seconds for all services to start..."
sleep 30
echo -n -e " done\n"

# Run db migrations against the seeded db (idempotent safety check)
docker exec openstreetmap-website-web-1 bin/rails db:migrate RAILS_ENV=development

# Warm up the map environment at startup so rollouts are faster
if [ "${MAP_BROWSER_WARMUP:-1}" != "0" ]; then
    if [ -n "${MAP_WARMUP_PYTHON_BIN:-}" ]; then
        MAP_WARMUP_CMD=("$MAP_WARMUP_PYTHON_BIN")
    else
        MAP_WARMUP_CMD=(uv run python)
    fi
    "${MAP_WARMUP_CMD[@]}" "$SCRIPT_DIR/warmup.py" "$MAP_URL" \
        --timeout-ms "${MAP_WARMUP_TIMEOUT_MS:-90000}" \
        --retries "${MAP_WARMUP_RETRIES:-3}" \
        --retry-delay-s "${MAP_WARMUP_RETRY_DELAY_S:-5}"
fi
