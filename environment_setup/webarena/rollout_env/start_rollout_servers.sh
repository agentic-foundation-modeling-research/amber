#!/bin/bash
# Starts the FastAPI setup service in the background.

PUBLIC_HOSTNAME=$(curl -s ifconfig.me)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR/.."
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

bash "$SCRIPT_DIR/stop_rollout_servers.sh"

cd "$REPO_DIR"

if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
fi

SETUP_SERVER_PID_FILE="$SCRIPT_DIR/setup_server.pid"
SETUP_SERVER_LOG_FILE="$SCRIPT_DIR/setup_server.log"

if ! command -v uv > /dev/null 2>&1; then
    echo "Error: uv not found on PATH. Install uv before starting the setup service."
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
    echo "Error: $PYTHON_BIN not found on PATH."
    exit 1
fi

MAP_PORT=443
WIKIPEDIA_PORT=444
HOMEPAGE_PORT=7564
SETUP_SERVER_PORT=7565

# Start the maps and wikipedia services offline - these do not mutate
echo "Starting OpenStreetMaps"
bash "$BASE_DIR/maps/reset.sh" "$MAP_PORT"
echo "Starting Wikipedia"
bash "$BASE_DIR/wikipedia/reset.sh" "$WIKIPEDIA_PORT"
WIKIPEDIA_URL="http://${PUBLIC_HOSTNAME}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"

echo "Starting setup service"
nohup uv run uvicorn --app-dir "$SCRIPT_DIR" setup_server:app --host 0.0.0.0 --port "$SETUP_SERVER_PORT" > "$SETUP_SERVER_LOG_FILE" 2>&1 &
echo $! > "$SETUP_SERVER_PID_FILE"

# Start the homepage - this is only useful for using the calculator
# as each rollout will have a different URL for mutable services
HOMEPAGE_PID_FILE="$SCRIPT_DIR/homepage.pid"
HOMEPAGE_LOG_FILE="$SCRIPT_DIR/homepage.log"

nohup "$PYTHON_BIN" "$SCRIPT_DIR/homepage_server.py" \
    --host 0.0.0.0 \
    --port "$HOMEPAGE_PORT" \
    --wikipedia-url "$WIKIPEDIA_URL" \
    > "$HOMEPAGE_LOG_FILE" 2>&1 &
echo $! > "$HOMEPAGE_PID_FILE"

echo -n -e "Waiting 3 seconds for all servers to start..."
sleep 3
echo -n -e " done\n"

echo "Hostname: $PUBLIC_HOSTNAME"
echo "Setup service running on http://$PUBLIC_HOSTNAME:$SETUP_SERVER_PORT (pid $(cat "$SETUP_SERVER_PID_FILE")). Logs: $SETUP_SERVER_LOG_FILE"
echo "Homepage running on http://$PUBLIC_HOSTNAME:$HOMEPAGE_PORT (pid $(cat "$HOMEPAGE_PID_FILE")). Logs: $HOMEPAGE_LOG_FILE"
