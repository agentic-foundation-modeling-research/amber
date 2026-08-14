#!/bin/bash
# Stops the setup service and homepage started by start_rollout_servers.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR/.."

# Stop the maps and wikipedia containers
bash $BASE_DIR/maps/stop_and_remove.sh
bash $BASE_DIR/wikipedia/stop_and_remove.sh

SETUP_PID_FILE="$SCRIPT_DIR/setup_server.pid"
HOMEPAGE_PID_FILE="$SCRIPT_DIR/homepage.pid"

stop_server() {
    SERVER_NAME="$1"
    PID_FILE="$2"

    if [ -f "$PID_FILE" ]; then
        PID="$(cat "$PID_FILE")"
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping $SERVER_NAME (pid $PID)..."
            kill "$PID" 2>/dev/null || true
            sleep 1
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

stop_server "setup service" "$SETUP_PID_FILE"
stop_server "homepage" "$HOMEPAGE_PID_FILE"
