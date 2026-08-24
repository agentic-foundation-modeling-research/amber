#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PUBLIC_HOSTNAME=$(curl -s ifconfig.me)

PID_FILE="$SCRIPT_DIR/env_server.pid"
LOG_FILE="$SCRIPT_DIR/env_server.log"

PORT="${ROLLOUT_SERVER_PORT:-8082}"
ENV_FILE="${ENV_SERVER_ENV_FILE:-$REPO_DIR/.env}"

usage() {
    echo "Usage: $0 {start|stop}"
}

is_running() {
    PID="$1"
    kill -0 "$PID" 2>/dev/null
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "env_server is not running."
        return
    fi

    PID="$(cat "$PID_FILE")"
    if is_running "$PID"; then
        echo "Stopping env_server (pid $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 1
        if is_running "$PID"; then
            kill -9 "$PID" 2>/dev/null || true
        fi
    else
        echo "env_server pid file was stale (pid $PID)."
    fi

    rm -f "$PID_FILE"
}

start_server() {
    stop_server

    cd "$REPO_DIR"
    if [ -f "$ENV_FILE" ]; then
        echo "Loading env_server environment from $ENV_FILE"
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi

    missing_env_vars=()
    if [ -z "${OPENAI_BASE_URL:-}" ]; then
        missing_env_vars+=("OPENAI_BASE_URL")
    fi
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        missing_env_vars+=("OPENAI_API_KEY")
    fi
    if [ "${#missing_env_vars[@]}" -gt 0 ]; then
        echo "Error: missing required env_server environment variables: ${missing_env_vars[*]}"
        echo "Set them in the shell or in $ENV_FILE. Override the env file with ENV_SERVER_ENV_FILE=/path/to/file."
        exit 1
    fi

    if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$REPO_DIR/.venv/bin/activate"
    fi

    if ! command -v uv > /dev/null 2>&1; then
        echo "Error: uv not found on PATH. Install uv and run vm_utils/setup-env-server.sh first."
        exit 1
    fi

    echo "Starting env_server on port $PORT"
    nohup uv run --no-sync amber-env-server --port "$PORT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    sleep 2
    PID="$(cat "$PID_FILE")"
    if ! is_running "$PID"; then
        echo "Error: env_server failed to start. Logs: $LOG_FILE"
        tail -n 20 "$LOG_FILE" || true
        rm -f "$PID_FILE"
        exit 1
    fi

    echo "env_server running on port http://$PUBLIC_HOSTNAME:$PORT (pid $PID). Logs: $LOG_FILE"
}

case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    *)
        usage
        exit 1
        ;;
esac
