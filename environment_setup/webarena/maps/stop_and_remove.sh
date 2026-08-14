#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/openstreetmap-website"

if [ -d "$COMPOSE_DIR" ]; then
    (cd "$COMPOSE_DIR" && docker compose down -v)
fi
