#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

cd $REPO_DIR
uv sync --package amber-env-server
source .venv/bin/activate
uv run playwright install chromium --with-deps
uv run python -m nltk.downloader punkt_tab
