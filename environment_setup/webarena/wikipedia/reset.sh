#!/bin/bash

WIKIPEDIA_PORT="${1:-8081}"

# Stop and remove existing containers
bash "$(dirname "$0")/stop_and_remove.sh"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
WIKIPEDIA_ARCHIVE="wikipedia_en_all_maxi_2022-05.zim"

if [ ! -e "$DATA_DIR/$WIKIPEDIA_ARCHIVE" ]; then
    echo "Error: $DATA_DIR/$WIKIPEDIA_ARCHIVE not found. Run setup.sh first."
    exit 1
fi

# Create the container
docker create --name wikipedia --volume="${DATA_DIR}":/data -p $WIKIPEDIA_PORT:80 ghcr.io/kiwix/kiwix-serve:3.3.0 $WIKIPEDIA_ARCHIVE

# Start the container
docker start wikipedia
echo -n -e "Waiting 10 seconds for all services to start..."
sleep 10
echo -n -e " done\n"
