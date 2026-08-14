#!/bin/bash
# Sets up the wikipedia environment: pulls the kiwix-serve image and stages
# the wikipedia .zim archive into wiki/data/.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/archive"
DATA_DIR="$SCRIPT_DIR/data"
WIKIPEDIA_ARCHIVE="wikipedia_en_all_maxi_2022-05.zim"

mkdir -p "$DATA_DIR"

# Stage the wikipedia archive in data/
if [ -f "$DATA_DIR/$WIKIPEDIA_ARCHIVE" ] || [ -L "$DATA_DIR/$WIKIPEDIA_ARCHIVE" ]; then
    echo "Wikipedia archive already present at $DATA_DIR/$WIKIPEDIA_ARCHIVE"
elif [ -f "$ARCHIVE_DIR/$WIKIPEDIA_ARCHIVE" ]; then
    echo "Moving wikipedia archive from $ARCHIVE_DIR/ into $DATA_DIR/"
    mv "$ARCHIVE_DIR/$WIKIPEDIA_ARCHIVE" "$DATA_DIR/$WIKIPEDIA_ARCHIVE"
else
    echo "Error: $WIKIPEDIA_ARCHIVE not found."
    echo "Download it from:"
    echo "  https://archive.org/download/wikipedia_en_all_maxi_2022-05/${WIKIPEDIA_ARCHIVE}"
    echo "and place it in $ARCHIVE_DIR/ (or directly in $DATA_DIR/)."
    exit 1
fi

echo "Setup complete."
