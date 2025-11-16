#!/bin/bash
# Script to add test videos to the uploads directory

if [ $# -eq 0 ]; then
    echo "Usage: ./add-test-video.sh /path/to/video.mp4"
    echo ""
    echo "This will copy the video to ./uploads/ and it will be immediately"
    echo "visible in the containers (NO RESTART NEEDED)"
    exit 1
fi

SOURCE_FILE="$1"
DEST_DIR="./uploads"

# Normalize path (remove double slashes, resolve relative paths)
SOURCE_FILE=$(echo "$SOURCE_FILE" | sed 's|//|/|g')
SOURCE_FILE=$(realpath "$SOURCE_FILE" 2>/dev/null || echo "$SOURCE_FILE")

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: File not found: $SOURCE_FILE"
    exit 1
fi

# Create uploads directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Get absolute path of destination directory
DEST_DIR_ABS=$(cd "$DEST_DIR" && pwd)

# Get absolute path of source file
SOURCE_FILE_ABS=$(cd "$(dirname "$SOURCE_FILE")" && pwd)/$(basename "$SOURCE_FILE")

# Get filename
FILENAME=$(basename "$SOURCE_FILE")
DEST_FILE="$DEST_DIR/$FILENAME"

# Check if source and destination are the same file
if [ "$SOURCE_FILE_ABS" = "$DEST_DIR_ABS/$FILENAME" ]; then
    echo "ℹ File is already in the uploads directory: $FILENAME"
    echo ""
    echo "File is already available at:"
    echo "  Host: ./uploads/$FILENAME"
    echo "  Container: /app/uploads/$FILENAME"
    echo ""
    echo "No action needed - file is already accessible!"
    exit 0
fi

# Check if file already exists in destination
if [ -f "$DEST_FILE" ]; then
    echo "⚠ File already exists in uploads: $FILENAME"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipped. File remains unchanged."
        exit 0
    fi
fi

# Copy file
echo "Copying $SOURCE_FILE to $DEST_FILE..."
cp "$SOURCE_FILE" "$DEST_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Successfully copied!"
    echo ""
    echo "File is now available at:"
    echo "  Host: ./uploads/$FILENAME"
    echo "  Container: /app/uploads/$FILENAME"
    echo ""
    echo "No restart needed - file is immediately accessible!"
else
    echo "✗ Failed to copy file"
    exit 1
fi

