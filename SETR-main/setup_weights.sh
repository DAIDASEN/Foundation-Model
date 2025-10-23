#!/bin/bash
# Script to setup pretrained weights for SETR training
# 
# Usage:
#   1. If you have the weights file, create a symlink:
#      bash setup_weights.sh /path/to/your/VFM_Fundus_weights.pth
#   
#   2. If using weights from ViT folder (for testing):
#      bash setup_weights.sh ../ViT/checkpoint.pth

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET_NAME="VFM_Fundus_weights.pth"
TARGET_PATH="$SCRIPT_DIR/$TARGET_NAME"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_weights_file>"
    echo ""
    echo "This script creates a symbolic link to your pretrained weights file."
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/VFM_Fundus_weights.pth"
    echo "  $0 ../ViT/checkpoint.pth"
    echo ""
    exit 1
fi

SOURCE_PATH="$1"

# Convert to absolute path
if [[ "$SOURCE_PATH" != /* ]]; then
    SOURCE_PATH="$(cd "$(dirname "$SOURCE_PATH")" && pwd)/$(basename "$SOURCE_PATH")"
fi

# Check if source file exists
if [ ! -f "$SOURCE_PATH" ]; then
    echo "Error: Source file not found: $SOURCE_PATH"
    exit 1
fi

# Check if target already exists
if [ -e "$TARGET_PATH" ]; then
    echo "Warning: $TARGET_NAME already exists."
    read -p "Do you want to replace it? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm "$TARGET_PATH"
fi

# Create symbolic link
ln -s "$SOURCE_PATH" "$TARGET_PATH"

echo "✓ Created symbolic link:"
echo "  $TARGET_PATH -> $SOURCE_PATH"
echo ""
echo "File info:"
ls -lh "$TARGET_PATH"
echo ""
echo "You can now run training with:"
echo "  bash train_vitb16_4gpus.sh exp_name"
