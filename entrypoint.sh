#!/bin/bash

# Directory to check
MOUNTED_DIR="/workspace"

# Check if directory is mounted and not empty
if mountpoint -q "$MOUNTED_DIR" && [ "$(ls -A $MOUNTED_DIR)" ]; then
    echo "Directory is mounted and not empty. Installing the package."
    pip install -e .
else
    echo "Directory is not mounted or is empty. Skipping package installation."
fi

echo "Setup git config safe directory"
git config --global --add safe.directory /workspace

# Execute the command passed to the docker run
exec "$@"
