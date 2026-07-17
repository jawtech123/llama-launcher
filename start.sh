#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Start llama launcher GUI from the project root so the package is importable
cd "$SCRIPT_DIR"
exec python -m llama_launcher "$@"
