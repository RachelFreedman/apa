#!/bin/bash
# Setup script for APA project using uv
# Run this from any terminal (not through Claude Code)

set -e

cd /home/rachel/APA

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv with Python 3.9-3.12 (avoid 3.14)
uv venv --python 3.11 2>/dev/null || uv venv --python 3.10 2>/dev/null || uv venv --python 3.9

# Sync dependencies
uv sync

echo ""
echo "Setup complete! Activate with:"
echo "  source /home/rachel/APA/.venv/bin/activate"
echo ""
echo "Or run directly with:"
echo "  uv run python tests/test_imports.py"
