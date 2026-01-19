#!/bin/bash
# Setup script for APA project
# Run this from any terminal (not through Claude Code)

set -e

# Create venv on NAS for more space
rm -rf /home/rachel/APA/.venv /nas/ucb/rachel/APA/venv
python3.9 -m venv /nas/ucb/rachel/APA/venv
ln -s /nas/ucb/rachel/APA/venv /home/rachel/APA/.venv

# Install dependencies
/nas/ucb/rachel/APA/venv/bin/pip install --cache-dir=/nas/ucb/rachel/APA/pip_cache \
    torch \
    transformers \
    sentence-transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    pyyaml \
    pandas \
    tqdm \
    huggingface_hub

# Install APA in editable mode
/nas/ucb/rachel/APA/venv/bin/pip install -e /home/rachel/APA

echo ""
echo "Setup complete! Activate with:"
echo "  source /home/rachel/APA/.venv/bin/activate"
echo ""
echo "Then run tests with:"
echo "  python tests/test_imports.py"
