#!/bin/bash
# Run APA scripts using existing LoRe environment + cached packages
# Usage: ./run.sh <script_or_command> [args...]
#
# Examples:
#   ./run.sh scripts/prepare_prism_embeddings.py --n_samples 100
#   ./run.sh scripts/train_lore_prism.py --n_users 50
#   ./run.sh scripts/run_democratic_inference.py --query "What is life?"
#   ./run.sh -c "from apa.config import get_config; print(get_config())"
#   ./run.sh tests/test_imports.py

# Environment setup
export HF_HOME=/nas/ucb/rachel/APA/hf_cache
export TRANSFORMERS_CACHE=/nas/ucb/rachel/APA/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/nas/ucb/rachel/APA/hf_cache/sentence_transformers
export TMPDIR=/nas/ucb/rachel/APA/tmp
export TEMP=/nas/ucb/rachel/APA/tmp
export TMP=/nas/ucb/rachel/APA/tmp

# Ensure cache directories exist
mkdir -p /nas/ucb/rachel/APA/hf_cache/sentence_transformers
mkdir -p /nas/ucb/rachel/APA/tmp
mkdir -p /nas/ucb/rachel/APA/data/prism
mkdir -p /nas/ucb/rachel/APA/checkpoints/prism

# Use LoRe-venv site-packages (sentence-transformers must be installed there)
export PYTHONPATH=/home/rachel/scratch/LoRe-venv/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/home/rachel/APA

exec /usr/bin/python3.8 "$@"
