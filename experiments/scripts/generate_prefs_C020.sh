#!/usr/bin/env bash
# Resume script: generate synthetic preferences for C020 only (C016 already
# completed in a prior run). Uses the 70B HistLlama model with reduced
# gpu_memory_utilization to leave headroom on shared GPUs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments"
OUT_DIR="$EXP_DIR/synthetic_prefs_C016_C020"
PROFILES="$EXP_DIR/profiles.jsonl"
QUESTIONS_JSONL="$EXP_DIR/chosen_questions.jsonl"
QUESTIONS_IDS="$OUT_DIR/chosen_question_ids.txt"

mkdir -p "$OUT_DIR"

# Restrict vLLM to the four GPUs that are currently idle on this host.
export CUDA_VISIBLE_DEVICES=1,2,6,7

cd "$REPO_ROOT"

uv run python -m experiments.utils extract-question-ids \
    --input "$QUESTIONS_JSONL" \
    --output "$QUESTIONS_IDS"

uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
    --centuries C020 \
    --model-size 70B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --profiles "$PROFILES" \
    --questions "$QUESTIONS_IDS" \
    --output-dir "$OUT_DIR"
