#!/usr/bin/env bash
# Pilot v2: same as generate_prefs_C020_pilot.sh but with X/Y label
# randomization enabled (per-prompt random assignment of X/Y to first vs.
# second response). Output goes to a fresh directory so the v1 pilot is
# preserved for direct comparison.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments"
OUT_DIR="$EXP_DIR/synthetic_prefs_C020_pilot_v2"
PROFILES="$EXP_DIR/profiles.jsonl"
QUESTIONS_JSONL="$EXP_DIR/chosen_questions.jsonl"
QUESTIONS_IDS="$OUT_DIR/chosen_question_ids.txt"

mkdir -p "$OUT_DIR"

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
    --n-questions-cap 10 \
    --output-dir "$OUT_DIR"
