#!/usr/bin/env bash
# Pilot: generate synthetic preferences for C020 on the FIRST 10 questions
# only, using the new two-stage CoT prompt with system-role personas and X/Y
# labels. Result is written to a fresh output directory so it does NOT clobber
# the prior C016/C020 outputs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments"
OUT_DIR="$EXP_DIR/synthetic_prefs_C020_pilot"
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
    --n-questions-cap 10 \
    --output-dir "$OUT_DIR"
