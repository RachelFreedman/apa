#!/usr/bin/env bash
# Generate synthetic historical preferences on PRISM and evaluate LoRe suitability.
#
# Usage:
#   bash scripts/run_hist_prefs.sh [output_dir]
set -euo pipefail

# Last measured runtime: ~16 min (2 centuries x 5 profiles x 20 questions x 3 reps x 2 orders
#   = 1200 model queries, plus embedding + eval). First run may be slower due to model downloads.
echo "Expected runtime: ~16 minutes"

OUT_DIR="${1:-/nas/ucb/rachel/APA/synthetic_prefs}"
mkdir -p "$OUT_DIR"

echo "=== Step 1: Generate historical preferences ==="
uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
    --centuries C013 C019 \
    --n-questions 20 \
    --n-runs 3 \
    --seed 42 \
    --output-dir "$OUT_DIR"

echo ""
echo "=== Step 2: Evaluate C013 preferences ==="
uv run python -m apa.synthetic_prefs.eval_prefs \
    "$OUT_DIR/hist_prefs_C013.jsonl" \
    --name "HistLlama C013 (5 profiles x 20 questions)"

echo ""
echo "=== Step 3: Evaluate C019 preferences ==="
uv run python -m apa.synthetic_prefs.eval_prefs \
    "$OUT_DIR/hist_prefs_C019.jsonl" \
    --name "HistLlama C019 (5 profiles x 20 questions)"

echo ""
echo "=== Step 4: Evaluate combined ==="
uv run python -m apa.synthetic_prefs.eval_prefs \
    "$OUT_DIR/hist_prefs_all.jsonl" \
    --name "HistLlama C013+C019 (10 profiles x 20 questions)"

echo ""
echo "=== Done. Results in $OUT_DIR ==="
