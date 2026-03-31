#!/usr/bin/env bash
# Generate synthetic historical preferences (20 users) and compare against PRISM/Random baselines.
#
# Expected runtime: ~28 minutes (10 profiles x 2 centuries x 20 questions x 3 reps x 2 orders
#   = 2400 model queries, plus embedding + eval).
#
# Usage:
#   bash scripts/run_hist_prefs_full.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-/nas/ucb/rachel/APA/synthetic_prefs}"
mkdir -p "$OUT_DIR"

echo "Expected runtime: ~28 minutes"
echo ""

echo "=== Step 1: Generate historical preferences (20 users) ==="
uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
    --centuries C013 C019 \
    --n-questions 20 \
    --n-runs 3 \
    --seed 42 \
    --output-dir "$OUT_DIR"

echo ""
echo "=== Step 2: Run comparison (Synth vs PRISM vs Random, all at 20 users) ==="
uv run python scripts/compare_metrics.py \
    --synth-path "$OUT_DIR/hist_prefs_all.jsonl" \
    --n-baseline-users 20 \
    --seed 42

echo ""
echo "=== Done. Results in $OUT_DIR ==="
