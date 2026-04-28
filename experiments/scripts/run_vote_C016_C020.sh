#!/usr/bin/env bash
# Hold a fixed-jury democratic vote over experiments/query_responses.jsonl.
# Jury = all 10 C016 + all 10 C020 historical voters + 10 randomly-sampled
# PRISM voters. Frozen LoRe basis V_K8 (PRISM-trained); historical voters
# from W_adapted_hist_C016_C020_filtered.pt; PRISM voters from W_seen_K8.pt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESPONSES="$REPO_ROOT/experiments/query_responses.jsonl"
ADAPTED="/nas/ucb/rachel/APA/models/W_adapted_hist_C016_C020_filtered.pt"
OUT_DIR="$REPO_ROOT/experiments/vote_C016_C020"
LOG_DIR="$REPO_ROOT/experiments/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$REPO_ROOT"

uv run python -m experiments.run_vote_C016_C020 \
    --responses_file "$RESPONSES" \
    --adapted_users "$ADAPTED" \
    --n_prism 10 \
    --seed 42 \
    --output_dir "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/vote_C016_C020.log"
