#!/usr/bin/env bash
# Few-shot LoRe adaptation: fit per-user weight vectors w_u for the 20
# C016+C020 historical personas using the filtered synthetic preferences.
# The PRISM-trained LoRe basis V_K8.pt is loaded read-only (frozen).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFS="$REPO_ROOT/experiments/synthetic_prefs_C016_C020/hist_prefs_all_filtered.jsonl"
LOG_DIR="$REPO_ROOT/experiments/logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

uv run python -m apa.lore_adapt "$PREFS" \
    --K 8 \
    --name hist_C016_C020_filtered \
    2>&1 | tee "$LOG_DIR/train_user_weights_C016_C020.log"
