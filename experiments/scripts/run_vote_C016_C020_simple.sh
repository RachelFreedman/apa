#!/usr/bin/env bash
# Same fixed-jury vote as run_vote_C016_C020.sh, but over the simplified
# yes/no response set in experiments/query_responses_simple.jsonl.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESPONSES="$REPO_ROOT/experiments/query_responses_simple.jsonl"
ADAPTED="/nas/ucb/rachel/APA/models/W_adapted_hist_C016_C020_filtered.pt"
OUT_DIR="$REPO_ROOT/experiments/vote_C016_C020_simple"
LOG_DIR="$REPO_ROOT/experiments/logs"
AUDIT_LOG="$OUT_DIR/audit_log.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$REPO_ROOT"

uv run python -m apa.democratic_response \
    --responses_file "$RESPONSES" \
    --adapted_users "$ADAPTED" \
    --jury_sources "C16,C20,prism:10" \
    --methods borda_count,plurality,copeland,instant_runoff \
    --seed 42 \
    --log_file "$AUDIT_LOG" \
    2>&1 | tee "$LOG_DIR/vote_C016_C020_simple.log"

uv run python -m apa.vote_analysis "$AUDIT_LOG" --output_dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_DIR/vote_C016_C020_simple.log"
