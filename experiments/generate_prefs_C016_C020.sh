#!/usr/bin/env bash
# Generate synthetic preferences for C016 + C020 personas on the curated
# experiments/chosen_questions.jsonl prompts using the 70B HistLlama judge.
#
# - 10 profiles per century (from experiments/profiles.jsonl)
# - All prompts from experiments/chosen_questions.jsonl
# - Each (user, question) is asked 2x in original order + 2x in reversed
#   order = 4 generations per (user, question) (n-runs=2)
#
# Disconnect-resilient: launched under `setsid nohup` so SSH disconnect
# won't kill it. Logs are timestamped and flushed (PYTHONUNBUFFERED=1).
# Per-century JSONL is written at the end of each century, so a crash
# during C020 still leaves C016 output on disk.
#
# Estimated runtime: ~2h40m (2 x 10 x 99 x 4 = 7920 generations @ ~1.2s).
#
# Usage:
#   bash experiments/generate_prefs_C016_C020.sh           # launch in background
#   bash experiments/generate_prefs_C016_C020.sh --fg      # run in foreground
#   tail -f experiments/logs/gen_C016_C020_*.log           # monitor

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILES="experiments/profiles.jsonl"
QUESTIONS_JSONL="experiments/chosen_questions.jsonl"
OUTPUT_DIR="experiments/synthetic_prefs_C016_C020"
LOG_DIR="experiments/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Extract question IDs from the JSONL into a tmp .txt that the generator
# expects (one ID per line; '#' comments allowed).
QIDS_TXT="$OUTPUT_DIR/chosen_question_ids.txt"
.venv/bin/python -c "
import json, sys
ids = [json.loads(l)['question_id'] for l in open('$QUESTIONS_JSONL') if l.strip()]
print(f'# auto-generated from $QUESTIONS_JSONL  ({len(ids)} ids)')
for i in ids: print(i)
" > "$QIDS_TXT"
N_Q=$(grep -cv '^#' "$QIDS_TXT" || true)
echo "Wrote $N_Q question IDs to $QIDS_TXT"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/gen_C016_C020_${TS}.log"

run() {
  PYTHONUNBUFFERED=1 .venv/bin/python -m apa.synthetic_prefs.historical_prefs \
      generate-synth \
      --centuries C016 C020 \
      --profiles "$PROFILES" \
      --questions "$QIDS_TXT" \
      --model-size 70B \
      --n-runs 2 \
      --temperature 0.3 \
      --output-dir "$OUTPUT_DIR" \
      2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
}

if [[ "${1:-}" == "--fg" ]]; then
  echo "Running in foreground; logging to $LOG"
  run | tee "$LOG"
else
  echo "Launching in background (setsid nohup); log: $LOG"
  echo "  Estimated runtime: ~2h40m  (2 centuries x 10 profiles x $N_Q questions x 4 runs)"
  echo "  Monitor with: tail -f $LOG"
  setsid nohup bash -c "$(declare -f run); run" > "$LOG" 2>&1 < /dev/null &
  PID=$!
  disown "$PID" || true
  echo "PID: $PID"
fi
