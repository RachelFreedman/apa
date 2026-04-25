#!/usr/bin/env bash
# Generate synthetic preferences for C016 + C020 personas on the curated
# experiments/chosen_questions.jsonl prompts using the 70B HistLlama judge.
#
# - 10 profiles per century (experiments/profiles.jsonl)
# - All prompts from experiments/chosen_questions.jsonl
# - Each (user, question) is asked 2x original order + 2x reversed (n-runs=2)
#
# Disconnect-resilient: default mode re-invokes the script under
# `setsid nohup` with `--fg`, detaching from the terminal. Per-century
# JSONL is written as each century finishes, so a crash during C020 still
# preserves C016 on disk.
#
# Estimated runtime: ~2h40m (2 x 10 x 100 x 4 = 8000 generations @ ~1.2s).
#
# Usage:
#   bash experiments/generate_prefs_C016_C020.sh           # launch detached
#   bash experiments/generate_prefs_C016_C020.sh --fg      # run attached
#   tail -f experiments/logs/gen_C016_C020_*.log           # monitor

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILES="experiments/profiles.jsonl"
QUESTIONS_JSONL="experiments/chosen_questions.jsonl"
OUTPUT_DIR="experiments/synthetic_prefs_C016_C020"
LOG_DIR="experiments/logs"
QIDS_TXT="$OUTPUT_DIR/chosen_question_ids.txt"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Extract PRISM question IDs from the JSONL into a one-ID-per-line .txt
# (the format load_curated_question_ids expects). Single-line python,
# reads path from argv to avoid shell interpolation inside python source.
.venv/bin/python -c 'import json, sys; ids=[json.loads(l)["question_id"] for l in open(sys.argv[1]) if l.strip()]; print(f"# {len(ids)} ids from {sys.argv[1]}"); [print(i) for i in ids]' "$QUESTIONS_JSONL" > "$QIDS_TXT"
N_Q=$(grep -cv '^#' "$QIDS_TXT")

if [[ "${1:-}" == "--fg" ]]; then
  echo "Wrote $N_Q question IDs to $QIDS_TXT"
  echo "Running in foreground ($(date '+%F %T'))"
  exec env PYTHONUNBUFFERED=1 .venv/bin/python -m apa.synthetic_prefs.historical_prefs \
      generate-synth \
      --centuries C016 C020 \
      --profiles "$PROFILES" \
      --questions "$QIDS_TXT" \
      --model-size 70B \
      --n-runs 2 \
      --temperature 0.3 \
      --output-dir "$OUTPUT_DIR"
fi

# Background mode: self-respawn under setsid+nohup, pipe through awk for
# line-level timestamps, log to a fresh file.
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/gen_C016_C020_${TS}.log"
SCRIPT_PATH="$REPO_ROOT/experiments/generate_prefs_C016_C020.sh"

echo "Wrote $N_Q question IDs to $QIDS_TXT"
echo "Launching in background (setsid nohup); log: $LOG"
echo "  Estimated runtime: ~2h40m  (2 centuries x 10 profiles x $N_Q questions x 4 runs)"
echo "  Monitor with: tail -f $LOG"

setsid nohup bash -c "cd '$REPO_ROOT' && '$SCRIPT_PATH' --fg 2>&1 | awk '{ print strftime(\"[%Y-%m-%d %H:%M:%S]\"), \$0; fflush(); }' >> '$LOG' 2>&1" \
  < /dev/null > /dev/null 2>&1 &
PID=$!
disown "$PID" || true
echo "PID: $PID"
