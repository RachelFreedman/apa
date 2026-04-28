#!/usr/bin/env bash
# Durable wrapper around generate_prefs_C016_C020.sh: re-launches the
# underlying script on transient failure. The Python entrypoint
# (cmd_generate_synth) supports per-century resume — it skips a century
# whose per-century outputs already exist on disk and match the current
# {question_id, user_id} cover sets. So a mid-run kill costs at most one
# in-progress century, and the retry continues from there.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../synthetic_prefs_C016_C020"
INNER="$SCRIPT_DIR/generate_prefs_C016_C020.sh"
MAX_ATTEMPTS=5

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[durable] attempt $attempt / $MAX_ATTEMPTS — $(date -Is)"
    if bash "$INNER"; then
        echo "[durable] inner script exited 0 on attempt $attempt"
        break
    fi
    echo "[durable] attempt $attempt failed; sleeping 30s before resume..." >&2
    sleep 30
done

# Final completeness check — both per-century raw outputs must exist.
if [[ -f "$OUT_DIR/hist_prefs_C016_raw.json" && -f "$OUT_DIR/hist_prefs_C020_raw.json" ]]; then
    echo "[durable] both centuries' outputs present in $OUT_DIR"
    exit 0
fi
echo "[durable] FAILED: missing per-century outputs after $MAX_ATTEMPTS attempts" >&2
exit 1
