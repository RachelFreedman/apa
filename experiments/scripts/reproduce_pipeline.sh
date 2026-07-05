#!/bin/bash
# Full end-to-end reproduction of the APA pipeline, used to verify that the
# refactor is behavior-preserving. Regenerates outputs into a scratch dir and
# compares against the canonical NAS artifacts (per CLAUDE.md #9: regenerate,
# never copy prior outputs).
#
# WARNING: multi-hour, GPU-heavy (embedding regen + K=8 training + HistLlama
# download/generation). Run only with explicit approval.
#
# Usage:
#   experiments/scripts/reproduce_pipeline.sh [SCRATCH_DIR] [K_LIST] [CENTURY] [QUERY]
#
# Defaults:
#   SCRATCH_DIR = ./repro_out
#   K_LIST      = 0,1,8
#   CENTURY     = C013
#   QUERY       = "What is the meaning of life?"

set -euo pipefail

SCRATCH_DIR="${1:-./repro_out}"
K_LIST="${2:-0,1,8}"
CENTURY="${3:-C013}"
QUERY="${4:-What is the meaning of life?}"

REF_EMB_DIR="/nas/ucb/rachel/APA/embeddings"
EMB_DIR="${SCRATCH_DIR}/embeddings"
MODELS_DIR="${SCRATCH_DIR}/models"
DATA_DIR="${SCRATCH_DIR}/data/prism"
LOG_DIR="${SCRATCH_DIR}/logs"

mkdir -p "$EMB_DIR" "$MODELS_DIR" "$DATA_DIR" "$LOG_DIR"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
banner() { echo ""; echo "[$(ts)] ============================================================"; echo "[$(ts)] $1"; echo "[$(ts)] ============================================================"; }

# --- Stage 5: regenerate embeddings from scratch and compare to canonical -----
banner "Stage 5/8: Regenerate PRISM embeddings -> ${EMB_DIR}"
python -m apa.load_prism --split both \
    --output_dir "$EMB_DIR" --data_dir "$DATA_DIR" \
    2>&1 | tee "${LOG_DIR}/load_prism.log"

banner "Compare regenerated train.pkl vs canonical ${REF_EMB_DIR}/train.pkl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/compare_embeddings.py" \
    --new "${EMB_DIR}/train.pkl" --ref "${REF_EMB_DIR}/train.pkl" --n 500 --atol 1e-2 \
    2>&1 | tee "${LOG_DIR}/compare_train.log"

# --- Stage 6: train LoRe over multiple ranks ----------------------------------
banner "Stage 6/8: Train LoRe (K_list=${K_LIST}) -> ${MODELS_DIR}"
python -m apa.train_lore_bases --K_list "$K_LIST" \
    --embeddings_dir "$EMB_DIR" --output_dir "$MODELS_DIR" \
    2>&1 | tee "${LOG_DIR}/train_lore.log"

# --- Stage 7: historical user vector ------------------------------------------
banner "Stage 7/8: Historical prefs generate + train (${CENTURY})"
python -m apa.historical_prefs generate --century "$CENTURY" --n_questions 200 \
    --output_dir "$MODELS_DIR" \
    2>&1 | tee "${LOG_DIR}/hist_generate_${CENTURY}.log"
python -m apa.historical_prefs train \
    --preferences_file "${MODELS_DIR}/preferences_historical_${CENTURY}.json" \
    --lore_checkpoint "${MODELS_DIR}/V_K8.pt" --output_dir "$MODELS_DIR" \
    2>&1 | tee "${LOG_DIR}/hist_train_${CENTURY}.log"

# --- Stage 8: democratic inference (default + a non-default aggregator) --------
banner "Stage 8/8: Democratic inference"
python -m apa.democratic_response --query "$QUERY" --show_all \
    --lore_checkpoint "${MODELS_DIR}/V_K8.pt" \
    --prism_users "${MODELS_DIR}/W_seen_K8.pt" \
    --historical_dir "$MODELS_DIR" \
    2>&1 | tee "${LOG_DIR}/inference_default.log"

python -m apa.democratic_response --query "$QUERY" --aggregate_strategy plurality \
    --lore_checkpoint "${MODELS_DIR}/V_K8.pt" \
    --prism_users "${MODELS_DIR}/W_seen_K8.pt" \
    --historical_dir "$MODELS_DIR" \
    2>&1 | tee "${LOG_DIR}/inference_plurality.log"

banner "Reproduction complete. Outputs + logs under ${SCRATCH_DIR}"
