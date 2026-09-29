#!/usr/bin/env bash
# Reproduce the Section 5 results, table and figures from the frozen generated responses and blind ratings.
# Usage: bash run.sh            (uses `python` on PATH; set PYTHON=... to override)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
echo "== 1/4 features from blind ratings";        "$PY" features.py
echo "== 2/4 experiments (about 1 minute)";        "$PY" experiments.py 200
echo "== 3/4 table and figures -> outputs/figs";   "$PY" figures.py
echo "== 4/4 compare with the paper";              "$PY" check.py
