#!/usr/bin/env bash
# Reproduce the README baselines: PRISM subset (50 users) and random null (200 users).
set -euo pipefail

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Generate datasets
uv run python -m apa.synthetic_prefs.sample_data sample-emb -n 50  -o "$TMP/prism50.pt"
uv run python -m apa.synthetic_prefs.sample_data sample-emb -n 200  -o "$TMP/prism200.pt"
uv run python -m apa.synthetic_prefs.sample_data random-emb -n 200 -o "$TMP/random200.pt"

# Evaluate
uv run python -m apa.synthetic_prefs.eval_prefs "$TMP/prism50.pt"    --embeddings --name "PRISM (50 users)"
uv run python -m apa.synthetic_prefs.eval_prefs "$TMP/prism200.pt"    --embeddings --name "PRISM (200 users)"
uv run python -m apa.synthetic_prefs.eval_prefs "$TMP/random200.pt"  --embeddings --name "Random (200 users)"
