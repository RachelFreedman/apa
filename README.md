# APA: Aggregated Preference Alignment

A democratic preference aggregation pipeline that:
1. Learns individual user reward models using LoRe (Low-rank Reward modeling) on PRISM
2. Simulates historical users via ProgressGym HistLlama models
3. Generates diverse response slates and aggregates preferences democratically

## Quick Start

```bash
cd /home/rachel/APA

# Set up environment (creates .venv symlink on NAS to avoid disk quota issues)
source setup_uv.sh

# Install dependencies
uv sync

# Verify installation
uv run python tests/test_imports.py

# Run tests
uv run pytest tests/ -v
```

## Pipeline Steps

### 1. Prepare PRISM embeddings

```bash
uv run python -m apa.load_prism --split both

# Or for testing with fewer samples:
uv run python -m apa.load_prism --split both --n_samples 1000
```

### 2. Train LoRe model

```bash
uv run python -m apa.train_lore_bases --K_list 0,1,8

# Or for testing:
uv run python -m apa.train_lore_bases --K_list 0,1 --n_users 50
```

### 3. Historical user vectors (optional)

#### 3a. Generate historical preferences

```bash
uv run python -m apa.historical_prefs generate --century C013 --n_questions 500
uv run python -m apa.historical_prefs generate --century C017 --n_questions 500
```

#### 3b. Train historical user vectors

```bash
uv run python -m apa.historical_prefs train --century C013
uv run python -m apa.historical_prefs train --century C017
```

### 4. Run democratic inference

```bash
# Single query
uv run python -m apa.democratic_response --query "What is the meaning of life?"

# Interactive mode
uv run python -m apa.democratic_response --interactive

# With all responses shown
uv run python -m apa.democratic_response --query "..." --show_all
```

## Project Structure

```
APA/
├── apa/
│   ├── config.py              # Configuration and paths
│   ├── utils.py               # Shared helpers (timestamped logging)
│   ├── load_prism.py          # PRISM data loading and embedding
│   ├── train_lore_bases.py    # LoRe reward model training
│   ├── historical_prefs.py    # Historical preference generation
│   ├── democratic_response.py # Democratic inference pipeline
│   └── levers/                # Modular strategy functions (+ dispatch registry)
│       ├── __init__.py        # SAMPLERS/AGGREGATORS/... registries + get_* resolvers
│       ├── voter_sampling.py
│       ├── voter_aggregation.py
│       ├── query_selection.py
│       └── slate_generation.py
├── tests/
├── setup_uv.sh
└── pyproject.toml
```

## Levers (Strategy Modules)

The system has four modular strategy functions that can be swapped. Strategies
are selected by name through `InferenceConfig` (defaults) and resolved via the
registries in `apa/levers/__init__.py` (`get_sampler`, `get_aggregator`, …).
Democratic inference exposes `--sample_strategy` / `--aggregate_strategy` CLI
flags; the defaults (`random` + `borda_count`) reproduce the original behavior.

### 1. Response Generation (`slate_generation.py`)
How to generate diverse responses.
- `temperature_sampling` (default)

### 2. User Sampling (`voter_sampling.py`)
How to select which users vote.
- `random_sampling` (default)
- `stratified_sampling`
- `weighted_sampling`
- `temporal_mix_sampling`

### 3. Ranking Aggregation (`voter_aggregation.py`)
How to combine rankings.
- `borda_count` (default)
- `plurality`
- `copeland`
- `instant_runoff`

### 4. Question Selection (`query_selection.py`)
How to select training questions.
- `random_subset` (default)

## Reproducibility

Every CLI accepts `--seed` (and `--deterministic`), applied via
`apa.utils.set_seed` (seeds `random`, `numpy`, and `torch`). A fixed seed makes
training, voter sampling, and generation reproducible:

```bash
uv run python -m apa.train_lore_bases --K_list 0,1 --seed 42
uv run python -m apa.democratic_response --query "..." --seed 42
```

- `--seed` defaults to `42` for training/sampling/generation; `load_prism`
  defaults to `123` (`SPLIT_SEED`) so the seen/unseen split reproduces the
  canonical dataset.
- `--deterministic` additionally enables strict deterministic algorithms
  (`torch.use_deterministic_algorithms`, cuDNN deterministic) for
  bitwise-reproducible runs on the same hardware. Off by default (it can slow
  training).

## Configuration

Default parameters in `apa/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| k_responses | 5 | Number of responses to generate |
| m_voters | 10 | Number of users to sample for voting |
| lore.K_list | [0, 1] | Low-rank dimensions to train |
| lore.alpha | 10000 | Regularization coefficient |
| hist_llama.size | 8B | HistLlama model size |
| historical_centuries | [C013, C017, C019, C021] | Historical centuries to use |

## Storage

Large files are stored on NAS:

```
/nas/ucb/rachel/APA/
├── embeddings/      # PRISM embeddings (train.pkl, test.pkl)
├── models/          # Trained models (lore_K8.pt, V_K8.pt, W_*.pt)
├── hf_cache/        # HuggingFace model cache
└── tmp/             # Temporary files
```

## External Resources

- [LoRe Paper](https://arxiv.org/abs/2504.14439)
- [LoRe Code](https://github.com/facebookresearch/LoRe)
- [PRISM Dataset](https://github.com/HannahKirk/prism-alignment)
- [ProgressGym](https://github.com/PKU-Alignment/ProgressGym)
- [HistLlama Models](https://huggingface.co/collections/PKU-Alignment/progressgym-666735fcf3e4efa276226eaa)
