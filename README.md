# APA: Aggregated Preference Alignment

A democratic preference aggregation pipeline that:
1. Learns individual user reward models using LoRe (Low-rank Reward modeling) on PRISM
2. Simulates "future" users via ProgressGym HistLlama models
3. Generates diverse response slates and aggregates preferences democratically

## Quick Start

```bash
cd /home/rachel/APA

# Set up environment variables (add to your .bashrc for persistence)
export UV_CACHE_DIR=/nas/ucb/rachel/APA/uv_cache
export TMPDIR=/nas/ucb/rachel/APA/tmp

# Create symlink so .venv lives on /nas (avoids disk quota issues)
# Skip if symlink already exists
[ -L .venv ] || (rm -rf .venv && ln -s /nas/ucb/rachel/APA/.venv .venv)

# Install/sync dependencies with uv
uv sync

# Verify installation
uv run python tests/test_imports.py

# Run tests
uv run pytest tests/ -v
```

**Note:** The home directory has limited disk quota. The symlink approach stores the virtual environment on `/nas` where there is ample space.

## Pipeline Steps

### 1. Prepare PRISM embeddings

```bash
uv run python scripts/prepare_prism_embeddings.py

# Or for testing with fewer samples:
uv run python scripts/prepare_prism_embeddings.py --n_samples 1000
```

### 2. Train LoRe model

```bash
uv run python scripts/train_lore_prism.py

# Or for testing:
uv run python scripts/train_lore_prism.py --n_users 50 --epochs 5
```

### 3. Historical user vectors (optional)

#### 3a. Generate historical preferences

```bash
uv run python scripts/generate_historical_preferences.py --century C013 --n_questions 500
uv run python scripts/generate_historical_preferences.py --century C017 --n_questions 500
```

#### 3b. Train historical user vectors

```bash
uv run python scripts/train_historical_users.py --preferences_file /nas/ucb/rachel/APA/checkpoints/prism/historical/preferences_historical_C013.json
uv run python scripts/train_historical_users.py --preferences_file /nas/ucb/rachel/APA/checkpoints/prism/historical/preferences_historical_C017.json
```

### 4. Run democratic inference

```bash
# Single query
uv run python scripts/run_democratic_inference.py --query "What is the meaning of life?"

# Interactive mode
uv run python scripts/run_democratic_inference.py --interactive

# With all responses shown
uv run python scripts/run_democratic_inference.py --query "..." --show_all
```

## Project Structure

```
APA/
├── apa/
│   ├── config.py           # Configuration and paths
│   ├── data/               # Data loading
│   │   └── prism_loader.py
│   ├── reward/             # LoRe reward modeling
│   │   └── lore_model.py
│   ├── historical/         # Historical LLM preferences
│   │   ├── hist_llama.py
│   │   └── preference_gen.py
│   ├── inference/          # Democratic inference
│   │   ├── response_generator.py
│   │   ├── voter.py
│   │   └── democratic_inference.py
│   ├── levers/             # Injection points
│   │   ├── lever_generate.py
│   │   ├── lever_sample.py
│   │   ├── lever_aggregate.py
│   │   └── lever_questions.py
│   └── utils/
│       ├── embedding_utils.py
│       └── file_utils.py
├── scripts/
│   ├── prepare_prism_embeddings.py
│   ├── train_lore_prism.py
│   ├── generate_historical_preferences.py
│   ├── train_historical_users.py
│   └── run_democratic_inference.py
├── tests/
└── pyproject.toml
```

## Levers (Injection Points)

The system has four "levers" - modular functions that can be easily swapped:

### 1. Response Generation (`lever_generate.py`)
How to generate diverse responses. Default: temperature sampling.
- `temperature_sampling` (default)
- `diverse_beam` (placeholder)
- `contrastive_decode` (placeholder)

### 2. User Sampling (`lever_sample.py`)
How to select which users vote. Default: random.
- `random` (default)
- `stratified`
- `weighted`
- `temporal_mix`

### 3. Ranking Aggregation (`lever_aggregate.py`)
How to combine rankings. Default: Borda count.
- `borda_count` (default)
- `plurality`
- `copeland`
- `instant_runoff`
- `schulze` (placeholder)
- `kemeny_young` (placeholder)

### 4. Question Selection (`lever_questions.py`)
How to select training questions. Default: random.
- `random_subset` (default)
- `diverse_topics` (placeholder)
- `controversial` (placeholder)
- `high_agreement` (placeholder)
- `temporal_relevant` (placeholder)

## Configuration

Default parameters in `apa/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| k_responses | 5 | Number of responses to generate |
| m_voters | 10 | Number of users to sample for voting |
| lore.rank | 8 | Low-rank dimension |
| lore.alpha | 10000 | Regularization coefficient |
| hist_llama_size | 8B | HistLlama model size |
| centuries | [13, 17, 19, 21] | Historical centuries to use |

## Storage

Large files are stored on NAS:
- **NAS base**: `/nas/ucb/rachel/APA/`
- **Embeddings**: `/nas/ucb/rachel/APA/data/prism/embeddings.pkl`
- **Checkpoints**: `/nas/ucb/rachel/APA/checkpoints/`
- **HF cache**: `/nas/ucb/rachel/APA/hf_cache/`

## External Resources

- [LoRe Paper](https://arxiv.org/abs/2504.14439)
- [LoRe Code](https://github.com/facebookresearch/LoRe)
- [PRISM Dataset](https://github.com/HannahKirk/prism-alignment)
- [ProgressGym](https://github.com/PKU-Alignment/ProgressGym)
- [HistLlama Models](https://huggingface.co/collections/PKU-Alignment/progressgym-666735fcf3e4efa276226eaa)
