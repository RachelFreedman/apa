# APA: Aggregated Preference Alignment

A democratic preference aggregation pipeline that:
1. Learns individual user reward models using LoRe (Low-rank Reward modeling) on PRISM
2. Simulates "future" users via ProgressGym HistLlama models
3. Generates diverse response slates and aggregates preferences democratically

## Quick Start

### Option A: Use existing LoRe environment (recommended)

The project can use the existing LoRe virtual environment which has torch and transformers already installed.

```bash
cd /home/rachel/APA

# Run any script with ./run.sh
./run.sh tests/test_imports.py               # Verify imports
./run.sh scripts/prepare_prism_embeddings.py # Generate embeddings
./run.sh scripts/train_lore_prism.py         # Train LoRe model
```

### Option B: Create fresh virtual environment with uv

```bash
cd /home/rachel/APA
./setup_uv.sh
# Then use: uv run python <script>
```

### Option C: Create fresh virtual environment with pip

```bash
cd /home/rachel/APA
./setup_env.sh
source .venv/bin/activate
```

### Pipeline Steps

#### 1. Prepare PRISM embeddings

```bash
./run.sh scripts/prepare_prism_embeddings.py
# Or for testing with fewer samples:
./run.sh scripts/prepare_prism_embeddings.py --n_samples 1000
```

#### 2. Train LoRe model

```bash
./run.sh scripts/train_lore_prism.py
# Or for testing:
./run.sh scripts/train_lore_prism.py --n_users 50 --epochs 5
```

#### 3. Train historical user vectors (optional)

```bash
./run.sh scripts/train_historical_users.py --century C013 --n_questions 500
./run.sh scripts/train_historical_users.py --century C017 --n_questions 500
```

#### 4. Run democratic inference

```bash
# Single query
./run.sh scripts/run_democratic_inference.py --query "What is the meaning of life?"

# Interactive mode
./run.sh scripts/run_democratic_inference.py --interactive

# With all responses shown
./run.sh scripts/run_democratic_inference.py --query "..." --show_all
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
│   ├── train_historical_users.py
│   └── run_democratic_inference.py
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
- `stratified_by_type`

## Configuration

Default parameters in `apa/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| k_responses | 5 | Number of responses to generate |
| m_voters | 10 | Number of users to sample for voting |
| lore.rank | 8 | Low-rank dimension |
| lore.alpha | 10000 | Regularization coefficient |
| inference_llm | Llama-3.1-8B-Instruct | Base LLM for generation |
| hist_llama_size | 8B | HistLlama model size |
| centuries | [13, 17, 19, 21] | Historical centuries to use |

## Storage

Large files are stored on NAS with symlinks:
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
