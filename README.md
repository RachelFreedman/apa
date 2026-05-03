# APA: Aggregated Preference Alignment

Code for **democratic preference aggregation with personalized reward
models**. The pipeline:

1. learns individual user reward models on PRISM using **LoRe**
   (low-rank reward modeling),
2. simulates historical "future" users via the **ProgressGym HistLlama**
   century-conditioned models,
3. forms a jury of those users and aggregates their per-response rankings
   into a single democratic ranking over candidate responses.

The repository contains the library (`apa/`), validation pipelines
(`scripts/`), and the end-to-end paper experiment (`experiments/`),
including its tracked input data and generated artifacts so reviewers can
inspect intermediate outputs without rerunning the heavy steps.

## Installation

```bash
cd /home/rachel/APA

# Create the .venv (symlinked onto NAS to avoid disk-quota issues)
source setup_uv.sh

# Install dependencies
uv sync

# Smoke test
uv run python -m apa.config           # imports
uv run pytest -m "not slow" -q        # ~30s, all fast tests
```

The slow tests (`tests/test_lore.py`, `tests/test_suitability.py`) take
~30 minutes total and exercise the full LoRe training and PRISM
suitability pipelines; run them as a final verification with
`uv run pytest`.

## Reproducing the paper

The paper experiment lives under `experiments/` and runs the C016 (16th
century) + C020 (20th century) jury composition on a curated, deliberately
time-varying subset of PRISM. All intermediate artifacts are tracked in
the repo so each step can be skipped if its inputs are unchanged.

**Prerequisites**

- PRISM access at `/nas/ucb/rachel/historical-prefs/data/prism/` (already
  populated on the lab cluster).
- Write access to `/nas/ucb/rachel/APA/` (paths in `apa/config.py`).
- ≥4× A100 (80GB) for step 2 (70B HistLlama via vLLM); 1× A100 for steps
  1 and 3–6.

### Step 1 — Train the LoRe basis on PRISM

```bash
uv run python -m apa.load_prism --split both
uv run python -m apa.train_lore_bases --K_list 0,1,8
```

Writes `V_K{0,1,8}.pt`, `W_seen_K8.pt`, and `user_to_idx.json` under
`/nas/ucb/rachel/APA/models/`. `V_K8.pt` is the basis the rest of the
pipeline reuses.

### Step 2 — Generate synthetic preferences for C016 and C020

```bash
bash experiments/scripts/generate_prefs_C016_C020.sh
```

Reads:
- `experiments/chosen_questions.jsonl` (500 PRISM questions where moral
  consensus is expected to vary across centuries; produced by
  `scripts/select_time_varying_questions.py`),
- `experiments/profiles_C016_C020.jsonl` (the 20 paper personas: 10 from
  C016 + 10 from C020).

Writes per-century outputs under
`experiments/synthetic_prefs_C016_C020/`:
`hist_prefs_C{016,020}.jsonl` (eval_prefs format) and
`hist_prefs_C{016,020}_raw.json` (full reasoning + logprobs).

Each preference pair is judged via a two-stage CoT chat-template flow:
stage 1 asks the persona (in the system role) to reason briefly about
"Response X" vs "Response Y"; stage 2 commits to a guided-decoded token
from `{"X","Y"}` so the calibrated soft-preference logprobs land on the
first emitted token. Both orderings of the pair are run and averaged to
cancel the model's residual letter-prior.

### Step 3 — Filter the synthetic preferences

```bash
uv run python -m experiments.filter_output filter \
    --input  experiments/synthetic_prefs_C016_C020/hist_prefs_C016_raw.json \
             experiments/synthetic_prefs_C016_C020/hist_prefs_C020_raw.json \
    --output experiments/synthetic_prefs_C016_C020/hist_prefs_all_filtered.jsonl \
    --min-records-per-user 5
```

Three-stage filter: drop pair with `consistency != 1.0`; drop questions
where every persona picked the same side; drop personas left with too few
records to fit a per-user W vector.

### Step 4 — Few-shot LoRe adaptation: fit per-user W vectors

```bash
bash experiments/scripts/train_user_weights_C016_C020.sh
```

Loads the filtered preferences, embeds them with Skywork-Reward, and runs
500 iterations of gradient descent against the frozen `V_K8.pt` basis from
step 1. Writes `W_adapted_hist_C016_C020_filtered.pt` to
`/nas/ucb/rachel/APA/models/`.

### Step 5 — Hold the democratic vote

```bash
bash experiments/scripts/run_vote_C016_C020.sh
bash experiments/scripts/run_vote_C016_C020_simple.sh
```

Both scripts call `apa.democratic_response` over a fixed query/response
slate (`experiments/query_responses{,_simple}.jsonl`) with
`--jury_sources "C16,C20,prism:10"` (all 10 C016 + all 10 C020 +
10 randomly sampled PRISM voters; seed 42). Then call `apa.vote_analysis`
to compute per-group aggregations and intra/inter-group rank agreement.

Outputs land in `experiments/vote_C016_C020/` and
`experiments/vote_C016_C020_simple/`: `audit_log.json` (raw per-voter
rankings + jury manifest), `vote_results.json` (aggregated rankings per
method), `vote_analysis.json` (group/agreement metrics), and
`vote_report.txt` (human-readable summary).

### Step 6 — Render figures

```bash
uv run python -m experiments.figs all
```

Reads the audit logs from step 5 and the W checkpoints from step 4 and
writes `experiments/figs/{user_weights_grid,jury_agreement_heatmap}.{png,pdf}`.
Individual figures available via `... figs user_weights_grid`,
`... figs jury_agreement_heatmap`.

## Library pipeline (generic)

The four `apa.*` entry points underneath the paper scripts are also
usable directly:

```bash
# 1. PRISM data + Skywork embeddings (writes to /nas/.../embeddings/)
uv run python -m apa.load_prism --split both
uv run python -m apa.load_prism --split both --n_samples 1000   # quick test

# 2. LoRe basis training (writes V_K*.pt and W_seen_K*.pt to /nas/.../models/)
uv run python -m apa.train_lore_bases --K_list 0,1,8

# 3a. Generate synthetic historical preferences for arbitrary centuries
uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
    --centuries C013 C017 C019 C021 --n-questions 20

# Multi-GPU 70B variant
uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
    --centuries C013 --n-questions 20 --model-size 70B --tensor-parallel-size 4

# 3b. Few-shot adapt per-user W vectors against an existing V basis
uv run python -m apa.lore_adapt path/to/prefs.jsonl --K 8 --name my_run

# 4. Hold a democratic vote on a single query
uv run python -m apa.democratic_response --query "What is the meaning of life?"

# 4-alt. Or vote on a pre-supplied response slate
uv run python -m apa.democratic_response --responses_file slate.jsonl --show_all
```

## Project structure

```
APA/
├── apa/
│   ├── _logging.py                   # Shared timestamped logger
│   ├── config.py                     # Paths and dataclass configs
│   ├── load_prism.py                 # PRISM load + Skywork embedding
│   ├── train_lore_bases.py           # LoRe basis (V) training
│   ├── lore_adapt.py                 # Few-shot W adaptation + LoReScorer
│   ├── democratic_response.py        # Jury → vote orchestrator
│   ├── vote_analysis.py              # Audit-log post-processing + reports
│   ├── synthetic_prefs/
│   │   ├── historical_prefs.py       # HistLlama-driven preference generation
│   │   ├── eval_prefs.py             # LoRe suitability metrics
│   │   ├── sample_data.py            # PRISM/random baseline samplers
│   │   ├── profiles.jsonl            # Canonical 90 personas (10 / century)
│   │   └── curated_questions.txt     # Curated value-laden PRISM questions
│   └── levers/                       # Pluggable strategies (see below)
│       ├── slate_generation.py
│       ├── voter_sampling.py
│       ├── voter_aggregation.py
│       └── query_selection.py
├── scripts/                          # Generic / validation pipelines
│   ├── select_time_varying_questions.py
│   ├── compare_metrics.py
│   ├── run_hist_prefs_full.sh
│   ├── run_all_centuries_70b.sh
│   └── run_hist_adapt.sh
├── experiments/                      # Paper experiment + figures
│   ├── scripts/
│   │   ├── generate_prefs_C016_C020.sh
│   │   ├── train_user_weights_C016_C020.sh
│   │   ├── run_vote_C016_C020.sh
│   │   └── run_vote_C016_C020_simple.sh
│   ├── filter_output.py              # 3-stage preference filter
│   ├── figs.py                       # Paper figures
│   ├── utils.py                      # extract-question-ids helper
│   ├── chosen_questions.jsonl        # 500 time-varying PRISM questions (input)
│   ├── profiles_C016_C020.jsonl      # 20 paper personas (input)
│   ├── query_responses.jsonl         # Vote slate (input)
│   ├── query_responses_simple.jsonl  # Yes/no vote slate (input)
│   ├── synthetic_prefs_C016_C020/    # Step-2/3 outputs (tracked)
│   ├── vote_C016_C020/               # Step-5 outputs (tracked)
│   ├── vote_C016_C020_simple/        # Step-5 outputs (tracked)
│   └── figs/                         # Step-6 outputs (tracked)
├── tests/                            # Fast (~30s) + 2 slow (~30 min total)
├── pyproject.toml
├── setup_uv.sh
└── README.md
```

## Levers (pluggable strategies)

The four lever modules under `apa/levers/` factor out the strategy
choices made by the pipeline. Production code dispatches by name where
applicable.

| Lever | Module | Strategies | Used by |
|-------|--------|------------|---------|
| Response generation | `slate_generation.py` | `temperature_sampling` | `democratic_response` |
| Voter sampling (jury composition) | `voter_sampling.py` | `random`, `stratified`, `weighted`, `temporal_mix`, `per_group_sampling` | `democratic_response` |
| Ranking aggregation | `voter_aggregation.py` | `borda_count`, `plurality`, `copeland`, `instant_runoff` | `democratic_response` |
| Question selection | `query_selection.py` | `random_subset`, `select_by_ids` | `historical_prefs` (which questions to put to a persona) |

`per_group_sampling` is the lever that backs the `--jury_sources` flag —
e.g. `--jury_sources "C16,C20,prism:10"` means *all C016 voters + all
C020 voters + 10 randomly-sampled PRISM voters*. It also exposes
`parse_jury_source_spec` for parsing those flag tokens.

## Configuration

Defaults live in `apa/config.py`. The most relevant fields:

| Section | Field | Default | Notes |
|---------|-------|---------|-------|
| Paths | `NAS_BASE` | `/nas/ucb/rachel/APA` | Root for all large artifacts. |
| `LoReConfig` | `K_list` | `[0, 1]` | Ranks to train (paper uses 8). |
| `LoReConfig` | `alpha` | `10000.0` | Regularization strength (matches LoRe paper). |
| `LoReConfig` | `num_iterations` | `20000` | Basis training iterations. |
| `LoReConfig` | `learning_rate` | `0.5` | **Critical**: 0.5, not 1e-4. |
| `LoReConfig` | `few_shot_iterations` | `500` | Per-user W fitting iterations. |
| `LoReConfig` | `few_shot_lr` | `0.5` | |
| `LoReConfig` | `embedding_model` | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` | |
| `LoReConfig` | `embedding_dim` | `4096` | Llama 3.1 8B hidden dim. |
| `InferenceLLMConfig` | `model_name` | `Qwen/Qwen2.5-7B-Instruct` | LLM for response generation. |
| `InferenceLLMConfig` | `temperature` | `1.2` | Higher → more diverse slate. |
| `InferenceConfig` | `k_responses` | `5` | Slate size. |
| `InferenceConfig` | `m_voters` | `10` | Jury size when not using `--jury_sources`. |
| `InferenceConfig` | `aggregate_strategy` | `borda_count` | CLI `--methods` default. |
| `APAConfig` | `historical_centuries` | `[C013, C017, C019, C021]` | Used by `historical_prefs` when no `--centuries` given. |

## Storage

Large artifacts live on NAS:

```
/nas/ucb/rachel/APA/
├── data/prism/                           # PRISM raw data
├── embeddings/{train,test}.pkl           # Skywork embeddings
├── models/
│   ├── V_K{0,1,8}.pt                     # LoRe bases
│   ├── W_seen_K{0,1,8}.pt                # PRISM seen-user W matrix
│   ├── user_to_idx.json                  # PRISM uid → row index
│   └── W_adapted_*.pt                    # Few-shot adapted per-user W (one per run)
├── hf_cache/                             # HuggingFace + sentence-transformers cache
└── tmp/                                  # TMPDIR for transient files
```

Paper-experiment outputs live inside the repo under
`experiments/synthetic_prefs_C016_C020/`,
`experiments/vote_C016_C020{,_simple}/`, and `experiments/figs/`.

## Testing

```bash
uv run pytest -m "not slow" -q   # ~30s
uv run pytest -q                 # ~30 min (includes test_lore.py + test_suitability.py)
```

## External resources

- [LoRe paper](https://arxiv.org/abs/2504.14439)
- [LoRe code](https://github.com/facebookresearch/LoRe)
- [PRISM dataset](https://github.com/HannahKirk/prism-alignment)
- [ProgressGym](https://github.com/PKU-Alignment/ProgressGym)
- [HistLlama models](https://huggingface.co/collections/PKU-Alignment/progressgym-666735fcf3e4efa276226eaa)
