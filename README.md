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

The paper experiment runs a fixed C016 (16th century) + C020 (20th
century) jury over a curated, deliberately time-varying subset of PRISM.
There are two reproduction paths:

- **A — exact reproduction of the paper's votes** (recommended for
  reviewers). Uses the LoRe basis, PRISM weights, and per-user
  C016/C020 weights that produced the paper, all tracked in
  `experiments/checkpoints/`. Skips PRISM data prep, V/W training, and
  70B HistLlama generation — just runs the vote.
- **B — full pipeline from scratch**. Retrains everything, including the
  LoRe basis on PRISM and the synthetic preferences via 70B HistLlama.
  Produces *comparable* but **not** byte-identical votes (see "Why B
  diverges" below).

Both paths share the same input slate (`experiments/query_responses*.jsonl`),
seed (`42`), and aggregation methods. Path A always yields byte-identical
rankings to the tracked `experiments/vote_C016_C020{,_simple}/audit_log.json`.

### A. Exact-reproduction path

**Prerequisites**: one CUDA GPU with ≥16 GB free for the Skywork-Reward
embedding of the candidate response slates. No NAS, PRISM, or 70B model
required. Total wall time: a couple of minutes.

```bash
bash experiments/scripts/reproduce_paper_votes.sh
```

That wrapper runs both votes — `run_vote_C016_C020.sh` (regular slate)
and `run_vote_C016_C020_simple.sh` (yes/no slate) — against the
repo-tracked checkpoints:

```
experiments/checkpoints/V_K8.pt                            # LoRe basis (4096 × 8)
experiments/checkpoints/W_seen_K8.pt                       # PRISM jury voters (182 × 8)
experiments/checkpoints/W_adapted_hist_C016_C020_filtered.pt  # 20 historical voters
```

Each script writes `audit_log.json`, `vote_results.json` (regular slate
only), `vote_analysis.json`, and `vote_report.txt` under
`experiments/vote_C016_C020{,_simple}/`. The new `audit_log.json`'s
`sampled_user_ids`, `per_voter_rankings`, `aggregations`, and
`average_ranks` will match the tracked versions byte-for-byte; only the
`jury_manifest.path`, `embeddings_hash`, and timestamps differ.

You can rebuild the W_adapted checkpoint yourself before voting if you
want to verify that step too:

```bash
# Refits W_adapted_hist_C016_C020_filtered.pt against the tracked V and the
# tracked filtered prefs. Cosine similarity to the shipped W is 1.0 across
# all 20 users; vote rankings remain byte-identical.
bash experiments/scripts/train_user_weights_C016_C020.sh
bash experiments/scripts/reproduce_paper_votes.sh
```

### B. Full pipeline from scratch

**Prerequisites**:
- PRISM raw data (downloaded automatically by `apa.load_prism` from
  HuggingFace; written under `$NAS_BASE/data/prism/`).
- Write access to `$NAS_BASE` (`/nas/ucb/rachel/APA` by default; see
  `apa/config.py`).
- 1× A100/A6000 (≥40 GB) for steps 1 and 3–6.
- 4× A100/A6000 (each ≥35 GB free) for step 2 (70B HistLlama via vLLM
  with `tensor_parallel_size=4`).

#### Step 1 — Train the LoRe basis on PRISM

```bash
uv run python -m apa.load_prism --split both
uv run python -m apa.train_lore_bases --K_list 0,1,8
```

Writes `V_K{0,1,8}.pt` and `W_seen_K{0,1,8}.pt` under `$NAS_BASE/models/`.
`V_K8.pt` is the basis the rest of the pipeline reuses.

#### Step 2 — Generate synthetic preferences for C016 and C020

```bash
bash experiments/scripts/generate_prefs_C016_C020.sh
```

Reads:
- `experiments/chosen_questions.jsonl` — 500 PRISM questions where moral
  consensus is expected to vary across centuries (produced by
  `scripts/select_time_varying_questions.py`).
- `experiments/profiles_C016_C020.jsonl` — the 20 paper personas (10
  from C016 + 10 from C020).

Writes `hist_prefs_C{016,020}.jsonl` (eval_prefs format) and
`hist_prefs_C{016,020}_raw.json` (full reasoning + logprobs) under
`experiments/synthetic_prefs_C016_C020/`. Each preference pair is judged
via a two-stage CoT chat-template flow (stage 1 reasoning, stage 2
guided-decode commit to `{"X","Y"}`); both orderings are averaged to
cancel the model's letter-prior.

#### Step 3 — Filter the synthetic preferences

```bash
uv run python -m experiments.filter_output filter \
    --input  experiments/synthetic_prefs_C016_C020/hist_prefs_C016_raw.json \
             experiments/synthetic_prefs_C016_C020/hist_prefs_C020_raw.json \
    --output experiments/synthetic_prefs_C016_C020/hist_prefs_all_filtered.jsonl \
    --min-records-per-user 5
```

#### Step 4 — Few-shot LoRe adaptation: fit per-user W vectors

```bash
bash experiments/scripts/train_user_weights_C016_C020.sh
```

Defaults to fitting against `experiments/checkpoints/V_K8.pt`. Override
with `V_CHECKPOINT=$NAS_BASE/models/V_K8.pt bash …` to use the V you
just trained in step 1.

#### Step 5 — Hold the democratic vote

```bash
bash experiments/scripts/run_vote_C016_C020.sh
bash experiments/scripts/run_vote_C016_C020_simple.sh
```

Each script defaults to the repo-tracked checkpoints; override with the
`V_CHECKPOINT` / `PRISM_USERS` / `ADAPTED` env vars to use freshly
trained ones.

#### Step 6 — Render figures

```bash
uv run python -m experiments.figs all
```

Writes `experiments/figs/{user_weights_grid,jury_agreement_heatmap}.{png,pdf}`.

#### Why B diverges from the paper

Path B regenerates the LoRe basis, which means the PRISM jury pool
changes too. The original paper's `V_K8.pt` was trained against an
upstream LoRe-supplied PRISM split with **182 seen users**; the current
`apa/load_prism.py` derives the seen/unseen split locally from the raw
PRISM conversations (`min_dialogs > 5`, `seen_ratio = 0.8`) and yields
**1030 seen users**. With the same seed but a 5.6× larger pool,
`random.sample(prism_pool, 10)` lands on different voters, so the
audit logs in path B will diverge from the in-repo ones — even though
the methodology is unchanged. The 70B preference generation itself is
near-deterministic on the chosen questions (verified: 100% match on
`chosen` text in our reruns), and the filter output is byte-identical
given the same input.

If you want path B to land *closer* to the paper, you can shrink the
seen pool by setting a higher `min_dialogs` threshold in
`apa/load_prism.py:split_users` before retraining V.

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
