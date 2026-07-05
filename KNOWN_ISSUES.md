# Known Issues

These are pre-existing, behavior-affecting issues identified during the
refactor. They were intentionally **left unchanged** so the refactor stays
strictly behavior-preserving (see the refactor plan). Fix them in dedicated,
separately-verified changes — not as part of a "no functional change" refactor.

## 1. OOM-detection operator precedence (`apa/load_prism.py`)

In `_embed_conversation` (formerly inlined in `_generate_embeddings`):

```python
if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
```

`and` binds tighter than `or`, so this parses as
`A or (B and C)` — i.e. *any* error containing "out of memory" is treated as
OOM, and other errors only count when they contain both "cuda" and "memory".
This is likely not the intended grouping. The condition is preserved verbatim
in the extracted helper.

## 2. Non-deterministic historical user IDs (`apa/historical_prefs.py`)

`cmd_generate` / `cmd_train` build user IDs with
`hash(user_profile) % 10000`. Python's builtin `hash()` for `str` is salted per
process (`PYTHONHASHSEED`), so the same profile yields different IDs across
runs, hurting reproducibility. A stable hash (e.g. `hashlib.md5`) would fix it.
Note the checkpoint *filename* is `W_{century}.pt` (deterministic); only the
embedded `user_id` field varies.

## 3. `embed_texts` does not actually batch (`apa/train_lore_bases.py`)

`embed_texts` iterates in `batch_size` windows but then embeds each text
individually via `_extract_embedding`, so `batch_size` provides no batching
speedup. Real batching (padded tokenization + a single forward pass) would be
faster but changes numerics slightly, so it is out of scope here.

## 4. Broad exception swallowing (`apa/historical_prefs.py`)

`find_latest_model_version` catches bare `Exception` and silently falls back to
`"v0.2"`, which can mask real HuggingFace/network errors.

## 5. `torch.load(weights_only=False)` FutureWarnings

Several loaders (`LoReRewardModel.load`, `VoterPool.load_prism_users`,
`load_historical_users`) call `torch.load` without `weights_only`. PyTorch will
flip the default in a future release; these should pass `weights_only=...`
explicitly once the loaded checkpoint contents are audited.
