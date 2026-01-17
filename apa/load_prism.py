"""
PRISM dataset loading, embedding generation, and checkpoint management.

This module provides:
- PRISMDataset: PyTorch Dataset for pairwise preferences
- load_prism_pairwise: Load PRISM CSV data
- group_embeddings_by_user: Group embeddings for LoRe training
- CheckpointManager: Long-running training state management
- CLI for generating embeddings

CLI Usage:
    python -m apa.load_prism --split both
    python -m apa.load_prism --split train --n_samples 100
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =============================================================================
# Data Loading
# =============================================================================

def get_user_column(df: pd.DataFrame) -> str | None:
    """Find the user identifier column ('user_id' or 'interaction_id')."""
    if 'user_id' in df.columns:
        return 'user_id'
    if 'interaction_id' in df.columns:
        return 'interaction_id'
    return None


def get_unique_users(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique user identifiers from DataFrame."""
    user_col = get_user_column(df)
    if user_col is None:
        return []
    return sorted(df[user_col].unique().tolist())


def load_prism_pairwise(
    path: Path | str | None = None,
    min_pairs_per_user: int = 0,
) -> pd.DataFrame:
    """
    Load PRISM pairwise preference data.

    Args:
        path: Path to CSV. If None, uses default from config.
        min_pairs_per_user: Filter to users with at least this many pairs

    Returns:
        DataFrame with user_id, question_id, prompt, response_1, response_2, etc.
    """
    if path is None:
        from apa.config import HISTORICAL_PREFS_DATA
        path = HISTORICAL_PREFS_DATA / "prism" / "questions_pairwise.csv"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PRISM pairwise data not found at {path}")

    df = pd.read_csv(path)

    if 'user_id' not in df.columns and 'interaction_id' in df.columns:
        df['user_id'] = df['interaction_id']

    if min_pairs_per_user > 0:
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_pairs_per_user].index
        df = df[df['user_id'].isin(valid_users)]
        print(f"Filtered to {len(valid_users)} users with >= {min_pairs_per_user} pairs")

    return df


# =============================================================================
# Dataset Class
# =============================================================================

class PRISMDataset(Dataset):
    """PyTorch Dataset for PRISM pairwise preference data."""

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        labels: np.ndarray,
        user_ids: np.ndarray | None = None,
    ):
        """
        Args:
            embeddings: Dict with 'response_1_embeddings' and 'response_2_embeddings'
            labels: Array of shape (n_samples,) with 0 or 1 indicating preference
            user_ids: Optional array of user IDs for each sample
        """
        self.response_1 = torch.tensor(embeddings['response_1_embeddings'], dtype=torch.float32)
        self.response_2 = torch.tensor(embeddings['response_2_embeddings'], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        assert len(self.response_1) == len(self.response_2) == len(self.labels)

        if user_ids is not None:
            unique_users = sorted(set(user_ids))
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
            self.user_indices = torch.tensor(
                [self.user_to_idx[uid] for uid in user_ids],
                dtype=torch.long
            )
            self._n_users = len(unique_users)
        else:
            self.user_to_idx = None
            self.idx_to_user = None
            self.user_indices = None
            self._n_users = 1

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            'response_1_embedding': self.response_1[idx],
            'response_2_embedding': self.response_2[idx],
            'label': self.labels[idx],
        }
        if self.user_indices is not None:
            item['user_idx'] = self.user_indices[idx]
        return item

    @property
    def embedding_dim(self) -> int:
        return self.response_1.shape[1]

    @property
    def n_users(self) -> int:
        return self._n_users

    def get_user_data(self, user_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all data for a specific user."""
        if self.user_to_idx is None:
            raise ValueError("Dataset was not initialized with user_ids")
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            raise KeyError(f"User {user_id} not found in dataset")
        mask = self.user_indices == user_idx
        return self.response_1[mask], self.response_2[mask], self.labels[mask]


# =============================================================================
# Embedding Grouping
# =============================================================================

def group_embeddings_by_user(
    train_embeddings: list[dict],
    test_embeddings: list[dict],
    device: str | torch.device = "cuda:0",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Group embeddings by user and compute difference (chosen - rejected).

    This follows the exact logic from LoRe/train_basis.py.

    Returns:
        (train_seen, train_unseen, test_seen, test_unseen) - lists of per-user tensors
    """
    def log(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def process_dataset(dataset, seen_value, split_name):
        split_label = "seen" if seen_value else "unseen"
        log(f"Processing {split_name} {split_label} dataset ({len(dataset)} examples)...")
        start_time = time.time()
        grouped = defaultdict(lambda: {"embeddings": []})
        skipped = 0
        processed = 0

        for idx, example in enumerate(dataset):
            extra_info = example.get("extra_info", {})
            if extra_info.get("seen") == seen_value and extra_info.get("split") == split_name:
                user_id = extra_info.get("user_id")
                if user_id:
                    chosen_emb = extra_info.get("chosen_conv_embedding")
                    rejected_emb = extra_info.get("rejected_conv_embedding")
                    if chosen_emb is None or rejected_emb is None:
                        skipped += 1
                        continue
                    chosen = torch.tensor(chosen_emb, dtype=torch.float32, device=device)
                    rejected = torch.tensor(rejected_emb, dtype=torch.float32, device=device)
                    grouped[user_id]["embeddings"].append(chosen - rejected)
                    processed += 1

            if (idx + 1) % max(1, len(dataset) // 10) == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(dataset) - idx - 1) / rate if rate > 0 else 0
                log(f"  Progress: {idx+1}/{len(dataset)} ({100*(idx+1)/len(dataset):.1f}%) | "
                    f"Processed: {processed} | Skipped: {skipped} | ETA: {remaining:.1f}s")

        log(f"  Stacking embeddings for {len(grouped)} users...")
        sorted_grouped = []
        count = 0
        for user_id in sorted(grouped.keys()):
            count += len(grouped[user_id]["embeddings"])
            sorted_grouped.append(torch.stack(grouped[user_id]["embeddings"]))

        elapsed = time.time() - start_time
        log(f"  Completed {split_name} {split_label}: {count} embeddings from {len(grouped)} users "
            f"({processed} processed, {skipped} skipped) in {elapsed:.1f}s")
        return sorted_grouped

    log("=" * 60)
    log("Grouping embeddings by user...")
    log("=" * 60)
    grouping_start = time.time()

    train_seen = process_dataset(train_embeddings, seen_value=True, split_name="train")
    train_unseen = process_dataset(train_embeddings, seen_value=False, split_name="train")
    test_seen = process_dataset(test_embeddings, seen_value=True, split_name="test")
    test_unseen = process_dataset(test_embeddings, seen_value=False, split_name="test")

    grouping_time = time.time() - grouping_start
    log(f"Embedding grouping completed in {grouping_time:.1f}s ({grouping_time/60:.1f} min)")
    log("=" * 60)

    return train_seen, train_unseen, test_seen, test_unseen


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """Manages checkpointing for long-running training processes."""

    def __init__(
        self,
        checkpoint_dir: Path,
        name: str,
        checkpoint_interval: int = 100,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.name = name
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.name}_checkpoint.pt"

    def save_checkpoint(self, state: dict[str, Any], iteration: int) -> None:
        checkpoint = {'iteration': iteration, 'state': state}
        torch.save(checkpoint, self.checkpoint_path)
        print(f"[Checkpoint] Saved at iteration {iteration}")

    def load_checkpoint(self) -> tuple[dict[str, Any], int] | None:
        if not self.checkpoint_path.exists():
            return None
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        print(f"[Checkpoint] Loaded from iteration {checkpoint['iteration']}")
        return checkpoint['state'], checkpoint['iteration']

    def maybe_save(self, state: dict[str, Any], iteration: int, force: bool = False) -> None:
        if force or (iteration > 0 and iteration % self.checkpoint_interval == 0):
            self.save_checkpoint(state, iteration)

    def cleanup(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"[Checkpoint] Removed: {self.checkpoint_path}")


def save_with_symlink(
    data: pd.DataFrame | torch.Tensor | dict,
    nas_path: Path,
    local_path: Path | None = None,
    sep: str = '\t',
) -> None:
    """Save data to NAS and optionally create symlink in local directory."""
    nas_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(nas_path, sep=sep, index=False)
    elif isinstance(data, (torch.Tensor, dict)):
        torch.save(data, nas_path)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    print(f"Saved to {nas_path}")

    if local_path is not None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() or local_path.is_symlink():
            local_path.unlink()
        local_path.symlink_to(nas_path)
        print(f"Created symlink: {local_path} -> {nas_path}")


def load_from_nas(nas_path: Path) -> Any:
    """Load data from NAS path (auto-detects file type)."""
    if not nas_path.exists():
        raise FileNotFoundError(f"File not found: {nas_path}")

    suffix = nas_path.suffix.lower()
    if suffix in ['.pt', '.pth']:
        return torch.load(nas_path, map_location='cpu')
    elif suffix == '.pkl':
        import pickle
        with open(nas_path, 'rb') as f:
            return pickle.load(f)
    elif suffix in ['.csv', '.tsv']:
        sep = '\t' if suffix == '.tsv' else ','
        return pd.read_csv(nas_path, sep=sep)
    else:
        raise ValueError(f"Unknown file extension: {suffix}")


# =============================================================================
# CLI: Embedding Generation
# =============================================================================

def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _generate_embeddings(dataset, model, tokenizer, device: str, output_path: Path, n_samples: int | None = None) -> list[dict]:
    """Generate embeddings for the dataset (internal CLI helper)."""
    from tqdm import tqdm

    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    dataset_size = len(dataset)
    _log(f"Generating embeddings for {dataset_size} examples...")
    start_time = time.time()
    embeddings_data = []

    for idx, entry in enumerate(tqdm(dataset, desc="Generating embeddings", total=dataset_size), 1):
        if not isinstance(entry, dict):
            entry = dict(entry)
        else:
            entry = dict(entry)
            if "extra_info" in entry and isinstance(entry["extra_info"], dict):
                entry["extra_info"] = dict(entry["extra_info"])

        extra_info = entry.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        entry["extra_info"] = extra_info

        prompt = entry.get("prompt", [])
        chosen_utterance = extra_info.get("chosen_utterance", "")
        rejected_utterance = extra_info.get("rejected_utterance", "")

        chosen = [{"content": chosen_utterance, "role": "assistant"}]
        rejected = [{"content": rejected_utterance, "role": "assistant"}]
        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected

        # Generate chosen embedding
        try:
            tokenized = tokenizer.apply_chat_template(chosen_conv, tokenize=True, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["chosen_conv_embedding"] = embedding
            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
                _log(f"CUDA OOM at example {idx} (chosen), skipping...")
                entry["extra_info"]["chosen_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

        # Generate rejected embedding
        try:
            tokenized = tokenizer.apply_chat_template(rejected_conv, tokenize=True, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["rejected_conv_embedding"] = embedding
            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
                _log(f"CUDA OOM at example {idx} (rejected), skipping...")
                entry["extra_info"]["rejected_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

        embeddings_data.append(entry)

        if idx % 100 == 0 and device.startswith("cuda"):
            torch.cuda.empty_cache()

        if idx % max(1, dataset_size // 10) == 0 or idx == dataset_size:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (dataset_size - idx) / rate if rate > 0 else 0
            _log(f"Progress: {idx}/{dataset_size} ({100*idx/dataset_size:.1f}%) | Rate: {rate:.1f} ex/s | ETA: {remaining:.1f}s")

    total_time = time.time() - start_time
    _log(f"Completed {dataset_size} examples in {total_time:.1f}s ({total_time/60:.1f} min)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_data, output_path)
    _log(f"Saved embeddings to {output_path}")

    return embeddings_data


def main() -> None:
    """CLI entry point for embedding generation."""
    from apa.config import configure_environment, EMBEDDINGS_DIR, NAS_BASE

    parser = argparse.ArgumentParser(
        description="Prepare PRISM embeddings for LoRe training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Limit to first N samples (for testing)")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"], help="Which split(s) to process")
    parser.add_argument("--model", type=str, default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", help="Embedding model to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run model on")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (uses EMBEDDINGS_DIR if not specified)")
    parser.add_argument("--data_dir", type=str, default=None, help="Input data directory containing parquet files")
    args = parser.parse_args()

    configure_environment()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _log("=" * 60)
    _log("PRISM Embedding Generation")
    _log("=" * 60)
    _log(f"Model: {args.model}")
    _log(f"Device: {args.device}")
    _log(f"Split: {args.split}")
    if args.n_samples:
        _log(f"Limiting to {args.n_samples} samples per split")

    data_dir = Path(args.data_dir) if args.data_dir else NAS_BASE / "data" / "prism"
    output_dir = Path(args.output_dir) if args.output_dir else EMBEDDINGS_DIR

    _log(f"Data directory: {data_dir}")
    _log(f"Output directory: {output_dir}")

    _log("Loading model and tokenizer...")
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        _log(f"GPU Memory: {total_mem:.2f} GB total")

    try:
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto" if args.device.startswith("cuda") else None,
            attn_implementation="eager", num_labels=1, low_cpu_mem_usage=True,
        )
        _log(f"Model loaded on {args.device}")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        _log(f"CUDA error during model loading: {e}")
        _log("Falling back to CPU...")
        args.device = "cpu"
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map=None,
            attn_implementation="eager", num_labels=1, low_cpu_mem_usage=True,
        )
        model = model.to(args.device)
        _log("Model loaded on CPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    _log("Loading datasets...")
    from datasets import load_dataset

    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_path}")

    if args.split in ["train", "both"]:
        _log("=" * 60)
        _log("Processing TRAIN split")
        _log("=" * 60)
        train_dataset = load_dataset("parquet", data_files=str(train_path))["train"]
        _log(f"Train dataset: {len(train_dataset)} examples")
        _generate_embeddings(train_dataset, model, tokenizer, args.device, output_dir / "train.pkl", args.n_samples)

    if args.split in ["test", "both"]:
        _log("=" * 60)
        _log("Processing TEST split")
        _log("=" * 60)
        test_dataset = load_dataset("parquet", data_files=str(test_path))["train"]
        _log(f"Test dataset: {len(test_dataset)} examples")
        _generate_embeddings(test_dataset, model, tokenizer, args.device, output_dir / "test.pkl", args.n_samples)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log("=" * 60)
    _log("Done!")
    _log("=" * 60)


if __name__ == "__main__":
    main()
