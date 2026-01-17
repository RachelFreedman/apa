#!/usr/bin/env python3
"""
Prepare PRISM embeddings for LoRe training.

This script loads the PRISM pairwise dataset and generates embeddings
using the Skywork-Reward model (following the LoRe paper methodology).

The script generates embeddings for each chosen/rejected pair by:
1. Formatting the conversation with chat template
2. Extracting the last token's hidden state from the model
3. Storing embeddings in a format compatible with train_lore_prism.py

Usage:
    python scripts/prepare_prism_embeddings.py
    python scripts/prepare_prism_embeddings.py --n_samples 100  # For testing
    python scripts/prepare_prism_embeddings.py --split train    # Train only
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig, LoReConfig, DATA_DIR


def log(message: str) -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    lore_config = LoReConfig()

    parser = argparse.ArgumentParser(
        description="Prepare PRISM embeddings for LoRe training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limit to first N samples (for testing)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which split(s) to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=lore_config.embedding_model,
        help="Embedding model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (uses default data/prism if not specified)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Input data directory containing parquet files",
    )
    return parser.parse_args()


def generate_embeddings(
    dataset,
    model,
    tokenizer,
    device: str,
    output_path: Path,
    n_samples: int | None = None,
) -> list[dict]:
    """
    Generate embeddings for the dataset.

    Following LoRe methodology: extracts last hidden state of last token.

    Args:
        dataset: HuggingFace dataset with PRISM data
        model: Pre-loaded embedding model
        tokenizer: Pre-loaded tokenizer
        device: Device string
        output_path: Path to save embeddings
        n_samples: Optional limit on samples

    Returns:
        List of examples with embeddings added to extra_info
    """
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    dataset_size = len(dataset)
    log(f"Generating embeddings for {dataset_size} examples...")

    start_time = time.time()
    embeddings_data = []

    for idx, entry in enumerate(tqdm(dataset, desc="Generating embeddings", total=dataset_size), 1):
        # Convert to dict if needed
        if not isinstance(entry, dict):
            entry = dict(entry)
        else:
            # Make a copy to avoid modifying the original
            entry = dict(entry)
            if "extra_info" in entry and isinstance(entry["extra_info"], dict):
                entry["extra_info"] = dict(entry["extra_info"])

        extra_info = entry.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        entry["extra_info"] = extra_info

        prompt = entry.get("prompt", [])

        # Build chosen and rejected conversations
        chosen_utterance = extra_info.get("chosen_utterance", "")
        rejected_utterance = extra_info.get("rejected_utterance", "")

        chosen = [{"content": chosen_utterance, "role": "assistant"}]
        rejected = [{"content": rejected_utterance, "role": "assistant"}]
        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected

        # Generate chosen embedding
        try:
            tokenized = tokenizer.apply_chat_template(
                chosen_conv,
                tokenize=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["chosen_conv_embedding"] = embedding

            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        except Exception as e:
            error_str = str(e).lower()
            is_oom = ("out of memory" in error_str or
                     isinstance(e, torch.cuda.OutOfMemoryError) or
                     ("cuda" in error_str and "memory" in error_str))

            if is_oom:
                log(f"CUDA OOM at example {idx} (chosen), skipping...")
                entry["extra_info"]["chosen_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                log(f"Error at example {idx} (chosen): {e}")
                raise

        # Generate rejected embedding
        try:
            tokenized = tokenizer.apply_chat_template(
                rejected_conv,
                tokenize=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["rejected_conv_embedding"] = embedding

            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        except Exception as e:
            error_str = str(e).lower()
            is_oom = ("out of memory" in error_str or
                     isinstance(e, torch.cuda.OutOfMemoryError) or
                     ("cuda" in error_str and "memory" in error_str))

            if is_oom:
                log(f"CUDA OOM at example {idx} (rejected), skipping...")
                entry["extra_info"]["rejected_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                log(f"Error at example {idx} (rejected): {e}")
                raise

        embeddings_data.append(entry)

        # Periodic cache clearing
        if idx % 100 == 0 and device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Log progress every 10%
        if idx % max(1, dataset_size // 10) == 0 or idx == dataset_size:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (dataset_size - idx) / rate if rate > 0 else 0
            log(f"Progress: {idx}/{dataset_size} ({100*idx/dataset_size:.1f}%) | "
                f"Rate: {rate:.1f} ex/s | ETA: {remaining:.1f}s")

    total_time = time.time() - start_time
    log(f"Completed {dataset_size} examples in {total_time:.1f}s "
        f"({total_time/60:.1f} min)")

    # Save embeddings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_data, output_path)
    log(f"Saved embeddings to {output_path}")

    return embeddings_data


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure environment
    configure_environment()

    # Set CUDA memory optimization
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log("=" * 60)
    log("PRISM Embedding Generation")
    log("=" * 60)
    log(f"Model: {args.model}")
    log(f"Device: {args.device}")
    log(f"Split: {args.split}")
    if args.n_samples:
        log(f"Limiting to {args.n_samples} samples per split")

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = DATA_DIR / "prism"

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DATA_DIR / "prism"

    log(f"Data directory: {data_dir}")
    log(f"Output directory: {output_dir}")

    # Load model and tokenizer
    log("Loading model and tokenizer...")
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU Memory: {total_mem:.2f} GB total")

    try:
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if args.device.startswith("cuda") else None,
            attn_implementation="eager",
            num_labels=1,
            low_cpu_mem_usage=True,
        )
        log(f"Model loaded on {args.device}")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        log(f"CUDA error during model loading: {e}")
        log("Falling back to CPU...")
        args.device = "cpu"
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map=None,
            attn_implementation="eager",
            num_labels=1,
            low_cpu_mem_usage=True,
        )
        model = model.to(args.device)
        log("Model loaded on CPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load datasets
    log("Loading datasets...")
    from datasets import load_dataset

    # Find parquet files
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_path}")

    # Process train split
    if args.split in ["train", "both"]:
        log("=" * 60)
        log("Processing TRAIN split")
        log("=" * 60)
        train_dataset = load_dataset("parquet", data_files=str(train_path))["train"]
        log(f"Train dataset: {len(train_dataset)} examples")

        train_output = output_dir / "train_embeddings.pkl"
        generate_embeddings(
            train_dataset,
            model,
            tokenizer,
            args.device,
            train_output,
            args.n_samples,
        )

    # Process test split
    if args.split in ["test", "both"]:
        log("=" * 60)
        log("Processing TEST split")
        log("=" * 60)
        test_dataset = load_dataset("parquet", data_files=str(test_path))["train"]
        log(f"Test dataset: {len(test_dataset)} examples")

        test_output = output_dir / "test_embeddings.pkl"
        generate_embeddings(
            test_dataset,
            model,
            tokenizer,
            args.device,
            test_output,
            args.n_samples,
        )

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log("=" * 60)
    log("Done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
