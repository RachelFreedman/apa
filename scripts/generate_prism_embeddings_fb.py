#!/usr/bin/env python3
"""
Generate PRISM embeddings using FB's exact methodology.

This script is a direct copy of https://github.com/facebookresearch/LoRe/blob/main/PRISM/generate-prism-embeddings.py
with only path adaptations for our directory structure.

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Usage:
    python scripts/generate_prism_embeddings_fb.py
    python scripts/generate_prism_embeddings_fb.py --device cuda:3
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use NAS storage for data
DATA_DIR = Path("/nas/ucb/rachel/APA/data/prism_fb")
HF_CACHE_DIR = Path("/nas/ucb/rachel/APA/hf_cache")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PRISM embeddings - FB replication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing prepared PRISM data",
    )
    return parser.parse_args()


def generate_prism_embeddings(
    dataset,
    model,
    tokenizer,
    device,
    output_path
):
    """
    Generate embeddings for each user in the dataset.

    Structure: chosen_embeddings[user_id][dialog_id] = [embedding_turn_0, ..., embedding_turn_n]

    Alternate:
    embeddings[user_id][dialog_id][turn_nb][chosen/rejected][seen : True or False][train : True or False]

    Later for given user_id (and specific chosen/rejected value, seen True or False value)
    gather all chosen embeddings as a tensor
    """
    embeddings_data = []
    skipped = 0

    for entry in tqdm(dataset, desc="Generating embeddings"):
        # Skip entries without chosen/rejected utterances (last turn of dialog)
        if 'chosen_utterance' not in entry["extra_info"]:
            skipped += 1
            continue

        user_id = entry["extra_info"]["user_id"]
        dialog_id = entry["extra_info"]["dialog_id"]
        prompt = entry["prompt"]

        # FB's approach: single chosen, but rejected can be a list
        # We use the first rejected for the difference vector
        chosen = [{"content": entry["extra_info"]["chosen_utterance"], "role": "assistant"}]
        rejected_list = entry["extra_info"]["rejected_utterance"]

        # Handle both single string and list of strings for rejected
        if isinstance(rejected_list, list):
            rejected_text = rejected_list[0] if rejected_list else ""
        else:
            rejected_text = rejected_list

        rejected = [{"content": rejected_text, "role": "assistant"}]

        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected

        # Tokenize and embed chosen conversation
        tokenized = tokenizer.apply_chat_template(
            chosen_conv,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(tokenized)
            embedding = output.last_hidden_state[0, -1].cpu()  # [hidden_dim]

        entry["extra_info"]["chosen_conv_embedding"] = embedding

        # Tokenize and embed rejected conversation
        tokenized = tokenizer.apply_chat_template(
            rejected_conv,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(tokenized)
            embedding = output.last_hidden_state[0, -1].cpu()  # [hidden_dim]

        entry["extra_info"]["rejected_conv_embedding"] = embedding

        embeddings_data.append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_data, output_path)
    print(f"Saved embeddings to {output_path}")
    print(f"  Total entries: {len(embeddings_data)}")
    print(f"  Skipped (no chosen/rejected): {skipped}")

    return embeddings_data


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"\n{'='*60}")
    print("PRISM Embedding Generation - FB Replication")
    print(f"{'='*60}")
    print(f"  Data directory: {data_dir}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # Check that prepared data exists
    train_parquet = data_dir / "train.parquet"
    test_parquet = data_dir / "test.parquet"

    if not train_parquet.exists() or not test_parquet.exists():
        print("ERROR: Prepared data not found!")
        print(f"  Expected: {train_parquet}")
        print(f"  Expected: {test_parquet}")
        print("\nPlease run prepare_prism_fb.py first.")
        sys.exit(1)

    # Load model and tokenizer (FB uses AutoModel, not AutoModelForSequenceClassification)
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

    print(f"Loading model: {model_name}")
    print("  Note: Using AutoModel (not AutoModelForSequenceClassification) per FB")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
        num_labels=1,
        cache_dir=str(HF_CACHE_DIR),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(HF_CACHE_DIR),
    )

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset("parquet", data_files=str(train_parquet))["train"]
    test_dataset = load_dataset("parquet", data_files=str(test_parquet))["train"]

    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Test examples: {len(test_dataset)}")

    # Generate embeddings
    print("\nGenerating train embeddings...")
    train_embeddings = generate_prism_embeddings(
        train_dataset, model, tokenizer, args.device,
        str(data_dir / "train_embeddings.pkl")
    )

    print("\nGenerating test embeddings...")
    test_embeddings = generate_prism_embeddings(
        test_dataset, model, tokenizer, args.device,
        str(data_dir / "test_embeddings.pkl")
    )

    print(f"\n{'='*60}")
    print("Embedding Generation Complete!")
    print(f"{'='*60}")
    print(f"  Train embeddings: {len(train_embeddings)}")
    print(f"  Test embeddings: {len(test_embeddings)}")
    print(f"  Output: {data_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
