#!/usr/bin/env python3
"""
Prepare PRISM embeddings for LoRe training.

This script loads the PRISM pairwise dataset and generates sentence embeddings
for all prompts and responses using sentence-transformers.

Usage:
    python scripts/prepare_prism_embeddings.py
    python scripts/prepare_prism_embeddings.py --n_samples 1000  # For testing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig, LoReConfig, DATA_DIR
from apa.data.prism_loader import load_prism_pairwise
from apa.utils.embedding_utils import (
    get_embedding_model,
    embed_response_pairs,
    save_embeddings,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence transformer model to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (uses default if not specified)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure environment
    configure_environment()

    # Load PRISM data
    df = load_prism_pairwise(n_samples=args.n_samples)

    # Get texts
    prompts = df['prompt'].tolist()
    responses_1 = df['response_1'].tolist()
    responses_2 = df['response_2'].tolist()

    print(f"\nDataset statistics:")
    print(f"  Total pairs: {len(df)}")
    if 'user_id' in df.columns:
        print(f"  Unique users: {df['user_id'].nunique()}")

    # Load embedding model
    model = get_embedding_model(args.model)

    # Generate embeddings
    print(f"\nGenerating embeddings with {args.model}...")
    embeddings = embed_response_pairs(
        prompts=prompts,
        responses_1=responses_1,
        responses_2=responses_2,
        model=model,
        batch_size=args.batch_size,
    )

    # Add metadata
    embeddings['model_name'] = args.model
    embeddings['n_samples'] = len(df)

    # Include user_ids and labels for convenience
    # Use user_id if available, otherwise fall back to interaction_id
    if 'user_id' in df.columns:
        embeddings['user_ids'] = df['user_id'].values
    elif 'interaction_id' in df.columns:
        embeddings['user_ids'] = df['interaction_id'].values
        print("Note: Using 'interaction_id' as user identifier")
    embeddings['labels'] = (df['human_preferred'].astype(str) == '2').astype(int).values
    embeddings['question_ids'] = df['question_id'].values

    # Save embeddings
    if args.output:
        output_path = Path(args.output)
    else:
        dataset_config = DatasetConfig()
        output_path = dataset_config.embeddings_path

    save_embeddings(embeddings, output_path)

    print(f"\nEmbeddings saved to: {output_path}")
    print(f"  Shape: {embeddings['response_1_embeddings'].shape}")


if __name__ == "__main__":
    main()
