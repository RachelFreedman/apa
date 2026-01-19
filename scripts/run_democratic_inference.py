#!/usr/bin/env python3
"""
Run democratic inference.

This script runs the full democratic inference pipeline:
1. Generates diverse responses using the base LLM
2. Samples user models from PRISM and historical users
3. Collects rankings from each user
4. Aggregates rankings democratically
5. Returns the winning response

Usage:
    python scripts/run_democratic_inference.py --query "What is the meaning of life?"
    python scripts/run_democratic_inference.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig, InferenceConfig
from apa.inference.democratic_inference import DemocraticInference


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run democratic inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to run inference on",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of responses to generate",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=10,
        help="Number of voters to sample",
    )
    parser.add_argument(
        "--lore_checkpoint",
        type=str,
        default=None,
        help="Path to LoRe model checkpoint",
    )
    parser.add_argument(
        "--prism_users",
        type=str,
        default=None,
        help="Path to PRISM user vectors (W_lore_seen.pt)",
    )
    parser.add_argument(
        "--historical_dir",
        type=str,
        default=None,
        help="Directory with historical user vectors",
    )
    parser.add_argument(
        "--generate_strategy",
        type=str,
        default="temperature_sampling",
        help="Response generation strategy",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="random",
        help="User sampling strategy",
    )
    parser.add_argument(
        "--aggregate_strategy",
        type=str,
        default="borda_count",
        help="Ranking aggregation strategy",
    )
    parser.add_argument(
        "--show_all",
        action="store_true",
        help="Show all responses and rankings",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.query and not args.interactive:
        print("Error: Either --query or --interactive is required")
        sys.exit(1)

    # Configure environment
    configure_environment()
    dataset_config = DatasetConfig()

    # Set up paths
    if args.lore_checkpoint:
        lore_checkpoint = Path(args.lore_checkpoint)
    else:
        lore_checkpoint = dataset_config.checkpoints_dir / "lore_K8_best.pt"

    if args.prism_users:
        prism_users = Path(args.prism_users)
    else:
        prism_users = dataset_config.checkpoints_dir / "W_lore_seen_8_10000.0.pt"

    if args.historical_dir:
        historical_dir = Path(args.historical_dir)
    else:
        historical_dir = dataset_config.checkpoints_dir / "historical"

    # Check if checkpoints exist
    if not lore_checkpoint.exists():
        print(f"ERROR: LoRe checkpoint not found: {lore_checkpoint}")
        print("Please train LoRe first using: python scripts/train_lore_prism.py")
        sys.exit(1)

    # Create inference config
    inference_config = InferenceConfig(
        k_responses=args.k,
        m_voters=args.m,
        generate_strategy=args.generate_strategy,
        sample_strategy=args.sample_strategy,
        aggregate_strategy=args.aggregate_strategy,
    )

    print("\n" + "="*60)
    print("Democratic Inference")
    print("="*60)
    print(f"LoRe checkpoint: {lore_checkpoint}")
    print(f"PRISM users: {prism_users if prism_users.exists() else 'Not found'}")
    print(f"Historical users: {historical_dir if historical_dir.exists() else 'Not found'}")
    print(f"K responses: {args.k}")
    print(f"M voters: {args.m}")
    print(f"Generate strategy: {args.generate_strategy}")
    print(f"Sample strategy: {args.sample_strategy}")
    print(f"Aggregate strategy: {args.aggregate_strategy}")
    print("="*60 + "\n")

    # Create inference pipeline
    inference = DemocraticInference.from_checkpoints(
        lore_checkpoint=lore_checkpoint,
        prism_users_path=prism_users if prism_users.exists() else None,
        historical_dir=historical_dir if historical_dir.exists() else None,
        config=inference_config,
    )

    print(f"Total voters: {len(inference.voter_pool.get_all_user_ids())}\n")

    if args.interactive:
        inference.run_interactive()
    else:
        # Run single query
        print(f"Query: {args.query}\n")
        print("Running democratic inference...")

        result = inference(args.query)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)

        if args.show_all:
            print("\nAll generated responses:")
            for i, resp in enumerate(result.responses):
                print(f"\n--- Response {i+1} ---")
                print(resp[:500] + "..." if len(resp) > 500 else resp)

            print("\n\nRankings from sampled voters:")
            for user_id, ranking in result.rankings.items():
                print(f"  {user_id}: {ranking}")

            print(f"\nAggregate ranking: {result.aggregate_ranking}")

        print(f"\n{'='*60}")
        print(f"WINNER: Response #{result.winner_idx + 1}")
        print(f"{'='*60}")
        print(result.winner_response)
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
