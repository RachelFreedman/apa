#!/usr/bin/env python3
"""
Generate historical preferences using HistLlama models.

This script generates preference data from historical LLM perspectives,
which can then be used by train_historical_users.py to train user vectors.

Usage:
    python scripts/generate_historical_preferences.py --century C013
    python scripts/generate_historical_preferences.py --century C017 --n_questions 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import (
    configure_environment,
    DatasetConfig,
    HistLlamaConfig,
)
from apa.data.prism_loader import load_prism_pairwise
from apa.historical.hist_llama import load_hist_llama, century_to_name
from apa.historical.preference_gen import generate_historical_preferences
from apa.levers import lever_select_questions


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate historical preferences using HistLlama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--century",
        type=str,
        required=True,
        choices=HistLlamaConfig.VALID_CENTURIES,
        help="Century to generate preferences for (e.g., C013)",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=500,
        help="Number of questions to use",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of times to run each preference comparison",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="8B",
        choices=["8B", "70B"],
        help="HistLlama model size",
    )
    parser.add_argument(
        "--user_profile",
        type=str,
        default=None,
        help="Optional user profile description to condition preferences",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for preference files",
    )
    parser.add_argument(
        "--question_strategy",
        type=str,
        default="random_subset",
        help="Question selection strategy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question selection",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure environment
    configure_environment()
    dataset_config = DatasetConfig()

    print(f"\n{'='*60}")
    print(f"Generating Historical Preferences for {century_to_name(args.century)}")
    print(f"{'='*60}\n")

    # Set up output directory
    output_dir = Path(args.output_dir) if args.output_dir else dataset_config.checkpoints_dir
    output_dir = output_dir / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PRISM questions
    df = load_prism_pairwise()
    print(f"Loaded {len(df)} PRISM questions")

    # Select subset using lever
    config = {'questions': args.question_strategy, 'seed': args.seed}
    selected_df = lever_select_questions(df, args.n_questions, config)
    print(f"Selected {len(selected_df)} questions using '{args.question_strategy}' strategy")

    # Convert to list of dicts
    questions = [
        {
            'question_id': row['question_id'],
            'prompt': row['prompt'],
            'response_1': row['response_1'],
            'response_2': row['response_2'],
        }
        for _, row in selected_df.iterrows()
    ]

    # Load HistLlama model
    print(f"\nLoading HistLlama {args.model_size} for {args.century}...")
    hist_model, hist_tokenizer = load_hist_llama(
        century=args.century,
        size=args.model_size,
    )

    # Generate preferences
    print(f"\nGenerating preferences with {args.n_runs} run(s) per question...")
    hist_config = HistLlamaConfig(size=args.model_size, century=args.century)

    preferences = generate_historical_preferences(
        hist_model,
        hist_tokenizer,
        questions,
        config=hist_config,
        user_profile=args.user_profile,
        n_runs=args.n_runs,
        show_progress=True,
    )

    # Compute statistics
    valid_count = sum(1 for p in preferences if p.get('final_preference') in ['1', '2'])
    consistencies = [p['consistency'] for p in preferences]
    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0

    print(f"\nResults:")
    print(f"  Total questions: {len(preferences)}")
    print(f"  Valid preferences: {valid_count} ({valid_count/len(preferences)*100:.1f}%)")
    print(f"  Average consistency: {avg_consistency:.2%}")

    # Build output filename
    user_id = f"historical_{args.century}"
    if args.user_profile:
        user_id += f"_{hash(args.user_profile) % 10000}"

    # Save preferences
    output_data = {
        'century': args.century,
        'user_profile': args.user_profile,
        'n_questions': len(questions),
        'n_runs': args.n_runs,
        'model_size': args.model_size,
        'question_strategy': args.question_strategy,
        'seed': args.seed,
        'valid_count': valid_count,
        'avg_consistency': avg_consistency,
        'questions': questions,
        'preferences': preferences,
    }

    output_path = output_dir / f"preferences_{user_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved preferences to: {output_path}")

    print(f"\n{'='*60}")
    print("Preference generation complete!")
    print(f"{'='*60}\n")
    print(f"Next step: Train user vector with:")
    print(f"  python scripts/train_historical_users.py --century {args.century} --preferences_file {output_path}")


if __name__ == "__main__":
    main()
