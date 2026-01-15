#!/usr/bin/env python3
"""
Train LoRe model using Facebook's approach.

Implements the FB LoRe training protocol:
- 80/20 seen/unseen user split
- 50/50 train/test dialog split per user
- 20,000 iterations with lr=0.5
- Alternating minimization (W then V)
- Cosine similarity regularization with warmup
- Few-shot learning for unseen users
- Comprehensive logging

Usage:
    python scripts/train_lore_fb.py --rank 5
    python scripts/train_lore_fb.py --ranks 1 5 50
    python scripts/train_lore_fb.py --n_iterations 1000 --log_interval 100  # Quick test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig
from apa.data.user_splits import FBDataSplitter
from apa.utils.embedding_utils import load_embeddings
from apa.reward.lore_model import LoReRewardModel
from apa.reward.lore_fb_trainer import LoReFBTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LoRe model using Facebook's approach",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs='+',
        default=[1, 5, 50],
        help="Ranks to train (space-separated list)",
    )
    parser.add_argument(
        "--full_grid",
        action="store_true",
        help="Use full FB rank grid [0, 1, 5, 10, 15, 20, 25, 50]",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=20000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--fewshot_iterations",
        type=int,
        default=500,
        help="Number of few-shot adaptation iterations",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.5,
        help="Learning rate for optimizers",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10000.0,
        help="Regularization coefficient",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log every N iterations",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings file (uses default if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory (creates timestamped subdir)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--seen_user_ratio",
        type=float,
        default=0.8,
        help="Fraction of users for seen set",
    )
    parser.add_argument(
        "--dialog_train_ratio",
        type=float,
        default=0.5,
        help="Fraction of each user's dialogs for training",
    )
    return parser.parse_args()


def extract_v_sft(embeddings_path: Path, device: str) -> torch.Tensor:
    """
    Extract V_sft from the pretrained Skywork-Reward model.

    This follows the FB approach of finding the last linear layer in the model
    (which is the MLP down_proj in the final transformer block) and using its
    first column as the reference direction for regularization.
    """
    import os
    from transformers import AutoModel

    cache_dir = os.environ.get("HF_HOME", "/nas/ucb/rachel/APA/hf_cache")
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

    print(f"Loading model for V_sft extraction: {model_name}")

    # FB uses AutoModel (not AutoModelForSequenceClassification)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
        attn_implementation="eager",
        num_labels=1,
    )

    # FB approach: Find the last linear layer by iterating through modules
    # This will be the MLP down_proj in the final transformer block
    last_linear_layer = None
    last_linear_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module
            last_linear_name = name

    if last_linear_layer is None:
        raise RuntimeError("Could not find any linear layer in model")

    print(f"Found last linear layer: {last_linear_name}")
    print(f"Layer weight shape: {last_linear_layer.weight.shape}")

    # FB extraction: Take first column and reshape to (hidden_dim, 1)
    V_sft = last_linear_layer.weight[:, 0].float().reshape(-1, 1)
    print(f"Extracted V_sft with shape: {V_sft.shape}")

    # Clean up model to free memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return V_sft


def create_log_dir(base_dir: Path) -> Path:
    """Create timestamped log directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base_dir / f"lore_prism_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def train_rank(
    rank: int,
    splitter_result,
    V_sft: torch.Tensor,
    args: argparse.Namespace,
    log_dir: Path,
) -> dict:
    """
    Train LoRe model for a single rank.

    Returns:
        Dictionary with final metrics
    """
    print(f"\n{'='*60}")
    print(f"Training rank={rank}")
    print(f"{'='*60}")

    device = args.device

    # Create dataloaders
    train_seen_loader = DataLoader(
        splitter_result.train_seen,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_seen_loader = DataLoader(
        splitter_result.test_seen,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_unseen_loader = DataLoader(
        splitter_result.train_unseen,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_unseen_loader = DataLoader(
        splitter_result.test_unseen,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Handle rank 0 case (reference model with no personalization)
    effective_rank = max(1, rank)

    # Create model
    model = LoReRewardModel(
        embed_dim=splitter_result.train_seen.embedding_dim,
        rank=effective_rank,
        n_users=splitter_result.n_seen_users,
        alpha=args.alpha,
    )

    print(f"Model: embed_dim={model.embed_dim}, rank={effective_rank}, "
          f"n_seen_users={splitter_result.n_seen_users}")

    # Create trainer
    trainer = LoReFBTrainer(
        model=model,
        V_sft=V_sft,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        device=device,
    )

    # Open log file for per-iteration metrics
    log_file_path = log_dir / f"training_log_K{rank}.jsonl"
    log_file = open(log_file_path, 'w')

    def log_callback(metrics):
        log_file.write(json.dumps(metrics) + '\n')
        log_file.flush()
        if metrics['iteration'] % (args.log_interval * 10) == 0:
            print(f"  Iter {metrics['iteration']:5d}: "
                  f"bce={metrics['bce_loss']:.4f}, "
                  f"reg={metrics['reg_loss']:.4f}, "
                  f"acc={metrics['accuracy']:.4f}, "
                  f"alpha={metrics['alpha']:.1f}")

    # Training
    train_start = time.time()
    print(f"\nTraining for {args.n_iterations} iterations...")

    trainer.train(
        dataloader=train_seen_loader,
        n_iterations=args.n_iterations,
        log_interval=args.log_interval,
        log_callback=log_callback,
    )

    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.1f}s")
    log_file.close()

    # Evaluate seen users
    print("\nEvaluating seen users...")
    seen_train_metrics = trainer.evaluate(train_seen_loader)
    seen_test_metrics = trainer.evaluate(test_seen_loader)

    print(f"  Seen train accuracy: {seen_train_metrics['accuracy']:.4f}")
    print(f"  Seen test accuracy:  {seen_test_metrics['accuracy']:.4f}")

    # Few-shot for unseen users
    print(f"\nFew-shot adaptation for {splitter_result.n_unseen_users} unseen users...")

    # Reset W for unseen users
    trainer.reset_user_weights(n_users=splitter_result.n_unseen_users)

    # Open few-shot log file
    fewshot_log_path = log_dir / f"fewshot_log_K{rank}.jsonl"
    fewshot_log = open(fewshot_log_path, 'w')

    def fewshot_callback(metrics):
        fewshot_log.write(json.dumps(metrics) + '\n')
        fewshot_log.flush()

    trainer.fewshot_adapt(
        dataloader=train_unseen_loader,
        n_iterations=args.fewshot_iterations,
        log_interval=args.log_interval,
        log_callback=fewshot_callback,
    )
    fewshot_log.close()

    # Evaluate unseen users
    print("Evaluating unseen users...")
    unseen_train_metrics = trainer.evaluate(train_unseen_loader)
    unseen_test_metrics = trainer.evaluate(test_unseen_loader)

    print(f"  Unseen train accuracy: {unseen_train_metrics['accuracy']:.4f}")
    print(f"  Unseen test accuracy:  {unseen_test_metrics['accuracy']:.4f}")

    # Compile final results
    final_metrics = {
        'rank': rank,
        'seen_train_acc': seen_train_metrics['accuracy'],
        'seen_test_acc': seen_test_metrics['accuracy'],
        'unseen_train_acc': unseen_train_metrics['accuracy'],
        'unseen_test_acc': unseen_test_metrics['accuracy'],
        'training_time_seconds': train_time,
    }

    # Save final results for this rank
    results_path = log_dir / f"results_K{rank}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'rank': rank,
                'n_iterations': args.n_iterations,
                'fewshot_iterations': args.fewshot_iterations,
                'learning_rate': args.learning_rate,
                'alpha': args.alpha,
                'batch_size': args.batch_size,
                'seed': args.seed,
            },
            'data_stats': {
                'n_seen_users': splitter_result.n_seen_users,
                'n_unseen_users': splitter_result.n_unseen_users,
                'train_seen_samples': len(splitter_result.train_seen),
                'test_seen_samples': len(splitter_result.test_seen),
                'train_unseen_samples': len(splitter_result.train_unseen),
                'test_unseen_samples': len(splitter_result.test_unseen),
            },
            'final_metrics': final_metrics,
        }, f, indent=2)

    # Save model checkpoint
    model_path = log_dir / f"lore_fb_K{rank}.pt"
    model.save(str(model_path))

    return final_metrics


def main() -> None:
    """Main entry point."""
    args = parse_args()
    start_time = time.time()

    # Configure environment
    configure_environment()
    dataset_config = DatasetConfig()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine ranks to train
    if args.full_grid:
        ranks = [0, 1, 5, 10, 15, 20, 25, 50]
    else:
        ranks = args.ranks

    # Print configuration
    print(f"\n{'='*60}")
    print("FB LoRe Training Configuration")
    print(f"{'='*60}")
    print(f"  Ranks:             {ranks}")
    print(f"  Iterations:        {args.n_iterations}")
    print(f"  Few-shot iters:    {args.fewshot_iterations}")
    print(f"  Learning rate:     {args.learning_rate}")
    print(f"  Alpha:             {args.alpha}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Log interval:      {args.log_interval}")
    print(f"  Device:            {args.device}")
    print(f"  Seed:              {args.seed}")
    print(f"  Seen user ratio:   {args.seen_user_ratio}")
    print(f"  Dialog train ratio:{args.dialog_train_ratio}")
    print(f"{'='*60}\n")

    # Set up output directory
    base_output_dir = Path(args.output_dir) if args.output_dir else Path("logs")
    log_dir = create_log_dir(base_output_dir)
    print(f"Logging to: {log_dir}")

    # Save configuration
    config_path = log_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load embeddings
    embeddings_path = Path(args.embeddings) if args.embeddings else dataset_config.embeddings_path
    print(f"\nLoading embeddings from {embeddings_path}")
    embeddings = load_embeddings(embeddings_path)

    # Create user splits
    print("\nSplitting data by users...")
    splitter = FBDataSplitter(
        embeddings=embeddings,
        seen_user_ratio=args.seen_user_ratio,
        dialog_train_ratio=args.dialog_train_ratio,
        seed=args.seed,
    )
    splitter_result = splitter.split()

    stats = splitter.get_stats()
    print(f"  Total users:         {stats['total_users']}")
    print(f"  Seen users:          {stats['n_seen_users']}")
    print(f"  Unseen users:        {stats['n_unseen_users']}")
    print(f"  Train seen samples:  {stats['train_seen_samples']}")
    print(f"  Test seen samples:   {stats['test_seen_samples']}")
    print(f"  Train unseen samples:{stats['train_unseen_samples']}")
    print(f"  Test unseen samples: {stats['test_unseen_samples']}")
    print(f"  Embedding dim:       {stats['embedding_dim']}")

    # Save split statistics
    with open(log_dir / "data_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Extract V_sft
    print("\nExtracting V_sft from pretrained model...")
    V_sft = extract_v_sft(embeddings_path, args.device)

    # Save V_sft for reference
    torch.save(V_sft, log_dir / "V_sft.pt")

    # Train for each rank
    all_results = []
    for rank in ranks:
        results = train_rank(
            rank=rank,
            splitter_result=splitter_result,
            V_sft=V_sft,
            args=args,
            log_dir=log_dir,
        )
        all_results.append(results)

    # Create summary
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")
    print(f"{'Rank':>6} {'Seen Train':>12} {'Seen Test':>12} {'Unseen Train':>12} {'Unseen Test':>12}")
    print(f"{'-'*60}")

    summary = {
        'ranks': [],
        'seen_train_acc': [],
        'seen_test_acc': [],
        'unseen_train_acc': [],
        'unseen_test_acc': [],
    }

    for result in all_results:
        print(f"{result['rank']:>6} "
              f"{result['seen_train_acc']:>12.4f} "
              f"{result['seen_test_acc']:>12.4f} "
              f"{result['unseen_train_acc']:>12.4f} "
              f"{result['unseen_test_acc']:>12.4f}")

        summary['ranks'].append(result['rank'])
        summary['seen_train_acc'].append(result['seen_train_acc'])
        summary['seen_test_acc'].append(result['seen_test_acc'])
        summary['unseen_train_acc'].append(result['unseen_train_acc'])
        summary['unseen_test_acc'].append(result['unseen_test_acc'])

    # Save summary
    with open(log_dir / "results_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved to: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
