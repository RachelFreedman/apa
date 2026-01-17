#!/usr/bin/env python3
"""
Train LoRe model on PRISM dataset.

This script trains the Low-rank Reward model on PRISM pairwise preferences,
learning both shared bases (V) and user-specific vectors (W).

Performance targets (on PRISM dataset):
| Rank | Train Acc | Seen/Unseen Prompts | Few-Shot Train | Unseen/Unseen |
|------|-----------|---------------------|----------------|---------------|
| 0    | 71.56%    | 71.56%              | 73.55%         | 71.20%        |
| 1    | 76.18%    | 76.59%              | 76.90%         | 76.06%        |
| 5    | 87.90%    | 87.75%              | 88.30%         | 87.92%        |
| 10   | 90.05%    | 89.76%              | 91.57%         | 91.25%        |

Usage:
    python scripts/train_lore_prism.py
    python scripts/train_lore_prism.py --K_list 0,1,5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig, LoReConfig, DATA_DIR, CHECKPOINTS_DIR
from apa.data.prism_loader import group_embeddings_by_user
from apa.reward.lore_model import (
    LoReTrainer,
    eval_multiple,
    learn_multiple_few_shot,
    get_device,
)


def log(message: str) -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LoRe model on PRISM dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--K_list",
        type=str,
        default="0,1",
        help="Comma-separated list of ranks to train (e.g., '0,1,5,10')",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10000.0,
        help="Regularization coefficient (default matches LoRe paper)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=20000,
        help="Number of training iterations (default matches LoRe paper)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.5,
        help="Learning rate (default 0.5 matches LoRe paper)",
    )
    parser.add_argument(
        "--few_shot_iterations",
        type=int,
        default=500,
        help="Iterations for few-shot personalization",
    )
    parser.add_argument(
        "--few_shot_lr",
        type=float,
        default=0.5,
        help="Learning rate for few-shot personalization",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=2000,
        help="Log training diagnostics every N iterations",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Directory containing embeddings (uses DATA_DIR/prism if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (uses CHECKPOINTS_DIR/prism if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        default=True,
        help="Save accuracy vs rank plot",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        help="Embedding model (used to extract V_final)",
    )
    return parser.parse_args()


def run_regularized(
    K_list: list[int],
    alpha_list: list[float],
    V_final: torch.Tensor,
    train_features: list[torch.Tensor],
    test_features_sparse: list[torch.Tensor],
    train_features_unseen: list[torch.Tensor],
    test_features_sparse_unseen: list[torch.Tensor],
    N: int,
    N_unseen: int,
    device: torch.device,
    checkpoint_dir: Path,
    num_iterations: int = 20000,
    learning_rate: float = 0.5,
    few_shot_iterations: int = 500,
    few_shot_lr: float = 0.5,
    log_interval: int = 2000,
):
    """
    Compute accuracies for joint and few-shot learning.

    This follows the run_regularized function from LoRe/utils.py.
    """
    # Initialize result lists
    train_accuracies_joint = []
    seen_user_unseen_prompts_accuracies_joint = []
    few_shot_train_accuracies_few_shot = []
    unseen_user_unseen_prompts_accuracies_few_shot = []
    train_accuracies_joint_std = []
    seen_user_unseen_prompts_accuracies_joint_std = []
    few_shot_train_accuracies_few_shot_std = []
    unseen_user_unseen_prompts_accuracies_few_shot_std = []

    # Store training histories
    all_training_histories = {}

    for alpha in alpha_list:
        log(f"Alpha: {alpha}")

        for K in K_list:
            log("")
            log("=" * 50)
            log(f"Training K={K}, alpha={alpha}")
            log("=" * 50)

            if K == 0:
                # Reference model: use V_final directly
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for _ in range(N)]
                training_history = None
            else:
                # Train LoRe model
                num_features = 4096  # Llama 3.1 8B hidden dim
                trainer = LoReTrainer(
                    V_sft=V_final,
                    alpha=alpha,
                    num_classes=N,
                    num_features=num_features,
                    num_basis_vectors=K,
                    num_iterations=num_iterations,
                    learning_rate=learning_rate,
                    log_interval=log_interval,
                )

                W_joint, V_joint = trainer.train_model(train_features)
                training_history = trainer.training_history
                all_training_histories[f"K{K}_alpha{alpha}"] = training_history

                # Log training summary
                if training_history:
                    log("")
                    log(f"Training Summary for K={K}:")
                    log(f"  Initial NLL: {training_history['nll_V'][0]:.4f}")
                    log(f"  Final NLL:   {training_history['nll_V'][-1]:.4f}")
                    log(f"  NLL change:  {training_history['nll_V'][-1] - training_history['nll_V'][0]:.4f}")

                # Save checkpoints
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                v_path = checkpoint_dir / f"V_lore_K_{K}_alpha_{alpha}.pt"
                torch.save(V_joint.detach().cpu(), v_path)
                log(f"Saved V to {v_path}")

                w_path = checkpoint_dir / f"W_lore_seen_{K}_{alpha}.pt"
                torch.save(W_joint.detach().cpu(), w_path)
                log(f"Saved W to {w_path}")

            # Evaluate on train set (seen users, seen prompts)
            log("Train Performance")
            accuracies_train = eval_multiple(
                W_joint,
                [V_joint.detach() for _ in range(N)],
                train_features
            )
            train_accuracies_joint.append(np.mean(accuracies_train))
            train_accuracies_joint_std.append(np.std(accuracies_train))

            # Evaluate on test set (seen users, unseen prompts)
            log("Seen User Unseen Prompts")
            accuracies_seen_unseen = eval_multiple(
                W_joint,
                [V_joint.detach() for _ in range(N)],
                test_features_sparse
            )
            seen_user_unseen_prompts_accuracies_joint.append(np.mean(accuracies_seen_unseen))
            seen_user_unseen_prompts_accuracies_joint_std.append(np.std(accuracies_seen_unseen))

            # Few-shot learning for unseen users
            if K <= 1:
                W_few_shot = [torch.tensor([1.0]).to(device) for _ in range(N_unseen)]
            else:
                W_few_shot = learn_multiple_few_shot(
                    train_features_unseen,
                    V_joint.detach(),
                    num_iterations=few_shot_iterations,
                    learning_rate=few_shot_lr,
                )

            # Evaluate few-shot on train (unseen users, seen prompts)
            log("Few Shot Train Performance")
            accuracies_few_shot_train = eval_multiple(
                W_few_shot,
                [V_joint.detach() for _ in range(N_unseen)],
                train_features_unseen
            )
            few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
            few_shot_train_accuracies_few_shot_std.append(np.std(accuracies_few_shot_train))

            # Evaluate few-shot on test (unseen users, unseen prompts)
            log("Unseen User Unseen Prompts")
            accuracies_unseen_unseen = eval_multiple(
                W_few_shot,
                [V_joint.detach() for _ in range(N_unseen)],
                test_features_sparse_unseen
            )
            unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_unseen))
            unseen_user_unseen_prompts_accuracies_few_shot_std.append(np.std(accuracies_unseen_unseen))

    # Convert to numpy arrays
    fac = 0.25
    return (
        np.array(train_accuracies_joint),
        np.array(seen_user_unseen_prompts_accuracies_joint),
        np.array(few_shot_train_accuracies_few_shot),
        np.array(unseen_user_unseen_prompts_accuracies_few_shot),
        fac * np.array(train_accuracies_joint_std),
        fac * np.array(seen_user_unseen_prompts_accuracies_joint_std),
        fac * np.array(few_shot_train_accuracies_few_shot_std),
        fac * np.array(unseen_user_unseen_prompts_accuracies_few_shot_std),
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure environment
    configure_environment()

    # Parse K_list
    K_list = [int(k.strip()) for k in args.K_list.split(",")]
    alpha_list = [args.alpha]

    script_start = time.time()
    log("=" * 60)
    log("Starting PRISM basis training")
    log("=" * 60)
    log(f"K values: {K_list}")
    log(f"Alpha: {args.alpha}")
    log(f"Iterations: {args.num_iterations}")
    log(f"Learning rate: {args.learning_rate}")
    log(f"Device: {args.device}")

    # Determine paths
    if args.embeddings_dir:
        embeddings_dir = Path(args.embeddings_dir)
    else:
        embeddings_dir = DATA_DIR / "prism"

    if args.output_dir:
        checkpoint_dir = Path(args.output_dir)
    else:
        checkpoint_dir = CHECKPOINTS_DIR / "prism"

    log(f"Embeddings dir: {embeddings_dir}")
    log(f"Checkpoint dir: {checkpoint_dir}")

    # Load embeddings
    log("Loading embeddings...")
    load_start = time.time()
    train_embeddings = torch.load(embeddings_dir / "train_embeddings.pkl")
    test_embeddings = torch.load(embeddings_dir / "test_embeddings.pkl")
    load_time = time.time() - load_start
    log(f"Loaded embeddings in {load_time:.1f}s")
    log(f"  Train embeddings: {len(train_embeddings)} examples")
    log(f"  Test embeddings: {len(test_embeddings)} examples")

    # Group embeddings by user
    device = args.device
    train_seen, train_unseen, test_seen, test_unseen = group_embeddings_by_user(
        train_embeddings, test_embeddings, device
    )

    N = len(train_seen)
    N_unseen = len(train_unseen)
    log(f"Dataset statistics:")
    log(f"  Train seen users: {N}")
    log(f"  Train unseen users: {N_unseen}")
    log(f"  Test seen users: {len(test_seen)}")
    log(f"  Test unseen users: {len(test_unseen)}")

    # Load V_final from reward model
    log("=" * 60)
    log("Loading reward model on CPU to extract V_final...")
    model_start = time.time()

    from transformers import AutoModel

    model_name = args.embedding_model
    log(f"  Model: {model_name}")

    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        num_labels=1,
        low_cpu_mem_usage=True,
    )
    model_time = time.time() - model_start
    log(f"Model loaded in {model_time:.1f}s")

    # Extract final linear layer weights
    log("Extracting final linear layer weights...")
    last_linear_layer = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module

    if last_linear_layer is None:
        raise RuntimeError("Could not find a linear layer in the model")

    V_final = last_linear_layer.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)
    log(f"  V_final shape: {V_final.shape}")

    # Free memory
    del rm
    gc.collect()
    log("Model deleted to free memory")

    # Run training
    log("=" * 60)
    log("Starting training with run_regularized...")
    log("=" * 60)
    training_start = time.time()

    results = run_regularized(
        K_list=K_list,
        alpha_list=alpha_list,
        V_final=V_final,
        train_features=train_seen,
        test_features_sparse=test_seen,
        train_features_unseen=train_unseen,
        test_features_sparse_unseen=test_unseen,
        N=N,
        N_unseen=N_unseen,
        device=torch.device(device),
        checkpoint_dir=checkpoint_dir,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        few_shot_iterations=args.few_shot_iterations,
        few_shot_lr=args.few_shot_lr,
        log_interval=args.log_interval,
    )

    (train_acc, seen_unseen_acc, few_shot_train_acc, unseen_unseen_acc,
     train_std, seen_unseen_std, few_shot_train_std, unseen_unseen_std) = results

    training_time = time.time() - training_start
    log("=" * 60)
    log(f"Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    log("=" * 60)

    # Print final results table
    log("")
    log("Final Results:")
    log("-" * 80)
    log(f"{'Rank':<6} {'Train Acc':<12} {'Seen/Unseen':<14} {'Few-Shot':<12} {'Unseen/Unseen':<14}")
    log("-" * 80)
    for i, K in enumerate(K_list):
        log(f"{K:<6} {train_acc[i]*100:>10.2f}%  {seen_unseen_acc[i]*100:>12.2f}%  "
            f"{few_shot_train_acc[i]*100:>10.2f}%  {unseen_unseen_acc[i]*100:>12.2f}%")
    log("-" * 80)

    # Save results
    results_path = checkpoint_dir / f"results_alpha_{args.alpha}.json"
    results_dict = {
        "K_list": K_list,
        "alpha": args.alpha,
        "train_accuracy": train_acc.tolist(),
        "seen_unseen_accuracy": seen_unseen_acc.tolist(),
        "few_shot_train_accuracy": few_shot_train_acc.tolist(),
        "unseen_unseen_accuracy": unseen_unseen_acc.tolist(),
    }
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    log(f"Saved results to {results_path}")

    # Generate plot
    if args.save_plot:
        log("Generating plot...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(K_list, seen_unseen_acc, marker='o', linestyle='-', label="Seen Users")
            plt.plot(K_list, unseen_unseen_acc, marker='o', linestyle='-', label="Unseen Users")
            plt.plot(K_list, train_acc, marker='o', linestyle='-', label="Train Seen Users")
            plt.plot(K_list, few_shot_train_acc, marker='o', linestyle='-', label="Train Unseen Users Fewshot")
            plt.xlabel('Rank')
            plt.ylabel('Accuracy')
            plt.title('Generalization Accuracy vs. Rank')
            plt.xticks(K_list, labels=["ref" if k == 0 else str(k) for k in K_list])
            plt.legend()

            plot_path = checkpoint_dir / f"accuracy_vs_rank_alpha_{args.alpha}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log(f"Plot saved to {plot_path}")
            plt.close()
        except ImportError:
            log("matplotlib not available, skipping plot generation")

    total_time = time.time() - script_start
    log("=" * 60)
    log(f"All done! Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    log("=" * 60)


if __name__ == "__main__":
    main()
