#!/usr/bin/env python3
"""
Train LoRe on PRISM - Facebook replication.

This script replicates the FB LoRe training protocol exactly:
- Uses FB's data format (List[Tensor] per user with difference vectors)
- Uses FB's LoRe_regularized class with cosine similarity regularization
- Uses FB's hyperparameters (20k iterations, lr=0.5, alpha=10000)
- Produces the same accuracy vs rank plot as FB

Usage:
    python scripts/train_lore_fb.py --fb_data          # Use FB's exact data format (recommended)
    python scripts/train_lore_fb.py                    # Quick test with ranks 0,1,5
    python scripts/train_lore_fb.py --full_grid        # Full FB rank grid
    python scripts/train_lore_fb.py --n_iterations 1000  # Quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting disabled")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig
from apa.data.user_splits import convert_to_fb_format
from apa.reward.lore_fb import run_regularized
from apa.utils.embedding_utils import load_embeddings

# FB data directory
FB_DATA_DIR = Path("/nas/ucb/rachel/APA/data/prism_fb")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LoRe on PRISM - FB replication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fb_data",
        action="store_true",
        help="Use FB's exact data format (from prepare_prism_fb.py and generate_prism_embeddings_fb.py)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs='+',
        default=[0, 1, 5],
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
        help="Number of training iterations (FB uses 20000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10000.0,
        help="Regularization coefficient (FB uses 10000)",
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
        default="logs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (FB uses 123)",
    )
    return parser.parse_args()


def group_embeddings_by_user(train_embeddings, test_embeddings, device):
    """
    Group embeddings by user and compute difference vectors.

    This is FB's exact function from train_basis.py.

    Args:
        train_embeddings: List of dicts from train_embeddings.pkl
        test_embeddings: List of dicts from test_embeddings.pkl
        device: Torch device

    Returns:
        Tuple of (train_seen, train_unseen, test_seen, test_unseen)
        Each is a List[Tensor] where tensor[i] has shape (n_samples_for_user_i, embed_dim)
    """
    def process_dataset(dataset, seen_value, split_name):
        grouped = defaultdict(lambda: {"embeddings": []})
        for example in dataset:
            extra_info = example.get("extra_info", {})
            if extra_info.get("seen") == seen_value and extra_info.get("split") == split_name:
                user_id = extra_info.get("user_id")
                if user_id:
                    chosen = torch.tensor(extra_info["chosen_conv_embedding"], dtype=torch.float32, device=device)
                    rejected = torch.tensor(extra_info["rejected_conv_embedding"], dtype=torch.float32, device=device)
                    grouped[user_id]["embeddings"].append(chosen - rejected)
        # Stack and sort by user_id
        sorted_grouped = []
        count = 0
        for user_id in sorted(grouped.keys()):
            count += len(grouped[user_id]["embeddings"])
            sorted_grouped.append(
                torch.stack(grouped[user_id]["embeddings"]))
        print(f"  {split_name} seen={seen_value}: {count} samples from {len(sorted_grouped)} users")
        return sorted_grouped

    print("Grouping embeddings by user...")
    train_seen = process_dataset(train_embeddings, seen_value=True, split_name="train")
    train_unseen = process_dataset(train_embeddings, seen_value=False, split_name="train")
    test_seen = process_dataset(test_embeddings, seen_value=True, split_name="test")
    test_unseen = process_dataset(test_embeddings, seen_value=False, split_name="test")

    return train_seen, train_unseen, test_seen, test_unseen


def extract_v_sft(device: str, cache_dir: str | None = None, use_automodel: bool = True) -> torch.Tensor:
    """
    Extract V_sft from the pretrained Skywork-Reward model.

    FB's approach: Uses AutoModel (not AutoModelForSequenceClassification) and
    finds the last linear layer by iterating through named_modules.

    Args:
        device: Device to place tensor on
        cache_dir: HuggingFace cache directory
        use_automodel: If True, use AutoModel (FB's approach). If False, use AutoModelForSequenceClassification.

    Returns:
        V_sft tensor of shape (hidden_dim, 1)
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", "/nas/ucb/rachel/APA/hf_cache")

    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    print(f"Loading model for V_sft extraction: {model_name}")

    if use_automodel:
        # FB's approach: use AutoModel
        from transformers import AutoModel
        print("  Using AutoModel (FB's approach)")
        rm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_dir,
            attn_implementation="eager",
            num_labels=1,
        )
    else:
        # Our previous approach: use AutoModelForSequenceClassification
        from transformers import AutoModelForSequenceClassification
        print("  Using AutoModelForSequenceClassification")
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_dir,
            attn_implementation="eager",
            num_labels=1,
        )

    # Find the last linear layer by iterating through all modules
    last_linear_layer = None
    last_linear_name = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module
            last_linear_name = name

    if last_linear_layer is None:
        raise RuntimeError("Could not find any linear layer in model")

    print(f"Found last linear layer: {last_linear_name}")
    print(f"Layer weight shape: {last_linear_layer.weight.shape}")

    weight = last_linear_layer.weight.to(torch.float32)

    # Handle different layer shapes:
    # - AutoModelForSequenceClassification score layer: shape [1, hidden_dim] -> need .T
    # - AutoModel MLP layer: shape [hidden_dim, intermediate_size] -> take [:, 0]
    if last_linear_name == "score":
        # Score layer: weight has shape [num_labels, hidden_dim] = [1, 4096]
        # We want V_sft with shape [4096, 1], so transpose
        V_sft = weight.T.to(device)  # [4096, 1]
        print(f"  Score layer detected, transposing weights")
    else:
        # FB's approach for other layers: take first column
        V_sft = weight[:, 0].to(device).reshape(-1, 1)
        print(f"  Non-score layer, taking first column")

    print(f"Extracted V_sft with shape: {V_sft.shape}")

    # Clean up model to free memory
    del rm
    if 'cuda' in device:
        torch.cuda.empty_cache()

    return V_sft


def plot_results(
    K_list: list[int],
    results: tuple,
    alpha: float,
    output_path: Path,
) -> None:
    """
    Generate plot matching FB's generalization_accuracy_vs_rank_lore_alpha_10000.0.png.

    Args:
        K_list: List of ranks
        results: Tuple of 8 arrays from run_regularized
        alpha: Alpha value used
        output_path: Path to save plot
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot (matplotlib not available): {output_path}")
        return

    (train_acc, seen_test_acc, unseen_train_acc, unseen_test_acc,
     train_std, seen_test_std, unseen_train_std, unseen_test_std) = results

    plt.figure(figsize=(8, 5))
    plt.plot(K_list, seen_test_acc, marker='o', linestyle='-', label="Seen Users")
    plt.plot(K_list, unseen_test_acc, marker='o', linestyle='-', label="Unseen Users")
    plt.plot(K_list, train_acc, marker='o', linestyle='-', label="Train Seen Users")
    plt.plot(K_list, unseen_train_acc, marker='o', linestyle='-', label="Train Unseen Users Fewshot")
    plt.xlabel('rank')
    plt.ylabel('Accuracies')
    plt.title('Generalization Accuracy vs. Rank')
    plt.xticks(K_list, labels=["ref" if k == 0 else str(k) for k in K_list])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    start_time = time.time()

    # Configure environment
    configure_environment()
    dataset_config = DatasetConfig()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine ranks to train
    if args.full_grid:
        K_list = [0, 1, 5, 10, 15, 20, 25, 50]
    else:
        K_list = args.ranks

    # Print configuration
    print(f"\n{'='*60}")
    print("FB LoRe Training - Replication")
    print(f"{'='*60}")
    print(f"  FB Data Mode:      {args.fb_data}")
    print(f"  Ranks:             {K_list}")
    print(f"  Iterations:        {args.n_iterations}")
    print(f"  Alpha:             {args.alpha}")
    print(f"  Device:            {args.device}")
    print(f"  Seed:              {args.seed}")
    print(f"{'='*60}\n")

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"lore_fb_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load data based on mode
    if args.fb_data:
        # FB's exact data format from prepare_prism_fb.py and generate_prism_embeddings_fb.py
        train_emb_path = FB_DATA_DIR / "train_embeddings.pkl"
        test_emb_path = FB_DATA_DIR / "test_embeddings.pkl"

        if not train_emb_path.exists() or not test_emb_path.exists():
            print("ERROR: FB data files not found!")
            print(f"  Expected: {train_emb_path}")
            print(f"  Expected: {test_emb_path}")
            print("\nPlease run these scripts first:")
            print("  python scripts/prepare_prism_fb.py")
            print("  python scripts/generate_prism_embeddings_fb.py")
            sys.exit(1)

        print(f"\nLoading FB embeddings from {FB_DATA_DIR}")
        train_embeddings = torch.load(train_emb_path)
        test_embeddings = torch.load(test_emb_path)

        # Group by user (FB's approach)
        train_seen, train_unseen, test_seen, test_unseen = group_embeddings_by_user(
            train_embeddings, test_embeddings, args.device
        )
        N = len(train_seen)
        N_unseen = len(train_unseen)
    else:
        # Our previous approach: load from embeddings.pkl and convert
        embeddings_path = Path(args.embeddings) if args.embeddings else dataset_config.embeddings_path
        print(f"\nLoading embeddings from {embeddings_path}")
        embeddings = load_embeddings(embeddings_path)

        # Convert to FB format
        print("\nConverting to FB data format...")
        fb_data = convert_to_fb_format(
            embeddings,
            seen_user_ratio=0.8,
            dialog_train_ratio=0.5,
            min_samples_per_user=6,
            seed=args.seed,
            device=args.device,
        )

        train_seen = fb_data['train_seen']
        test_seen = fb_data['test_seen']
        train_unseen = fb_data['train_unseen']
        test_unseen = fb_data['test_unseen']
        N = fb_data['n_seen']
        N_unseen = fb_data['n_unseen']

    # Save data statistics
    data_stats = {
        'n_seen_users': N,
        'n_unseen_users': N_unseen,
        'train_seen_samples': sum(t.shape[0] for t in train_seen),
        'test_seen_samples': sum(t.shape[0] for t in test_seen),
        'train_unseen_samples': sum(t.shape[0] for t in train_unseen),
        'test_unseen_samples': sum(t.shape[0] for t in test_unseen),
        'embedding_dim': train_seen[0].shape[1] if train_seen else 0,
    }
    with open(output_dir / "data_stats.json", 'w') as f:
        json.dump(data_stats, f, indent=2)

    print(f"\nData Statistics:")
    print(f"  Seen users: {N}")
    print(f"  Unseen users: {N_unseen}")
    print(f"  Train seen samples: {data_stats['train_seen_samples']}")
    print(f"  Test seen samples: {data_stats['test_seen_samples']}")

    # Extract V_sft
    print("\nExtracting V_sft from pretrained model...")
    # Always use AutoModelForSequenceClassification for V_sft extraction
    # AutoModel finds the wrong layer (MLP.down_proj instead of score layer)
    V_final = extract_v_sft(args.device, use_automodel=False)

    # Save V_sft
    torch.save(V_final, output_dir / "V_sft.pt")

    # Modify the lore_fb module to use custom iterations if needed
    # This is a bit hacky but matches how FB would do quick tests
    import apa.reward.lore_fb as lore_fb_module
    original_run = lore_fb_module.run_regularized

    def custom_run_regularized(K_list, alpha_list, V_final, train_features, test_features_sparse,
                               train_features_unseen, test_features_sparse_unseen, N, N_unseen, device):
        """Wrapper to allow custom iteration count."""
        # Temporarily modify solve_regularized_simplex
        original_solve = lore_fb_module.solve_regularized_simplex

        def custom_solve(V_sft, alpha, train_features, num_basis_vectors, num_iterations=20000, learning_rate=0.5):
            return original_solve(V_sft, alpha, train_features, num_basis_vectors,
                                  num_iterations=args.n_iterations, learning_rate=learning_rate)

        lore_fb_module.solve_regularized_simplex = custom_solve

        try:
            return original_run(K_list, alpha_list, V_final, train_features, test_features_sparse,
                                train_features_unseen, test_features_sparse_unseen, N, N_unseen, device)
        finally:
            lore_fb_module.solve_regularized_simplex = original_solve

    # Run training
    print(f"\n{'='*60}")
    print(f"Training with {args.n_iterations} iterations per rank")
    print(f"{'='*60}\n")

    alpha_list = [args.alpha]

    results = custom_run_regularized(
        K_list, alpha_list, V_final,
        train_seen, test_seen,
        train_unseen, test_unseen,
        N, N_unseen, args.device
    )

    # Unpack results
    (train_acc, seen_test_acc, unseen_train_acc, unseen_test_acc,
     train_std, seen_test_std, unseen_train_std, unseen_test_std) = results

    # Print summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"{'Rank':>6} {'Train Seen':>12} {'Seen Test':>12} {'Unseen Train':>12} {'Unseen Test':>12}")
    print(f"{'-'*60}")

    for i, K in enumerate(K_list):
        print(f"{K:>6} {train_acc[i]:>12.4f} {seen_test_acc[i]:>12.4f} "
              f"{unseen_train_acc[i]:>12.4f} {unseen_test_acc[i]:>12.4f}")

    # Save results
    results_dict = {
        'ranks': K_list,
        'train_seen_acc': train_acc.tolist(),
        'seen_test_acc': seen_test_acc.tolist(),
        'unseen_train_acc': unseen_train_acc.tolist(),
        'unseen_test_acc': unseen_test_acc.tolist(),
        'train_seen_std': train_std.tolist(),
        'seen_test_std': seen_test_std.tolist(),
        'unseen_train_std': unseen_train_std.tolist(),
        'unseen_test_std': unseen_test_std.tolist(),
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Generate plot
    plot_path = output_dir / f"generalization_accuracy_vs_rank_lore_alpha_{args.alpha}.png"
    plot_results(K_list, results, args.alpha, plot_path)

    # Also save to current directory for easy comparison
    plot_results(K_list, results, args.alpha,
                 Path(f"generalization_accuracy_vs_rank_lore_alpha_{args.alpha}.png"))

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}\n")

    # Print expected vs actual for comparison
    print("\nComparison with FB expected results:")
    print(f"{'Rank':>6} {'Expected':>12} {'Actual':>12} {'Diff':>10}")
    expected = {0: 0.71, 1: 0.77, 5: 0.88, 50: 0.95}
    for i, K in enumerate(K_list):
        if K in expected:
            exp = expected[K]
            act = seen_test_acc[i]
            diff = act - exp
            print(f"{K:>6} {exp:>12.2f} {act:>12.4f} {diff:>+10.4f}")


if __name__ == "__main__":
    main()
