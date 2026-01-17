#!/usr/bin/env python3
"""
Test that all modules can be imported correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Test config module."""
    from apa.config import (
        APAConfig,
        DatasetConfig,
        HistLlamaConfig,
        LoReConfig,
        InferenceConfig,
        configure_environment,
        get_config,
    )
    config = get_config()
    assert config.lore.alpha == 10000.0
    print("  config: OK")


def test_data():
    """Test data module."""
    from apa.data import load_prism_pairwise, PRISMDataset
    print("  data: OK")


def test_reward():
    """Test reward module."""
    from apa.reward import LoReRewardModel
    from apa.reward.lore_model import LoReTrainer
    print("  reward: OK")


def test_levers():
    """Test levers module."""
    from apa.levers import (
        lever_generate_responses,
        lever_sample_users,
        lever_aggregate_rankings,
        lever_select_questions,
    )
    print("  levers: OK")


def test_historical():
    """Test historical module."""
    from apa.historical import (
        load_hist_llama,
        generate_historical_preferences,
    )
    print("  historical: OK")


def test_inference():
    """Test inference module."""
    from apa.inference import DemocraticInference
    from apa.inference.response_generator import (
        generate_responses,
        load_inference_llm,
    )
    from apa.inference.voter import UserVoter, VoterPool
    print("  inference: OK")


def test_utils():
    """Test utils module."""
    from apa.utils import (
        get_embedding_model,
        embed_text,
        embed_texts,
        save_with_symlink,
        CheckpointManager,
    )
    print("  utils: OK")


def test_lever_aggregate():
    """Test aggregation lever with sample data."""
    from apa.levers.lever_aggregate import lever_aggregate_rankings

    rankings = {
        'user_1': [0, 1, 2],
        'user_2': [1, 0, 2],
        'user_3': [0, 2, 1],
    }
    config = {'aggregate': 'borda_count'}

    result = lever_aggregate_rankings(rankings, config)
    assert len(result) == 3
    assert result[0] == 0  # Response 0 should win
    print("  lever_aggregate: OK")


def test_lever_sample():
    """Test sampling lever."""
    from apa.levers.lever_sample import lever_sample_users

    all_users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
    config = {'sample': 'random'}

    result = lever_sample_users(all_users, None, 3, config)
    assert len(result) == 3
    assert all(u in all_users for u in result)
    print("  lever_sample: OK")


def main():
    """Run all import tests."""
    print("\nTesting APA module imports...")
    print("-" * 40)

    tests = [
        test_config,
        test_data,
        test_reward,
        test_levers,
        test_historical,
        test_inference,
        test_utils,
        test_lever_aggregate,
        test_lever_sample,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  {test_fn.__name__}: FAILED - {e}")
            failed += 1

    print("-" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
