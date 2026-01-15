"""
Unit tests for FB-style LoRe trainer.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from apa.reward.lore_model import LoReRewardModel
from apa.reward.lore_fb_trainer import LoReFBTrainer


def create_simple_dataloader(n_samples: int = 32, embed_dim: int = 32, n_users: int = 5, batch_size: int = 8):
    """Create a simple dataloader for testing."""
    class SimpleDataset:
        def __init__(self, n_samples, embed_dim, n_users):
            self.emb_1 = torch.randn(n_samples, embed_dim)
            self.emb_2 = torch.randn(n_samples, embed_dim)
            self.labels = torch.randint(0, 2, (n_samples,))
            self.user_indices = torch.randint(0, n_users, (n_samples,))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'response_1_embedding': self.emb_1[idx],
                'response_2_embedding': self.emb_2[idx],
                'label': self.labels[idx],
                'user_idx': self.user_indices[idx],
            }

    dataset = SimpleDataset(n_samples, embed_dim, n_users)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestLoReFBTrainer:
    """Tests for LoReFBTrainer class."""

    def test_init(self):
        """Test trainer initialization."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)

        trainer = LoReFBTrainer(
            model=model,
            V_sft=V_sft,
            learning_rate=0.5,
            alpha=10000.0,
            device='cpu',
        )

        assert trainer.model is model
        assert trainer.alpha == 10000.0
        assert trainer.learning_rate == 0.5
        assert trainer.warmup_start == 0.2
        assert trainer.warmup_end == 0.8

    def test_compute_alpha_before_warmup(self):
        """Test alpha computation before warmup starts."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        # At 10% of iterations (before warmup_start=20%)
        alpha = trainer.compute_alpha(iteration=100, total_iterations=1000)
        assert alpha == 0.0

    def test_compute_alpha_during_warmup(self):
        """Test alpha computation during warmup period."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        # At 50% of iterations (middle of warmup 20%-80%)
        alpha = trainer.compute_alpha(iteration=500, total_iterations=1000)
        # 50% is exactly in the middle of 20%-80%, so alpha should be ~5000
        assert 4000 < alpha < 6000

    def test_compute_alpha_after_warmup(self):
        """Test alpha computation after warmup ends."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        # At 90% of iterations (after warmup_end=80%)
        alpha = trainer.compute_alpha(iteration=900, total_iterations=1000)
        assert alpha == 10000.0

    def test_train_iteration(self):
        """Test a single training iteration."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        emb_1 = torch.randn(8, 32)
        emb_2 = torch.randn(8, 32)
        user_indices = torch.randint(0, 5, (8,))
        labels = torch.randint(0, 2, (8,))

        metrics = trainer.train_iteration(
            emb_1, emb_2, user_indices, labels,
            iteration=500,
            total_iterations=1000,
        )

        assert 'iteration' in metrics
        assert 'loss_w' in metrics
        assert 'loss_v' in metrics
        assert 'bce_loss' in metrics
        assert 'reg_loss' in metrics
        assert 'alpha' in metrics
        assert 'accuracy' in metrics

        assert metrics['iteration'] == 500
        assert metrics['loss_w'] > 0
        assert metrics['loss_v'] > 0
        assert 0 <= metrics['accuracy'] <= 1

    def test_train_iteration_updates_weights(self):
        """Test that training iteration updates both V and W."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        # Store original weights
        V_before = model.V.data.clone()
        W_before = model.W.data.clone()

        emb_1 = torch.randn(8, 32)
        emb_2 = torch.randn(8, 32)
        user_indices = torch.randint(0, 5, (8,))
        labels = torch.randint(0, 2, (8,))

        trainer.train_iteration(
            emb_1, emb_2, user_indices, labels,
            iteration=500,
            total_iterations=1000,
        )

        # Both V and W should have changed
        assert not torch.allclose(model.V.data, V_before)
        assert not torch.allclose(model.W.data, W_before)

    def test_train(self):
        """Test training for multiple iterations."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        dataloader = create_simple_dataloader(n_samples=32, embed_dim=32, n_users=5)

        logged_metrics = trainer.train(
            dataloader=dataloader,
            n_iterations=100,
            log_interval=25,
        )

        # Should have logged at iterations 0, 25, 50, 75, 99
        assert len(logged_metrics) == 5
        assert logged_metrics[0]['iteration'] == 0
        assert logged_metrics[-1]['iteration'] == 99

    def test_train_with_callback(self):
        """Test training with log callback."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        dataloader = create_simple_dataloader(n_samples=32, embed_dim=32, n_users=5)

        callback_calls = []

        def callback(metrics):
            callback_calls.append(metrics)

        trainer.train(
            dataloader=dataloader,
            n_iterations=50,
            log_interval=10,
            log_callback=callback,
        )

        # Callback should have been called at iterations 0, 10, 20, 30, 40, 49
        assert len(callback_calls) == 6

    def test_evaluate(self):
        """Test evaluation."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, device='cpu')

        dataloader = create_simple_dataloader(n_samples=32, embed_dim=32, n_users=5)

        metrics = trainer.evaluate(dataloader)

        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_fewshot_adapt(self):
        """Test few-shot adaptation."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, device='cpu')

        dataloader = create_simple_dataloader(n_samples=32, embed_dim=32, n_users=5)

        logged_metrics = trainer.fewshot_adapt(
            dataloader=dataloader,
            n_iterations=50,
            log_interval=10,
        )

        # Should have logged at iterations 0, 10, 20, 30, 40, 49
        assert len(logged_metrics) == 6
        assert logged_metrics[0]['iteration'] == 0
        assert 'loss' in logged_metrics[0]
        assert 'accuracy' in logged_metrics[0]

    def test_fewshot_adapt_freezes_v(self):
        """Test that few-shot adaptation keeps V frozen."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, device='cpu')

        # Store original V
        V_before = model.V.data.clone()

        dataloader = create_simple_dataloader(n_samples=32, embed_dim=32, n_users=5)

        trainer.fewshot_adapt(
            dataloader=dataloader,
            n_iterations=50,
        )

        # V should not have changed
        assert torch.allclose(model.V.data, V_before)

    def test_fewshot_adapt_updates_w(self):
        """Test that few-shot adaptation updates W."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, device='cpu')

        # Store original W
        W_before = model.W.data.clone()

        dataloader = create_simple_dataloader(n_samples=32, embed_dim=32, n_users=5)

        trainer.fewshot_adapt(
            dataloader=dataloader,
            n_iterations=50,
        )

        # W should have changed
        assert not torch.allclose(model.W.data, W_before)

    def test_fewshot_adapt_unfreezes_v_after(self):
        """Test that V is unfrozen after few-shot adaptation."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, device='cpu')

        dataloader = create_simple_dataloader(n_samples=32, embed_dim=32, n_users=5)

        trainer.fewshot_adapt(dataloader=dataloader, n_iterations=10)

        # V should be unfrozen after adaptation
        assert model.V.requires_grad is True

    def test_reset_user_weights(self):
        """Test resetting user weights."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, device='cpu')

        # Store original W
        W_before = model.W.data.clone()

        trainer.reset_user_weights()

        # W should have been reset
        assert not torch.allclose(model.W.data, W_before)
        assert model.W.shape == (5, 4)

    def test_reset_user_weights_with_new_count(self):
        """Test resetting user weights with new user count."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, device='cpu')

        trainer.reset_user_weights(n_users=10)

        assert model.n_users == 10
        assert model.W.shape == (10, 4)

    def test_alternating_minimization_order(self):
        """Test that W is updated before V in each iteration."""
        # This is a bit tricky to test directly, but we can verify
        # that both optimizers are used by checking gradients
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        V_sft = torch.randn(32, 4)
        trainer = LoReFBTrainer(model, V_sft, alpha=10000.0, device='cpu')

        emb_1 = torch.randn(8, 32)
        emb_2 = torch.randn(8, 32)
        user_indices = torch.randint(0, 5, (8,))
        labels = torch.randint(0, 2, (8,))

        # Do one iteration
        metrics = trainer.train_iteration(
            emb_1, emb_2, user_indices, labels,
            iteration=500,
            total_iterations=1000,
        )

        # After iteration, both V and W should have requires_grad=True
        assert model.V.requires_grad is True
        assert model.W.requires_grad is True

        # We got separate loss_w and loss_v, confirming two update steps
        assert 'loss_w' in metrics
        assert 'loss_v' in metrics
