"""
Unit tests for LoRe reward model.
"""

import pytest
import torch

from apa.reward.lore_model import LoReRewardModel, LoReTrainer


class TestLoReRewardModel:
    """Tests for LoReRewardModel class."""

    def test_init(self):
        """Test model initialization."""
        model = LoReRewardModel(embed_dim=768, rank=8, n_users=10)

        assert model.embed_dim == 768
        assert model.rank == 8
        assert model.n_users == 10
        assert model.V.shape == (768, 8)
        assert model.W.shape == (10, 8)

    def test_forward(self):
        """Test forward pass."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        embeddings = torch.randn(3, 32)
        user_indices = torch.tensor([0, 1, 2])

        rewards = model(embeddings, user_indices)

        assert rewards.shape == (3,)
        assert rewards.dtype == torch.float32

    def test_forward_batch(self):
        """Test forward pass with larger batch."""
        model = LoReRewardModel(embed_dim=64, rank=8, n_users=10)

        batch_size = 16
        embeddings = torch.randn(batch_size, 64)
        user_indices = torch.randint(0, 10, (batch_size,))

        rewards = model(embeddings, user_indices)

        assert rewards.shape == (batch_size,)

    def test_compute_preference_logits(self):
        """Test preference logit computation."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        emb_1 = torch.randn(4, 32)
        emb_2 = torch.randn(4, 32)
        user_indices = torch.tensor([0, 1, 2, 3])

        logits = model.compute_preference_logits(emb_1, emb_2, user_indices)

        assert logits.shape == (4,)
        # Logits should be (r2 - r1) / temperature
        r1 = model(emb_1, user_indices)
        r2 = model(emb_2, user_indices)
        expected = (r2 - r1) / model.temperature
        assert torch.allclose(logits, expected)

    def test_compute_loss(self):
        """Test loss computation."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        emb_1 = torch.randn(4, 32)
        emb_2 = torch.randn(4, 32)
        user_indices = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(emb_1, emb_2, user_indices, labels)

        assert loss.shape == ()
        assert loss.item() > 0  # Loss should be positive

    def test_cosine_regularization(self):
        """Test cosine regularization computation."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        # Create V_sft with same shape as V
        V_sft = torch.randn(32, 4)

        reg_loss = model.cosine_regularization(V_sft)

        assert reg_loss.shape == ()
        # Regularization should be between 0 and 2 (1 - cos_sim where cos_sim is [-1, 1])
        assert 0 <= reg_loss.item() <= 2

    def test_cosine_regularization_identical(self):
        """Test cosine regularization is 0 when V equals V_sft."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        # Use model's V as V_sft (they should be identical)
        V_sft = model.V.data.clone()

        reg_loss = model.cosine_regularization(V_sft)

        # Should be approximately 0 (perfect similarity)
        assert reg_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_cosine_regularization_smaller_v_sft(self):
        """Test cosine regularization with smaller V_sft (rank 1)."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        # V_sft with only 1 column (like from pretrained model)
        V_sft = torch.randn(32, 1)

        reg_loss = model.cosine_regularization(V_sft)

        assert reg_loss.shape == ()
        assert 0 <= reg_loss.item() <= 2

    def test_compute_loss_alternating_basic(self):
        """Test alternating loss computation without regularization."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        emb_1 = torch.randn(4, 32)
        emb_2 = torch.randn(4, 32)
        user_indices = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 0, 1])

        loss, metrics = model.compute_loss_alternating(
            emb_1, emb_2, user_indices, labels
        )

        assert loss.shape == ()
        assert loss.item() > 0
        assert 'bce_loss' in metrics
        assert 'reg_loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['reg_loss'] == 0.0  # No V_sft provided

    def test_compute_loss_alternating_with_regularization(self):
        """Test alternating loss computation with regularization."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        emb_1 = torch.randn(4, 32)
        emb_2 = torch.randn(4, 32)
        user_indices = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 0, 1])
        V_sft = torch.randn(32, 4)

        loss, metrics = model.compute_loss_alternating(
            emb_1, emb_2, user_indices, labels,
            V_sft=V_sft, alpha=1.0
        )

        assert loss.shape == ()
        assert metrics['reg_loss'] > 0  # Should have regularization
        # Total loss should be bce_loss + alpha * reg_loss
        expected_loss = metrics['bce_loss'] + 1.0 * metrics['reg_loss']
        assert loss.item() == pytest.approx(expected_loss, abs=1e-5)

    def test_compute_loss_alternating_accuracy(self):
        """Test that accuracy is computed correctly in alternating loss."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        # Create a batch where we know the predictions
        emb_1 = torch.randn(4, 32)
        emb_2 = torch.randn(4, 32)
        user_indices = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 1, 0, 1])

        _, metrics = model.compute_loss_alternating(
            emb_1, emb_2, user_indices, labels
        )

        # Accuracy should be between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1

    def test_score_responses(self):
        """Test scoring multiple responses for a single user."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        embeddings = torch.randn(10, 32)
        scores = model.score_responses(embeddings, user_idx=2)

        assert scores.shape == (10,)

    def test_rank_responses(self):
        """Test ranking responses for a single user."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        embeddings = torch.randn(5, 32)
        ranking = model.rank_responses(embeddings, user_idx=0)

        assert len(ranking) == 5
        assert set(ranking) == {0, 1, 2, 3, 4}
        # Should be sorted by score (descending)

    def test_get_user_reward_function(self):
        """Test getting a user-specific reward function."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        reward_fn = model.get_user_reward_function(user_idx=1)
        embeddings = torch.randn(3, 32)

        rewards = reward_fn(embeddings)

        assert rewards.shape == (3,)
        # Should match direct call
        expected = model.score_responses(embeddings, user_idx=1)
        assert torch.allclose(rewards, expected)

    def test_freeze_unfreeze_basis(self):
        """Test freezing and unfreezing the basis matrix."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        assert model.V.requires_grad is True

        model.freeze_basis()
        assert model.V.requires_grad is False

        model.unfreeze_basis()
        assert model.V.requires_grad is True

    def test_add_users(self):
        """Test adding new users to the model."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        assert model.n_users == 5
        assert model.W.shape == (5, 4)

        model.add_users(3)

        assert model.n_users == 8
        assert model.W.shape == (8, 4)

    def test_save_load(self, tmp_path):
        """Test saving and loading model."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)

        # Modify weights to have non-random values
        model.V.data = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
        model.W.data = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

        path = str(tmp_path / "test_model.pt")
        model.save(path)

        loaded = LoReRewardModel.load(path)

        assert loaded.embed_dim == model.embed_dim
        assert loaded.rank == model.rank
        assert loaded.n_users == model.n_users
        assert torch.allclose(loaded.V, model.V)
        assert torch.allclose(loaded.W, model.W)


class TestLoReTrainer:
    """Tests for LoReTrainer class."""

    def test_init(self):
        """Test trainer initialization."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        trainer = LoReTrainer(model, learning_rate=1e-3, device='cpu')

        assert trainer.model is model
        assert trainer.device == 'cpu'

    def test_train_epoch(self):
        """Test training for one epoch."""
        model = LoReRewardModel(embed_dim=32, rank=4, n_users=5)
        trainer = LoReTrainer(model, learning_rate=1e-3, device='cpu')

        # Create simple dataloader
        from torch.utils.data import DataLoader, TensorDataset

        n_samples = 16
        dataset = TensorDataset(
            torch.randn(n_samples, 32),  # emb_1
            torch.randn(n_samples, 32),  # emb_2
            torch.randint(0, 2, (n_samples,)),  # labels
        )

        class SimpleDataLoader:
            def __init__(self, dataset, batch_size=4):
                self.dataset = dataset
                self.batch_size = batch_size

            def __iter__(self):
                for i in range(0, len(self.dataset), self.batch_size):
                    batch = self.dataset[i:i+self.batch_size]
                    yield {
                        'response_1_embedding': batch[0],
                        'response_2_embedding': batch[1],
                        'label': batch[2],
                    }

        dataloader = SimpleDataLoader(dataset)
        loss = trainer.train_epoch(dataloader, verbose=False)

        assert isinstance(loss, float)
        assert loss > 0
