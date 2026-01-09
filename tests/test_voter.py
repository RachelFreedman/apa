"""
Unit tests for voter module.
"""

import pytest
import torch

from apa.inference.voter import UserVoter, VoterPool


class TestUserVoter:
    """Tests for UserVoter class."""

    def test_init(self):
        """Test voter initialization."""
        V = torch.randn(32, 8)
        w = torch.randn(8)

        voter = UserVoter(
            user_id='test_user',
            user_vector=w,
            basis_matrix=V,
            metadata={'source': 'test'},
        )

        assert voter.user_id == 'test_user'
        assert voter.w.shape == (8,)
        assert voter.V.shape == (32, 8)
        assert voter.metadata['source'] == 'test'
        assert voter.reward_direction.shape == (32,)

    def test_score_embeddings(self):
        """Test scoring embeddings."""
        V = torch.randn(32, 8)
        w = torch.randn(8)

        voter = UserVoter('user', w, V)
        embeddings = torch.randn(5, 32)

        scores = voter.score_embeddings(embeddings)

        assert scores.shape == (5,)
        # Verify computation
        expected = embeddings @ (V @ w)
        assert torch.allclose(scores, expected)

    def test_rank_embeddings(self):
        """Test ranking embeddings."""
        V = torch.randn(32, 8)
        w = torch.randn(8)

        voter = UserVoter('user', w, V)
        embeddings = torch.randn(5, 32)

        ranking = voter.rank_embeddings(embeddings)

        assert len(ranking) == 5
        assert set(ranking) == {0, 1, 2, 3, 4}

        # Verify it's sorted by score (descending)
        scores = voter.score_embeddings(embeddings)
        for i in range(len(ranking) - 1):
            assert scores[ranking[i]] >= scores[ranking[i + 1]]


class TestVoterPool:
    """Tests for VoterPool class."""

    def test_init(self):
        """Test pool initialization."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)

        assert pool.V.shape == (32, 8)
        assert len(pool.voters) == 0

    def test_add_voter(self):
        """Test adding a voter."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)

        w = torch.randn(8)
        pool.add_voter('user_1', w, {'century': 'C013'})

        assert 'user_1' in pool.voters
        assert pool.voters['user_1'].metadata['century'] == 'C013'

    def test_get_voter(self):
        """Test getting a voter."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)
        pool.add_voter('user_1', torch.randn(8))

        voter = pool.get_voter('user_1')
        assert voter is not None
        assert voter.user_id == 'user_1'

        missing = pool.get_voter('nonexistent')
        assert missing is None

    def test_get_all_user_ids(self):
        """Test getting all user IDs."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)
        pool.add_voter('user_1', torch.randn(8))
        pool.add_voter('user_2', torch.randn(8))
        pool.add_voter('user_3', torch.randn(8))

        ids = pool.get_all_user_ids()

        assert set(ids) == {'user_1', 'user_2', 'user_3'}

    def test_get_user_metadata(self):
        """Test getting user metadata."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)
        pool.add_voter('user_1', torch.randn(8), {'source': 'prism'})
        pool.add_voter('user_2', torch.randn(8), {'source': 'historical'})

        metadata = pool.get_user_metadata()

        assert metadata['user_1']['source'] == 'prism'
        assert metadata['user_2']['source'] == 'historical'

    def test_collect_rankings(self):
        """Test collecting rankings from voters."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)
        pool.add_voter('user_1', torch.randn(8))
        pool.add_voter('user_2', torch.randn(8))
        pool.add_voter('user_3', torch.randn(8))

        embeddings = torch.randn(5, 32)

        rankings = pool.collect_rankings(embeddings)

        assert len(rankings) == 3
        assert set(rankings.keys()) == {'user_1', 'user_2', 'user_3'}
        for ranking in rankings.values():
            assert len(ranking) == 5
            assert set(ranking) == {0, 1, 2, 3, 4}

    def test_collect_rankings_subset(self):
        """Test collecting rankings from a subset of voters."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)
        pool.add_voter('user_1', torch.randn(8))
        pool.add_voter('user_2', torch.randn(8))
        pool.add_voter('user_3', torch.randn(8))

        embeddings = torch.randn(5, 32)

        rankings = pool.collect_rankings(embeddings, user_ids=['user_1', 'user_3'])

        assert len(rankings) == 2
        assert set(rankings.keys()) == {'user_1', 'user_3'}

    def test_load_prism_users(self, tmp_path):
        """Test loading PRISM users from checkpoint."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)

        # Create mock W matrix
        W = torch.randn(5, 8)
        path = tmp_path / "W_test.pt"
        torch.save(W, path)

        pool.load_prism_users(path)

        assert len(pool.voters) == 5
        # Users should be named prism_user_0, etc.
        assert 'prism_user_0' in pool.voters
        assert 'prism_user_4' in pool.voters

    def test_load_historical_users(self, tmp_path):
        """Test loading historical users from directory."""
        V = torch.randn(32, 8)
        pool = VoterPool(V)

        # Create mock historical user files
        for century in ['C013', 'C017']:
            checkpoint = {
                'user_id': f'historical_{century}',
                'century': century,
                'w': torch.randn(8),
            }
            path = tmp_path / f"W_historical_{century}.pt"
            torch.save(checkpoint, path)

        pool.load_historical_users(tmp_path)

        assert len(pool.voters) == 2
        assert 'historical_C013' in pool.voters
        assert pool.voters['historical_C013'].metadata['century'] == 'C013'
