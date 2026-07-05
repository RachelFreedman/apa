"""
Tests for the DemocraticInference orchestrator and its lever dispatch.

These use a tiny in-memory VoterPool plus one-hot response embeddings so voter
rankings are fully controllable, and monkeypatch response generation/embedding
so no LLM is loaded. They verify that:
- default strategies resolve to random_sampling + borda_count (the previous
  hardcoded behavior), and
- selecting a non-default aggregator actually changes the winner.
"""

import torch

from apa.democratic_response import DemocraticInference, VoterPool
import apa.democratic_response as dr


class _Fake:
    """Placeholder model/tokenizer so __init__ never loads a real LLM."""


def _weight_for_ranking(ranking):
    """Weight vector w such that scoring one-hot embeddings yields `ranking`.

    With V = I and embedding e_i = one-hot(i), score_i = w[i]; ranking is
    argsort(w) descending, so assign descending weights along `ranking`.
    """
    n = len(ranking)
    w = torch.zeros(n)
    for pos, cand in enumerate(ranking):
        w[cand] = float(n - pos)
    return w


def _make_pool():
    """VoterPool over 3 candidates whose voters split borda vs. plurality.

    Rankings: three [0,1,2], plus [1,2,0], [2,1,0], [1,2,0].
    Plurality winner = 0 (3 first-place votes); Borda winner = 1.
    """
    V = torch.eye(3)
    pool = VoterPool(V)
    rankings = [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
    ]
    for i, r in enumerate(rankings):
        pool.add_voter(f"u{i}", _weight_for_ranking(r))
    return pool


def _patch_generation(monkeypatch, pool):
    """Stub response generation + embedding so no models are needed."""
    monkeypatch.setattr(
        dr, "generate_responses",
        lambda query, k, model, tokenizer: [f"r{i}" for i in range(k)],
    )
    # One-hot embeddings for the 3 responses -> score_i == w[i].
    monkeypatch.setattr(
        pool, "embed_responses",
        lambda responses, query=None: torch.eye(len(responses)),
    )


def test_default_strategies_match_previous_hardcoded_behavior():
    from apa.levers.voter_sampling import random_sampling
    from apa.levers.voter_aggregation import borda_count

    pool = _make_pool()
    di = DemocraticInference(pool, model=_Fake(), tokenizer=_Fake())

    assert di.sampler is random_sampling
    assert di.aggregator is borda_count
    assert di.sample_strategy == "random"
    assert di.aggregate_strategy == "borda_count"


def test_default_borda_winner(monkeypatch):
    pool = _make_pool()
    # m_voters >= n_voters -> all voters sampled, so the winner is deterministic
    di = DemocraticInference(pool, k_responses=3, m_voters=6, model=_Fake(), tokenizer=_Fake())
    _patch_generation(monkeypatch, pool)

    result = di("q")
    assert result.winner_idx == 1  # Borda winner


def test_plurality_strategy_changes_winner(monkeypatch):
    from apa.levers.voter_aggregation import plurality

    pool = _make_pool()
    di = DemocraticInference(
        pool, k_responses=3, m_voters=6, model=_Fake(), tokenizer=_Fake(),
        aggregate_strategy="plurality",
    )
    assert di.aggregator is plurality

    _patch_generation(monkeypatch, pool)
    result = di("q")
    assert result.winner_idx == 0  # Plurality winner differs from Borda's (1)


def test_unknown_strategy_raises():
    import pytest

    pool = _make_pool()
    with pytest.raises(ValueError):
        DemocraticInference(pool, model=_Fake(), tokenizer=_Fake(), aggregate_strategy="nope")
