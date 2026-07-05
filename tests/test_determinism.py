"""
Determinism tests: a fixed seed must reproduce results exactly.

All GPU-free — the LoRe trainer is forced onto CPU so the reproduction check is
robust to GPU reduction-order nondeterminism (that caveat is covered by the
opt-in strict --deterministic mode, exercised separately below).
"""

import random

import numpy as np
import pytest
import torch

from apa.utils import set_seed
from apa.train_lore_bases import LoReTrainer
from apa.levers.voter_sampling import random_sampling, weighted_sampling
from apa.historical_prefs import _profile_suffix


def _force_cpu(monkeypatch):
    monkeypatch.setattr("apa.train_lore_bases.get_device", lambda: torch.device("cpu"))


def test_set_seed_torch_reproducible():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_python_and_numpy_reproducible():
    set_seed(7)
    a = (np.random.rand(4).tolist(), [random.random() for _ in range(4)])
    set_seed(7)
    b = (np.random.rand(4).tolist(), [random.random() for _ in range(4)])
    assert a == b


def test_lore_trainer_init_reproducible(monkeypatch):
    _force_cpu(monkeypatch)
    V_sft = torch.randn(16, 1)

    def build():
        return LoReTrainer(
            V_sft=V_sft, alpha=1e4, num_classes=5, num_features=16,
            num_basis_vectors=4, num_iterations=10, learning_rate=0.5,
        )

    set_seed(0)
    t1 = build()
    set_seed(0)
    t2 = build()
    assert torch.equal(t1.W, t2.W)
    assert torch.equal(t1.V, t2.V)


def test_lore_training_reproducible(monkeypatch):
    _force_cpu(monkeypatch)

    def make_X():
        # Fixed generator so the training data is identical across runs and
        # independent of the global RNG consumed by trainer init.
        g = torch.Generator().manual_seed(999)
        return [torch.randn(8, 16, generator=g) for _ in range(5)]

    def run():
        set_seed(0)
        trainer = LoReTrainer(
            V_sft=torch.zeros(16, 1), alpha=0.0, num_classes=5, num_features=16,
            num_basis_vectors=4, num_iterations=50, learning_rate=0.1, log_interval=10_000,
        )
        return trainer.train_model(make_X())

    W1, V1 = run()
    W2, V2 = run()
    assert torch.equal(W1, W2)
    assert torch.equal(V1, V2)


def test_seeded_voter_sampling_reproducible():
    users = [f"u{i}" for i in range(50)]
    assert random_sampling(users, None, 10, {"seed": 1}) == random_sampling(users, None, 10, {"seed": 1})

    meta = {u: {"weight": i + 1} for i, u in enumerate(users)}
    assert weighted_sampling(users, meta, 10, {"seed": 1}) == weighted_sampling(users, meta, 10, {"seed": 1})


def test_profile_suffix_stable_and_bounded():
    assert _profile_suffix("a helpful assistant") == _profile_suffix("a helpful assistant")
    assert 0 <= _profile_suffix("anything") < 10000


def test_strict_deterministic_mode_runs():
    try:
        set_seed(3, deterministic=True)
        _ = torch.randn(4) @ torch.randn(4)
    finally:
        # Reset so strict mode doesn't leak into other tests in the session.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
