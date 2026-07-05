"""Shared utilities for the APA package."""

from __future__ import annotations

import os
import random
from datetime import datetime


def log_timestamped(message: str) -> None:
    """Print a timestamped, flushed log line.

    Format matches the loggers previously copy-pasted across load_prism.py and
    train_lore_bases.py: ``[YYYY-MM-DD HH:MM:SS] message``.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs used by the pipeline for reproducible runs.

    Seeds Python ``random``, NumPy, and PyTorch (CPU + all CUDA devices). Call
    this once at the start of an entry point, after ``configure_environment()``.

    Args:
        seed: The seed value.
        deterministic: If True, additionally request strict deterministic
            algorithms (bitwise-reproducible on the same hardware). This can
            slow training and, for ops lacking a deterministic kernel, emit a
            warning (``warn_only=True``) rather than raising. Off by default.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required for deterministic cuBLAS GEMMs; must be set before first use.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
