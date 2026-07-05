"""Shared utilities for the APA package."""

from __future__ import annotations

from datetime import datetime


def log_timestamped(message: str) -> None:
    """Print a timestamped, flushed log line.

    Format matches the loggers previously copy-pasted across load_prism.py and
    train_lore_bases.py: ``[YYYY-MM-DD HH:MM:SS] message``.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
