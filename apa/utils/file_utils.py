"""
File utilities for APA project.

Provides functions for saving files to NAS with local symlinks,
and checkpoint management for long-running processes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Set

import pandas as pd
import torch


def save_with_symlink(
    data: pd.DataFrame | torch.Tensor | dict,
    nas_path: Path,
    local_path: Path | None = None,
    sep: str = '\t',
) -> None:
    """
    Save data to NAS and optionally create symlink in local directory.

    Args:
        data: DataFrame, tensor, or dict to save
        nas_path: Path on NAS storage for the actual file
        local_path: Optional path for the local symlink
        sep: CSV separator (default: tab) - only for DataFrames
    """
    # Ensure parent directories exist
    nas_path.parent.mkdir(parents=True, exist_ok=True)

    # Save based on type
    if isinstance(data, pd.DataFrame):
        data.to_csv(nas_path, sep=sep, index=False)
    elif isinstance(data, torch.Tensor):
        torch.save(data, nas_path)
    elif isinstance(data, dict):
        torch.save(data, nas_path)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    print(f"Saved to {nas_path}")

    # Create/update symlink if local path provided
    if local_path is not None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() or local_path.is_symlink():
            local_path.unlink()
        local_path.symlink_to(nas_path)
        print(f"Created symlink: {local_path} -> {nas_path}")


def load_from_nas(nas_path: Path) -> Any:
    """
    Load data from NAS path.

    Automatically detects file type based on extension.
    """
    if not nas_path.exists():
        raise FileNotFoundError(f"File not found: {nas_path}")

    suffix = nas_path.suffix.lower()

    if suffix in ['.pt', '.pth']:
        return torch.load(nas_path, map_location='cpu')
    elif suffix == '.pkl':
        import pickle
        with open(nas_path, 'rb') as f:
            return pickle.load(f)
    elif suffix in ['.csv', '.tsv']:
        sep = '\t' if suffix == '.tsv' else ','
        return pd.read_csv(nas_path, sep=sep)
    else:
        raise ValueError(f"Unknown file extension: {suffix}")


class CheckpointManager:
    """
    Manages checkpointing for long-running training processes.

    Tracks progress and enables resumption from saved state.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        name: str,
        checkpoint_interval: int = 100,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            name: Base name for checkpoint files
            checkpoint_interval: Save checkpoint every N iterations
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.name = name
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._iteration = 0
        self._state: dict[str, Any] = {}

    @property
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint file."""
        return self.checkpoint_dir / f"{self.name}_checkpoint.pt"

    def save_checkpoint(self, state: dict[str, Any], iteration: int) -> None:
        """
        Save checkpoint to disk.

        Args:
            state: Dictionary of state to save
            iteration: Current iteration number
        """
        checkpoint = {
            'iteration': iteration,
            'state': state,
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"[Checkpoint] Saved at iteration {iteration}")

    def load_checkpoint(self) -> tuple[dict[str, Any], int] | None:
        """
        Load checkpoint from disk if it exists.

        Returns:
            Tuple of (state dict, iteration) or None if no checkpoint
        """
        if not self.checkpoint_path.exists():
            return None

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        print(f"[Checkpoint] Loaded from iteration {checkpoint['iteration']}")
        return checkpoint['state'], checkpoint['iteration']

    def maybe_save(self, state: dict[str, Any], iteration: int, force: bool = False) -> None:
        """
        Save checkpoint if interval reached or forced.

        Args:
            state: State to save
            iteration: Current iteration
            force: If True, save regardless of interval
        """
        if force or (iteration > 0 and iteration % self.checkpoint_interval == 0):
            self.save_checkpoint(state, iteration)

    def cleanup(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"[Checkpoint] Removed: {self.checkpoint_path}")
