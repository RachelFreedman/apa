"""
Centralized configuration for APA (Aggregated Preference Alignment) project.

This module provides path configuration, model parameters, and inference settings.
Paths are configured to use NAS storage for large files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os


# =============================================================================
# Base Paths
# =============================================================================

NAS_BASE = Path("/nas/ucb/rachel/APA")
LOCAL_BASE = Path(__file__).parent.parent

# NAS paths
EMBEDDINGS_DIR = NAS_BASE / "embeddings"
MODELS_DIR = NAS_BASE / "models"
HF_CACHE_DIR = NAS_BASE / "hf_cache"
PRISM_DATA_DIR = NAS_BASE / "data" / "prism"

# Historical prefs data (already processed)
HISTORICAL_PREFS_DATA = Path("/nas/ucb/rachel/historical-prefs/data")


# =============================================================================
# Environment Configuration
# =============================================================================

def configure_environment() -> None:
    """Configure environment variables for HuggingFace and temp directories."""
    os.environ['HF_HOME'] = str(HF_CACHE_DIR)
    os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR)
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(HF_CACHE_DIR / "sentence_transformers")
    os.environ['TMPDIR'] = str(NAS_BASE / "tmp")
    os.environ['TEMP'] = str(NAS_BASE / "tmp")
    os.environ['TMP'] = str(NAS_BASE / "tmp")

    # Ensure directories exist
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (HF_CACHE_DIR / "sentence_transformers").mkdir(parents=True, exist_ok=True)
    (NAS_BASE / "tmp").mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset Configuration
# =============================================================================

DatasetName = Literal["prism"]


@dataclass
class DatasetConfig:
    """Configuration for PRISM dataset."""
    name: DatasetName = "prism"

    @property
    def questions_pairwise_path(self) -> Path:
        """Path to pairwise questions CSV."""
        return HISTORICAL_PREFS_DATA / "prism" / "questions_pairwise.csv"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for precomputed embeddings."""
        return EMBEDDINGS_DIR

    @property
    def models_dir(self) -> Path:
        """Directory for model checkpoints."""
        return MODELS_DIR

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class InferenceLLMConfig:
    """Configuration for the base LLM used in inference."""

    # Using Qwen2.5 as it's ungated (no HF auth required)
    # Alternatives: "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 1.2  # Higher for diversity
    do_sample: bool = True


# =============================================================================
# LoRe Configuration
# =============================================================================

@dataclass
class LoReConfig:
    """
    Configuration for Low-rank Reward modeling.

    CRITICAL: These hyperparameters match the original LoRe paper exactly.
    Changing them may result in performance mismatch with published results.

    Performance targets (from LoRe paper):
    | Rank | Train Acc | Seen/Unseen | Few-Shot | Unseen/Unseen |
    |------|-----------|-------------|----------|---------------|
    | 0    | 71.56%    | 71.56%      | 73.55%   | 71.20%        |
    | 1    | 76.18%    | 76.59%      | 76.90%   | 76.06%        |
    | 5    | 87.90%    | 87.75%      | 88.30%   | 87.92%        |
    | 10   | 90.05%    | 89.76%      | 91.57%   | 91.25%        |
    """

    # Training hyperparameters (MUST match LoRe paper)
    K_list: list[int] = field(default_factory=lambda: [0, 1])  # Start with tested ranks
    alpha: float = 10000.0  # Regularization strength
    num_iterations: int = 20000  # Training iterations (NOT epochs!)
    learning_rate: float = 0.5  # CRITICAL: 0.5, NOT 1e-4
    logits_scale: float = 100.0  # Division factor in NLL loss
    threshold: float = 1e-2  # Dimension filtering threshold

    # Few-shot personalization for unseen users
    few_shot_iterations: int = 500
    few_shot_lr: float = 0.5

    # Embedding model (Skywork-Reward for alignment with LoRe paper)
    embedding_model: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    embedding_dim: int = 4096  # Llama 3.1 8B hidden dimension

    # Logging
    log_interval: int = 2000


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for democratic inference."""

    k_responses: int = 5  # Number of alternative responses to generate
    m_voters: int = 10  # Number of user models to sample

    # Default aggregation method (read by apa.democratic_response CLI default)
    aggregate_strategy: str = "borda_count"


# =============================================================================
# Main Configuration Class
# =============================================================================

@dataclass
class APAConfig:
    """Main configuration container for APA project."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inference_llm: InferenceLLMConfig = field(default_factory=InferenceLLMConfig)
    lore: LoReConfig = field(default_factory=LoReConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Centuries to use for historical users
    historical_centuries: list[str] = field(
        default_factory=lambda: ["C013", "C017", "C019", "C021"]
    )

# =============================================================================
# Global instances
# =============================================================================

def get_config() -> APAConfig:
    """Get default configuration."""
    return APAConfig()
