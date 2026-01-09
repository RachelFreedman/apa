"""
Centralized configuration for APA (Aggregated Preference Alignment) project.

This module provides path configuration, model parameters, and inference settings.
Paths are configured to use NAS storage for large files with symlinks in local workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal
import os
import yaml


# =============================================================================
# Base Paths
# =============================================================================

NAS_BASE = Path("/nas/ucb/rachel/APA")
LOCAL_BASE = Path(__file__).parent.parent
DATA_DIR = NAS_BASE / "data"
CHECKPOINTS_DIR = NAS_BASE / "checkpoints"
HF_CACHE_DIR = NAS_BASE / "hf_cache"

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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


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
    def embeddings_path(self) -> Path:
        """Path to precomputed embeddings."""
        return DATA_DIR / "prism" / "embeddings.pkl"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for LoRe checkpoints."""
        return CHECKPOINTS_DIR / "prism"

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "prism").mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class HistLlamaConfig:
    """Configuration for ProgressGym historical models."""

    size: str = "8B"  # "8B" or "70B"
    century: str = "C013"  # C013-C021

    VALID_SIZES: ClassVar[tuple[str, ...]] = ("8B", "70B")
    VALID_CENTURIES: ClassVar[tuple[str, ...]] = (
        "C013", "C014", "C015", "C016", "C017",
        "C018", "C019", "C020", "C021"
    )

    # Generation parameters
    max_new_tokens: int = 20
    temperature: float = 0.9
    do_sample: bool = True

    # HuggingFace model naming
    hf_org: str = "PKU-Alignment"
    default_version: str = "v0.2"

    @property
    def model_name(self) -> str:
        """Full HuggingFace model path."""
        return f"{self.hf_org}/ProgressGym-HistLlama3-{self.size}-{self.century}-instruct-{self.default_version}"


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
    """Configuration for Low-rank Reward modeling."""

    rank: int = 8  # Low-rank dimension K
    alpha: float = 10000.0  # Regularization coefficient
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 32

    # Embedding model
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768  # Dimension of all-mpnet-base-v2


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for democratic inference."""

    k_responses: int = 5  # Number of alternative responses to generate
    m_voters: int = 10  # Number of user models to sample

    # Lever defaults
    generate_strategy: str = "temperature_sampling"
    sample_strategy: str = "random"
    aggregate_strategy: str = "borda_count"
    question_strategy: str = "random_subset"


# =============================================================================
# Main Configuration Class
# =============================================================================

@dataclass
class APAConfig:
    """Main configuration container for APA project."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hist_llama: HistLlamaConfig = field(default_factory=HistLlamaConfig)
    inference_llm: InferenceLLMConfig = field(default_factory=InferenceLLMConfig)
    lore: LoReConfig = field(default_factory=LoReConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Centuries to use for historical users
    historical_centuries: list[str] = field(
        default_factory=lambda: ["C013", "C017", "C019", "C021"]
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "APAConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Build nested configs
        config = cls()
        if 'dataset' in data:
            config.dataset = DatasetConfig(**data['dataset'])
        if 'hist_llama' in data:
            config.hist_llama = HistLlamaConfig(**data['hist_llama'])
        if 'inference_llm' in data:
            config.inference_llm = InferenceLLMConfig(**data['inference_llm'])
        if 'lore' in data:
            config.lore = LoReConfig(**data['lore'])
        if 'inference' in data:
            config.inference = InferenceConfig(**data['inference'])
        if 'historical_centuries' in data:
            config.historical_centuries = data['historical_centuries']

        return config

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        import dataclasses

        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()
                        if not k.startswith('VALID_')}
            elif isinstance(obj, (list, tuple)):
                return [to_dict(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = to_dict(self)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# =============================================================================
# Global instances
# =============================================================================

def get_config() -> APAConfig:
    """Get default configuration."""
    return APAConfig()
