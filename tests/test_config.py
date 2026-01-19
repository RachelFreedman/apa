"""
Unit tests for configuration module.
"""

import pytest
from pathlib import Path

from apa.config import (
    APAConfig,
    DatasetConfig,
    HistLlamaConfig,
    InferenceConfig,
    LoReConfig,
    InferenceLLMConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetConfig()

        assert config.name == "prism"
        assert "prism" in str(config.questions_pairwise_path)

    def test_path_properties(self):
        """Test path property returns Path objects."""
        config = DatasetConfig()

        assert isinstance(config.questions_pairwise_path, Path)
        assert isinstance(config.embeddings_dir, Path)
        assert isinstance(config.models_dir, Path)


class TestHistLlamaConfig:
    """Tests for HistLlamaConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HistLlamaConfig()

        assert config.size == "8B"
        assert config.century == "C013"
        assert config.max_new_tokens == 20
        assert config.temperature == 0.9

    def test_model_name_property(self):
        """Test model_name property."""
        config = HistLlamaConfig(size="8B", century="C017")

        assert "HistLlama3-8B-C017" in config.model_name
        assert config.hf_org in config.model_name

    def test_valid_centuries(self):
        """Test valid centuries tuple."""
        config = HistLlamaConfig()

        assert "C013" in config.VALID_CENTURIES
        assert "C021" in config.VALID_CENTURIES
        assert len(config.VALID_CENTURIES) == 9

    def test_valid_sizes(self):
        """Test valid sizes tuple."""
        config = HistLlamaConfig()

        assert "8B" in config.VALID_SIZES
        assert "70B" in config.VALID_SIZES


class TestLoReConfig:
    """Tests for LoReConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoReConfig()

        assert config.K_list == [0, 1]
        assert config.alpha == 10000.0
        assert config.num_iterations == 20000
        assert config.learning_rate == 0.5
        assert config.logits_scale == 100.0
        assert config.threshold == 1e-2
        assert config.few_shot_iterations == 500
        assert config.few_shot_lr == 0.5
        assert config.embedding_dim == 4096
        assert config.log_interval == 2000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoReConfig(K_list=[0, 1, 5], alpha=5000.0, num_iterations=10000)

        assert config.K_list == [0, 1, 5]
        assert config.alpha == 5000.0
        assert config.num_iterations == 10000


class TestInferenceConfig:
    """Tests for InferenceConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InferenceConfig()

        assert config.k_responses == 5
        assert config.m_voters == 10
        assert config.generate_strategy == "temperature_sampling"
        assert config.sample_strategy == "random"
        assert config.aggregate_strategy == "borda_count"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = InferenceConfig(
            k_responses=10,
            m_voters=20,
            aggregate_strategy="plurality",
        )

        assert config.k_responses == 10
        assert config.m_voters == 20
        assert config.aggregate_strategy == "plurality"


class TestInferenceLLMConfig:
    """Tests for InferenceLLMConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InferenceLLMConfig()

        # Model name should be a valid HuggingFace model
        assert "/" in config.model_name  # Format: org/model
        assert config.max_new_tokens == 512
        assert config.temperature == 1.2
        assert config.do_sample is True


class TestAPAConfig:
    """Tests for APAConfig class."""

    def test_default_values(self):
        """Test default configuration creates nested configs."""
        config = APAConfig()

        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.hist_llama, HistLlamaConfig)
        assert isinstance(config.inference_llm, InferenceLLMConfig)
        assert isinstance(config.lore, LoReConfig)
        assert isinstance(config.inference, InferenceConfig)

    def test_historical_centuries(self):
        """Test historical centuries default."""
        config = APAConfig()

        assert "C013" in config.historical_centuries
        assert "C021" in config.historical_centuries

    def test_yaml_roundtrip(self, tmp_path):
        """Test saving and loading from YAML."""
        config = APAConfig()
        config.inference.k_responses = 10
        config.lore.alpha = 5000.0

        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        loaded = APAConfig.from_yaml(yaml_path)

        assert loaded.inference.k_responses == 10
        assert loaded.lore.alpha == 5000.0
