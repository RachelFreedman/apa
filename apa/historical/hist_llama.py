"""
Historical LLM loading utilities.

Provides functions for loading ProgressGym HistLlama models
for different historical centuries.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from apa.config import HistLlamaConfig, HF_CACHE_DIR, configure_environment


def find_latest_model_version(config: HistLlamaConfig) -> str:
    """
    Find the latest available model version on HuggingFace.

    Args:
        config: HistLlama configuration

    Returns:
        Version string (e.g., "v0.2")
    """
    api = HfApi()
    base_name = f"ProgressGym-HistLlama3-{config.size}-{config.century}-instruct"

    try:
        models = api.list_models(author=config.hf_org, search=base_name)

        versions: list[tuple[float, str]] = []
        for model in models:
            model_id = getattr(model, 'id', None) or getattr(model, 'modelId', None)
            if model_id and base_name in model_id:
                version_part = model_id.split('-')[-1]
                if version_part.startswith('v'):
                    try:
                        version_num = float(version_part[1:])
                        versions.append((version_num, version_part))
                    except ValueError:
                        pass

        if versions:
            versions.sort(reverse=True)
            latest = versions[0][1]
            print(f"Found versions: {[v[1] for v in versions]}, using {latest}")
            return latest
        else:
            print(f"No matching model versions found, using {config.default_version}")
            return config.default_version

    except Exception as e:
        print(f"Could not query HuggingFace for model versions: {e}")
        print(f"Using default {config.default_version}")
        return config.default_version


def load_hist_llama(
    century: str,
    size: str = "8B",
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> Tuple[Any, Any]:
    """
    Load HistLlama model and tokenizer for a specific century.

    Args:
        century: Century code (e.g., "C013" for 13th century)
        size: Model size ("8B" or "70B")
        device_map: Device mapping strategy
        cache_dir: Cache directory (uses default if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Configure environment
    configure_environment()

    config = HistLlamaConfig(size=size, century=century)

    if cache_dir is None:
        cache_dir = str(HF_CACHE_DIR)

    # Find latest version
    version = find_latest_model_version(config)
    model_name = f"{config.hf_org}/ProgressGym-HistLlama3-{size}-{century}-instruct-{version}"

    print(f"Loading HistLlama model: {model_name}")
    print("This may take a while on first run as it downloads the model...")
    print(f"Cache directory: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    print("Model loaded successfully.")
    return model, tokenizer


def get_available_centuries() -> list[str]:
    """Get list of available century codes."""
    return list(HistLlamaConfig.VALID_CENTURIES)


def century_to_name(century: str) -> str:
    """Convert century code to human-readable name."""
    mapping = {
        "C013": "13th Century",
        "C014": "14th Century",
        "C015": "15th Century",
        "C016": "16th Century",
        "C017": "17th Century",
        "C018": "18th Century",
        "C019": "19th Century",
        "C020": "20th Century",
        "C021": "21st Century",
    }
    return mapping.get(century, century)
