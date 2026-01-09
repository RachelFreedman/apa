"""
Embedding utilities for APA project.

Provides functions for generating sentence embeddings using
sentence-transformers models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Set HuggingFace cache to NAS if not already set
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/nas/ucb/rachel/APA/hf_cache"
if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/nas/ucb/rachel/APA/hf_cache/sentence_transformers"

# Global cache for embedding model
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_NAME = None


def get_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str | None = None,
    cache_folder: str | None = None,
) -> Any:
    """
    Get or load the sentence embedding model.

    Uses a global cache to avoid reloading the model multiple times.

    Args:
        model_name: HuggingFace model name for sentence-transformers
        device: Device to load model on (auto-detected if None)
        cache_folder: Optional folder to cache model weights

    Returns:
        SentenceTransformer model
    """
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_NAME

    if _EMBEDDING_MODEL is not None and _EMBEDDING_MODEL_NAME == model_name:
        return _EMBEDDING_MODEL

    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cache_folder is None:
        cache_folder = os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME",
            "/nas/ucb/rachel/APA/hf_cache/sentence_transformers"
        )

    print(f"Loading embedding model: {model_name}")
    print(f"Cache folder: {cache_folder}")
    _EMBEDDING_MODEL = SentenceTransformer(model_name, device=device, cache_folder=cache_folder)
    _EMBEDDING_MODEL_NAME = model_name

    return _EMBEDDING_MODEL


def embed_text(
    text: str,
    model: Any | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a single text string.

    Args:
        text: Text to embed
        model: Pre-loaded model (optional)
        model_name: Model name if model not provided
        normalize: Whether to L2-normalize the embedding

    Returns:
        Embedding vector as numpy array
    """
    if model is None:
        model = get_embedding_model(model_name)

    embedding = model.encode(text, normalize_embeddings=normalize)
    return embedding


def embed_texts(
    texts: list[str],
    model: Any | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed multiple text strings.

    Args:
        texts: List of texts to embed
        model: Pre-loaded model (optional)
        model_name: Model name if model not provided
        batch_size: Batch size for encoding
        normalize: Whether to L2-normalize embeddings
        show_progress: Whether to show progress bar

    Returns:
        Embeddings as numpy array of shape (n_texts, embed_dim)
    """
    if model is None:
        model = get_embedding_model(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
    )

    return embeddings


def embed_response_pairs(
    prompts: list[str],
    responses_1: list[str],
    responses_2: list[str],
    model: Any | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 32,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """
    Embed prompts and response pairs for preference learning.

    Concatenates prompt with each response before embedding to capture
    the context of the response.

    Args:
        prompts: List of prompt texts
        responses_1: List of first responses
        responses_2: List of second responses
        model: Pre-loaded model (optional)
        model_name: Model name if model not provided
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with:
            - 'prompt_embeddings': (n, d) prompt embeddings
            - 'response_1_embeddings': (n, d) response 1 embeddings (with context)
            - 'response_2_embeddings': (n, d) response 2 embeddings (with context)
    """
    if model is None:
        model = get_embedding_model(model_name)

    n = len(prompts)
    assert len(responses_1) == n and len(responses_2) == n

    # Create contextualized response texts
    ctx_responses_1 = [f"{p}\n\n{r}" for p, r in zip(prompts, responses_1)]
    ctx_responses_2 = [f"{p}\n\n{r}" for p, r in zip(prompts, responses_2)]

    print("Embedding prompts...")
    prompt_embeddings = embed_texts(
        prompts, model=model, batch_size=batch_size, show_progress=show_progress
    )

    print("Embedding response 1s...")
    response_1_embeddings = embed_texts(
        ctx_responses_1, model=model, batch_size=batch_size, show_progress=show_progress
    )

    print("Embedding response 2s...")
    response_2_embeddings = embed_texts(
        ctx_responses_2, model=model, batch_size=batch_size, show_progress=show_progress
    )

    return {
        'prompt_embeddings': prompt_embeddings,
        'response_1_embeddings': response_1_embeddings,
        'response_2_embeddings': response_2_embeddings,
    }


def save_embeddings(embeddings: dict[str, np.ndarray], path: Path) -> None:
    """Save embeddings to disk."""
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {path}")


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load embeddings from disk."""
    import pickle
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings from {path}")
    return embeddings
