"""
Embedding utilities for APA project.

Provides functions for generating embeddings using the Skywork-Reward model,
following the LoRe paper methodology (last token hidden state extraction).
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

# Global cache for embedding model and tokenizer
_EMBEDDING_MODEL = None
_EMBEDDING_TOKENIZER = None
_EMBEDDING_MODEL_NAME = None


def get_embedding_model(
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    device: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any]:
    """
    Get or load the Skywork-Reward embedding model and tokenizer.

    Uses a global cache to avoid reloading the model multiple times.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on (auto-detected if None)
        torch_dtype: Torch dtype for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    global _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER, _EMBEDDING_MODEL_NAME

    if _EMBEDDING_MODEL is not None and _EMBEDDING_MODEL_NAME == model_name:
        return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = os.environ.get("HF_HOME", "/nas/ucb/rachel/APA/hf_cache")

    print(f"Loading embedding model: {model_name}")
    print(f"Device: {device}, dtype: {torch_dtype}")

    _EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    _EMBEDDING_MODEL = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        cache_dir=cache_dir,
        num_labels=1,
    )
    _EMBEDDING_MODEL.eval()
    _EMBEDDING_MODEL_NAME = model_name

    return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER


def _format_for_embedding(prompt: str, response: str, tokenizer: Any) -> str:
    """
    Format prompt and response as a chat conversation for embedding.

    Uses the tokenizer's chat template to format properly.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return formatted


def _extract_embedding(
    model: Any,
    tokenizer: Any,
    text: str,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract embedding from the last token's hidden state.

    Following LoRe paper: uses the last hidden state of the last token.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )
        # Get last layer hidden state, last token
        embedding = outputs.hidden_states[-1][0, -1, :]

    return embedding.float().cpu().numpy()


def embed_text(
    text: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
) -> np.ndarray:
    """
    Embed a single text string.

    Args:
        text: Text to embed (should be formatted as chat if prompt+response)
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        model_name: Model name if model not provided

    Returns:
        Embedding vector as numpy array (4096-dim for Llama 3.1 8B)
    """
    if model is None or tokenizer is None:
        model, tokenizer = get_embedding_model(model_name)

    device = next(model.parameters()).device
    return _extract_embedding(model, tokenizer, text, str(device))


def embed_texts(
    texts: list[str],
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    batch_size: int = 4,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed multiple text strings.

    Args:
        texts: List of texts to embed
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        model_name: Model name if model not provided
        batch_size: Batch size (smaller due to large model)
        show_progress: Whether to show progress bar

    Returns:
        Embeddings as numpy array of shape (n_texts, 4096)
    """
    if model is None or tokenizer is None:
        model, tokenizer = get_embedding_model(model_name)

    device = next(model.parameters()).device
    embeddings = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding texts")

    for i in iterator:
        batch_texts = texts[i:i + batch_size]

        # Process one at a time within batch due to variable lengths
        batch_embeddings = []
        for text in batch_texts:
            emb = _extract_embedding(model, tokenizer, text, str(device))
            batch_embeddings.append(emb)

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def embed_response_pairs(
    prompts: list[str],
    responses_1: list[str],
    responses_2: list[str],
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    batch_size: int = 4,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """
    Embed prompts and response pairs for preference learning.

    Formats each prompt+response as a chat conversation before embedding.

    Args:
        prompts: List of prompt texts
        responses_1: List of first responses
        responses_2: List of second responses
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        model_name: Model name if model not provided
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with:
            - 'prompt_embeddings': (n, 4096) prompt-only embeddings
            - 'response_1_embeddings': (n, 4096) prompt+response1 embeddings
            - 'response_2_embeddings': (n, 4096) prompt+response2 embeddings
    """
    if model is None or tokenizer is None:
        model, tokenizer = get_embedding_model(model_name)

    n = len(prompts)
    assert len(responses_1) == n and len(responses_2) == n

    # Format as chat conversations
    formatted_1 = [
        _format_for_embedding(p, r, tokenizer)
        for p, r in zip(prompts, responses_1)
    ]
    formatted_2 = [
        _format_for_embedding(p, r, tokenizer)
        for p, r in zip(prompts, responses_2)
    ]

    print("Embedding prompts...")
    prompt_embeddings = embed_texts(
        prompts, model=model, tokenizer=tokenizer,
        batch_size=batch_size, show_progress=show_progress
    )

    print("Embedding response 1s...")
    response_1_embeddings = embed_texts(
        formatted_1, model=model, tokenizer=tokenizer,
        batch_size=batch_size, show_progress=show_progress
    )

    print("Embedding response 2s...")
    response_2_embeddings = embed_texts(
        formatted_2, model=model, tokenizer=tokenizer,
        batch_size=batch_size, show_progress=show_progress
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


def load_embeddings(path: Path, remap_user_ids: bool = True) -> dict[str, np.ndarray]:
    """
    Load embeddings from disk.

    Args:
        path: Path to embeddings file
        remap_user_ids: If True and embeddings have interaction_id format (intXXX),
                       remap to proper user_id (userXXX) using HuggingFace mapping

    Returns:
        Embeddings dictionary with properly mapped user_ids
    """
    import pickle
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings from {path}")

    # Remap user_ids if they're in interaction_id format
    if remap_user_ids and 'user_ids' in embeddings:
        sample_id = str(embeddings['user_ids'][0])
        if sample_id.startswith('int'):
            embeddings = remap_interaction_to_user_ids(embeddings)

    return embeddings


def remap_interaction_to_user_ids(embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Remap interaction_ids to proper user_ids using HuggingFace PRISM data.

    The original pairwise CSV didn't have user_id, only interaction_id and conversation_id.
    This function maps conversation_id -> user_id to get actual participant identifiers.

    Args:
        embeddings: Embeddings dict with 'user_ids' containing interaction_id format (intXXX)

    Returns:
        Embeddings dict with 'user_ids' remapped to user_id format (userXXX)
    """
    from apa.data.prism_loader import load_prism_pairwise

    print("Remapping interaction_ids to user_ids...")

    # Load the pairwise data with user_id mapping
    df = load_prism_pairwise()

    # Create interaction_id to user_id mapping via question_id
    # The embeddings have question_ids that match the dataframe
    if 'question_ids' not in embeddings:
        raise ValueError("Embeddings must have 'question_ids' for remapping")

    # Create mapping from question_id to user_id
    question_to_user = dict(zip(df['question_id'], df['user_id']))

    # Remap user_ids
    old_user_ids = embeddings['user_ids']
    question_ids = embeddings['question_ids']
    new_user_ids = np.array([question_to_user.get(qid, None) for qid in question_ids])

    n_mapped = np.sum(new_user_ids != None)  # noqa: E711
    n_users = len(set(uid for uid in new_user_ids if uid is not None))
    print(f"  Remapped {n_mapped}/{len(new_user_ids)} samples to {n_users} unique users")

    embeddings['user_ids'] = new_user_ids
    return embeddings
