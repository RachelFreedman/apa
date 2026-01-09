"""
Response generation for democratic inference.

Provides utilities for loading the base LLM and generating
diverse responses to queries.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from apa.config import InferenceLLMConfig, HF_CACHE_DIR, configure_environment
from apa.levers import lever_generate_responses


# Global cache for inference model
_MODEL = None
_TOKENIZER = None
_MODEL_NAME = None


def load_inference_llm(
    model_name: str | None = None,
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> Tuple[Any, Any]:
    """
    Load the base LLM for response generation.

    Uses a global cache to avoid reloading.

    Args:
        model_name: HuggingFace model name (uses default if None)
        device_map: Device mapping strategy
        cache_dir: Cache directory (uses default if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    global _MODEL, _TOKENIZER, _MODEL_NAME

    config = InferenceLLMConfig()
    if model_name is None:
        model_name = config.model_name

    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL, _TOKENIZER

    configure_environment()

    if cache_dir is None:
        cache_dir = str(HF_CACHE_DIR)

    print(f"Loading inference LLM: {model_name}")
    print(f"Cache directory: {cache_dir}")

    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    _MODEL_NAME = model_name

    print("Inference LLM loaded successfully.")
    return _MODEL, _TOKENIZER


def generate_responses(
    query: str,
    k: int = 5,
    model: Any | None = None,
    tokenizer: Any | None = None,
    config: dict | None = None,
) -> list[str]:
    """
    Generate k diverse responses to a query.

    This is a convenience wrapper around the lever_generate_responses function.

    Args:
        query: The user query
        k: Number of responses to generate
        model: Pre-loaded model (loads default if None)
        tokenizer: Pre-loaded tokenizer (loads default if None)
        config: Configuration dict for the lever

    Returns:
        List of k response strings
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_inference_llm()

    if config is None:
        config = {
            'generate': 'temperature_sampling',
            'temperature': 1.2,
            'max_new_tokens': 512,
        }

    return lever_generate_responses(model, tokenizer, query, k, config)


def generate_single_response(
    query: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a single response to a query.

    Args:
        query: The user query
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        temperature: Sampling temperature
        max_new_tokens: Maximum new tokens to generate

    Returns:
        Response string
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_inference_llm()

    # Format as chat message
    messages = [{"role": "user", "content": query}]

    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"User: {query}\n\nAssistant:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt):].strip()

    return response
