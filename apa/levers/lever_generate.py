"""
Lever: Response Generation Strategies.

INJECTION POINT: This module controls how diverse responses are generated
for a given query. The default is temperature sampling, but this can be
replaced with more sophisticated methods.

To add a new strategy:
1. Create a new function following the same signature
2. Register it in the STRATEGIES dict
3. Update config to use the new strategy name
"""

from __future__ import annotations

from typing import Any, Callable

import torch


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGIES: dict[str, Callable] = {}


def register_strategy(name: str):
    """Decorator to register a generation strategy."""
    def decorator(fn: Callable) -> Callable:
        STRATEGIES[name] = fn
        return fn
    return decorator


# =============================================================================
# Main Interface
# =============================================================================

def lever_generate_responses(
    model: Any,
    tokenizer: Any,
    query: str,
    k: int,
    config: dict,
) -> list[str]:
    """
    INJECTION POINT: Generate k diverse responses to a query.

    This is the main entry point. It dispatches to the appropriate strategy
    based on the config.

    Args:
        model: The language model (e.g., Llama 3.1)
        tokenizer: The model's tokenizer
        query: The user query to respond to
        k: Number of responses to generate
        config: Configuration dict with 'generate' key for strategy name

    Returns:
        List of k response strings
    """
    strategy_name = config.get('generate', 'temperature_sampling')

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown generation strategy: {strategy_name}. "
                        f"Available: {list(STRATEGIES.keys())}")

    return STRATEGIES[strategy_name](model, tokenizer, query, k, config)


# =============================================================================
# Strategy Implementations
# =============================================================================

@register_strategy("temperature_sampling")
def temperature_sampling(
    model: Any,
    tokenizer: Any,
    query: str,
    k: int,
    config: dict,
) -> list[str]:
    """
    Generate responses using temperature sampling.

    Default strategy: Simple temperature sampling with high temperature
    to encourage diversity.

    Args:
        model: The language model
        tokenizer: The tokenizer
        query: The query to respond to
        k: Number of responses
        config: Config dict with optional 'temperature' key

    Returns:
        List of k responses
    """
    temperature = config.get('temperature', 1.2)
    max_new_tokens = config.get('max_new_tokens', 512)

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

    responses = []
    for _ in range(k):
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
        responses.append(response)

    return responses


@register_strategy("diverse_beam")
def diverse_beam_search(
    model: Any,
    tokenizer: Any,
    query: str,
    k: int,
    config: dict,
) -> list[str]:
    """
    Generate responses using diverse beam search.

    PLACEHOLDER: This is a stub for future implementation.
    Uses diversity penalty to encourage different responses.
    """
    # TODO: Implement diverse beam search
    # For now, fall back to temperature sampling
    print("[lever_generate] diverse_beam not implemented, using temperature_sampling")
    return temperature_sampling(model, tokenizer, query, k, config)


@register_strategy("contrastive_decode")
def contrastive_decoding(
    model: Any,
    tokenizer: Any,
    query: str,
    k: int,
    config: dict,
) -> list[str]:
    """
    Generate responses using contrastive decoding.

    PLACEHOLDER: This is a stub for future implementation.
    Uses contrastive decoding against a smaller model.
    """
    # TODO: Implement contrastive decoding
    print("[lever_generate] contrastive_decode not implemented, using temperature_sampling")
    return temperature_sampling(model, tokenizer, query, k, config)
