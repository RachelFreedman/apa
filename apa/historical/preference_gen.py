"""
Historical preference generation.

Provides functions for generating preference labels using
HistLlama models. Adapted from the historical-prefs repository.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from tqdm import tqdm

from apa.config import HistLlamaConfig


def parse_model_response(response: str) -> str:
    """
    Parse model response to extract preference choice.

    Args:
        response: Raw model response text

    Returns:
        '1' if option 1 preferred, '2' if option 2 preferred, '-1' if ambiguous
    """
    response_lower = response.lower().strip()

    # Empty response
    if not response_lower:
        return '-1'

    # Patterns that indicate choice 1 or 2
    patterns_1 = [
        r'\boption\s*1\b',
        r'\bption\s*1\b',
        r'\btion\s*1\b',
        r'\bthe\s+1\s+option\b',
        r'\b1\s*(?:st)?\s*option\b',
        r'\bchoice\s*1\b',
        r'\bresponse\s*1\b',
        r'\banswer\s*1\b',
        r'\bfirst\s+option\b',
        r'\bfirst\s+response\b',
        r'\b(?:is|be)\s+1\b',
        r'#1\b',
    ]

    patterns_2 = [
        r'\boption\s*2\b',
        r'\bption\s*2\b',
        r'\btion\s*2\b',
        r'\bthe\s+2\s+option\b',
        r'\b2\s*(?:nd)?\s*option\b',
        r'\bchoice\s*2\b',
        r'\bresponse\s*2\b',
        r'\banswer\s*2\b',
        r'\bsecond\s+option\b',
        r'\bsecond\s+response\b',
        r'\b(?:is|be)\s+2\b',
        r'#2\b',
    ]

    has_1 = any(re.search(p, response_lower) for p in patterns_1)
    has_2 = any(re.search(p, response_lower) for p in patterns_2)

    if has_1 and not has_2:
        return '1'
    elif has_2 and not has_1:
        return '2'
    elif has_1 and has_2:
        return '-1'  # Both mentioned - ambiguous

    # Fallback: check if response starts with '1' or '2'
    if response_lower.startswith('1'):
        return '1'
    elif response_lower.startswith('2'):
        return '2'

    # Secondary fallback for very short responses
    if len(response_lower) < 15:
        has_digit_1 = bool(re.search(r'\b1\b', response_lower))
        has_digit_2 = bool(re.search(r'\b2\b', response_lower))

        if has_digit_1 and not has_digit_2:
            return '1'
        elif has_digit_2 and not has_digit_1:
            return '2'

    return '-1'  # Invalid/unparseable response


def generate_single_preference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response_1: str,
    response_2: str,
    config: HistLlamaConfig | None = None,
    user_profile: str | None = None,
) -> str:
    """
    Generate a preference between two responses using the model.

    Args:
        model: The loaded HistLlama model
        tokenizer: The model's tokenizer
        prompt: The original question/prompt
        response_1: First response option
        response_2: Second response option
        config: Model configuration
        user_profile: Optional user profile description to condition response

    Returns:
        '1' if option 1 preferred, '2' if option 2 preferred, '-1' if invalid
    """
    if config is None:
        config = HistLlamaConfig()

    # Build comparison prompt with optional user profile
    if user_profile:
        comparison_content = (
            f"You are: {user_profile}\n\n"
            f"Given your perspective, which response do you prefer?\n\n"
            f"Question: {prompt}\n\n"
            f"Option 1: {response_1}\n\n"
            f"Option 2: {response_2}\n\n"
            f"Answer with only the number 1 or 2."
        )
    else:
        comparison_content = (
            f"Question: {prompt}\n\n"
            f"Option 1: {response_1}\n\n"
            f"Option 2: {response_2}\n\n"
            f"Which option is better? Answer with only the number 1 or 2."
        )

    # Use chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": comparison_content}]
        comparison_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        comparison_prompt = (
            f"{comparison_content}\n\n"
            "Answer with just the number \"1\" or just the number \"2\". "
            "Do not include any other text."
        )

    # Tokenize
    inputs = tokenizer(comparison_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(comparison_prompt):].strip()

    return parse_model_response(response)


def generate_historical_preferences(
    model: Any,
    tokenizer: Any,
    questions: list[dict],
    config: HistLlamaConfig | None = None,
    user_profile: str | None = None,
    n_runs: int = 1,
    show_progress: bool = True,
) -> list[dict]:
    """
    Generate preferences for multiple question pairs.

    For each question, runs in both original and reversed order to
    check for consistency.

    Args:
        model: The loaded HistLlama model
        tokenizer: The model's tokenizer
        questions: List of dicts with 'prompt', 'response_1', 'response_2'
        config: Model configuration
        user_profile: Optional user profile description
        n_runs: Number of times to run each comparison in each order
        show_progress: Whether to show progress bar

    Returns:
        List of preference results with original and reversed choices
    """
    if config is None:
        config = HistLlamaConfig()

    results = []
    iterator = tqdm(questions, desc="Generating preferences") if show_progress else questions

    for q in iterator:
        result = {
            'prompt': q['prompt'],
            'response_1': q.get('response_1', ''),
            'response_2': q.get('response_2', ''),
            'question_id': q.get('question_id', ''),
        }

        # Run original order
        original_choices = []
        for _ in range(n_runs):
            choice = generate_single_preference(
                model, tokenizer,
                q['prompt'],
                q['response_1'],
                q['response_2'],
                config=config,
                user_profile=user_profile,
            )
            original_choices.append(choice)

        # Run reversed order
        reversed_choices = []
        for _ in range(n_runs):
            choice = generate_single_preference(
                model, tokenizer,
                q['prompt'],
                q['response_2'],  # Swapped
                q['response_1'],  # Swapped
                config=config,
                user_profile=user_profile,
            )
            reversed_choices.append(choice)

        result['original_choices'] = original_choices
        result['reversed_choices'] = reversed_choices

        # Compute consistency: reversed choice should be opposite of original
        # If original='1', reversed should='2' for consistency
        consistent_count = 0
        total_count = len(original_choices) * len(reversed_choices)
        for orig in original_choices:
            for rev in reversed_choices:
                if (orig == '1' and rev == '2') or (orig == '2' and rev == '1'):
                    consistent_count += 1
                elif orig == '-1' or rev == '-1':
                    pass  # Invalid doesn't count

        result['consistency'] = consistent_count / total_count if total_count > 0 else 0

        # Determine final preference (majority vote on original order)
        valid_original = [c for c in original_choices if c in ['1', '2']]
        if valid_original:
            count_1 = valid_original.count('1')
            count_2 = valid_original.count('2')
            result['final_preference'] = '1' if count_1 >= count_2 else '2'
        else:
            result['final_preference'] = '-1'

        results.append(result)

    return results


def preferences_to_labels(
    preferences: list[dict],
    as_binary: bool = True,
) -> list[int]:
    """
    Convert preference results to training labels.

    Args:
        preferences: List of preference dicts from generate_historical_preferences
        as_binary: If True, return 0/1 labels. If False, return 1/2/-1.

    Returns:
        List of labels
    """
    labels = []
    for p in preferences:
        pref = p.get('final_preference', '-1')
        if as_binary:
            if pref == '1':
                labels.append(0)  # Prefer response 1
            elif pref == '2':
                labels.append(1)  # Prefer response 2
            else:
                labels.append(-1)  # Invalid
        else:
            labels.append(int(pref) if pref in ['1', '2', '-1'] else -1)

    return labels
