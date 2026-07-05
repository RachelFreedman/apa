"""
Historical preference generation and training.

This module provides:
- HistLlama model loading (ProgressGym historical models)
- Preference generation using historical LLMs
- Training user vectors from historical preferences

CLI Usage:
    python -m apa.historical_prefs generate --century C017
    python -m apa.historical_prefs train --preferences_file ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
from tqdm import tqdm

from apa.config import HistLlamaConfig


# =============================================================================
# HistLlama Model Loading
# =============================================================================

# Single source of truth for valid centuries lives in config.
VALID_CENTURIES = HistLlamaConfig.VALID_CENTURIES

CENTURY_NAMES = {
    "C013": "13th Century", "C014": "14th Century", "C015": "15th Century",
    "C016": "16th Century", "C017": "17th Century", "C018": "18th Century",
    "C019": "19th Century", "C020": "20th Century", "C021": "21st Century",
}


def century_to_name(century: str) -> str:
    """Convert century code to human-readable name."""
    return CENTURY_NAMES.get(century, century)


def get_available_centuries() -> list[str]:
    """Get list of available century codes."""
    return list(VALID_CENTURIES)


def find_latest_model_version(size: str, century: str, hf_org: str = "PKU-Alignment") -> str:
    """Find the latest available model version on HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()
    base_name = f"ProgressGym-HistLlama3-{size}-{century}-instruct"

    try:
        models = api.list_models(author=hf_org, search=base_name)
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
            print("No matching model versions found, using v0.2")
            return "v0.2"
    except Exception as e:
        print(f"Could not query HuggingFace for model versions: {e}")
        return "v0.2"


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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from apa.config import configure_environment, HF_CACHE_DIR

    configure_environment()

    if cache_dir is None:
        cache_dir = str(HF_CACHE_DIR)

    version = find_latest_model_version(size, century)
    model_name = f"PKU-Alignment/ProgressGym-HistLlama3-{size}-{century}-instruct-{version}"

    print(f"Loading HistLlama model: {model_name}")
    print(f"Cache directory: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device_map,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    print("Model loaded successfully.")
    return model, tokenizer


# =============================================================================
# Preference Generation
# =============================================================================

def parse_model_response(response: str) -> str:
    """
    Parse model response to extract preference choice.

    Returns:
        '1' if option 1 preferred, '2' if option 2 preferred, '-1' if ambiguous
    """
    response_lower = response.lower().strip()

    if not response_lower:
        return '-1'

    patterns_1 = [
        r'\boption\s*1\b', r'\bption\s*1\b', r'\btion\s*1\b', r'\bthe\s+1\s+option\b',
        r'\b1\s*(?:st)?\s*option\b', r'\bchoice\s*1\b', r'\bresponse\s*1\b',
        r'\banswer\s*1\b', r'\bfirst\s+option\b', r'\bfirst\s+response\b',
        r'\b(?:is|be)\s+1\b', r'#1\b',
    ]
    patterns_2 = [
        r'\boption\s*2\b', r'\bption\s*2\b', r'\btion\s*2\b', r'\bthe\s+2\s+option\b',
        r'\b2\s*(?:nd)?\s*option\b', r'\bchoice\s*2\b', r'\bresponse\s*2\b',
        r'\banswer\s*2\b', r'\bsecond\s+option\b', r'\bsecond\s+response\b',
        r'\b(?:is|be)\s+2\b', r'#2\b',
    ]

    has_1 = any(re.search(p, response_lower) for p in patterns_1)
    has_2 = any(re.search(p, response_lower) for p in patterns_2)

    if has_1 and not has_2:
        return '1'
    elif has_2 and not has_1:
        return '2'
    elif has_1 and has_2:
        return '-1'

    if response_lower.startswith('1'):
        return '1'
    elif response_lower.startswith('2'):
        return '2'

    if len(response_lower) < 15:
        has_digit_1 = bool(re.search(r'\b1\b', response_lower))
        has_digit_2 = bool(re.search(r'\b2\b', response_lower))
        if has_digit_1 and not has_digit_2:
            return '1'
        elif has_digit_2 and not has_digit_1:
            return '2'

    return '-1'


def generate_single_preference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response_1: str,
    response_2: str,
    max_new_tokens: int = 20,
    temperature: float = 0.9,
    user_profile: str | None = None,
) -> str:
    """Generate a preference between two responses using the model."""
    if user_profile:
        comparison_content = (
            f"You are: {user_profile}\n\n"
            f"Given your perspective, which response do you prefer?\n\n"
            f"Question: {prompt}\n\nOption 1: {response_1}\n\nOption 2: {response_2}\n\n"
            f"Answer with only the number 1 or 2."
        )
    else:
        comparison_content = (
            f"Question: {prompt}\n\nOption 1: {response_1}\n\nOption 2: {response_2}\n\n"
            f"Which option is better? Answer with only the number 1 or 2."
        )

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": comparison_content}]
        comparison_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        comparison_prompt = f"{comparison_content}\n\nAnswer with just the number \"1\" or just the number \"2\"."

    inputs = tokenizer(comparison_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(comparison_prompt):].strip()

    return parse_model_response(response)


def generate_historical_preferences(
    model: Any,
    tokenizer: Any,
    questions: list[dict],
    max_new_tokens: int = 20,
    temperature: float = 0.9,
    user_profile: str | None = None,
    n_runs: int = 1,
    show_progress: bool = True,
) -> list[dict]:
    """
    Generate preferences for multiple question pairs.

    For each question, runs in both original and reversed order for consistency checking.
    """
    results = []
    iterator = tqdm(questions, desc="Generating preferences") if show_progress else questions

    for q in iterator:
        result = {
            'prompt': q['prompt'],
            'response_1': q.get('response_1', ''),
            'response_2': q.get('response_2', ''),
            'question_id': q.get('question_id', ''),
        }

        original_choices = []
        for _ in range(n_runs):
            choice = generate_single_preference(
                model, tokenizer, q['prompt'], q['response_1'], q['response_2'],
                max_new_tokens=max_new_tokens, temperature=temperature, user_profile=user_profile,
            )
            original_choices.append(choice)

        reversed_choices = []
        for _ in range(n_runs):
            choice = generate_single_preference(
                model, tokenizer, q['prompt'], q['response_2'], q['response_1'],
                max_new_tokens=max_new_tokens, temperature=temperature, user_profile=user_profile,
            )
            reversed_choices.append(choice)

        result['original_choices'] = original_choices
        result['reversed_choices'] = reversed_choices

        # Compute consistency
        consistent_count = 0
        total_count = len(original_choices) * len(reversed_choices)
        for orig in original_choices:
            for rev in reversed_choices:
                if (orig == '1' and rev == '2') or (orig == '2' and rev == '1'):
                    consistent_count += 1

        result['consistency'] = consistent_count / total_count if total_count > 0 else 0

        valid_original = [c for c in original_choices if c in ['1', '2']]
        if valid_original:
            count_1 = valid_original.count('1')
            count_2 = valid_original.count('2')
            result['final_preference'] = '1' if count_1 >= count_2 else '2'
        else:
            result['final_preference'] = '-1'

        results.append(result)

    return results


def preferences_to_labels(preferences: list[dict], as_binary: bool = True) -> list[int]:
    """Convert preference results to training labels."""
    labels = []
    for p in preferences:
        pref = p.get('final_preference', '-1')
        if as_binary:
            if pref == '1':
                labels.append(0)
            elif pref == '2':
                labels.append(1)
            else:
                labels.append(-1)
        else:
            labels.append(int(pref) if pref in ['1', '2', '-1'] else -1)
    return labels


# =============================================================================
# User Vector Training
# =============================================================================

def train_user_vector(
    V: torch.Tensor,
    embeddings_1: torch.Tensor,
    embeddings_2: torch.Tensor,
    labels: torch.Tensor,
    rank: int,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
) -> torch.Tensor:
    """Train a single user vector given frozen basis V."""
    w = torch.randn(rank, device=device) * 0.01
    w = w.clone().detach().requires_grad_(True)

    V = V.to(device)
    embeddings_1 = embeddings_1.to(device)
    embeddings_2 = embeddings_2.to(device)
    labels = labels.float().to(device)

    optimizer = torch.optim.Adam([w], lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        r1 = embeddings_1 @ V @ w
        r2 = embeddings_2 @ V @ w
        logits = r2 - r1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                preds = (logits > 0).float()
                acc = (preds == labels).float().mean()
                print(f"  Epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    return w.detach().cpu()


# =============================================================================
# CLI: Generate Preferences
# =============================================================================

def cmd_generate(args) -> None:
    """Generate historical preferences using HistLlama."""
    from apa.config import configure_environment, MODELS_DIR
    from apa.load_prism import load_prism_pairwise
    from apa.levers.query_selection import random_subset

    configure_environment()

    print(f"\n{'='*60}")
    print(f"Generating Historical Preferences for {century_to_name(args.century)}")
    print(f"{'='*60}\n")

    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_prism_pairwise()
    print(f"Loaded {len(df)} PRISM questions")

    selected_df = random_subset(df, args.n_questions, {'seed': args.seed})
    print(f"Selected {len(selected_df)} questions")

    questions = [
        {'question_id': row['question_id'], 'prompt': row['prompt'],
         'response_1': row['response_1'], 'response_2': row['response_2']}
        for _, row in selected_df.iterrows()
    ]

    print(f"\nLoading HistLlama {args.model_size} for {args.century}...")
    hist_model, hist_tokenizer = load_hist_llama(century=args.century, size=args.model_size)

    print(f"\nGenerating preferences with {args.n_runs} run(s) per question...")
    preferences = generate_historical_preferences(
        hist_model, hist_tokenizer, questions,
        user_profile=args.user_profile, n_runs=args.n_runs, show_progress=True,
    )

    valid_count = sum(1 for p in preferences if p.get('final_preference') in ['1', '2'])
    consistencies = [p['consistency'] for p in preferences]
    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0

    print(f"\nResults:")
    print(f"  Total questions: {len(preferences)}")
    print(f"  Valid preferences: {valid_count} ({valid_count/len(preferences)*100:.1f}%)")
    print(f"  Average consistency: {avg_consistency:.2%}")

    user_id = f"historical_{args.century}"
    if args.user_profile:
        user_id += f"_{hash(args.user_profile) % 10000}"

    output_data = {
        'century': args.century, 'user_profile': args.user_profile,
        'n_questions': len(questions), 'n_runs': args.n_runs,
        'model_size': args.model_size, 'seed': args.seed,
        'valid_count': valid_count, 'avg_consistency': avg_consistency,
        'questions': questions, 'preferences': preferences,
    }

    output_path = output_dir / f"preferences_{user_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved preferences to: {output_path}")
    print(f"\nNext step: Train user vector with:")
    print(f"  python -m apa.historical_prefs train --preferences_file {output_path}")


# =============================================================================
# CLI: Train User Vector
# =============================================================================

def cmd_train(args) -> None:
    """Train historical user vectors from preference files."""
    from apa.config import (
        configure_environment,
        MODELS_DIR,
        DEFAULT_INFERENCE_RANK,
        v_checkpoint_path,
    )
    from apa.train_lore_bases import LoReRewardModel, embed_texts, get_embedding_model

    configure_environment()

    prefs_path = Path(args.preferences_file)
    if not prefs_path.exists():
        print(f"ERROR: Preferences file not found: {prefs_path}")
        print("Generate preferences first: python -m apa.historical_prefs generate --century C013")
        sys.exit(1)

    print(f"Loading preferences from {prefs_path}")
    with open(prefs_path, 'r') as f:
        pref_data = json.load(f)

    century = pref_data['century']
    questions = pref_data['questions']
    preferences = pref_data['preferences']

    print(f"\n{'='*60}")
    print(f"Training Historical User Vector for {century}")
    print(f"{'='*60}\n")

    print(f"Preferences loaded:")
    print(f"  Century: {century}")
    print(f"  Questions: {len(questions)}")
    print(f"  Valid preferences: {pref_data.get('valid_count', 'N/A')}")

    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lore_checkpoint:
        lore_path = Path(args.lore_checkpoint)
    else:
        lore_path = v_checkpoint_path(DEFAULT_INFERENCE_RANK)

    if not lore_path.exists():
        print(f"ERROR: LoRe checkpoint not found at {lore_path}")
        print("Please train LoRe first: python -m apa.train_lore_bases")
        sys.exit(1)

    print(f"\nLoading LoRe model from {lore_path}")
    lore_model = LoReRewardModel.load(str(lore_path), device='cpu')
    V = lore_model.V.data.clone()
    rank = lore_model.rank

    print(f"LoRe model: embed_dim={lore_model.embedding_dim}, rank={rank}")

    labels = preferences_to_labels(preferences, as_binary=True)
    valid_mask = [l != -1 for l in labels]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    labels = [labels[i] for i in valid_indices]
    valid_questions = [questions[i] for i in valid_indices]

    print(f"\nValid training samples: {len(labels)} / {len(preferences)}")

    if len(labels) == 0:
        print("ERROR: No valid preference labels found.")
        sys.exit(1)

    print("\nGenerating embeddings...")
    model, tokenizer = get_embedding_model()

    responses_1 = [f"{q['prompt']}\n\n{q['response_1']}" for q in valid_questions]
    responses_2 = [f"{q['prompt']}\n\n{q['response_2']}" for q in valid_questions]

    embeddings_1 = embed_texts(responses_1, model=model, tokenizer=tokenizer, show_progress=False)
    embeddings_2 = embed_texts(responses_2, model=model, tokenizer=tokenizer, show_progress=False)

    embeddings_1 = torch.tensor(embeddings_1, dtype=torch.float32)
    embeddings_2 = torch.tensor(embeddings_2, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    print(f"\nTraining user vector for {century}...")
    w = train_user_vector(
        V=V, embeddings_1=embeddings_1, embeddings_2=embeddings_2,
        labels=labels_tensor, rank=rank, epochs=args.epochs,
        learning_rate=args.learning_rate, device=args.device,
    )

    user_id = f"historical_{century}"
    if pref_data.get('user_profile'):
        user_id += f"_{hash(pref_data['user_profile']) % 10000}"

    output_path = output_dir / f"W_{century}.pt"
    torch.save({
        'user_id': user_id, 'century': century,
        'user_profile': pref_data.get('user_profile'),
        'w': w, 'n_questions': len(valid_questions),
        'preferences_file': str(prefs_path),
    }, output_path)

    print(f"\nSaved user vector to {output_path}")
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


# =============================================================================
# Main CLI
# =============================================================================

def main() -> None:
    """CLI entry point for historical preference management."""
    parser = argparse.ArgumentParser(
        description="Historical preference generation and training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate subcommand
    gen_parser = subparsers.add_parser('generate', help='Generate historical preferences')
    gen_parser.add_argument("--century", type=str, required=True, choices=VALID_CENTURIES,
                           help="Century to generate preferences for")
    gen_parser.add_argument("--n_questions", type=int, default=500, help="Number of questions")
    gen_parser.add_argument("--n_runs", type=int, default=1, help="Runs per comparison")
    gen_parser.add_argument("--model_size", type=str, default="8B", choices=["8B", "70B"])
    gen_parser.add_argument("--user_profile", type=str, default=None, help="User profile description")
    gen_parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train user vector from preferences')
    train_parser.add_argument("--preferences_file", type=str, required=True, help="Preferences JSON file")
    train_parser.add_argument("--lore_checkpoint", type=str, default=None, help="LoRe model checkpoint")
    train_parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    train_parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'train':
        cmd_train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
