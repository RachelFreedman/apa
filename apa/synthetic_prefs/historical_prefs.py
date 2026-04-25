"""
Historical preference generation.

This module provides:
- HistLlama model loading (ProgressGym historical models)
- Preference generation using historical LLMs
- Synthetic preference datasets across centuries and user profiles
  (output format is JSONL, consumed by apa.lore_adapt for few-shot
  user adaptation).

CLI Usage:
    python -m apa.synthetic_prefs.historical_prefs generate-synth \\
        --centuries C013 C019 --n-questions 20
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


# =============================================================================
# HistLlama Model Loading
# =============================================================================

VALID_CENTURIES = ("C013", "C014", "C015", "C016", "C017", "C018", "C019", "C020", "C021")

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

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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


def preference_from_logprobs(
    prob_1_original: float,
    prob_2_original: float,
    prob_1_reversed: float,
    prob_2_reversed: float,
) -> dict:
    """Combine per-direction probabilities into a final preference + soft signal.

    The two arguments ending in ``_original`` are P("1") and P("2") when the
    pair was shown in the original order (Option 1 = physical response 1).  The
    ``_reversed`` arguments are the analogous probabilities when the pair was
    shown swapped (Option 1 = physical response 2).  Probabilities outside
    {1, 2} are assumed already collapsed by the caller (e.g. via guided
    decoding) but a missing entry can be passed as 0.0.

    Returns a dict with keys:
      - ``final_preference`` — ``"1"``, ``"2"``, or ``"-1"`` if the two
        orderings disagree on which physical response is preferred.
      - ``prob_1_original``, ``prob_2_original``,
        ``prob_1_reversed``, ``prob_2_reversed`` (echoed back).
      - ``soft_preference_1`` — mean probability that physical response 1 wins,
        averaged across both orderings.
      - ``consistency`` — 1.0 if the two orderings agree on the argmax over
        physical responses, else 0.0.
    """
    # In the reversed prompt, "Option 1" is physical response 2 and vice versa.
    p1_phys = 0.5 * (prob_1_original + prob_2_reversed)
    p2_phys = 0.5 * (prob_2_original + prob_1_reversed)

    arg_orig = '1' if prob_1_original >= prob_2_original else '2'
    arg_rev_phys = '1' if prob_2_reversed >= prob_1_reversed else '2'

    if arg_orig == arg_rev_phys:
        final = arg_orig
        consistency = 1.0
    else:
        final = '-1'
        consistency = 0.0

    return {
        "final_preference": final,
        "prob_1_original": prob_1_original,
        "prob_2_original": prob_2_original,
        "prob_1_reversed": prob_1_reversed,
        "prob_2_reversed": prob_2_reversed,
        "soft_preference_1": p1_phys / (p1_phys + p2_phys) if (p1_phys + p2_phys) > 0 else 0.5,
        "consistency": consistency,
    }


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
# Synthetic preference generation from profiles
# =============================================================================

CENTURY_SEED_OFFSETS = {c: i * 100 for i, c in enumerate(VALID_CENTURIES)}


def load_profiles(path: Path | str | None = None) -> dict[str, list[str]]:
    """Load user profiles from a JSONL file.

    Each line must be a JSON object with ``"century"`` and ``"profile"`` fields.

    Args:
        path: Path to the profiles JSONL file.  If *None*, uses the bundled
              ``profiles.jsonl`` next to this module.

    Returns:
        Dict mapping century code to list of profile description strings.
    """
    if path is None:
        path = Path(__file__).parent / "profiles.jsonl"
    path = Path(path)

    profiles: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            century = obj["century"]
            profiles.setdefault(century, []).append(obj["profile"])
    return profiles


def results_to_jsonl_records(results: list[dict]) -> list[dict]:
    """Convert :func:`generate_historical_preferences` output to eval_prefs JSONL format.

    Each result with ``final_preference`` in ``{'1', '2'}`` becomes one record
    with keys ``user_id``, ``prompt``, ``chosen``, ``rejected``.  Results with
    ``final_preference == '-1'`` (ambiguous) are skipped.
    """
    records = []
    for r in results:
        pref = r.get("final_preference")
        if pref == "1":
            chosen, rejected = r["response_1"], r["response_2"]
        elif pref == "2":
            chosen, rejected = r["response_2"], r["response_1"]
        else:
            continue
        records.append({
            "user_id": r["user_id"],
            "prompt": r["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        })
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")


def write_raw_results(results: list[dict], path: Path) -> None:
    """Write full provenance JSON including per-run choices and consistency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def generate_century_prefs(
    century: str,
    profiles: list[str],
    questions: list[dict],
    model_size: str = "8B",
    n_runs: int = 3,
    temperature: float = 0.3,
    show_progress: bool = True,
) -> list[dict]:
    """Load HistLlama once for *century* and generate preferences for all profiles.

    For each profile, calls :func:`generate_historical_preferences` which runs
    *n_runs* times in original order and *n_runs* times in reversed order per
    question.

    Args:
        century: Century code (e.g. ``"C013"``).
        profiles: List of user profile description strings.
        questions: List of dicts with keys ``question_id``, ``prompt``,
            ``response_1``, ``response_2``.
        model_size: HistLlama size (``"8B"`` or ``"70B"``).
        n_runs: Number of repetitions per order direction.
        temperature: Sampling temperature (lower = more deterministic).
        show_progress: Whether to show tqdm progress bars.

    Returns:
        Flat list of annotated result dicts, each containing ``user_id``,
        ``century``, ``profile_index``, ``user_profile`` in addition to the
        fields from :func:`generate_historical_preferences`.
    """
    print(f"\nLoading HistLlama {model_size} for {century_to_name(century)}...")
    model, tokenizer = load_hist_llama(century=century, size=model_size)

    all_results: list[dict] = []
    for idx, profile in enumerate(profiles):
        user_id = f"hist_{century}_{idx:02d}"
        print(f"\n--- Profile {idx}: {profile[:60]}... ---")

        prefs = generate_historical_preferences(
            model, tokenizer, questions,
            n_runs=n_runs, temperature=temperature,
            user_profile=profile, show_progress=show_progress,
        )

        for r in prefs:
            r["user_id"] = user_id
            r["century"] = century
            r["profile_index"] = idx
            r["user_profile"] = profile

        all_results.extend(prefs)

    # Free GPU memory before loading next century
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


# =============================================================================
# CLI: Generate Synthetic Preferences
# =============================================================================

def load_curated_question_ids(path: Path | str | None = None) -> list[int]:
    """Load question IDs from a text file (one ID per line, ``#`` comments allowed).

    Args:
        path: Path to the question IDs file.  If *None*, uses the bundled
              ``curated_questions.txt`` next to this module.
    """
    if path is None:
        path = Path(__file__).parent / "curated_questions.txt"
    path = Path(path)
    ids: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(int(line))
    return ids


def cmd_generate_synth(args) -> None:
    """Generate synthetic preferences across centuries and user profiles."""
    from apa.config import configure_environment, NAS_BASE
    from apa.load_prism import load_prism_pairwise
    from apa.levers.query_selection import random_subset, select_by_ids

    configure_environment()

    output_dir = Path(args.output_dir) if args.output_dir else NAS_BASE / "synthetic_prefs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load profiles
    profiles_all = load_profiles(args.profiles)
    centuries = args.centuries

    for c in centuries:
        if c not in profiles_all:
            print(f"WARNING: no profiles for {c} in profiles file, skipping.")
    centuries = [c for c in centuries if c in profiles_all]

    if not centuries:
        print("ERROR: no valid centuries with profiles.")
        sys.exit(1)

    # Load curated question IDs (if explicitly provided)
    curated_ids = None
    if args.questions is not None:
        curated_ids = load_curated_question_ids(args.questions)
        print(f"Using {len(curated_ids)} curated question IDs from {args.questions}")

    # Load PRISM questions once
    df = load_prism_pairwise()
    print(f"Loaded {len(df)} PRISM questions")

    all_records: list[dict] = []

    for century in centuries:
        profiles = profiles_all[century]
        seed = args.seed + CENTURY_SEED_OFFSETS.get(century, 0)

        print(f"\n{'='*60}")
        print(f"Century: {century_to_name(century)}  |  {len(profiles)} profiles  |  seed={seed}")
        print(f"{'='*60}")

        if curated_ids is not None:
            selected_df = select_by_ids(df, curated_ids)
        else:
            selected_df = random_subset(df, args.n_questions, {"seed": seed})
        print(f"Selected {len(selected_df)} questions")

        questions = [
            {
                "question_id": row["question_id"],
                "prompt": row["prompt"],
                "response_1": row["response_1"],
                "response_2": row["response_2"],
            }
            for _, row in selected_df.iterrows()
        ]

        results = generate_century_prefs(
            century, profiles, questions,
            model_size=args.model_size, n_runs=args.n_runs,
            temperature=args.temperature, show_progress=True,
        )

        records = results_to_jsonl_records(results)

        # Write per-century outputs
        write_jsonl(records, output_dir / f"hist_prefs_{century}.jsonl")
        write_raw_results(results, output_dir / f"hist_prefs_{century}_raw.json")

        valid = len(records)
        total = len(results)
        consistencies = [r["consistency"] for r in results]
        avg_c = sum(consistencies) / len(consistencies) if consistencies else 0

        print(f"\n{century} results: {valid}/{total} valid preferences, "
              f"avg consistency {avg_c:.2%}")

        all_records.extend(records)

    # Write combined output
    write_jsonl(all_records, output_dir / "hist_prefs_all.jsonl")
    print(f"\nTotal: {len(all_records)} preference records across {len(centuries)} centuries")
    print(f"Output: {output_dir}")


# =============================================================================
# Main CLI
# =============================================================================

def main() -> None:
    """CLI entry point for historical preference management."""
    parser = argparse.ArgumentParser(
        description="Historical preference generation (HistLlama personas).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate-synth subcommand
    synth_parser = subparsers.add_parser(
        'generate-synth',
        help='Generate synthetic preferences across centuries and user profiles',
    )
    synth_parser.add_argument("--centuries", nargs="+", default=["C013", "C019"],
                              choices=VALID_CENTURIES, help="Centuries to generate for")
    synth_parser.add_argument("--n-questions", type=int, default=20,
                              help="Number of PRISM questions per century")
    synth_parser.add_argument("--n-runs", type=int, default=3,
                              help="Repetitions per order direction (total queries = 2 * n_runs per question)")
    synth_parser.add_argument("--model-size", type=str, default="8B", choices=["8B", "70B"])
    synth_parser.add_argument("--temperature", type=float, default=0.3,
                              help="Sampling temperature (lower = more deterministic, default: 0.3)")
    synth_parser.add_argument("--profiles", type=str, default=None,
                              help="Path to profiles JSONL (default: bundled profiles.jsonl)")
    synth_parser.add_argument("--questions", type=str, default=None,
                              help="Path to curated question IDs file (default: bundled curated_questions.txt)")
    synth_parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    synth_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command == 'generate-synth':
        cmd_generate_synth(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
