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
import gc
import json
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
    cache_dir: str | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
) -> Tuple[Any, Any]:
    """
    Load a HistLlama model for a specific century as a vLLM ``LLM`` instance.

    The returned ``LLM`` does PagedAttention and continuous batching over the
    list of prompts passed to ``LLM.generate(...)``, which is ~10–50× faster
    than calling HF ``model.generate()`` once per query.  The tokenizer is
    returned alongside because callers still need ``apply_chat_template`` to
    format prompts and the token ids of "1" / "2" to extract logprobs.

    Args:
        century: Century code (e.g., "C013" for 13th century)
        size: Model size ("8B" or "70B")
        cache_dir: HF cache directory (uses default if None)
        tensor_parallel_size: Number of GPUs to shard across (default 1; use
            >1 for the 70B model).
        gpu_memory_utilization: Fraction of GPU memory vLLM may claim.
        max_model_len: Cap on context length; ``None`` lets vLLM use the
            model's native max.

    Returns:
        Tuple of (vllm.LLM, tokenizer)
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    from apa.config import configure_environment, HF_CACHE_DIR

    configure_environment()

    if cache_dir is None:
        cache_dir = str(HF_CACHE_DIR)

    version = find_latest_model_version(size, century)
    model_name = f"PKU-Alignment/ProgressGym-HistLlama3-{size}-{century}-instruct-{version}"

    print(f"Loading HistLlama model via vLLM: {model_name}")
    print(f"Cache directory: {cache_dir}  |  tensor_parallel_size={tensor_parallel_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    llm_kwargs = dict(
        model=model_name,
        dtype="float16",
        download_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)

    print("Model loaded successfully.")
    return llm, tokenizer


# =============================================================================
# Preference Generation
# =============================================================================

def _build_comparison_prompt(
    tokenizer: Any,
    prompt: str,
    response_1: str,
    response_2: str,
    user_profile: str | None,
) -> str:
    """Format a single pairwise comparison prompt with optional user profile."""
    if user_profile:
        content = (
            f"You are: {user_profile}\n\n"
            f"Given your perspective, which response do you prefer?\n\n"
            f"Question: {prompt}\n\nOption 1: {response_1}\n\nOption 2: {response_2}\n\n"
            f"Answer with only the number 1 or 2."
        )
    else:
        content = (
            f"Question: {prompt}\n\nOption 1: {response_1}\n\nOption 2: {response_2}\n\n"
            f"Which option is better? Answer with only the number 1 or 2."
        )

    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f'{content}\n\nAnswer with just the number "1" or just the number "2".'


def _resolve_choice_token_ids(tokenizer: Any) -> Tuple[int, int]:
    """Token id for "1" and "2" as emitted by ``guided_choice=["1","2"]``.

    vLLM's guided decoding constrains the output to one of the strings, but
    the actual emitted token is still tokenizer-dependent.  For all Llama-3
    tokenizers ``"1"`` and ``"2"`` are single tokens, so we look up the id
    directly.
    """
    ids_1 = tokenizer.encode("1", add_special_tokens=False)
    ids_2 = tokenizer.encode("2", add_special_tokens=False)
    if len(ids_1) != 1 or len(ids_2) != 1:
        raise RuntimeError(
            f"Expected '1' and '2' to be single tokens, got {ids_1} and {ids_2}."
        )
    return ids_1[0], ids_2[0]


def _probs_from_logprobs(
    logprobs: dict[int, Any] | None,
    token_id_1: int,
    token_id_2: int,
) -> Tuple[float, float]:
    """Extract P("1"), P("2") from a vLLM per-position logprobs dict.

    vLLM returns ``{token_id: Logprob(logprob=..., ...), ...}`` for the top-K
    tokens.  Under guided_choice=["1","2"] both ids are guaranteed to appear
    in the distribution; we still defend against missing keys by returning 0.
    """
    if not logprobs:
        return 0.0, 0.0

    def _p(tid: int) -> float:
        entry = logprobs.get(tid)
        if entry is None:
            return 0.0
        lp = getattr(entry, "logprob", entry)
        return float(torch.exp(torch.tensor(float(lp))).item())

    return _p(token_id_1), _p(token_id_2)


def generate_historical_preferences(
    llm: Any,
    tokenizer: Any,
    questions: list[dict],
    user_profile: str | None = None,
    show_progress: bool = True,
) -> list[dict]:
    """Generate preferences for many question pairs in a single batched vLLM call.

    For every question we issue two prompts — original order and reversed
    order — and combine the per-direction probabilities of "1" / "2" via
    :func:`preference_from_logprobs`.  The whole batch (``2 * len(questions)``
    prompts) goes through ``LLM.generate`` once, which is the entire point of
    the optimization: the GPU stays saturated with PagedAttention + continuous
    batching instead of doing one tiny forward pass per Python iteration.

    Decoding is greedy (``temperature=0.0``, ``max_tokens=1``) and constrained
    to ``["1", "2"]``, so the returned probability mass is exactly the
    calibrated soft preference signal.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    token_id_1, token_id_2 = _resolve_choice_token_ids(tokenizer)

    prompts: list[str] = []
    for q in questions:
        prompts.append(_build_comparison_prompt(
            tokenizer, q["prompt"], q["response_1"], q["response_2"], user_profile,
        ))
        prompts.append(_build_comparison_prompt(
            tokenizer, q["prompt"], q["response_2"], q["response_1"], user_profile,
        ))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20,
        guided_decoding=GuidedDecodingParams(choice=["1", "2"]),
    )

    if show_progress:
        print(f"Running vLLM batch over {len(prompts)} prompts "
              f"({len(questions)} questions × 2 orderings)...")

    outputs = llm.generate(prompts, sampling_params, use_tqdm=show_progress)

    results: list[dict] = []
    iterator = tqdm(questions, desc="Combining logprobs") if show_progress else questions
    for i, q in enumerate(iterator):
        out_orig = outputs[2 * i].outputs[0]
        out_rev = outputs[2 * i + 1].outputs[0]

        p1_o, p2_o = _probs_from_logprobs(out_orig.logprobs[0], token_id_1, token_id_2)
        p1_r, p2_r = _probs_from_logprobs(out_rev.logprobs[0], token_id_1, token_id_2)

        combined = preference_from_logprobs(p1_o, p2_o, p1_r, p2_r)

        results.append({
            "prompt": q["prompt"],
            "response_1": q.get("response_1", ""),
            "response_2": q.get("response_2", ""),
            "question_id": q.get("question_id", ""),
            **combined,
        })

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
    with the load-bearing keys ``user_id``, ``prompt``, ``chosen``,
    ``rejected`` (consumed by :func:`apa.synthetic_prefs.eval_prefs.load_prefs_jsonl`
    and :mod:`apa.lore_adapt`), plus the soft signal:

      - ``prob_chosen_original``, ``prob_rejected_original``
      - ``prob_chosen_reversed``, ``prob_rejected_reversed``
      - ``soft_preference_chosen`` — mean P(chosen wins) across the two
        orderings, in [0, 1].
      - ``consistency`` — 1.0 if both orderings agree, else 0.0.

    Records with ``final_preference == '-1'`` (orderings disagree) are
    skipped, matching prior behaviour.  Downstream readers that select only
    the four required keys ignore the extra fields.
    """
    records = []
    for r in results:
        pref = r.get("final_preference")
        if pref == "1":
            chosen, rejected = r["response_1"], r["response_2"]
            prob_chosen_o = r.get("prob_1_original", 0.0)
            prob_rejected_o = r.get("prob_2_original", 0.0)
            prob_chosen_r = r.get("prob_2_reversed", 0.0)
            prob_rejected_r = r.get("prob_1_reversed", 0.0)
            soft_chosen = r.get("soft_preference_1", 0.5)
        elif pref == "2":
            chosen, rejected = r["response_2"], r["response_1"]
            prob_chosen_o = r.get("prob_2_original", 0.0)
            prob_rejected_o = r.get("prob_1_original", 0.0)
            prob_chosen_r = r.get("prob_1_reversed", 0.0)
            prob_rejected_r = r.get("prob_2_reversed", 0.0)
            soft_chosen = 1.0 - r.get("soft_preference_1", 0.5)
        else:
            continue
        records.append({
            "user_id": r["user_id"],
            "prompt": r["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "prob_chosen_original": prob_chosen_o,
            "prob_rejected_original": prob_rejected_o,
            "prob_chosen_reversed": prob_chosen_r,
            "prob_rejected_reversed": prob_rejected_r,
            "soft_preference_chosen": soft_chosen,
            "consistency": r.get("consistency", 1.0),
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
    tensor_parallel_size: int = 1,
    show_progress: bool = True,
) -> list[dict]:
    """Load HistLlama once for *century* and generate preferences for all profiles.

    Each profile contributes ``2 * len(questions)`` prompts to a single
    batched ``vllm.LLM.generate`` call (one per ordering, for the position-bias
    consistency check).

    Args:
        century: Century code (e.g. ``"C013"``).
        profiles: List of user profile description strings.
        questions: List of dicts with keys ``question_id``, ``prompt``,
            ``response_1``, ``response_2``.
        model_size: HistLlama size (``"8B"`` or ``"70B"``).
        tensor_parallel_size: Number of GPUs to shard the model across
            (default 1; raise for the 70B model).
        show_progress: Whether to show tqdm progress bars.

    Returns:
        Flat list of annotated result dicts, each containing ``user_id``,
        ``century``, ``profile_index``, ``user_profile`` in addition to the
        fields from :func:`generate_historical_preferences`.
    """
    print(f"\nLoading HistLlama {model_size} for {century_to_name(century)}...")
    llm, tokenizer = load_hist_llama(
        century=century, size=model_size, tensor_parallel_size=tensor_parallel_size,
    )

    all_results: list[dict] = []
    for idx, profile in enumerate(profiles):
        user_id = f"hist_{century}_{idx:02d}"
        print(f"\n--- Profile {idx}: {profile[:60]}... ---")

        prefs = generate_historical_preferences(
            llm, tokenizer, questions,
            user_profile=profile, show_progress=show_progress,
        )

        for r in prefs:
            r["user_id"] = user_id
            r["century"] = century
            r["profile_index"] = idx
            r["user_profile"] = profile

        all_results.extend(prefs)

    # Free GPU memory before loading next century.  vLLM holds onto a worker
    # process; explicit destroy + gc + empty_cache is the documented sequence.
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    del llm, tokenizer
    gc.collect()
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
            model_size=args.model_size,
            tensor_parallel_size=args.tensor_parallel_size,
            show_progress=True,
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
    synth_parser.add_argument("--model-size", type=str, default="8B", choices=["8B", "70B"])
    synth_parser.add_argument("--tensor-parallel-size", type=int, default=1,
                              help="Number of GPUs to shard the model across (raise for 70B)")
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
