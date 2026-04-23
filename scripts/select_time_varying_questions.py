"""Select PRISM questions expected to vary most between 16th and 20th century.

Heuristic: prompts must (a) be value- or opinion-laden (not factual / task-y)
and (b) touch topics whose normative consensus has shifted over the last five
centuries (religion/church authority, gender roles, sexuality, race/slavery,
governance/monarchy, scientific vs religious authority, punishment, economics,
cosmology, colonialism, child-rearing, individual vs community, etc.).

We score each unique prompt against topic-specific keyword buckets. A prompt
scores once per bucket it hits (bucket-diversity matters more than raw count).
We also apply hard exclusions for clearly modern/technical/factual prompts
(code, modern brands, AI/tech tasks) where the 16th-century persona has no
meaningful opinion.

Usage:
    uv run python scripts/select_time_varying_questions.py \
        --out experiments/chosen_questions.jsonl --n 150
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from apa.load_prism import load_prism_pairwise

# Topic buckets: each bucket is a list of regex-ready lowercase substrings.
# A prompt scores +1 per bucket matched (capped at 1 per bucket).
TOPIC_BUCKETS: dict[str, list[str]] = {
    "religion": [
        "god", "religion", "religious", "faith", "christian", "christianity",
        "church", "priest", "bible", "scripture", "prayer", "atheis", "islam",
        "muslim", "jewish", "judaism", "hindu", "buddhis", "sin", "soul",
        "heaven", "hell", "afterlife", "salvation", "blasphem", "sacred",
        "miracle", "spiritual", "clergy", "pope", "worship",
    ],
    "gender_sex": [
        "women", "woman", "female", "feminis", "gender", "sexis", "patriarch",
        "men ", "masculin", "marriage", "wife", "husband", "divorce",
        "dating", "romantic", "romance",
        "sex", "sexual", "sexuality", "lgbt", "gay", "lesbian", "queer",
        "transgender", "trans ", "homosexual", "bisexual", "chastity",
        "virginity", "modesty", "adultery",
    ],
    "reproduction": [
        "abortion", "contracep", "birth control", "pregnan", "reproductive",
        "family planning", "ivf", "surrogacy",
    ],
    "race_slavery": [
        "race", "racism", "racial", "slavery", "slave", "black people",
        "white people", "colonial", "imperial", "indigenous", "native ",
        "minorit", "ethnic", "segregat", "apartheid", "civil rights",
    ],
    "governance": [
        "monarch", "king ", "queen ", "democracy", "democratic", "republic",
        "government", "dictator", "authoritarian", "revolution", "voting",
        "election", "parliament", "communis", "socialis", "capitalis",
        "fascis", "anarch", "tyran", "libert",
    ],
    "punishment_justice": [
        "death penalty", "capital punishment", "execut", "torture",
        "corporal punishment", "flog", "prison", "incarcer", "criminal",
        "punish", "justice", "vengeance", "retribution",
    ],
    "war_violence": [
        "war ", "warfare", "military", "soldier", "just war", "pacifis",
        "violence", "nonviolen", "self-defense", "killing", "honor",
        "duel", "revenge",
    ],
    "economy_class": [
        "wealth", "rich", "poor", "poverty", "inequality", "class ",
        "aristocra", "noble", "peasant", "labor", "worker", "union",
        "money", "usury", "interest rate", "taxes", "taxation",
        "welfare", "inherit", "property",
    ],
    "science_authority": [
        "science", "scientif", "evolution", "darwin", "creationis",
        "universe", "cosmolog", "flat earth", "astrology", "vaccin",
        "medicine", "doctor", "traditional medicine", "witch",
        "superstiti",
    ],
    "family_children": [
        "children", "child ", "parent", "parenting", "raise a child",
        "corporal punishment", "spank", "obedience", "discipline",
        "elderly", "ancestor", "filial",
    ],
    "morality_virtue": [
        "moral", "immoral", "ethic", "virtue", "vice", "honor ", "duty",
        "tradition", "custom", "piety", "humility", "pride", "vanity",
        "gluttony",
    ],
    "freedom_individual": [
        "freedom", "liberty", "individual", "community", "collective",
        "autonomy", "free speech", "censor", "privacy",
    ],
    "death_life": [
        "euthanasia", "suicide", "assisted dying", "end of life",
        "meaning of life", "purpose of life", "what happens when we die",
    ],
    "animals_environment": [
        "animal rights", "vegetarian", "vegan", "meat", "hunting",
        "environment", "nature ", "climate", "pollut",
    ],
    "drugs_intoxicants": [
        "alcohol", "drinking", "drunk", "drug", "opium", "cannabis",
        "marijuana", "tobacco", "smoking",
    ],
    "art_culture": [
        "art ", "music", "dance", "theater", "theatre", "literature",
        "poetry", "beauty", "aesthetic",
    ],
}

# Prompts we want to exclude outright (modern/technical/task-y).
EXCLUDE_PATTERNS = [
    r"\bpython\b", r"\bjavascript\b", r"\bjava \b", r"\bcode\b", r"\bsql\b",
    r"\bhtml\b", r"\bcss\b", r"\bregex\b", r"\bapi\b", r"\bgithub\b",
    r"\bchatgpt\b", r"\bllm\b", r"\bgpt-?\d", r"\bai model\b",
    r"\bmachine learning\b", r"\bneural network\b",
    r"\bcomputer\b", r"\bsoftware\b", r"\bapp\b", r"\bwebsite\b",
    r"\bemail\b", r"\bsmartphone\b", r"\biphone\b", r"\bandroid\b",
    r"\bexcel\b", r"\bspreadsheet\b", r"\bpowerpoint\b", r"\bword document\b",
    r"\bresume\b", r"\bcv\b", r"\bcover letter\b",
    r"recipe for", r"how do i cook", r"how to cook",
    r"translate", r"summarize", r"rewrite",
    r"what is the capital of", r"how many \w+ are",
    r"write a (poem|story|essay|song) about (my|a specific)",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_PATTERNS), re.IGNORECASE)


# Pre-compile: each keyword becomes a word-boundary regex. Trailing spaces
# in keywords are preserved (they already imply a following break).
def _compile(kw: str) -> re.Pattern:
    kw_stripped = kw.rstrip()
    # Use \b on both sides so "king" doesn't match "talking", etc.
    return re.compile(r"\b" + re.escape(kw_stripped) + r"\b", re.IGNORECASE)


_BUCKET_REGEXES = {
    bucket: [_compile(kw) for kw in kws]
    for bucket, kws in TOPIC_BUCKETS.items()
}


def score_prompt(prompt: str) -> tuple[int, list[str]]:
    """Return (bucket-diversity score, list of matched bucket names)."""
    matched: list[str] = []
    for bucket, patterns in _BUCKET_REGEXES.items():
        for pat in patterns:
            if pat.search(prompt):
                matched.append(bucket)
                break
    return len(matched), matched


def looks_modern_taskish(prompt: str) -> bool:
    return bool(EXCLUDE_RE.search(prompt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=150,
                        help="Target number of prompts to write (pre-user-filtering)")
    parser.add_argument("--min-score", type=int, default=1,
                        help="Minimum bucket-diversity score to keep")
    parser.add_argument("--max-len", type=int, default=400,
                        help="Max prompt character length")
    parser.add_argument("--min-len", type=int, default=20,
                        help="Min prompt character length")
    args = parser.parse_args()

    df = load_prism_pairwise()
    print(f"Loaded {len(df)} PRISM rows ({df['question_id'].nunique()} unique questions)")

    # Deduplicate by question_id (keep first row with responses for context).
    df_unique = df.drop_duplicates(subset=["question_id"]).copy()
    print(f"Unique questions: {len(df_unique)}")

    kept: list[dict] = []
    for _, row in df_unique.iterrows():
        prompt = str(row["prompt"]).strip()
        if len(prompt) < args.min_len or len(prompt) > args.max_len:
            continue
        if looks_modern_taskish(prompt):
            continue
        score, buckets = score_prompt(prompt)
        if score < args.min_score:
            continue
        kept.append({
            "question_id": int(row["question_id"]),
            "prompt": prompt,
            "response_1": row["response_1"],
            "response_2": row["response_2"],
            "topic_score": score,
            "topics": buckets,
        })

    kept.sort(key=lambda r: (-r["topic_score"], r["question_id"]))
    print(f"Candidates after filtering: {len(kept)}")
    print("Score distribution (top scores first):")
    from collections import Counter
    for s, c in sorted(Counter(r["topic_score"] for r in kept).items(), reverse=True):
        print(f"  score={s}: {c}")

    selected = kept[: args.n]
    print(f"Writing top {len(selected)} to {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    # Print a few samples across the score range.
    print("\n=== Sample entries (first 5, middle 3, last 3) ===")
    sample_idx = list(range(5)) + [len(selected)//2 + i for i in range(3)] + list(range(len(selected)-3, len(selected)))
    for i in sample_idx:
        r = selected[i]
        topics = ",".join(sorted(set(r["topics"])))
        print(f"  [{i}] qid={r['question_id']} score={r['topic_score']} topics={topics}")
        print(f"      {r['prompt'][:150]}")


if __name__ == "__main__":
    main()
