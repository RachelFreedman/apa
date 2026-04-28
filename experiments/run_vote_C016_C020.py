"""
Run a fixed-jury democratic vote over a query_responses file.

Jury: all C016 + C020 historical voters from a W_adapted_*.pt checkpoint, plus
N PRISM voters sampled (seeded) from W_seen_K{K}.pt. The PRISM-trained
LoRe basis V_K{K}.pt is loaded read-only.

For each query, computes:
  - per-voter scores and rankings over the supplied responses,
  - aggregate rankings under Borda / Plurality / Copeland / Instant-Runoff,
  - intra/inter-group agreement (mean pairwise Spearman + Kendall-tau),
  - per-group aggregate winners (so we can see whether C016 and C020 disagree).

Outputs a single JSON results file plus a concise text report.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apa.config import MODELS_DIR, configure_environment
from apa.democratic_response import _embed_responses, load_query_cases
from apa.levers.voter_aggregation import (
    borda_count,
    copeland,
    instant_runoff,
    plurality,
)
from apa.lore_adapt import LoReScorer


AGG_METHODS = {
    "borda_count": borda_count,
    "plurality": plurality,
    "copeland": copeland,
    "instant_runoff": instant_runoff,
}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def spearman(rank_a: list[int], rank_b: list[int]) -> float:
    """Spearman rho between two full rankings (lists of response indices)."""
    n = len(rank_a)
    pos_a = [0] * n
    pos_b = [0] * n
    for p, idx in enumerate(rank_a):
        pos_a[idx] = p
    for p, idx in enumerate(rank_b):
        pos_b[idx] = p
    a = np.asarray(pos_a, dtype=float)
    b = np.asarray(pos_b, dtype=float)
    a -= a.mean(); b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def kendall_tau(rank_a: list[int], rank_b: list[int]) -> float:
    """Kendall tau-b between two full rankings (no ties expected)."""
    n = len(rank_a)
    pos_a = [0] * n
    pos_b = [0] * n
    for p, idx in enumerate(rank_a):
        pos_a[idx] = p
    for p, idx in enumerate(rank_b):
        pos_b[idx] = p
    concordant = discordant = 0
    for i, j in combinations(range(n), 2):
        s = (pos_a[i] - pos_a[j]) * (pos_b[i] - pos_b[j])
        if s > 0:
            concordant += 1
        elif s < 0:
            discordant += 1
    total = concordant + discordant
    if total == 0:
        return float("nan")
    return (concordant - discordant) / total


def mean_pairwise(rankings: list[list[int]], fn) -> float:
    if len(rankings) < 2:
        return float("nan")
    vals = [fn(a, b) for a, b in combinations(rankings, 2)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def score_voter(scorer: LoReScorer, uid: str, embeddings: torch.Tensor) -> tuple[list[float], list[int]]:
    scores = [scorer.score_embedding(uid, embeddings[i]) for i in range(embeddings.shape[0])]
    ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return scores, ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_file", type=Path, required=True,
                        help="Path to query_responses (.jsonl) with query+responses per case.")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--V_checkpoint", type=Path, default=None)
    parser.add_argument("--prism_users", type=Path, default=None,
                        help="Path to W_seen_K{K}.pt for PRISM users.")
    parser.add_argument("--adapted_users", type=Path, required=True,
                        help="Path to W_adapted_*.pt with the C016+C020 personas.")
    parser.add_argument("--n_prism", type=int, default=10,
                        help="Number of PRISM voters to randomly sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    configure_environment()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    V_path = args.V_checkpoint or (MODELS_DIR / f"V_K{args.K}.pt")
    prism_path = args.prism_users or (MODELS_DIR / f"W_seen_K{args.K}.pt")

    log(f"Loading V from {V_path}")
    scorer = LoReScorer.from_checkpoint(V_path)
    log(f"Loading PRISM users from {prism_path}")
    scorer.load_prism_users(prism_path)
    prism_user_ids = list(scorer.get_user_ids())
    log(f"  PRISM users available: {len(prism_user_ids)}")

    log(f"Loading adapted users from {args.adapted_users}")
    scorer.load_adapted_users(args.adapted_users)
    all_users = scorer.get_user_ids()
    adapted_users = [u for u in all_users if u not in set(prism_user_ids)]
    log(f"  Adapted users loaded: {len(adapted_users)}")

    c016_users = sorted(u for u in adapted_users if "C016" in u)
    c020_users = sorted(u for u in adapted_users if "C020" in u)
    log(f"  C016 voters: {len(c016_users)}; C020 voters: {len(c020_users)}")

    sampled_prism = sorted(rng.sample(prism_user_ids, args.n_prism))
    log(f"  PRISM voters sampled: {len(sampled_prism)} (seed={args.seed})")

    voter_groups = {
        "C016": c016_users,
        "C020": c020_users,
        "PRISM": sampled_prism,
    }
    full_jury = c016_users + c020_users + sampled_prism
    voter_to_group = {u: g for g, users in voter_groups.items() for u in users}

    log(f"Loading query cases from {args.responses_file}")
    cases = load_query_cases(args.responses_file)
    log(f"  {len(cases)} case(s)")

    all_results: list[dict[str, Any]] = []

    for case_i, case in enumerate(cases, 1):
        log(f"--- Case {case_i}/{len(cases)} (qid={case.query_id}) ---")
        log(f"  query: {case.query[:100] if case.query else '<no query>'}")
        log(f"  n_responses: {len(case.responses)}")

        embeddings = _embed_responses(case.query, case.responses, scorer)
        n_resp = embeddings.shape[0]

        per_voter_scores: dict[str, list[float]] = {}
        per_voter_rankings: dict[str, list[int]] = {}
        for uid in full_jury:
            s, r = score_voter(scorer, uid, embeddings)
            per_voter_scores[uid] = s
            per_voter_rankings[uid] = r

        # Aggregate over full jury and per group.
        aggregations: dict[str, dict[str, Any]] = {}
        for scope, users in {"full": full_jury, **voter_groups}.items():
            sub = {u: per_voter_rankings[u] for u in users}
            per_method = {}
            for method, fn in AGG_METHODS.items():
                ranking = list(fn(sub, {}))
                per_method[method] = {
                    "ranking": ranking,
                    "winner_idx": int(ranking[0]),
                }
            aggregations[scope] = per_method

        # Average rank per response across the full jury.
        rank_sums = [0.0] * n_resp
        for r in per_voter_rankings.values():
            for pos, idx in enumerate(r):
                rank_sums[idx] += pos + 1
        avg_ranks = [s / len(per_voter_rankings) for s in rank_sums]

        # Intra-group and inter-group agreement.
        agreement = {}
        for g, users in voter_groups.items():
            ranks = [per_voter_rankings[u] for u in users]
            agreement[f"intra_{g}"] = {
                "mean_spearman": mean_pairwise(ranks, spearman),
                "mean_kendall_tau": mean_pairwise(ranks, kendall_tau),
                "n_pairs": len(users) * (len(users) - 1) // 2,
            }
        for g1, g2 in combinations(voter_groups.keys(), 2):
            r1 = [per_voter_rankings[u] for u in voter_groups[g1]]
            r2 = [per_voter_rankings[u] for u in voter_groups[g2]]
            sp = [spearman(a, b) for a in r1 for b in r2]
            kt = [kendall_tau(a, b) for a in r1 for b in r2]
            sp = [v for v in sp if not np.isnan(v)]
            kt = [v for v in kt if not np.isnan(v)]
            agreement[f"inter_{g1}_{g2}"] = {
                "mean_spearman": float(np.mean(sp)) if sp else float("nan"),
                "mean_kendall_tau": float(np.mean(kt)) if kt else float("nan"),
                "n_pairs": len(sp),
            }

        all_results.append({
            "query_id": case.query_id,
            "query": case.query,
            "n_responses": n_resp,
            "responses": case.responses,
            "average_ranks_full_jury": avg_ranks,
            "per_voter_rankings": per_voter_rankings,
            "per_voter_scores": per_voter_scores,
            "aggregations": aggregations,
            "agreement": agreement,
        })

    out_json = args.output_dir / "vote_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "jury": {
                "C016": c016_users,
                "C020": c020_users,
                "PRISM": sampled_prism,
            },
            "config": {
                "K": args.K,
                "V_checkpoint": str(V_path),
                "prism_users": str(prism_path),
                "adapted_users": str(args.adapted_users),
                "responses_file": str(args.responses_file),
                "n_prism": args.n_prism,
                "seed": args.seed,
            },
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    log(f"Wrote {out_json}")

    # Compact human-readable report.
    report_path = args.output_dir / "vote_report.txt"
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Democratic vote — {args.responses_file.name}")
    lines.append(f"Jury: {len(c016_users)} C016 + {len(c020_users)} C020 + "
                 f"{len(sampled_prism)} PRISM (seed={args.seed})")
    lines.append("=" * 72)
    for res in all_results:
        lines.append("")
        lines.append(f"Q{res['query_id']}: {res['query']}")
        lines.append(f"  n_responses={res['n_responses']}")
        lines.append("  Aggregate winners (response_id is 1-indexed):")
        for scope in ("full", "C016", "C020", "PRISM"):
            row = res["aggregations"][scope]
            lines.append(f"    [{scope:<5}]  " + "  ".join(
                f"{m}=#{row[m]['winner_idx']+1}" for m in AGG_METHODS
            ))
        lines.append("  Group rankings (Borda):")
        for scope in ("full", "C016", "C020", "PRISM"):
            r = res["aggregations"][scope]["borda_count"]["ranking"]
            lines.append(f"    [{scope:<5}] " + " > ".join(f"#{i+1}" for i in r))
        lines.append("  Agreement (mean pairwise):")
        for k, v in res["agreement"].items():
            lines.append(
                f"    {k:<22} spearman={v['mean_spearman']:+.3f}  "
                f"kendall={v['mean_kendall_tau']:+.3f}  (n_pairs={v['n_pairs']})"
            )
    report = "\n".join(lines) + "\n"
    report_path.write_text(report)
    log(f"Wrote {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
