"""
Compare two PRISM embedding pickles (e.g. a freshly regenerated subset vs. the
canonical NAS embeddings) to validate the deterministic data/embedding path
after a refactor.

Embedding generation is deterministic given seed=123 splits and a fixed model,
but GPU float ops are not bit-exact, so we compare the chosen-minus-rejected
difference vectors within a tolerance rather than requiring exact equality.

Usage:
    python experiments/scripts/compare_embeddings.py \
        --new /path/to/new/train.pkl --ref /nas/ucb/rachel/APA/embeddings/train.pkl \
        --n 200 --atol 1e-2
"""

from __future__ import annotations

import argparse

import torch


def _diff_vectors(data, n):
    """Return the (chosen - rejected) vectors for the first n usable examples."""
    diffs = []
    for ex in data:
        info = ex.get("extra_info", {})
        c = info.get("chosen_conv_embedding")
        r = info.get("rejected_conv_embedding")
        if c is None or r is None:
            continue
        diffs.append(torch.as_tensor(c, dtype=torch.float32) - torch.as_tensor(r, dtype=torch.float32))
        if len(diffs) >= n:
            break
    return torch.stack(diffs) if diffs else torch.empty(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PRISM embedding pickles")
    parser.add_argument("--new", required=True, help="Newly regenerated embeddings .pkl")
    parser.add_argument("--ref", required=True, help="Reference (canonical) embeddings .pkl")
    parser.add_argument("--n", type=int, default=200, help="Number of examples to compare")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance on diff vectors")
    args = parser.parse_args()

    new = torch.load(args.new, map_location="cpu")
    ref = torch.load(args.ref, map_location="cpu")

    new_d = _diff_vectors(new, args.n)
    ref_d = _diff_vectors(ref, args.n)

    n = min(len(new_d), len(ref_d))
    if n == 0:
        raise SystemExit("No comparable embedding pairs found.")

    new_d, ref_d = new_d[:n], ref_d[:n]
    max_abs = (new_d - ref_d).abs().max().item()
    mean_abs = (new_d - ref_d).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(new_d, ref_d, dim=1).mean().item()

    print(f"Compared {n} example diff-vectors (dim={new_d.shape[1]})")
    print(f"  max |Δ|  = {max_abs:.6f}")
    print(f"  mean |Δ| = {mean_abs:.6f}")
    print(f"  mean cosine similarity = {cos:.6f}")

    ok = max_abs <= args.atol
    print(f"RESULT: {'MATCH' if ok else 'MISMATCH'} (atol={args.atol})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
