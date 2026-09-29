"""Blind ratings -> response features phi(i) (mean of the two raters), plus the agreement statistics.
Writes outputs/features.json and outputs/rating_agreement.txt."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = lambda f: os.path.join(HERE, 'data', f)
OUT = os.path.join(HERE, 'outputs'); os.makedirs(OUT, exist_ok=True)
design = json.load(open(D('design.json'))); DIMS, DEMO = design['dims'], design['demo']
key = json.load(open(D('rating_key.json')))            # shuffled answer id (R01..R78) -> item key (e.g. Q1_03)
A, B = json.load(open(D('ratings_raterA.json'))), json.load(open(D('ratings_raterB.json')))
assert set(A) == set(B) == set(key)
inv = {v: k for k, v in key.items()}; items = sorted(inv)
RA = np.array([A[inv[i]] for i in items], float); RB = np.array([B[inv[i]] for i in items], float)
F = (RA + RB) / 2
json.dump({i: F[n].tolist() for n, i in enumerate(items)}, open(os.path.join(OUT, 'features.json'), 'w'), indent=0)

def lin_kappa(x, y, cats=range(-2, 3)):
    cats = list(cats); k = len(cats); O = np.zeros((k, k))
    for a, b in zip(x, y): O[cats.index(int(a)), cats.index(int(b))] += 1
    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0))
    Wt = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    return 1 - (Wt * O).sum() / (Wt * E).sum()
T = np.array([p for q in DEMO.values() for _, p in q['targets']], float)          # design targets, 30 demo items
Fd = np.array([F[items.index(f"{q}_{t:02d}")] for q in DEMO for t in range(1, 11)])
stats = dict(n_items=len(items), exact_agreement=float(np.mean(RA == RB)), within_one=float(np.mean(np.abs(RA - RB) <= 1)),
             pearson_r=float(np.corrcoef(RA.ravel(), RB.ravel())[0, 1]), linear_weighted_kappa=float(lin_kappa(RA.ravel(), RB.ravel())),
             target_vs_rating_r=float(np.corrcoef(T.ravel(), Fd.ravel())[0, 1]),
             target_sign_agreement=float(np.mean(np.sign(Fd[T != 0]) == np.sign(T[T != 0]))))
lines = [f"Inter-rater agreement (rater A vs B), {len(items)} items x 6 foundations:",
         f"  exact {stats['exact_agreement']:.3f}; within one point {stats['within_one']:.3f}; "
         f"Pearson r {stats['pearson_r']:.3f}; linear-weighted kappa {stats['linear_weighted_kappa']:.3f}"]
lines += [f"    {n:15s} r = {np.corrcoef(RA[:, d], RB[:, d])[0, 1]:.2f}" for d, n in enumerate(DIMS)]
lines += ["Design target vs mean rating, 30 demonstration items:",
          f"  Pearson r {stats['target_vs_rating_r']:.3f}; sign agreement on nonzero targets {stats['target_sign_agreement']:.3f}"]
open(os.path.join(OUT, 'rating_agreement.txt'), 'w').write('\n'.join(lines) + '\n')
json.dump(stats, open(os.path.join(OUT, 'rating_agreement.json'), 'w'), indent=1)
print('\n'.join(lines))
