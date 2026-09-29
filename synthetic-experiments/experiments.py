"""Experiments reported in Section 5. Writes outputs/results.json.

  votes             Table tab:synth_votes. For each jury (16C, 20C, 21C: 30 jurors each; mixed: 10 per period),
                    200 independent draws of new jurors, each fitted from m = 20 comparisons, voting on the three
                    demonstration questions under four rules. Also one walkthrough draw, for the rho-bar column.
  ranking_agreement Fraction of response pairs on the demonstration slates that a juror's fitted reward model
                    orders the same way as their true one (1,000 jurors per period, m = 20).

Usage: python experiments.py [n_draws=200]
"""
import json, os, sys
from collections import Counter
import numpy as np
from world import sample_cohort, fit_jurors, slate, winners, mean_spearman, RULES, DEMO_Q, COHORTS, OUT

SEED = 20260928      # each analysis uses its own stream: SEED + offset
M, LAM = 20, 0.5     # Stage-3 comparisons per new juror; ridge weight
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
JURIES = {'16C': {'16C': 30}, '20C': {'20C': 30}, '21C': {'21C': 30},
          'Mixed (10 each)': {'16C': 10, '20C': 10, '21C': 10}}

def jury(rng, spec):
    """Draw true weights for a jury with spec {cohort: count}, then fit each juror from M comparisons."""
    W = np.concatenate([sample_cohort(rng, c, n) for c, n in spec.items() if n > 0])
    return W, fit_jurors(rng, W, M, LAM)

def resolute(w):
    """Lowest-index tie-breaking for rules that return several winners."""
    return w[0]

def walkthrough(rng):
    """One jury draw per composition: mean pairwise Spearman correlation among fitted jurors (rho-bar)."""
    out = {}
    for jname, spec in JURIES.items():
        _, Wh = jury(rng, spec)
        out[jname] = {q: mean_spearman(Wh, slate(q)) for q in DEMO_Q}
    return out

def votes(rng, n_draws):
    """Winner counts over independent jury draws (fitted weights)."""
    res = {j: {q: {r: Counter() for r in RULES} for q in DEMO_Q} for j in JURIES}
    for _ in range(n_draws):
        for jname, spec in JURIES.items():
            _, Wh = jury(rng, spec)
            for q in DEMO_Q:
                w = winners(Wh, slate(q))
                for r in RULES: res[jname][q][r][resolute(w[r])] += 1
    return {j: {q: {r: dict(c.most_common()) for r, c in d.items()} for q, d in dq.items()} for j, dq in res.items()}

def pair_agreement(Wt, Wh):
    """Fraction of (juror, response pair) cases ordered the same way by true and fitted weights, averaged over questions."""
    out = []
    for q in DEMO_Q:
        F = slate(q); Rt, Rh = Wt @ F.T, Wh @ F.T; iu = np.triu_indices(F.shape[0], 1)
        st = np.sign(Rt[:, :, None] - Rt[:, None, :])[:, iu[0], iu[1]]
        sh = np.sign(Rh[:, :, None] - Rh[:, None, :])[:, iu[0], iu[1]]
        out.append(float((st == sh).mean()))
    return float(np.mean(out))

def ranking_agreement(rng, n=1000):
    out = {}
    for c in COHORTS:
        W = sample_cohort(rng, c, n)
        out[c] = pair_agreement(W, fit_jurors(rng, W, M, LAM))
    return out

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = dict(config=dict(seed=SEED, m=M, lam=LAM, n_draws=N_DRAWS, juries=JURIES, cohorts=COHORTS),
               walkthrough_rho=walkthrough(np.random.default_rng(SEED + 1)),
               votes=votes(np.random.default_rng(SEED + 11), N_DRAWS),
               ranking_agreement=ranking_agreement(np.random.default_rng(SEED + 5)))
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    print('ranking agreement:', {c: round(v, 3) for c, v in res['ranking_agreement'].items()})
