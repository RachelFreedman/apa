"""Synthetic APA world: MFQ-2 feature bases, cohort weight distributions, Stage-3 fitting, and voting.

Juror n's reward for response i is R_n(i) = w_n . phi(i), where phi(i) in [-2, 2]^6 is the mean of two blind
ratings of response i on the six MFQ-2 foundations (outputs/features.json, written by features.py).
"""
import json, os, warnings
from collections import Counter
import numpy as np
warnings.filterwarnings('ignore')
from pref_voting.profiles import Profile
from pref_voting.voting_methods import plurality, borda, instant_runoff_put, copeland

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, 'data'), os.path.join(HERE, 'outputs')
DESIGN = json.load(open(os.path.join(DATA, 'design.json')))
DIMS = DESIGN['dims']                       # Care, Equality, Proportionality, Loyalty, Authority, Purity
DEMO_Q, CALIB_Q = list(DESIGN['demo']), list(DESIGN['calib'])
FEAT = {k: np.array(v) for k, v in json.load(open(os.path.join(OUT, 'features.json'))).items()}

def slate(qid):
    """Feature matrix of a question's candidate responses (10 per demonstration question, 8 per calibration question)."""
    n = 10 if qid in DESIGN['demo'] else 8
    return np.stack([FEAT[f"{qid}_{i:02d}"] for i in range(1, n + 1)])

# ---------------- annotator cohorts (hand-specified two-component mixtures) ----------------
COHORTS = {
    '16C': [(0.75, (0.5, -1.0, 0.8, 0.8, 1.2, 1.2)),    # traditionalist majority
            (0.25, (1.0, -0.3, 0.8, 0.5, 0.6, 0.3))],   # reformist minority
    '20C': [(0.50, (0.8, -0.2, 0.8, 0.8, 0.8, 0.6)),    # traditionalists
            (0.50, (1.0, 0.8, 0.5, 0.3, 0.0, -0.2))],   # liberals
    '21C': [(0.70, (1.2, 1.2, 0.2, 0.0, -0.4, -0.6)),   # progressive majority
            (0.30, (0.8, 0.0, 0.8, 0.7, 0.7, 0.7))],    # conservative minority
}
SIGMA_WITHIN = 0.3                          # isotropic spread around each component mean

def sample_cohort(rng, cohort, n):
    comps = COHORTS[cohort]
    p = np.array([c[0] for c in comps]); mu = np.array([c[1] for c in comps])
    z = rng.choice(len(comps), size=n, p=p)
    return mu[z] + SIGMA_WITHIN * rng.standard_normal((n, len(DIMS)))

# ---------------- Stage 3: Bradley-Terry comparisons + ridge-regularized MLE ----------------
CALIB_PAIRS = [(q, i, j) for q in CALIB_Q for i in range(8) for j in range(i + 1, 8)]     # 6 x 28 = 168 pairs
CALIB_DIFF = np.stack([FEAT[f"{q}_{i+1:02d}"] - FEAT[f"{q}_{j+1:02d}"] for q, i, j in CALIB_PAIRS])

def simulate_comparisons(rng, w, m, beta=1.0):
    """m Bradley-Terry choices by a juror with true weights w, on random within-question calibration pairs."""
    idx = rng.choice(len(CALIB_PAIRS), size=m, replace=m > len(CALIB_PAIRS))
    X = CALIB_DIFF[idx]
    y = (rng.random(m) < 1 / (1 + np.exp(-beta * X @ w))).astype(float)   # 1: first response preferred
    return X, y

def fit_bt(X, y, lam=0.5, iters=50):
    """argmin_w sum_k -log sigma(s_k w.x_k) + lam ||w||^2, by Newton's method (strictly convex)."""
    s = 2 * y - 1; Z = X * s[:, None]; w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = -Z.T @ (1 - p) + 2 * lam * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + 2 * lam * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8: break
    return w

def fit_jurors(rng, W, m, lam=0.5):
    return np.stack([fit_bt(*simulate_comparisons(rng, w, m), lam=lam) for w in W])

# ---------------- Stage 2: voting ----------------
RULES = {'Plurality': plurality, 'Borda': borda, 'IRV-PUT': instant_runoff_put, 'Copeland': copeland}

def rankings(W, F):
    return np.argsort(-(W @ F.T), axis=1, kind='stable')

def profile(W, F):
    c = Counter(tuple(int(x) for x in r) for r in rankings(W, F))
    return Profile([list(k) for k in c], rcounts=list(c.values()))

def winners(W, F):
    """Winner set (1-indexed response numbers) under each rule."""
    P = profile(W, F)
    return {n: sorted(int(x) + 1 for x in f(P)) for n, f in RULES.items()}

def mean_spearman(W, F):
    """Mean pairwise Spearman correlation between jurors' rankings of the slate."""
    pos = np.argsort(rankings(W, F), axis=1).astype(float)
    Z = pos - pos.mean(1, keepdims=True); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    C = Z @ Z.T; return float(C[np.triu_indices(len(W), 1)].mean())
