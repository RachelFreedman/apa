# LoRe Suitability Evaluation

Diagnostic metrics that predict how well LoRe will learn distinct, predictive
user representations on a new dataset.  Each metric is calibrated so that
**random data FAILs** and **PRISM (a known-good dataset) PASSes**.

## Background

LoRe decomposes reward as:

```
reward(x) = x @ V @ w
```

where `V` is a shared basis `[D, K]` (pretrained) and `w` is a per-user
weight vector `[K]` (fitted at personalisation time).

Preference embeddings are `e = embed(chosen) - embed(rejected)`, always
oriented so that a correct prediction means `e @ V @ w > 0`.

## Metric reference

All metrics take `user_pref_embeddings`: a list of per-user tensors, each of
shape `[n_prefs_i, D]`.  Metrics that need the pretrained basis also take
`V: Tensor[D, K]`.

---

### Annotation Density

**What it measures:**
Whether each user has enough preference pairs to reliably constrain a
K-dimensional weight vector.

**Math:**
For each user `i`, count `n_i = |pairs_i|`.  Report the median count and the
fraction of users below the `2K` rule-of-thumb minimum:

```
fraction_below = (1/U) * sum(1[n_i < 2K] for i in 1..U)
```

**Intuition:**
A K-dimensional vector has K free parameters.  With fewer than 2K observations
the least-squares fit is underdetermined and the user vector will be dominated
by noise.

**Pass/Fail:**
- PASS: median pairs/user >= 5
- WARN: median in [2, 5)
- FAIL: median < 2

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 10 pairs/user |
| PRISM (750 users) | 9 pairs/user |
| Random (200 users) | 10 pairs/user |

> Annotation density does not distinguish random from structured data --
> it only checks whether there is *enough* data, not whether it is *learnable*.

---

### Label Balance (normalised)

**What it measures:**
Per-user directional consistency of preference embeddings, normalised against
the expected value for random data.

**Math:**
For each user `i` with `n_i` preference vectors:

```
x_bar_i  = mean(x_1, ..., x_n_i)
raw_i    = ||x_bar_i|| / mean(||x_j|| for j in 1..n_i)
```

For iid random vectors in D dimensions, `E[raw] = 1/sqrt(n)` (the mean of n
random unit-ish vectors has norm ~`1/sqrt(n)`).  We normalise out this
baseline:

```
normalised_i = raw_i * sqrt(n_i)
```

Report the mean across users: `mean(normalised_i for i in 1..U)`.

**Intuition:**
If a user's preferences all point in a consistent direction (they have a clear
preference axis), the mean embedding will be large relative to per-vector norms.
For random preferences, the mean shrinks as `1/sqrt(n)` by CLT.  After
normalisation, a value of 1.0 means "exactly as consistent as random" and
values above 1.0 indicate genuine directional structure.

**Pass/Fail:**
- PASS: normalised consistency > 1.3
- WARN: in (1.1, 1.3]
- FAIL: <= 1.1

| Dataset | Value | Threshold | Status |
|---------|-------|-----------|--------|
| PRISM (50 users) | 2.196 | > 1.3 | PASS |
| PRISM (750 users) | 2.194 | > 1.3 | PASS |
| Random (200 users) | 1.000 | > 1.3 | FAIL |

---

### Krippendorff Alpha Proxy (noise-corrected ICC)

**What it measures:**
Whether user identity predicts meaningful variation in preferences, above and
beyond what sampling noise alone would produce.

**Math:**
Pool all `N = sum(n_i)` preference vectors.  Compute the grand mean `x_bar`
and per-user means `x_bar_i`:

```
between_var = (1/U) * sum(||x_bar_i - x_bar||^2 for i in 1..U)
total_var   = (1/N) * sum(||x_ij - x_bar||^2 for all i,j)
```

The raw ratio `r = between_var / total_var` is an ICC-style decomposition.
But for iid random data, `x_bar_i` has variance `total_var / n_i` purely from
sampling noise, giving:

```
E[r | random] = (1/U) * sum(1/n_i for i in 1..U) = mean_recip_n
```

We subtract this:

```
corrected_ratio = r - mean_recip_n
```

**Intuition:**
If all users drew from the same distribution, user means would still differ by
sampling noise.  The correction removes exactly that expected noise, so
`corrected_ratio ~= 0` for random data and `> 0` only when users are genuinely
distinct.

**Pass/Fail:**
- PASS: corrected ratio > 0.03
- WARN: in (0.01, 0.03]
- FAIL: <= 0.01

| Dataset | Value | Threshold | Status |
|---------|-------|-----------|--------|
| PRISM (50 users) | 0.0494 | > 0.03 | PASS |
| PRISM (750 users) | 0.0438 | > 0.03 | PASS |
| Random (200 users) | -0.0005 | > 0.03 | FAIL |

---

### Nearest-Neighbour Accuracy (split-half)

**What it measures:**
Whether users who are geometrically similar (close mean embeddings) actually
share individual preferences -- LoRe's core assumption.  Does not require the
pretrained basis V.

**Math:**
Split each user's data randomly into two halves: *train* and *test*.

1. Compute means from train halves: `mu_i = mean(X_i_train)`
2. Build NN graph on normalised train means:
   `j*(i) = argmax over j != i of cos(mu_i, mu_j)`
3. Score each user's *test* pairs using the NN's train mean:
   `acc_i = mean(1[x . mu_j*(i) > 0] for x in X_i_test)`
4. Report `mean(acc_i for i in 1..U)`

**Why split-half:**
Without splitting, there is a transitive correlation: `x_i -> mu_i -> (NN
selection) -> mu_j`.  Because `mu_i` is the mean of `x_i`'s group,
`x_i . mu_i > 0` on average, and NN selection makes `mu_j ~= mu_i`, inflating
accuracy above 0.5 even for random data.  Splitting breaks this chain: the test
pairs did not contribute to the mean used for NN selection.

**Intuition:**
If a user's nearest neighbour (by average preference direction) can predict
that user's individual preference pairs, then geometrically similar users
genuinely share preferences.  This is exactly what LoRe needs to work: users
in similar parts of embedding space should behave similarly.

**Pass/Fail:**
- PASS: mean NN accuracy > 0.6
- WARN: in (0.55, 0.6]
- FAIL: <= 0.55

| Dataset | Value | Threshold | Status |
|---------|-------|-----------|--------|
| PRISM (50 users) | 1.000 | > 0.6 | PASS |
| PRISM (750 users) | 0.998 | > 0.6 | PASS |
| Random (200 users) | 0.506 | > 0.6 | FAIL |

---

### Inter-User Agreement

**What it measures:**
Pairwise cosine similarity between user mean preference vectors.

**Math:**

```
mu_i     = mean(X_i)
mu_hat_i = mu_i / ||mu_i||
sim_ij   = mu_hat_i . mu_hat_j
```

Report mean, std, min, max of the off-diagonal entries of the similarity
matrix.

**Intuition:**
High mean similarity means users mostly agree (little room for
personalisation).  Low mean similarity with high variance means a mixed bag.
Very low similarity means users broadly disagree.  This is an informational
metric -- there is no hard pass/fail threshold.

**Pass/Fail:** INFO only (no threshold).

| Dataset | Value (mean sim) |
|---------|------------------|
| PRISM (50 users) | (varies) |
| PRISM (750 users) | (varies) |
| Random (200 users) | ~0.0 |

---

### Basis Space Coherence (noise-corrected ICC in V-space)

**What it measures:**
Whether users cluster meaningfully *in the pretrained basis space* -- i.e.,
whether V captures dimensions along which users differ.  Requires V.

**Math:**
Project all preferences into the K-dimensional basis space:
`z_ij = x_ij @ V`, giving per-user tensors `Z_i` of shape `[n_i, K]`.

Then apply the same noise-corrected ICC decomposition as Krippendorff proxy,
but on the projected data:

```
between_var_Z = (1/U) * sum(||z_bar_i - z_bar||^2 for i in 1..U)
total_var_Z   = (1/N) * sum(||z_ij - z_bar||^2 for all i,j)
corrected_ratio_Z = between_var_Z / total_var_Z - mean_recip_n
```

**Intuition:**
Users may be distinct in the full D-dimensional embedding space yet look
identical once projected onto V (if V is misaligned with this domain's
preference dimensions).  This metric specifically tests basis alignment: a
positive corrected ratio means user identity predicts variation *within the
basis space LoRe actually uses*.

**Pass/Fail:**
- PASS: corrected ratio > 0.03
- WARN: in (0.01, 0.03]
- FAIL: <= 0.01

| Dataset | Value | Threshold | Status |
|---------|-------|-----------|--------|
| PRISM (50 users) | 0.0173 | > 0.03 | WARN |
| PRISM (750 users) | 0.0433 | > 0.03 | PASS |
| Random (200 users) | -0.0060 | > 0.03 | FAIL |

> The small PRISM subset gets WARN here because 50 users is borderline for
> detecting between-user variance in K=8 space.

---

### Population Accuracy

**What it measures:**
Whether there is a universal preference signal in this domain and whether the
pretrained V captures it.  Requires V.

**Math:**
Pool all N preference vectors and split into 80% train / 20% test (shuffled).

1. Project: `Z_train = X_train @ V`
2. Fit a single weight vector via least squares: `w = lstsq(Z_train, ones)`
3. Evaluate on test: `acc = mean(1[x @ V @ w > 0] for x in X_test)`

**Intuition:**
This fits one user vector for *everyone* pooled together.  If accuracy is above
chance, there exists a shared preference direction that V can represent.
Random data fails because there is no shared signal; a misaligned V (trained
on a very different domain) fails because it cannot represent the signal even if
one exists.

**Pass/Fail:**
- PASS: accuracy > 0.6
- WARN: in (0.55, 0.6]
- FAIL: <= 0.55

| Dataset | Value | Threshold | Status |
|---------|-------|-----------|--------|
| PRISM (50 users) | 0.990 | > 0.6 | PASS |
| PRISM (750 users) | 0.991 | > 0.6 | PASS |
| Random (200 users) | 0.500 | > 0.6 | FAIL |

---

### User Vector Diversity

**What it measures:**
How spread out the fitted user vectors are in basis space.  Requires V.

**Math:**
Fit per-user vectors `w_i` via least squares (see below), apply softmax,
normalise, compute pairwise cosine similarity:

```
w_tilde_i = softmax(w_i)
w_hat_i   = w_tilde_i / ||w_tilde_i||
d = 1 - mean(w_hat_i . w_hat_j for i != j)
```

Also computes effective rank of the user vector covariance matrix
`Cov(W_tilde)` as the count of eigenvalues above 1% of the max.

**Intuition:**
Low diversity means all users have similar weights -- personalisation is not
helping.  High diversity means LoRe has found meaningfully different weight
configurations for different users.

**Pass/Fail:** INFO only (no threshold).

| Dataset | Value (mean dist) |
|---------|-------------------|
| PRISM (50 users) | 0.461 |
| PRISM (750 users) | 0.423 |
| Random (200 users) | 0.462 |

---

### Basis Utilization Entropy

**What it measures:**
How uniformly users spread their weight across the K basis vectors.  Requires V.

**Math:**
Apply softmax to each user vector, then compute per-user Shannon entropy:

```
w_tilde_i = softmax(w_i)
H_i       = -sum(w_tilde_ik * log(w_tilde_ik) for k in 1..K)
```

Report normalised mean: `mean(H_i) / log(K)`.

**Intuition:**
A normalised entropy near 1.0 means users spread weight uniformly across all
bases.  A low value means most users concentrate on 1-2 bases, suggesting the
full rank is not being utilised and the pretrained bases may not cover this
domain's preference dimensions.

**Pass/Fail:** INFO only (no threshold).

| Dataset | Value (norm entropy) |
|---------|----------------------|
| PRISM (50 users) | 0.705 |
| PRISM (750 users) | 0.743 |
| Random (200 users) | 0.770 |

---

### Held-Out Accuracy (per-user cross-validation)

**What it measures:**
Per-user generalisation: can a user vector fitted on part of a user's data
predict the rest?  This is the most faithful fast proxy for what LoRe will
achieve in production.  Requires V.

**Math:**
For each user `i` with `n_i >= 4` pairs, hold out the last 20%:

1. Project training pairs: `Z_i_train = X_i_train @ V`
2. Fit user vector: `w_i = lstsq(Z_i_train, ones)`
3. Evaluate on held-out pairs: `acc_i = mean(1[x @ V @ w_i > 0] for x in X_i_test)`

Report `mean(acc_i)` over users with >= 4 pairs.

**Intuition:**
This directly measures whether a user's preferences generalise beyond the
training pairs.  It uses the closed-form least-squares proxy for PersonalizeBatch
(LoRe's gradient-based adaptation), so it is fast but slightly conservative.

**Pass/Fail:**
- PASS: mean accuracy > 0.6
- WARN: in (0.55, 0.6]
- FAIL: <= 0.55

| Dataset | Value | Threshold | Status |
|---------|-------|-----------|--------|
| PRISM (50 users) | 0.903 | > 0.6 | PASS |
| PRISM (750 users) | 0.871 | > 0.6 | PASS |
| Random (200 users) | 0.497 | > 0.6 | FAIL |

---

## Closed-form user vector fitting

Several metrics (user vector diversity, basis utilization entropy, held-out
accuracy) rely on fitted user weight vectors.  These are computed via
closed-form least squares, a fast proxy for LoRe's gradient-based
PersonalizeBatch:

```
w_i = lstsq(X_i @ V, ones)
    = (V' X_i' X_i V)^-1 V' X_i' 1
```

The target vector of ones reflects that all preference embeddings are oriented
so that `e @ V @ w > 0` should hold for a correct prediction.

---

## Deprecated metrics

The following metrics are still available in `suitability.py` for backwards
compatibility but are **not included** in `evaluate_suitability()`:

- **Effective rank** -- threshold < 1.0 is vacuous when n\_users << D
  (Marchenko-Pastur distribution guarantees random matrices have ratio < 1.0).
- **Silhouette score** -- K-means always finds local minima in any distribution;
  no principled null without permutation testing.
- **Basis activation variance** -- replaced by basis space coherence (which
  applies the noise-corrected ICC decomposition instead of raw variance).
- **Fit quality** -- training accuracy is always 1.0 due to overfitting
  (underdetermined least-squares system when n\_i < D).

---

## Usage

```python
from apa.eval.suitability import evaluate_suitability, embed_preferences
import torch

# If you have raw text preferences:
user_pref_embeddings = embed_preferences(user_prefs, model, tokenizer)

# If you have pre-computed embeddings (e.g. from load_prism):
V = torch.load("models/V_K8.pt")
results = evaluate_suitability(user_pref_embeddings, V=V)
```

Or run the report script directly:

```bash
python -m apa.eval.check_suitability
```
