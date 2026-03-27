# LoRe Suitability Evaluation

Diagnostic metrics that predict how well LoRe will learn distinct, predictive
user representations on a new dataset.  Each metric is calibrated so that
**random data FAILs** and **PRISM (a known-good dataset) PASSes**.

## Background

LoRe decomposes reward as:

$$\text{reward}(x) = x \cdot V w$$

where $V \in \mathbb{R}^{D \times K}$ is a shared basis (pretrained) and
$w \in \mathbb{R}^K$ is a per-user weight vector (fitted at personalisation time).

Preference embeddings are $e = \text{embed}(\text{chosen}) - \text{embed}(\text{rejected})$,
always oriented so that a correct prediction means $e \cdot Vw > 0$.

## Metric reference

All metrics take `user_pref_embeddings`: a list of per-user tensors, each of
shape `[n_prefs_i, D]`.  Metrics that need the pretrained basis also take
`V: Tensor[D, K]`.

---

### Tier 0 -- Annotation Density

**What it measures:**
Whether each user has enough preference pairs to reliably constrain a
$K$-dimensional weight vector.

**Math:**
For each user $i$, count $n_i = |\text{pairs}_i|$.  Report the median count
and the fraction of users below the $2K$ rule-of-thumb minimum.

$$\text{fraction\_below} = \frac{1}{U} \sum_{i=1}^{U} \mathbf{1}[n_i < 2K]$$

**Intuition:**
A $K$-dimensional vector has $K$ free parameters.  With fewer than $2K$
observations the least-squares fit is underdetermined and the user vector will
be dominated by noise.

**Pass/Fail:**
- PASS: median pairs/user $\geq 5$
- WARN: median $\in [2, 5)$
- FAIL: median $< 2$

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 10 pairs/user |
| PRISM (750 users) | 9 pairs/user |
| Random (200 users) | 10 pairs/user |

> Annotation density does not distinguish random from structured data --
> it only checks whether there is *enough* data, not whether it is *learnable*.

---

### Tier 1 -- Label Balance (normalised)

**What it measures:**
Per-user directional consistency of preference embeddings, normalised against
the expected value for random data.

**Math:**
For each user $i$ with $n_i$ preference vectors $\{x_1, \ldots, x_{n_i}\}$:

$$\text{raw}_i = \frac{\|\bar{x}_i\|}{\frac{1}{n_i}\sum_j \|x_j\|}
\qquad\text{where}\quad \bar{x}_i = \frac{1}{n_i}\sum_j x_j$$

For iid random vectors in $D$ dimensions, $\mathbb{E}[\text{raw}] = 1/\sqrt{n}$
(the mean of $n$ random unit-ish vectors has norm $\sim 1/\sqrt{n}$).
We normalise out this baseline:

$$\text{normalised}_i = \text{raw}_i \cdot \sqrt{n_i}$$

Report $\bar{\text{normalised}} = \frac{1}{U}\sum_i \text{normalised}_i$.

**Intuition:**
If a user's preferences all point in a consistent direction (they have a clear
preference axis), the mean embedding will be large relative to per-vector norms.
For random preferences, the mean shrinks as $1/\sqrt{n}$ by CLT.  After
normalisation, a value of 1.0 means "exactly as consistent as random" and
values above 1.0 indicate genuine directional structure.

**Pass/Fail:**
- PASS: normalised consistency $> 1.3$
- WARN: $\in (1.1, 1.3]$
- FAIL: $\leq 1.1$

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 2.196 |
| PRISM (750 users) | 2.194 |
| Random (200 users) | 1.000 |

---

### Tier 1 -- Krippendorff Alpha Proxy (noise-corrected ICC)

**What it measures:**
Whether user identity predicts meaningful variation in preferences, above and
beyond what sampling noise alone would produce.

**Math:**
Pool all $N = \sum_i n_i$ preference vectors.  Compute the grand mean
$\bar{x}$ and per-user means $\bar{x}_i$:

$$\text{between\_var} = \frac{1}{U}\sum_{i=1}^{U} \|\bar{x}_i - \bar{x}\|^2$$

$$\text{total\_var} = \frac{1}{N}\sum_{i,j} \|x_{ij} - \bar{x}\|^2$$

The raw ratio $r = \text{between\_var} / \text{total\_var}$ is an ICC-style
decomposition.  But for iid random data, $\bar{x}_i$ has variance
$\text{total\_var}/n_i$ purely from sampling noise, giving:

$$\mathbb{E}[r \mid \text{random}] = \frac{1}{U}\sum_i \frac{1}{n_i} = \overline{1/n}$$

We subtract this:

$$\text{corrected\_ratio} = r - \overline{1/n}$$

**Intuition:**
If all users drew from the same distribution, user means would still differ by
sampling noise.  The correction removes exactly that expected noise, so
$\text{corrected\_ratio} \approx 0$ for random data and $> 0$ only when users
are genuinely distinct.

**Pass/Fail:**
- PASS: corrected ratio $> 0.03$
- WARN: $\in (0.01, 0.03]$
- FAIL: $\leq 0.01$

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 0.0494 |
| PRISM (750 users) | 0.0438 |
| Random (200 users) | -0.0005 |

---

### Tier 1 -- Nearest-Neighbour Accuracy (split-half)

**What it measures:**
Whether users who are geometrically similar (close mean embeddings) actually
share individual preferences -- LoRe's core assumption.

**Math:**
Split each user's data randomly into two halves: *train* and *test*.

1. Compute means from train halves: $\mu_i^{\text{train}} = \text{mean}(X_i^{\text{train}})$
2. Build NN graph on normalised train means:
   $j^*(i) = \arg\max_{j \neq i} \cos(\mu_i^{\text{train}},\, \mu_j^{\text{train}})$
3. Score each user's *test* pairs using the NN's train mean:
   $\text{acc}_i = \frac{1}{|X_i^{\text{test}}|} \sum_{x \in X_i^{\text{test}}} \mathbf{1}[x \cdot \mu_{j^*(i)}^{\text{train}} > 0]$
4. Report $\bar{\text{acc}} = \frac{1}{U}\sum_i \text{acc}_i$

**Why split-half:**
Without splitting, there is a transitive correlation:
$x_i \to \mu_i \to \text{(NN selection)} \to \mu_j$.  Because $\mu_i$ is the
mean of $x_i$'s group, $x_i \cdot \mu_i > 0$ on average, and NN selection
makes $\mu_j \approx \mu_i$, inflating accuracy above 0.5 even for random
data.  Splitting breaks this chain: the test pairs did not contribute to the
mean used for NN selection.

**Intuition:**
If a user's nearest neighbour (by average preference direction) can predict
that user's individual preference pairs, then geometrically similar users
genuinely share preferences.  This is exactly what LoRe needs to work: users
in similar parts of embedding space should behave similarly.  Does not require
the pretrained basis $V$.

**Pass/Fail:**
- PASS: mean NN accuracy $> 0.6$
- WARN: $\in (0.55, 0.6]$
- FAIL: $\leq 0.55$

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 1.000 |
| PRISM (750 users) | 0.998 |
| Random (200 users) | 0.506 |

---

### Tier 1 -- Inter-User Agreement

**What it measures:**
Pairwise cosine similarity between user mean preference vectors.

**Math:**
$$\mu_i = \text{mean}(X_i), \qquad \hat{\mu}_i = \mu_i / \|\mu_i\|$$

$$\text{sim}_{ij} = \hat{\mu}_i \cdot \hat{\mu}_j$$

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

### Tier 3 -- Basis Space Coherence (noise-corrected ICC in V-space)

**What it measures:**
Whether users cluster meaningfully *in the pretrained basis space* -- i.e.,
whether $V$ captures dimensions along which users differ.

**Math:**
Project all preferences into the $K$-dimensional basis space:
$z_{ij} = x_{ij} \cdot V$, giving $Z_i \in \mathbb{R}^{n_i \times K}$.

Then apply the same noise-corrected ICC decomposition as Krippendorff proxy,
but on the projected data:

$$\text{between\_var}_Z = \frac{1}{U}\sum_i \|\bar{z}_i - \bar{z}\|^2, \qquad
\text{total\_var}_Z = \frac{1}{N}\sum_{i,j} \|z_{ij} - \bar{z}\|^2$$

$$\text{corrected\_ratio}_Z = \frac{\text{between\_var}_Z}{\text{total\_var}_Z} - \overline{1/n}$$

**Intuition:**
Users may be distinct in the full $D$-dimensional embedding space yet look
identical once projected onto $V$ (if $V$ is misaligned with this domain's
preference dimensions).  This metric specifically tests basis alignment: a
positive corrected ratio means user identity predicts variation *within the
basis space LoRe actually uses*.

**Pass/Fail:**
- PASS: corrected ratio $> 0.03$
- WARN: $\in (0.01, 0.03]$
- FAIL: $\leq 0.01$

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 0.0173 (WARN) |
| PRISM (750 users) | 0.0433 |
| Random (200 users) | -0.0060 |

> The small PRISM subset gets WARN here because 50 users is borderline for
> detecting between-user variance in $K=8$ space.

---

### Tier 3 -- Population Accuracy

**What it measures:**
Whether there is a universal preference signal in this domain and whether the
pretrained $V$ captures it.

**Math:**
Pool all $N$ preference vectors and split into 80% train / 20% test (shuffled).

1. Project: $Z^{\text{train}} = X^{\text{train}} V$
2. Fit a single weight vector via least squares:
   $w = \arg\min_w \|Z^{\text{train}} w - \mathbf{1}\|^2$
3. Evaluate on test:
   $\text{acc} = \frac{1}{N_{\text{test}}} \sum_{x \in X^{\text{test}}} \mathbf{1}[x \cdot V w > 0]$

**Intuition:**
This fits one user vector for *everyone* pooled together.  If accuracy is above
chance, there exists a shared preference direction that $V$ can represent.
Random data fails because there is no shared signal; a misaligned $V$ (trained
on a very different domain) fails because it cannot represent the signal even if
one exists.

**Pass/Fail:**
- PASS: accuracy $> 0.6$
- WARN: $\in (0.55, 0.6]$
- FAIL: $\leq 0.55$

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 0.990 |
| PRISM (750 users) | 0.991 |
| Random (200 users) | 0.500 |

---

### Tier 3 -- User Vector Diversity

**What it measures:**
How spread out the fitted user vectors are in basis space.

**Math:**
Fit per-user vectors $w_i$ via least squares (see below), apply softmax
$\tilde{w}_i = \text{softmax}(w_i)$, normalise, compute pairwise cosine
similarity:

$$d = 1 - \frac{1}{U(U-1)} \sum_{i \neq j} \cos(\tilde{w}_i, \tilde{w}_j)$$

Also computes effective rank of the user vector covariance matrix
$\text{Cov}(\tilde{W})$ as the count of eigenvalues above $1\%$ of the max.

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

### Tier 3 -- Basis Utilization Entropy

**What it measures:**
How uniformly users spread their weight across the $K$ basis vectors.

**Math:**
Apply softmax to each user vector: $\tilde{w}_i = \text{softmax}(w_i)$.
Per-user Shannon entropy:

$$H_i = -\sum_{k=1}^{K} \tilde{w}_{ik} \log \tilde{w}_{ik}$$

Report normalised mean: $\bar{H} / \log K$.

**Intuition:**
$\bar{H}/\log K \approx 1.0$ means users spread weight uniformly across all
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

### Tier 5 -- Held-Out Accuracy (per-user cross-validation)

**What it measures:**
Per-user generalisation: can a user vector fitted on part of a user's data
predict the rest?  This is the most faithful fast proxy for what LoRe will
achieve in production.

**Math:**
For each user $i$ with $n_i \geq 4$ pairs, hold out the last $20\%$:

1. $Z_i^{\text{train}} = X_i^{\text{train}} V$
2. $w_i = \arg\min_w \|Z_i^{\text{train}} w - \mathbf{1}\|^2$
3. $\text{acc}_i = \frac{1}{|X_i^{\text{test}}|} \sum_{x \in X_i^{\text{test}}} \mathbf{1}[x \cdot V w_i > 0]$

Report $\bar{\text{acc}} = \frac{1}{U'}\sum_i \text{acc}_i$ where $U'$ is the
number of users with $\geq 4$ pairs.

**Intuition:**
This directly measures whether a user's preferences generalise beyond the
training pairs.  It uses the closed-form least-squares proxy for PersonalizeBatch
(LoRe's gradient-based adaptation), so it is fast but slightly conservative.

**Pass/Fail:**
- PASS: mean accuracy $> 0.6$
- WARN: $\in (0.55, 0.6]$
- FAIL: $\leq 0.55$

| Dataset | Value |
|---------|-------|
| PRISM (50 users) | 0.903 |
| PRISM (750 users) | 0.871 |
| Random (200 users) | 0.497 |

---

## Closed-form user vector fitting

Several metrics (user vector diversity, basis utilization entropy, held-out
accuracy) rely on fitted user weight vectors.  These are computed via
closed-form least squares, a fast proxy for LoRe's gradient-based
PersonalizeBatch:

$$w_i = \arg\min_w \|X_i V w - \mathbf{1}\|^2 = (V^\top X_i^\top X_i V)^{-1} V^\top X_i^\top \mathbf{1}$$

The target vector $\mathbf{1}$ reflects that all preference embeddings are
oriented so that $e \cdot Vw > 0$ should hold for a correct prediction.

---

## Deprecated metrics

The following metrics are still available in `suitability.py` for backwards
compatibility but are **not included** in `evaluate_suitability()`:

- **Effective rank** -- threshold $< 1.0$ is vacuous when $n_\text{users} \ll D$
  (Marchenko-Pastur distribution guarantees random matrices have ratio $< 1.0$).
- **Silhouette score** -- K-means always finds local minima in any distribution;
  no principled null without permutation testing.
- **Basis activation variance** -- replaced by basis space coherence (which
  applies the noise-corrected ICC decomposition instead of raw variance).
- **Fit quality** -- training accuracy is always 1.0 due to overfitting
  (underdetermined least-squares system when $n_i < D$).

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
