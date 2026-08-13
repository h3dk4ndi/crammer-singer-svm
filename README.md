# Crammer-Singer Multiclass SVM

NumPy implementation of the direct multiclass SVM from Crammer & Singer (2001),
with linear, polynomial and RBF kernels. Benchmarked on UCI white wine quality.

Most multiclass SVMs decompose into many binary classifiers (one-vs-one or
one-vs-rest). Crammer & Singer solve a single optimisation over all `k` classes
at once, using a generalised multiclass margin.

## Usage

```python
model = CrammerSingerSVM(X_train, y_train, beta=1.1, kernel="rbf", gamma=0.05)
model.fit(epsilon=0.3, max_rounds=500_000)
y_pred = model.predict(X_test)
```

`beta` is the paper's regularisation `ν` (`C = 1/beta`); `epsilon` sets the
stopping tolerance, halting when `max_i ψ_i < ε·β`.

## Benchmark

3961 unique samples after removing 937 exact duplicates, 11 features,
7 quality bands, standardised, stratified 80/20 split.

| model | conv | rounds | time | acc | macro F1 | MAE | ±1 |
|---|---|---|---|---|---|---|---|
| majority baseline | — | — | — | 0.4515 | 0.0889 | 0.6330 | 0.9218 |
| linear | yes | 98,910 | 28s | 0.5296 | 0.1777 | 0.5460 | 0.9294 |
| poly `d=2,3,4` | **no** | 1,000,000 | ~260s | — | — | — | — |
| **RBF `γ=0.05`** | yes | 5,597 | **2.2s** | **0.5599** | 0.2590 | **0.5044** | 0.9445 |
| RBF `γ=0.2` | yes | 5,151 | 2.0s | 0.5473 | **0.2773** | 0.5183 | 0.9445 |
| kernel ridge `λ=0.5` | — | — | — | 0.5624 | 0.2906 | 0.5032 | 0.9433 |

Random guessing over 7 classes is 14.3%.

## Notes

**RBF converges, polynomial does not.** RBF reaches the KKT tolerance in ~5,000
rounds and 2 seconds; polynomial kernels stall at $\max_i \psi_i \approx 2\text{–}4$
against a threshold of $\varepsilon\beta = 0.33$ after 1,000,000 rounds. The reduced
problem divides by $A_p = K(\bar{x}_p, \bar{x}_p)$, which is exactly $1$ for RBF
but spans orders of magnitude for a degree-4 polynomial.

**Exact solve instead of the fixed-point iteration.** The paper iterates

$$\theta_{l+1} \leftarrow \frac{1}{k}\left[\sum_{r=1}^{k} \max\{\theta_l, D_r\}\right] - \frac{1}{k}$$

at a rate of $1 - 1/k$ per step. At $k = 7$, sorting $\bar{D}$ and applying
Eq. (42), $\theta^* = \frac{1}{s}\left(\sum_{r \le s} D_{(r)} - 1\right)$,
is 27x faster and exact.

**Duplicate leakage.** The raw file has 937 exact duplicate rows, so a random
split puts an identical twin of ~31% of test rows into training. Accuracy is
$0.60$ before deduplication and $0.45$ after — the gap is memorisation.

**Ordinal structure.** Quality is ordered, but the Crammer-Singer loss treats
labels as unordered. Kernel ridge regression, which predicts a continuous score
and rounds, did not measurably improve ordinal error (MAE $0.5032$ vs $0.5044$,
inside the $\pm 0.018$ standard error at $n = 793$).

## Reference

Crammer, K. and Singer, Y. (2001). *On the Algorithmic Implementation of
Multiclass Kernel-based Vector Machines.* JMLR 2, 265-292.

Cortez, P. et al. (2009). *Modeling wine preferences by data mining from
physicochemical properties.* Decision Support Systems 47(4), 547-553.
