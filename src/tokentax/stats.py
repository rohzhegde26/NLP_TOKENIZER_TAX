"""Statistics (owner: Rohith). Working first versions so others can import; Rohith may refine."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(correct, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar(a, b) -> float:
    """Exact McNemar p-value for paired 0/1 outcomes a, b."""
    from scipy.stats import binomtest

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
