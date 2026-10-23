"""Statistical tests and confidence intervals (owner: Rohith).

Includes:
- bootstrap_ci: percentile bootstrap CI over items/qids (default: 1000 resamples)
- bootstrap_diff_ci: paired difference bootstrap CI for interventions (E3/E4)
- mcnemar: exact McNemar test for paired binary outcomes
- holm: Holm-Bonferroni step-down correction for multiple testing
- fit_slope: linear regression slope for accuracy vs fertility (E4)
- bootstrap_slope_ci: bootstrap CI for the fragmentation slope
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    data,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    stat_fn=np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a 1D sequence."""
    x = np.asarray(data, dtype=float)
    if len(x) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot_stats = stat_fn(x[indices], axis=1) if stat_fn == np.mean else np.array([stat_fn(x[idx]) for idx in indices])
    return float(np.quantile(boot_stats, alpha / 2)), float(np.quantile(boot_stats, 1 - alpha / 2))


def bootstrap_diff_ci(
    a,
    b,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired difference bootstrap CI (mean(a - b))."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)


def mcnemar(a, b) -> float:
    """Exact McNemar test p-value for paired 0/1 outcomes a and b.

    Uses statsmodels contingency tables with exact binomial test.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if len(a) != len(b):
        raise ValueError(f"Lengths must match: len(a)={len(a)}, len(b)={len(b)}")

    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0

    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
        # 2x2 table: [[a=1, b=1], [a=1, b=0]], [[a=0, b=1], [a=0, b=0]]
        table = [[int((a & b).sum()), n10], [n01, int((~a & ~b).sum())]]
        res = sm_mcnemar(table, exact=True)
        return float(res.pvalue)
    except Exception:
        from scipy.stats import binomtest
        return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals) -> list[float]:
    """Holm-Bonferroni step-down correction for multiple testing."""
    p = np.asarray(pvals, dtype=float)
    if len(p) == 0:
        return []
    try:
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(p, method="holm")
        return corrected.tolist()
    except Exception:
        order = np.argsort(p)
        adj = np.empty_like(p)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(p) - rank) * p[i]))
            adj[i] = running
        return adj.tolist()


def fit_slope(x, y) -> tuple[float, float, float]:
    """Compute OLS linear slope, intercept, and R^2 for accuracy vs fertility."""
    from scipy.stats import linregress

    res = linregress(x, y)
    r2 = float(res.rvalue ** 2) if res.rvalue is not None else 0.0
    return float(res.slope), float(res.intercept), r2


def bootstrap_slope_ci(
    x,
    y,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for linear slope (x=fertility, y=accuracy)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx, by = x[idx], y[idx]
        denom = np.sum((bx - np.mean(bx)) ** 2)
        if denom == 0:
            continue
        slope = np.sum((bx - np.mean(bx)) * (by - np.mean(by))) / denom
        slopes.append(slope)

    if not slopes:
        return (0.0, 0.0)
    return float(np.quantile(slopes, alpha / 2)), float(np.quantile(slopes, 1 - alpha / 2))
