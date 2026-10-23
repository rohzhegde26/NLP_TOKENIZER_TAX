"""Tests for statistics functions (Lane C, D2)."""
from __future__ import annotations

import numpy as np
from tokentax.stats import (
    bootstrap_ci,
    bootstrap_diff_ci,
    bootstrap_slope_ci,
    fit_slope,
    holm,
    mcnemar,
)


def test_bootstrap_ci():
    data = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    low, high = bootstrap_ci(data, n_boot=500, seed=42)
    assert 0.0 <= low <= high <= 1.0
    assert low < np.mean(data) < high


def test_bootstrap_diff_ci():
    a = [1, 1, 1, 0, 1, 1]
    b = [1, 0, 0, 0, 1, 0]
    low, high = bootstrap_diff_ci(a, b, n_boot=500, seed=42)
    assert low <= high
    assert (low + high) / 2 > 0  # a is generally better than b


def test_mcnemar_exact():
    # Identical predictions -> p = 1.0
    assert mcnemar([1, 0, 1], [1, 0, 1]) == 1.0
    # Discordants: 10 vs 0 -> significant difference
    a = [1] * 15 + [0] * 5
    b = [0] * 15 + [0] * 5
    p = mcnemar(a, b)
    assert p < 0.001


def test_holm_stepdown():
    raw_p = [0.01, 0.04, 0.03]
    adj = holm(raw_p)
    assert len(adj) == 3
    assert adj[0] <= adj[2] <= adj[1]
    assert all(a >= r for a, r in zip(adj, raw_p))


def test_linear_slope_and_ci():
    # y = -0.1 * x + 0.9 + noise
    x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    y = np.array([0.80, 0.74, 0.69, 0.65, 0.61, 0.55, 0.50])
    slope, intercept, r2 = fit_slope(x, y)
    assert slope < 0  # Fragmentation hurts accuracy
    assert 0.8 < r2 <= 1.0

    low, high = bootstrap_slope_ci(x, y, n_boot=500, seed=42)
    assert low <= slope <= high
