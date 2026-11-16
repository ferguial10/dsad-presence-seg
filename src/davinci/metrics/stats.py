"""Surgery-level paired statistics of the P2 protocol (DESIGN.md §6)."""
from __future__ import annotations

import itertools

import numpy as np


EXACT_MAX_N = 20


def sign_flip_p(diffs, two_sided: bool = True, draws: int = 100_000, seed: int = 2027) -> float:
    """Sign-flip p-value for the mean of paired differences.

    Exact over all 2^n assignments for n <= 20 (the test set has n = 8); above
    that, Monte Carlo with ``draws`` random assignments and p = (1 + k) / (1 + draws).
    """
    d = np.asarray(diffs, float)
    observed = d.mean()
    if len(d) <= EXACT_MAX_N:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
        means = (signs * d).mean(1)
        hits = np.abs(means) >= abs(observed) - 1e-12 if two_sided else means >= observed - 1e-12
        return float(np.mean(hits))
    signs = np.random.default_rng(seed).choice((1.0, -1.0), size=(draws, len(d)))
    means = (signs * d).mean(1)
    hits = np.abs(means) >= abs(observed) - 1e-12 if two_sided else means >= observed - 1e-12
    return float((1 + hits.sum()) / (1 + draws))


def bootstrap_ci(diffs, draws: int = 10_000, seed: int = 2027, level: float = 0.95):
    """Percentile bootstrap CI of the mean, resampling surgeries."""
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def paired_contrast(a, b, draws: int = 10_000, seed: int = 2027) -> dict:
    """Mean of a - b over surgeries with exact sign-flip p and bootstrap CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = bootstrap_ci(d, draws, seed)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci95": [lo, hi],
            "p_sign_flip": sign_flip_p(d), "diffs": d.tolist()}


def non_inferior(contrast: dict, margin: float) -> bool:
    """A - B is non-inferior when the lower CI bound exceeds -margin."""
    return contrast["ci95"][0] > -margin
