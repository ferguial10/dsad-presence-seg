"""Distribution-free control of structure hallucination (DESIGN.md amendment 2).

For one structure, every image gets a presence score ``q``; the structure is
claimed present when ``q >= lam``. The controlled risk is the hallucination
rate on images whose presence label is 0. With binary losses the risk is
non-increasing in ``lam``, so Learn-then-Test reduces to fixed-sequence testing
from the strictest threshold down, using the Hoeffding-Bentkus p-value
(Bates et al., JACM 2021; Angelopoulos et al., Ann. Appl. Stat. 2025).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def _kl_bernoulli(a: float, b: float) -> float:
    a = min(max(a, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: risk > alpha (binary losses)."""
    hoeffding = math.exp(-n * _kl_bernoulli(min(risk_hat, alpha), alpha))
    bentkus = math.e * binom.cdf(math.ceil(n * risk_hat), n, alpha)
    return min(1.0, hoeffding, bentkus)


def calibrate_threshold(scores_absent, alpha: float, delta: float, grid) -> dict:
    """Smallest threshold whose hallucination rate is certified <= alpha w.p. 1-delta.

    ``scores_absent`` are presence scores of calibration images where the
    structure is absent; an image hallucinates at ``lam`` when its score is
    >= lam. Returns ``lam = inf`` (always abstain) if no threshold is certified.
    """
    s = np.asarray(scores_absent, float)
    n = len(s)
    chosen, tested = math.inf, 0
    for lam in sorted(grid, reverse=True):
        tested += 1
        risk = float(np.mean(s >= lam))
        if hb_pvalue(risk, n, alpha) > delta:
            break
        chosen = lam
    return {"lambda": chosen, "n_calibration": n, "alpha": alpha, "delta": delta,
            "thresholds_tested": tested,
            "calibration_risk": float(np.mean(s >= chosen)) if math.isfinite(chosen) else 0.0}
