import math

import numpy as np
from scipy.stats import beta

from davinci.metrics import risk_control


def test_pvalue_small_when_risk_far_below_alpha():
    assert risk_control.hb_pvalue(0.0, 1000, 0.05) < 1e-10
    assert risk_control.hb_pvalue(0.05, 1000, 0.05) > 0.5


def test_threshold_controls_risk_on_fresh_data():
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 201)
    violations = 0
    for _ in range(200):
        cal = rng.beta(2, 5, 1500)
        lam = risk_control.calibrate_threshold(cal, alpha=0.05, delta=0.1, grid=grid)["lambda"]
        violations += beta.sf(lam, 2, 5) > 0.05  # true hallucination rate
    assert violations / 200 <= 0.1


def test_no_certified_threshold_means_abstain():
    out = risk_control.calibrate_threshold([1.0] * 10, alpha=0.05, delta=0.1, grid=[0.5, 1.0])
    assert math.isinf(out["lambda"])
