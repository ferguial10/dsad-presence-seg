import numpy as np

from davinci.metrics import stats


def test_sign_flip_all_positive_is_minimum_p():
    # With 8 equal positive differences only the identity and full flip are as extreme.
    assert stats.sign_flip_p([0.1] * 8) == 2 / 256


def test_sign_flip_symmetric_is_one():
    assert stats.sign_flip_p([0.1, -0.1, 0.2, -0.2]) == 1.0


def test_bootstrap_ci_contains_mean():
    d = np.linspace(-0.05, 0.15, 8)
    lo, hi = stats.bootstrap_ci(d, draws=2000, seed=0)
    assert lo < d.mean() < hi


def test_holm_matches_hand_computation():
    adj = stats.holm({"h1": 0.01, "h2": 0.04})
    assert adj == {"h1": 0.02, "h2": 0.04}
    adj = stats.holm({"h1": 0.03, "h2": 0.02})
    assert adj == {"h2": 0.04, "h1": 0.04}


def test_non_inferior():
    c = stats.paired_contrast([0.5] * 8, [0.51] * 8, draws=500)
    assert stats.non_inferior(c, 0.03) and not stats.non_inferior(c, 0.005)


def test_sign_flip_monte_carlo_for_large_n():
    p = stats.sign_flip_p([0.1] * 32)
    assert p < 1e-4  # only (1 + 0) / (1 + draws) up to rare exact ties
