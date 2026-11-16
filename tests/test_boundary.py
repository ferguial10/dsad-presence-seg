import numpy as np

from davinci.metrics import boundary


def _bar(x0, width=3):
    m = np.zeros((40, 40), bool)
    m[5:35, x0:x0 + width] = True
    return m


def test_identical_masks_are_perfect():
    m = _bar(10)
    assert boundary.cldice(m, m) == 1.0
    assert boundary.hd95(m, m) == 0.0
    assert boundary.nsd(m, m, 1.0) == 1.0


def test_shift_changes_distance_by_shift():
    a, b = _bar(10), _bar(14)
    assert boundary.hd95(a, b) == 4.0
    assert boundary.nsd(a, b, 4.0) == 1.0 and boundary.nsd(a, b, 1.0) < 0.5
    assert boundary.cldice(a, b) == 0.0  # disjoint bars


def test_empty_prediction():
    assert np.isnan(boundary.hd95(np.zeros((40, 40), bool), _bar(10)))
    assert boundary.nsd(np.zeros((40, 40), bool), _bar(10), 5) == 0.0
