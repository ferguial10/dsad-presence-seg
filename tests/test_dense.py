import numpy as np

from davinci.metrics import dense


def test_agreement_identical_annotators_is_one():
    m = np.zeros((4, 4), bool)
    m[1:3, 1:3] = True
    out = dense.annotator_agreement([m, m, m], published=m)
    assert out == {"loo_majority": 1.0, "loo_union": 1.0, "pairwise": 1.0, "vs_published": 1.0}


def test_agreement_loo_majority_is_intersection_of_two_others():
    a = np.array([[1, 1, 0, 0]], bool)
    b = np.array([[1, 1, 1, 0]], bool)
    c = np.array([[1, 0, 0, 0]], bool)
    out = dense.annotator_agreement([a, b, c])
    # a vs (b&c)=[1,0,0,0]: 1/2; b vs (a&c)=[1,0,0,0]: 1/3; c vs (a&b)=[1,1,0,0]: 1/2
    assert np.isclose(out["loo_majority"], (1 / 2 + 1 / 3 + 1 / 2) / 3)
    # a vs (b|c)=[1,1,1,0]: 2/3; b vs (a|c)=[1,1,0,0]: 2/3; c vs (a|b)=[1,1,1,0]: 1/3
    assert np.isclose(out["loo_union"], (2 / 3 + 2 / 3 + 1 / 3) / 3)
    assert np.isclose(out["pairwise"], (2 / 3 + 1 / 2 + 1 / 3) / 3)
    assert "vs_published" not in out


def test_agreement_all_empty_has_no_keys():
    z = np.zeros((3, 3), bool)
    assert dense.annotator_agreement([z, z, z], published=z) == {}


def test_scene_labels_argmax_background_and_gate():
    logits = np.array([[[2.0, -1.0, 0.5]], [[1.0, -2.0, 3.0]]])  # 2 organs, 1x3 image
    assert dense.scene_labels(logits, [True, True]).tolist() == [[1, 0, 2]]
    assert dense.scene_labels(logits, [True, False]).tolist() == [[1, 0, 1]]
    assert dense.scene_labels(logits, [False, False]).tolist() == [[0, 0, 0]]


def test_gt_labels_marks_overlap_ignore():
    a = np.array([[1, 1, 0]], bool)
    b = np.array([[0, 1, 1]], bool)
    lab, n = dense.gt_labels([a, b])
    assert lab.tolist() == [[1, 255, 2]] and n == 1


def test_confusion_counts_skip_ignore():
    pred = np.array([[1, 1, 0, 2]], np.uint8)
    gt = np.array([[1, 255, 0, 1]], np.uint8)
    c = dense.confusion_counts(pred, gt, n_organs=2)
    assert c["n_pixels"] == 3 and c["correct"] == 2
    assert c["inter"] == [1, 1, 0] and c["union"] == [1, 2, 1]
