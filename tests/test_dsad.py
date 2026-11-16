import json

import numpy as np
from PIL import Image

from davinci.data import dsad


def _frame(path, value):
    Image.fromarray(np.full((4, 4, 3), value, np.uint8)).save(path)


def _surgery(root, structure, surgery, values, weak_rows):
    d = root / structure / f"{surgery:02d}"
    for k in (1, 2, 3):
        (d / f"anno_{k}").mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(values):
        _frame(d / f"image{i:02d}.png", v)
        for target in (d, d / "anno_1", d / "anno_2", d / "anno_3"):
            Image.fromarray(np.zeros((4, 4), np.uint8)).save(target / f"mask{i:02d}.png")
    (d / "weak_labels.csv").write_text("".join(
        f"images{i:03d}.png," + ",".join(map(str, r)) + ",\n" for i, r in enumerate(weak_rows)))


def _row(**present):
    return [int(present.get(c, 0)) for c in dsad.WEAK_COLUMNS] + [0]


def _vesicular_row():
    return [0] * 11 + [1]


def test_fusion_states_and_split(tmp_path):
    for s in dsad.STRUCTURES:
        (tmp_path / s).mkdir()
    # Frame 10 appears in ureter and colon folders of surgery 2 (test).
    _surgery(tmp_path, "ureter", 2, [10, 20], [_row(ureter=1, colon=1), _row(ureter=1)])
    _surgery(tmp_path, "colon", 2, [10], [_row(colon=1, ureter=1)])
    _surgery(tmp_path, "vesicular_glands", 1, [30], [_vesicular_row()])
    m = dsad.build_manifest(tmp_path)
    recs = {(r["surgery"], tuple(r["sources"])): r for r in m["records"]}
    fused = recs[(2, ("colon/00", "ureter/00"))]
    assert fused["split"] == "test"
    assert fused["label_state"]["ureter"] == "mask" and fused["label_state"]["colon"] == "mask"
    assert fused["label_state"]["liver"] == "absent"
    assert fused["presence"]["liver"] == 0 and fused["presence"]["ureter"] == 1
    assert len(fused["annotators"]["ureter"]) == 3
    single = recs[(2, ("ureter/01",))]
    assert single["label_state"]["colon"] == "absent"
    ves = recs[(1, ("vesicular_glands/00",))]
    assert ves["split"] == "train"
    assert ves["label_state"]["liver"] == "unknown"  # no presence labels in this folder
    assert ves["presence"]["liver"] is None and ves["presence"]["vesicular_glands"] == 1
    assert m["summary"]["frames"] == 3
    json.dumps(m)


def test_malformed_row_gives_unknown(tmp_path):
    for s in dsad.STRUCTURES:
        (tmp_path / s).mkdir()
    _surgery(tmp_path, "liver", 3, [5], [_row(liver=1)[:-3]])
    m = dsad.build_manifest(tmp_path)
    r = m["records"][0]
    assert m["summary"]["missing_or_malformed_weak_rows"] == 1
    assert r["label_state"]["liver"] == "mask" and r["label_state"]["colon"] == "unknown"


def test_eleven_value_rows_are_valid(tmp_path):
    p = tmp_path / "weak_labels.csv"
    p.write_text("images000.png,0,1,0,0,0,0,0,0,0,0,1,\n")
    row = dsad.read_weak_labels(p)[0]
    assert row["colon"] == 1 and row["vesicular_glands"] == 1 and row["liver"] == 0
