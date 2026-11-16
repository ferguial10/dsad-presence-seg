"""Dresden Surgical Anatomy Dataset (DSAD): frame-level manifest with fused labels.

DSAD stores one folder per structure. Each surgery folder holds ``imageXX.png``,
a binary ``maskXX.png`` for that folder's structure, three single-annotator
masks (``anno_{1,2,3}/``), a ``merged/`` mask and ``weak_labels.csv`` with
image-level presence for every structure. The same frame can appear in several
folders, so records are keyed by the MD5 of the image bytes within a surgery and
the annotations of all folders are fused.

For every frame and structure the label state is ``mask`` (pixel labels),
``absent`` (presence label 0 and no mask) or ``unknown``. ``presence`` keeps
the image-level label (1, 0 or None when missing or conflicting).
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

STRUCTURES = (
    "abdominal_wall", "colon", "inferior_mesenteric_artery", "intestinal_veins",
    "liver", "pancreas", "small_intestine", "spleen", "stomach", "ureter",
    "vesicular_glands",
)
CRITICAL = ("ureter", "inferior_mesenteric_artery", "intestinal_veins", "pancreas")
# weak_labels.csv: the dataset article (Carstens et al. 2023) documents 11
# columns in alphabetical structure order. The release has 11 or 12 values per
# row; a 12th value is padding (always 0 outside ``vesicular_glands``).
WEAK_COLUMNS = STRUCTURES
# Folders whose weak_labels.csv is unreliable: in ``vesicular_glands`` every row
# marks only a 12th column, so presence of other structures is unknown.
FOLDERS_WITHOUT_PRESENCE = ("vesicular_glands",)
MULTILABEL_ORGANS = ("abdominal_wall", "colon", "liver", "pancreas",
                     "small_intestine", "spleen", "stomach")

# Kolbinger et al., Int J Surg 2023;109(10):2962-2974.
SPLIT = {
    "train": (1, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 19, 22, 23, 24, 25, 27, 28,
              29, 30, 31),
    "val": (3, 21, 26),
    "test": (2, 7, 11, 13, 14, 18, 20, 32),
}
SURGERY_SPLIT = {s: name for name, ids in SPLIT.items() for s in ids}

_IMAGE = re.compile(r"^image(\d+)\.png$")
_WEAK = re.compile(r"^images?(\d+)\.png$")
_MULTI_MASK = re.compile(r"^mask(\d+)_([a-z_]+)\.png$")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_weak_labels(path: Path) -> dict[int, dict[str, int] | None]:
    """Map frame index to presence labels; malformed rows map to ``None``."""
    labels: dict[int, dict[str, int] | None] = {}
    for row in csv.reader(path.read_text().splitlines()):
        if not row:
            continue
        match = _WEAK.match(row[0])
        if not match:
            raise ValueError(f"{path}: unexpected row name {row[0]!r}")
        values = [v for v in row[1:] if v != ""]
        index = int(match.group(1))
        if len(values) not in (11, 12) or any(v not in ("0", "1") for v in values):
            labels[index] = None
        else:
            labels[index] = dict(zip(WEAK_COLUMNS, map(int, values[:11])))
    return labels


def build_manifest(root: Path) -> dict:
    """Scan the DSAD release and return fused frame records plus a summary."""
    root = Path(root)
    frames: dict[tuple[int, str], dict] = {}
    malformed = 0
    for structure in STRUCTURES:
        for surgery_dir in sorted(p for p in (root / structure).iterdir() if p.is_dir()):
            surgery = int(surgery_dir.name)
            weak_path = surgery_dir / "weak_labels.csv"
            weak = read_weak_labels(weak_path) if weak_path.exists() else {}
            for image in sorted(surgery_dir.glob("image*.png")):
                index = int(_IMAGE.match(image.name).group(1))
                mask = surgery_dir / f"mask{index:02d}.png"
                if not mask.exists():
                    raise FileNotFoundError(mask)
                key = (surgery, _md5(image))
                record = frames.setdefault(key, {
                    "surgery": surgery, "split": SURGERY_SPLIT[surgery],
                    "frame_md5": key[1], "image": str(image.relative_to(root)),
                    "sources": [], "masks": {}, "annotators": {},
                    "presence_votes": {s: [] for s in STRUCTURES},
                    "multilabel_masks": {},
                })
                record["sources"].append(f"{structure}/{index:02d}")
                record["masks"][structure] = str(mask.relative_to(root))
                record["annotators"][structure] = [
                    str((surgery_dir / f"anno_{k}" / mask.name).relative_to(root))
                    for k in (1, 2, 3) if (surgery_dir / f"anno_{k}" / mask.name).exists()]
                row = weak.get(index)
                if row is None:
                    malformed += 1
                elif structure not in FOLDERS_WITHOUT_PRESENCE:
                    for s in STRUCTURES:
                        record["presence_votes"][s].append(row[s])
    multilabel_root = root / "multilabel"
    multilabel_frames = 0
    if multilabel_root.exists():
        for surgery_dir in sorted(p for p in multilabel_root.iterdir() if p.is_dir()):
            surgery = int(surgery_dir.name)
            for image in sorted(surgery_dir.glob("image*.png")):
                index = int(_IMAGE.match(image.name).group(1))
                key = (surgery, _md5(image))
                record = frames.setdefault(key, {
                    "surgery": surgery, "split": SURGERY_SPLIT[surgery],
                    "frame_md5": key[1], "image": str(image.relative_to(root)),
                    "sources": [], "masks": {}, "annotators": {},
                    "presence_votes": {s: [] for s in STRUCTURES},
                    "multilabel_masks": {},
                })
                record["sources"].append(f"multilabel/{index:02d}")
                for organ in MULTILABEL_ORGANS:
                    path = surgery_dir / f"mask{index:02d}_{organ}.png"
                    if path.exists():
                        record["multilabel_masks"][organ] = str(path.relative_to(root))
                multilabel_frames += 1

    records = []
    conflicts = 0
    for record in frames.values():
        states = {}
        for s in STRUCTURES:
            votes = record["presence_votes"][s]
            if len(set(votes)) > 1:
                conflicts += 1
            if s in record["masks"]:
                states[s] = "mask"
            elif votes and max(votes) == 0:
                states[s] = "absent"
            else:
                states[s] = "unknown"
        record["label_state"] = states
        # Image-level presence: 1/0 when the votes agree (a mask implies 1), else None.
        record["presence"] = {
            s: 1 if s in record["masks"] else
            (record["presence_votes"][s][0] if len(set(record["presence_votes"][s])) == 1 else None)
            for s in STRUCTURES}
        record["presence_conflict"] = any(
            len(set(record["presence_votes"][s])) > 1 for s in STRUCTURES)
        del record["presence_votes"]
        records.append(record)
    records.sort(key=lambda r: (r["surgery"], r["image"]))

    cross = {}
    for r in records:
        cross.setdefault(r["frame_md5"], set()).add(r["surgery"])
    leaked = sum(len(v) > 1 for v in cross.values())
    if leaked:
        raise ValueError(f"{leaked} frames appear in more than one surgery")

    summary = {"frames": len(records), "missing_or_malformed_weak_rows": malformed,
               "presence_conflicts": conflicts,
               "multilabel_frame_entries": multilabel_frames, "by_split": {}}
    for split in SPLIT:
        rs = [r for r in records if r["split"] == split]
        summary["by_split"][split] = {
            "surgeries": sorted({r["surgery"] for r in rs}),
            "frames": len(rs),
            "states": {s: {st: sum(r["label_state"][s] == st for r in rs)
                           for st in ("mask", "absent", "unknown")}
                       for s in STRUCTURES},
        }
    return {"kind": "dsad_frame_manifest", "version": 3, "root": str(root),
            "structures": list(STRUCTURES), "weak_columns": list(WEAK_COLUMNS),
            "split_source": "Kolbinger et al., Int J Surg 2023;109(10):2962-2974",
            "split": {k: list(v) for k, v in SPLIT.items()},
            "summary": summary, "records": records}
