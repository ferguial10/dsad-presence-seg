import csv, glob, hashlib, os, collections, sys
import numpy as np
from PIL import Image
R = os.path.expanduser("~/datasets/DSAD/release")
S = sorted(d for d in os.listdir(R) if d != "multilabel")
# 1) weak label column order: fraction of 1s per column within each structure folder
for s in S:
    rows = []
    nimg = 0
    for f in glob.glob(f"{R}/{s}/*/weak_labels.csv"):
        rows += [r for r in csv.reader(open(f)) if r]
        nimg += len(glob.glob(os.path.dirname(f) + "/image*.png"))
    a = np.array([[int(x) for x in r[1:12]] for r in rows])
    print(f"{s:28s} rows={len(rows):5d} imgs={nimg:5d} col_mean=" + " ".join(f"{v:.2f}" for v in a.mean(0)))
print("first names", rows[:2])
# 2) duplicated frames across folders (same surgery)
h = collections.defaultdict(list)
for s in S + ["multilabel"]:
    for f in glob.glob(f"{R}/{s}/*/image*.png"):
        h[hashlib.md5(open(f, "rb").read()).hexdigest()].append(f"{s}/{f.split('/')[-2]}")
c = collections.Counter(len(v) for v in h.values()); print("unique frames", len(h), "multiplicity", sorted(c.items()))
cross = [v for v in h.values() if len({x.split('/')[1] for x in v}) > 1]; print("frames shared across surgeries", len(cross))
# 3) mask vs merged vs annotators on a sample
d = []
for f in sorted(glob.glob(f"{R}/ureter/*/mask*.png"))[:40]:
    base = os.path.dirname(f); n = os.path.basename(f)
    m = np.array(Image.open(f)) > 0
    mg = np.array(Image.open(f"{base}/merged/{n}")) > 0
    an = [np.array(Image.open(f"{base}/anno_{k}/{n}")) > 0 for k in (1, 2, 3)]
    maj = (sum(an) >= 2)
    d.append(((m == mg).mean(), (m == maj).mean(), (mg == maj).mean()))
print("mask==merged, mask==majority, merged==majority", np.mean(d, 0))
