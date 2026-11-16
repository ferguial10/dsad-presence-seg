# Data

Place the datasets here (not tracked by git):

- `data/DSAD/`: Dresden Surgical Anatomy Dataset (CC BY 4.0),
  doi:10.6084/m9.figshare.21702600. Folder layout as released (one folder per structure,
  `multilabel/`, surgery subfolders with `imageXX.png`, `maskXX.png`, `anno_{1,2,3}/`, `weak_labels.csv`).
- `data/CholecSeg8k/`: CholecSeg8k (external zero-shot evaluation only).
- `data/endovis2018_anatomy_development_manifest_v1.json`: list of EndoVis 2018 RSS development
  frames and binary intestine masks (`{"records": {"train": [...], "validation": [...]}}`, each
  record with `unit`, `image`, `mask`); built from the official release, external evaluation only.
