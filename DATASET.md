# Dataset

## Source and terms

| | |
|---|---|
| **Dataset** | Tomato-Village, Variant-c (Object Detection) |
| **Authors** | M. Gehlot, R.K. Saxena, G.C. Gandhi |
| **Paper** | *Multimedia Systems* (2023), https://doi.org/10.1007/s00530-023-01158-y |
| **Repository** | https://github.com/mamta-joshi-gehlot/Tomato-Village |
| **Commit used** | `dfc246014558293987b381c1edb49398d9f570ac` |
| **Licence** | None declared in the upstream repository (checked October 2026) |

We **do not redistribute images or labels.** This repository contains only:
- file names, split assignments and cluster ids, in `splits/split_manifest.csv` and `artifacts/phase9/arms/*.tsv`;
- statistics computed from our own models.

Obtain the data from the authors, follow their terms, and cite their paper.

## Leakage structure of the released split

This is computed from file names and labels only (`experiments/9_posthoc/ph_leakage.py` → `artifacts/posthoc/leakage.json`).

| Property | Value |
|---|---|
| Files | 14,368 (official train 11,493 / official validation 2,875) |
| Source photographs (families) | 1,796, each with **exactly 8 files** (1 original + 7 augmented copies) |
| Families spanning official train and validation | 1,484 |
| Official validation files with a sibling in official train | **2,875 / 2,875 (100%)** |
| Official test split | none |

Inferred augmentation type of the 12,572 copies, from comparing their labels with the original's (photometric changes are invisible to labels):

| Type | Share |
|---|---|
| Label-identical (photometric only) | 32.7% |
| Crop / scale / shift | 28.3% |
| Horizontal flip | 14.2% |
| Vertical flip | 14.2% |
| 90° rotations | 7.2% |
| 180° rotation | 3.5% |

## Our family-aware split

Built by `src/adcf_yolo/data/prepare.py`; the result is committed as `splits/split_manifest.csv`.

1. Group files by source photograph (strip the `_augN` suffix).
2. Merge near-duplicate photographs (difference hash, Hamming distance ≤ 4). This found 59 pairs, giving 1,740 clusters.
3. Split clusters 70/15/15, stratified by majority class, with seed 0.
4. Keep augmented copies in **train only**.

| Split | Photographs | Files | Notes |
|---|---|---|---|
| Train | 1,260 | 10,080 | includes the 7 copies of each photograph |
| Validation | 268 | 268 | originals only |
| Test | 268 | 268 | originals only; 3,205 boxes, 261 clusters |

Two classes (Early Blight, Healthy) have no boxes. Several deficiency classes are very small.

## Development evaluation arms (Phase 9)

Both arms are built from the official validation files and exclude every test family. Manifests are in
`artifacts/phase9/arms/`, with hashes in `artifacts/phase9/arm_freeze_record.json`.

| Arm | Definition | Families | Files | Boxes |
|---|---|---|---|---|
| A, source-exposed | source family is in our training split | 1,047 | 2,028 | 23,225 |
| B, source-independent | source family is in our validation split | 213 | 411 | 4,683 |

## Rebuilding

```bash
uv run tv-download --commit dfc246014558293987b381c1edb49398d9f570ac
uv run tv-prepare --manifest splits/split_manifest.csv
```
