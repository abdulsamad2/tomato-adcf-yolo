# 01 · The dataset, and the leakage we found

## What Tomato-Village Variant-c is

Tomato-Village (Gehlot, Saxena & Gandhi, *Multimedia Systems*, 2023) has three
variants. We use **Variant-c (object detection)**: field photos from Jaipur and
Jodhpur (Rajasthan, India) with bounding boxes in YOLO format, in 8 classes:

`Early Blight, Healthy, Late Blight, Leaf Miner, Magnesium Deficiency,
Nitrogen Deficiency, Potassium Deficiency, Spotted Wilt Virus`

(The repository spells "Pottassium" and mixes `_`/spaces. We fix the display
names only; class ids are unchanged.)

## What we found by inspecting the file list

| Fact | Number |
|---|---|
| Files (train + val) | 14,368 (11,493 + 2,875) |
| Distinct original photos | **1,796** |
| Files per photo | exactly 8: the original + `_aug1` … `_aug7` |
| Photos whose files appear in **both** official train and val | **all 1,484** photos behind the val files |
| Official test split | none |

So the authors augmented each photo 7× **before** splitting, then split *files*
at random. Almost every validation image has an augmented twin of the same photo
in the training set. (Open a few `_augN` files next to their original to see which
transforms were used, and describe them in the paper.)

## Why this is a serious problem

**Data leakage** means information from the evaluation set is available during
training. A detector that has seen `IMG_123_aug2` in training will find the same
lesions in `IMG_123_aug5` easily, even if it would fail on a genuinely new
field photo. Scores on the official val split are therefore **inflated**, and:

- you can't tell whether ADCF *generalises* better or just *memorises* better;
- published numbers on the official split aren't comparable to honest numbers;
- a careful reviewer who looks at the file names will reject the paper.

There's a second, subtler problem: with no test split, people select the best
epoch (and tune hyperparameters) on the same val set they report. That is also
optimistic, because you pick whatever happened to score well by chance on that set.

## What `tv-prepare` does about it

1. **Group by original photo.** `IMG_123`, `IMG_123_aug1` … `_aug7` form one group.
2. **Merge near-duplicates.** Field photographers often shoot the same plant twice.
   We compute a 64-bit *difference hash* (dHash: shrink to 9×8 grey pixels, record
   whether each pixel is brighter than its right neighbour) and merge photos whose
   hashes differ in ≤ 4 bits. **Check the listed pairs by eye.** Hashing is a
   heuristic: if it merges unrelated photos, lower `--dup-dist`; if you find
   duplicates it missed, raise it.
3. **Stratified group split, 70/15/15.** Each group gets a *stratum* = its majority
   class. Within each stratum, groups are shuffled (fixed seed) and dealt into
   train/val/test. Rare classes therefore appear in every split.
4. **Offline copies only in train.** Train uses the original + 7 copies; val and
   test use **only originals**, because we want to measure performance on real
   photos, not synthetic transforms.
5. **Validate labels.** Class ids like `3.0` are normalised to `3`; boxes are
   clipped to the image; degenerate/duplicate boxes and malformed lines are
   dropped and counted in `prepare_report.json`.
6. **Manifest.** `split_manifest.csv` records which photo went where. **Commit it
   and publish it with the paper**, so anyone can reproduce your exact split
   (`tv-prepare --manifest splits/split_manifest.csv`).

## Roles of the three splits (memorise this)

| Split | Used for | Never used for |
|---|---|---|
| train | fitting weights | reporting |
| val | early stopping, picking `best.pt`, any tuning you do | the paper's final numbers |
| test | **only** the final numbers, once per model | any decision |

If you look at test results and then change the model, test has become a second
val set. Make design decisions on val.

## EXIF orientation (check `label_preview.png`)

Phone photos often store pixels sideways plus an EXIF "rotate 90°" flag. OpenCV
(used by Ultralytics) applies the flag, but labelling tools sometimes don't. If
that happened, boxes would be drawn on the wrong region. `tv-stats` draws the
labels on EXIF-corrected images: **if boxes look misplaced, stop and investigate
before training anything.**

## Point for the paper

Put the leakage analysis in the dataset section with the numbers from
`prepare_report.json`. Also consider an extra experiment: train YOLO11s on the
**official** split and report its (inflated) val score next to your honest test
score. The gap directly shows why the re-split was needed.
