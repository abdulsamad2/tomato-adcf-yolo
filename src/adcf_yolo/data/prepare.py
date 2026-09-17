"""Build a leakage-free train/val/test split of Tomato-Village Variant-c.

Why this exists (see docs/notes/01-dataset-and-leakage.md):
  * Variant-c ships each photo as 8 files: the original + `_aug1`..`_aug7` copies.
  * The official train/val split was made per *file*, so every val photo also has
    augmented copies in train. Validation scores on it measure memorisation.
  * There is no test split, so "val" is used both for model selection and reporting.

What this script does:
  1. Scan train+val, group every file by its original photo id.
  2. Validate images and labels (class ids, box coordinates, degenerate boxes).
  3. Merge near-duplicate photos (perceptual hash) into one group.
  4. Stratified *group* split into train/val/test (default 70/15/15).
  5. Train gets originals + offline aug copies; val/test get originals only.
  6. Write an Ultralytics dataset.yaml, a split manifest (commit it!) and a report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from adcf_yolo.data.download import SUBDIR

AUG_RE = re.compile(r"_aug(\d+)$")
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
LABELS_FILE = "Varient-C Labels.txt"  # sic, as named in the repository
# Official class order (Varient-C Labels.txt), with spelling/underscore fixed for the paper.
DEFAULT_NAMES = [
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Potassium Deficiency",
    "Spotted Wilt Virus",
]
MIN_BOX = 1e-3  # boxes thinner than 0.1% of the image are annotation noise


@dataclass
class Sample:
    image: Path
    label: Path
    source_split: str
    original_id: str
    aug: int | None  # None for the original photo
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)


def clean_class_name(raw: str) -> str:
    name = raw.strip().strip("'\"").replace("_", " ").replace("Pottassium", "Potassium")
    return " ".join(w if w.isupper() else w.capitalize() for w in name.split())


def read_class_names(variant_dir: Path) -> list[str]:
    path = variant_dir / LABELS_FILE
    if not path.exists():
        return list(DEFAULT_NAMES)
    names = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            idx, raw = line.split(":", 1)
            names[int(idx.strip())] = clean_class_name(raw)
    return [names[i] for i in sorted(names)]


def parse_label(path: Path, nc: int, problems: Counter) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file; clip boxes to the image and drop invalid ones."""
    boxes = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems["bad_line"] += 1
            continue
        try:
            cls_f, cx, cy, w, h = map(float, parts)
        except ValueError:
            problems["bad_line"] += 1
            continue
        # Labels are stored as "3.0"; accept integral floats only.
        if not cls_f.is_integer() or not 0 <= cls_f < nc:
            problems["bad_class"] += 1
            continue
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        cx1, cy1, cx2, cy2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            problems["clipped_box"] += 1
        if cx2 - cx1 < MIN_BOX or cy2 - cy1 < MIN_BOX:
            problems["degenerate_box"] += 1
            continue
        box = (int(cls_f), (cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1)
        key = (box[0],) + tuple(round(v, 6) for v in box[1:])
        if key in boxes:
            problems["duplicate_box"] += 1
            continue
        boxes.add(key)
    return sorted(boxes)


def scan(variant_dir: Path, nc: int, problems: Counter) -> list[Sample]:
    samples = []
    for split in ("train", "val"):
        img_dir, lbl_dir = variant_dir / split / "images", variant_dir / split / "yolo"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"missing {img_dir}; run tv-download first")
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                problems["missing_label"] += 1
                continue
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                problems["corrupt_image"] += 1
                continue
            m = AUG_RE.search(img.stem)
            original_id = img.stem[: m.start()] if m else img.stem
            s = Sample(img, lbl, split, original_id, int(m.group(1)) if m else None)
            s.boxes = parse_label(lbl, nc, problems)
            if not s.boxes:
                problems["empty_label"] += 1
            samples.append(s)
    return samples


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing/compression, cheap to compare."""
    with Image.open(path) as im:
        im.draft("RGB", (256, 256))  # JPEG: decode at reduced scale, much faster
        g = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_pairs(ids: list[str], hashes: list[int], max_dist: int) -> list[tuple[str, str, int]]:
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        dist = np.bitwise_count(h[i] ^ h[i + 1 :])
        for j in np.nonzero(dist <= max_dist)[0]:
            pairs.append((ids[i], ids[i + 1 + j], int(dist[j])))
    return pairs


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def stratified_group_split(
    strata: dict[str, str], ratios: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """Assign each group to a split so every stratum keeps roughly the same ratios.

    LEARN: splitting *groups* (not files) is what prevents leakage; stratifying keeps
    rare classes present in val and test.
    """
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for gid in sorted(strata):
        by_stratum[strata[gid]].append(gid)
    assignment = {}
    for stratum in sorted(by_stratum):
        gids = by_stratum[stratum]
        rng.shuffle(gids)
        n = len(gids)
        n_test = round(n * ratios[2])
        n_val = round(n * ratios[1])
        if n - n_test - n_val < 1:  # tiny stratum: keep at least one group for training
            n_test, n_val = min(n_test, max(n - 1, 0)), 0
        for k, gid in enumerate(gids):
            assignment[gid] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return assignment


def safe_name(stem: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]+", "_", stem)).strip("_")


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different filesystem: fall back to copying
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def prepare(
    raw_root: Path,
    out: Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    train_aug: bool = True,
    dup_dist: int = 4,
    manifest: Path | None = None,
    link: str = "hardlink",
    overwrite: bool = False,
) -> dict:
    variant_dir = raw_root / SUBDIR if (raw_root / SUBDIR).is_dir() else raw_root
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to rebuild it")

    names = read_class_names(variant_dir)
    problems: Counter = Counter()
    samples = scan(variant_dir, len(names), problems)
    if not samples:
        raise RuntimeError(f"no usable images found under {variant_dir}")
    if out.exists():
        shutil.rmtree(out)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.original_id].append(s)

    # The official split leaks: count originals whose files sit in both train and val.
    leaked = sum(1 for g in groups.values() if len({s.source_split for s in g}) > 1)
    missing_original = sorted(gid for gid, g in groups.items() if all(s.aug is not None for s in g))

    def representative(g: list[Sample]) -> Sample:  # the un-augmented photo if present
        return min(g, key=lambda s: -1 if s.aug is None else s.aug)

    # Near-duplicate photos (e.g. the same leaf shot twice) must share a split too.
    gids = sorted(groups)
    print(f"[prepare] hashing {len(gids)} original photos for near-duplicate detection")
    dup_pairs = near_duplicate_pairs(gids, [dhash(representative(groups[g]).image) for g in gids], dup_dist)
    uf = UnionFind(gids)
    for a, b, _ in dup_pairs:
        uf.union(a, b)
    clusters: dict[str, list[str]] = defaultdict(list)
    for g in gids:
        clusters[uf.find(g)].append(g)

    # Stratum = majority class (by box count) of the cluster's original photos.
    global_counts = Counter(b[0] for s in samples if s.aug is None for b in s.boxes)
    strata = {}
    for cid, members in clusters.items():
        counts = Counter(b[0] for g in members for b in representative(groups[g]).boxes)
        if not counts:
            strata[cid] = "background"
        else:  # ties go to the globally rarer class
            strata[cid] = names[max(counts, key=lambda c: (counts[c], -global_counts[c]))]

    if manifest:
        with open(manifest, newline="") as f:
            fixed = {row["original_id"]: row["split"] for row in csv.DictReader(f)}
        unknown = set(gids) - set(fixed)
        if unknown:
            raise ValueError(f"{len(unknown)} photos not in manifest, e.g. {sorted(unknown)[:3]}")
        split_of = {g: fixed[g] for g in gids}
    else:
        cluster_split = stratified_group_split(strata, ratios, seed)
        split_of = {g: cluster_split[uf.find(g)] for g in gids}

    # Write the Ultralytics layout: images/<split>/x.jpg + labels/<split>/x.txt
    rows, file_counts, seen = [], Counter(), set()
    for gid in gids:
        split = split_of[gid]
        chosen = groups[gid] if (split == "train" and train_aug) else [representative(groups[gid])]
        for s in chosen:
            stem = safe_name(s.image.stem)
            if stem in seen:
                raise RuntimeError(f"name collision after sanitising: {stem}")
            seen.add(stem)
            place(s.image, out / "images" / split / f"{stem}{s.image.suffix.lower()}", link)
            lbl = out / "labels" / split / f"{stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in s.boxes))
            file_counts[split] += 1
        rows.append(
            {
                "original_id": gid,
                "split": split,
                "cluster": uf.find(gid),
                "stratum": strata[uf.find(gid)],
                "n_files_written": len(chosen),
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True))

    source = raw_root / "SOURCE.txt"
    report = {
        "source": source.read_text() if source.exists() else None,
        "args": {
            "ratios": ratios,
            "seed": seed,
            "train_aug": train_aug,
            "dup_dist": dup_dist,
            "manifest": str(manifest) if manifest else None,
        },
        "files_scanned": len(samples),
        "original_photos": len(groups),
        "official_split_leaked_photos": leaked,
        "photos_without_unaugmented_file": len(missing_original),
        "near_duplicate_pairs": [{"a": a, "b": b, "hamming": d} for a, b, d in dup_pairs],
        "clusters": len(clusters),
        "problems": dict(problems),
        "photos_per_split": dict(Counter(split_of.values())),
        "files_per_split": dict(file_counts),
        "strata_per_split": {
            sp: dict(Counter(r["stratum"] for r in rows if r["split"] == sp)) for sp in SPLITS
        },
    }
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"[prepare] {len(groups)} photos ({leaked} leaked across official splits), "
        f"{len(dup_pairs)} near-duplicate pairs -> {len(clusters)} clusters"
    )
    print(f"[prepare] photos/split {report['photos_per_split']}  files/split {dict(file_counts)}")
    print(f"[prepare] problems {dict(problems)}")
    print(f"[prepare] wrote {out / 'dataset.yaml'}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("data/raw/Tomato-Village"))
    p.add_argument("--out", type=Path, default=Path("data/tomato_village"))
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-train-aug", action="store_true", help="exclude the offline _augN copies from train")
    p.add_argument("--dup-dist", type=int, default=4, help="max dHash Hamming distance for near-duplicates")
    p.add_argument("--manifest", type=Path, help="reuse an existing split_manifest.csv instead of re-splitting")
    p.add_argument("--link", choices=("hardlink", "symlink", "copy"), default="hardlink")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if abs(sum(args.ratios) - 1) > 1e-6:
        p.error("--ratios must sum to 1")
    prepare(
        args.raw,
        args.out,
        tuple(args.ratios),
        args.seed,
        not args.no_train_aug,
        args.dup_dist,
        args.manifest,
        args.link,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
