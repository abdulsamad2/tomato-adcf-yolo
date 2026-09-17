"""Dataset statistics for the paper's dataset table + a label sanity-check preview.

Object sizes use COCO buckets measured after resizing the long side to 640 px
(the training resolution): small < 32^2, medium < 96^2, large >= 96^2 pixels.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps
from ultralytics.data.utils import exif_size

from adcf_yolo.data.prepare import IMG_EXTS, SPLITS

REF_SIZE = 640
SMALL, MEDIUM = 32**2, 96**2


def load_dataset(data_yaml: Path) -> tuple[Path, dict]:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    return Path(cfg["path"]), cfg


def split_files(root: Path, cfg: dict, split: str) -> list[tuple[Path, Path]]:
    img_dir = root / cfg[split]
    lbl_dir = root / cfg[split].replace("images", "labels", 1)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    return [(p, lbl_dir / f"{p.stem}.txt") for p in images]


def read_boxes(label: Path) -> np.ndarray:
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def size_bucket(area: float) -> str:
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def compute_stats(data_yaml: Path) -> dict:
    root, cfg = load_dataset(data_yaml)
    names = cfg["names"]
    out = {}
    for split in SPLITS:
        if split not in cfg:
            continue
        boxes_per_class, images_per_class, sizes = Counter(), Counter(), Counter()
        per_image, rel_areas, resolutions = [], [], Counter()
        files = split_files(root, cfg, split)
        for img, lbl in files:
            with Image.open(img) as im:
                w, h = exif_size(im)
            resolutions[f"{w}x{h}"] += 1
            scale = REF_SIZE / max(w, h)
            b = read_boxes(lbl)
            per_image.append(len(b))
            for c in set(b[:, 0].astype(int).tolist()):
                images_per_class[names[c]] += 1
            for c, _, _, bw, bh in b:
                boxes_per_class[names[int(c)]] += 1
                sizes[size_bucket(bw * w * scale * bh * h * scale)] += 1
                rel_areas.append(bw * bh)
        n_boxes = int(sum(per_image))
        out[split] = {
            "images": len(files),
            "boxes": n_boxes,
            "background_images": int(sum(1 for n in per_image if n == 0)),
            "boxes_per_image_mean": float(np.mean(per_image)) if per_image else 0.0,
            "boxes_per_image_max": int(max(per_image, default=0)),
            "boxes_per_class": {n: boxes_per_class[n] for n in names.values()},
            "images_per_class": {n: images_per_class[n] for n in names.values()},
            "size_buckets": {k: sizes[k] for k in ("small", "medium", "large")},
            "relative_box_area_median": float(np.median(rel_areas)) if rel_areas else None,
            "top_resolutions": dict(resolutions.most_common(5)),
        }
    return out


def to_markdown(stats: dict, names: list[str]) -> str:
    splits = list(stats)
    lines = ["| Class | " + " | ".join(f"{s} boxes (imgs)" for s in splits) + " |"]
    lines.append("|---|" + "---|" * len(splits))
    for n in names:
        cells = [f"{stats[s]['boxes_per_class'][n]} ({stats[s]['images_per_class'][n]})" for s in splits]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("| **Total** | " + " | ".join(f"{stats[s]['boxes']} ({stats[s]['images']})" for s in splits) + " |")
    lines += ["", "| Split | small | medium | large | boxes/img | background imgs |", "|---|---|---|---|---|---|"]
    for s in splits:
        sb = stats[s]["size_buckets"]
        total = max(stats[s]["boxes"], 1)
        pct = {k: f"{sb[k]} ({100 * sb[k] / total:.1f}%)" for k in sb}
        lines.append(
            f"| {s} | {pct['small']} | {pct['medium']} | {pct['large']} | "
            f"{stats[s]['boxes_per_image_mean']:.2f} | {stats[s]['background_images']} |"
        )
    return "\n".join(lines) + "\n"


def preview(data_yaml: Path, out: Path, split: str = "val", n: int = 16, seed: int = 0) -> None:
    """Draw ground-truth boxes on random images. If boxes look rotated/shifted, suspect EXIF issues."""
    root, cfg = load_dataset(data_yaml)
    files = split_files(root, cfg, split)
    picks = random.Random(seed).sample(files, min(n, len(files)))
    cols, tile = 4, 400
    sheet = Image.new("RGB", (cols * tile, ((len(picks) + cols - 1) // cols) * tile), "white")
    for k, (img, lbl) in enumerate(picks):
        with Image.open(img) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        for c, cx, cy, bw, bh in read_boxes(lbl):
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            draw.rectangle(box, outline="red", width=max(2, w // 300))
            draw.text((box[0], max(box[1] - 12, 0)), cfg["names"][int(c)], fill="yellow")
        im.thumbnail((tile, tile))
        sheet.paste(im, ((k % cols) * tile, (k // cols) * tile))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[stats] preview -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/tomato_village/dataset.yaml"))
    p.add_argument("--out", type=Path, default=Path("results/dataset"))
    p.add_argument("--preview", type=int, default=16, help="number of preview images (0 to skip)")
    args = p.parse_args()
    stats = compute_stats(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    names = list(load_dataset(args.data)[1]["names"].values())
    md = to_markdown(stats, names)
    (args.out / "dataset_table.md").write_text(md)
    print(md)
    if args.preview:
        preview(args.data, args.out / "label_preview.png", n=args.preview)


if __name__ == "__main__":
    main()
