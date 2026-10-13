"""Phase 8: seeded stratified annotation sample + crop rendering. Train and val only.

Sampling procedure (fixed before inspection):
  * population = every box in a NON-AUGMENTED label file of the train or val split
  * strata = populated class (6) x split (2) x size tercile (3) = 36 cells
  * size terciles from the global train+val original-photo sqrt(area)@512 distribution
  * up to TARGET_PER_CELL boxes per cell, drawn with random.Random(SEED).shuffle
  * HARD CONSTRAINT: each source-photo family contributes AT MOST ONE box in the whole sample,
    so augmented copies can never appear and no photo is counted twice
Crops are rendered at 2x the box extent with the box outlined, so it is visible whether the
enclosed structure continues past the box edge.
"""
import csv, json, random
from pathlib import Path
import numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont

SEED = 20261007
TARGET_PER_CELL = 11
CONTEXT = 2.0
CELL = 256
COLS, ROWS = 4, 5

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
D = REPO / "data" / "tomato_village"
OUT = Path(r"C:\Users\absamad\cs\phase8"); OUT.mkdir(exist_ok=True)
(OUT / "crops").mkdir(exist_ok=True)
(OUT / "sheets").mkdir(exist_ok=True)

yn = yaml.safe_load((D / "dataset.yaml").read_text(encoding="utf-8")).get("names") or {}
NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))

# population
pop = []
for split in ("train", "val"):
    for f in sorted((D / "labels" / split).iterdir()):
        if f.suffix != ".txt" or "_aug" in f.stem:
            continue
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            p = D / "images" / split / (f.stem + ext)
            if p.exists():
                img = p; break
        if img is None:
            continue
        for li, line in enumerate(f.read_text().splitlines()):
            q = line.split()
            if len(q) < 5:
                continue
            cid, cx, cy, w, h = int(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])
            pop.append({"split": split, "stem": f.stem, "img": str(img), "line": li,
                        "class_id": cid, "class_name": NAMES.get(cid, f"class{cid}"),
                        "cx": cx, "cy": cy, "w": w, "h": h,
                        "norm_area": w * h, "px_w_512": w * 512, "px_h_512": h * 512,
                        "sqrt_area_512": float(np.sqrt(w * 512 * h * 512))})

side = np.array([r["sqrt_area_512"] for r in pop])
t1, t2 = float(np.percentile(side, 100 / 3)), float(np.percentile(side, 200 / 3))
for r in pop:
    r["size_tercile"] = "small" if r["sqrt_area_512"] < t1 else ("mid" if r["sqrt_area_512"] < t2 else "large")

classes = sorted({r["class_id"] for r in pop})
cells = {}
for r in pop:
    cells.setdefault((r["class_id"], r["split"], r["size_tercile"]), []).append(r)

rng = random.Random(SEED)
used_photos, sample = set(), []
for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
    bucket = cells[key][:]
    rng.shuffle(bucket)
    taken = 0
    for r in bucket:
        if taken >= TARGET_PER_CELL:
            break
        if r["stem"] in used_photos:
            continue
        used_photos.add(r["stem"]); sample.append(r); taken += 1
rng.shuffle(sample)
for i, r in enumerate(sample):
    r["sample_idx"] = i

# render crops
try:
    font = ImageFont.truetype("arial.ttf", 18)
    fsm = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = fsm = ImageFont.load_default()

for r in sample:
    im = Image.open(r["img"]).convert("RGB")
    W, H = im.size
    r["src_w"], r["src_h"] = W, H
    bw, bh = r["w"] * W, r["h"] * H
    cx, cy = r["cx"] * W, r["cy"] * H
    side_px = max(bw, bh) * CONTEXT
    x0, y0 = cx - side_px / 2, cy - side_px / 2
    x1, y1 = cx + side_px / 2, cy + side_px / 2
    x0c, y0c = max(0, int(x0)), max(0, int(y0))
    x1c, y1c = min(W, int(x1)), min(H, int(y1))
    crop = im.crop((x0c, y0c, x1c, y1c))
    sc = CELL / max(crop.size)
    crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.LANCZOS)
    canvas = Image.new("RGB", (CELL, CELL), (24, 24, 24))
    canvas.paste(crop, ((CELL - crop.width) // 2, (CELL - crop.height) // 2))
    dr = ImageDraw.Draw(canvas)
    bx0 = (cx - bw / 2 - x0c) * sc + (CELL - crop.width) // 2
    by0 = (cy - bh / 2 - y0c) * sc + (CELL - crop.height) // 2
    dr.rectangle([bx0, by0, bx0 + bw * sc, by0 + bh * sc], outline=(255, 215, 0), width=2)
    r["crop"] = str(OUT / "crops" / f"c{r['sample_idx']:03d}.jpg")
    r["box_px_in_source"] = [round(bw, 1), round(bh, 1)]
    canvas.save(r["crop"], quality=90)

# contact sheets
per = COLS * ROWS
LBL = 26
sheets = []
for s in range(0, len(sample), per):
    grp = sample[s:s + per]
    sw, sh = COLS * (CELL + 8) + 8, ROWS * (CELL + LBL + 8) + 34
    sheet = Image.new("RGB", (sw, sh), (248, 248, 248))
    dr = ImageDraw.Draw(sheet)
    n = s // per + 1
    dr.text((10, 8), f"Phase 8 annotation sample - sheet {n} of "
                     f"{(len(sample) + per - 1) // per}   (box outlined in gold, crop = 2x box extent)",
            fill=(20, 20, 20), font=font)
    for j, r in enumerate(grp):
        cx_ = 8 + (j % COLS) * (CELL + 8)
        cy_ = 34 + (j // COLS) * (CELL + LBL + 8)
        sheet.paste(Image.open(r["crop"]), (cx_, cy_))
        dr.text((cx_ + 2, cy_ + CELL + 4),
                f"#{r['sample_idx']:03d} {r['class_name'][:16]}  {r['px_w_512']:.0f}x{r['px_h_512']:.0f}px",
                fill=(20, 20, 20), font=fsm)
    p = OUT / "sheets" / f"sheet{n:02d}.jpg"
    sheet.save(p, quality=88)
    sheets.append(str(p))

keys = ["sample_idx", "split", "stem", "class_id", "class_name", "size_tercile", "norm_area",
        "px_w_512", "px_h_512", "sqrt_area_512", "src_w", "src_h", "box_px_in_source", "line", "img", "crop"]
with open(OUT / "sample_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(sample)
json.dump({"seed": SEED, "target_per_cell": TARGET_PER_CELL, "context_factor": CONTEXT,
           "size_tercile_thresholds_sqrt_area_512": [round(t1, 2), round(t2, 2)],
           "population_boxes": len(pop),
           "population_source_photos": len({r["stem"] for r in pop}),
           "sampled": len(sample), "distinct_source_photos": len(used_photos),
           "one_box_per_photo": len(used_photos) == len(sample),
           "augmented_files_in_sample": sum(1 for r in sample if "_aug" in r["stem"]),
           "cells_total": len(cells), "cells_filled": len({(r["class_id"], r["split"], r["size_tercile"]) for r in sample}),
           "per_cell_counts": {f"{NAMES.get(k[0],k[0])}|{k[1]}|{k[2]}": sum(
               1 for r in sample if (r["class_id"], r["split"], r["size_tercile"]) == k) for k in sorted(cells)},
           "sheets": sheets}, open(OUT / "sampling_record.json", "w"), indent=1)
print(json.dumps({"population_boxes": len(pop), "population_photos": len({r["stem"] for r in pop}),
                  "terciles": [round(t1, 2), round(t2, 2)], "sampled": len(sample),
                  "distinct_photos": len(used_photos), "one_per_photo": len(used_photos) == len(sample),
                  "aug_in_sample": sum(1 for r in sample if "_aug" in r["stem"]),
                  "n_sheets": len(sheets)}, indent=1))
print("SAMPDONE")
