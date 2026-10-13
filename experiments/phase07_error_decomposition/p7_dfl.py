"""Phase 7 part 3: what box-regression range (reg_max) does the data actually require?

DFL predicts each of the four box-side distances from an anchor point, in units of grid cells,
over the integer range 0..reg_max-1. For a box of width w, height h at stride r, the distance
from an interior anchor to a side is at most w/r (or h/r); from the CENTRE it is w/(2r).
Both are reported: centre-anchor is the typical case, full-extent is the worst case for an
anchor near the opposite edge. TRAIN+VAL labels only; test not opened.
"""
import json
from pathlib import Path
import numpy as np, yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
D = REPO / "data" / "tomato_village"
OUT = Path(r"C:\Users\absamad\cs\phase7")
IMGSZ = 512
STRIDES = [8, 16, 32]
REGMAX = [4, 6, 8, 10, 12, 16]

rows = []
for split in ("train", "val"):
    for f in sorted((D / "labels" / split).iterdir()):
        if f.suffix != ".txt":
            continue
        uniq = "_aug" not in f.stem
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                rows.append((split, uniq, float(p[3]) * IMGSZ, float(p[4]) * IMGSZ))

out = {"imgsz": IMGSZ, "test_opened": False, "scopes": {}}
for scope, sel in (("train_unique_photos", [r for r in rows if r[0] == "train" and r[1]]),
                   ("train_all_files", [r for r in rows if r[0] == "train"]),
                   ("val", [r for r in rows if r[0] == "val"])):
    w = np.array([r[2] for r in sel]); h = np.array([r[3] for r in sel])
    big = np.maximum(w, h)
    e = {"n_boxes": int(len(sel)), "strides": {}}
    for r_ in STRIDES:
        centre = big / (2 * r_)      # centre anchor -> half-side in cells
        extent = big / r_            # worst-case anchor -> full side in cells
        st = {"cells_centre": {k: round(float(np.percentile(centre, q)), 3)
                               for k, q in (("p50", 50), ("p95", 95), ("p99", 99),
                                            ("p99.9", 99.9), ("max", 100))},
              "cells_extent": {k: round(float(np.percentile(extent, q)), 3)
                               for k, q in (("p50", 50), ("p95", 95), ("p99", 99),
                                            ("p99.9", 99.9), ("max", 100))},
              "coverage": {}}
        for rm in REGMAX:
            st["coverage"][str(rm)] = {
                "pct_boxes_covered_centre": round(100 * float((centre <= rm - 1).mean()), 3),
                "pct_boxes_covered_extent": round(100 * float((extent <= rm - 1).mean()), 3)}
        e["strides"][str(r_)] = st
    out["scopes"][scope] = e
    print(f"--- {scope}: {len(sel):,} boxes ---")
    for r_ in STRIDES:
        s = e["strides"][str(r_)]
        print(f"  stride {r_:>2}: half-side cells p50 {s['cells_centre']['p50']:.2f} "
              f"p99 {s['cells_centre']['p99']:.2f} max {s['cells_centre']['max']:.2f} | "
              f"full-side cells p99 {s['cells_extent']['p99']:.2f} max {s['cells_extent']['max']:.2f}")
        cov = "  ".join(f"rm{rm}: {s['coverage'][str(rm)]['pct_boxes_covered_extent']:.2f}%"
                        for rm in (4, 6, 8, 12, 16))
        print(f"            full-extent coverage -> {cov}")
json.dump(out, open(OUT / "dfl_range.json", "w"), indent=1)
print("DFLDONE")
