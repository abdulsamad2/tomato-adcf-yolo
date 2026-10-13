"""Phase 6 annotation geometry: ground-truth box sizes on TRAIN and VAL only. Test NOT opened.

Boxes in Tomato-Village Variant-c enclose leaves/leaflets, not individual lesions, so this
characterises the object scale the detector is actually asked to localise.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
D = REPO / "data" / "tomato_village"
IMGSZ = 512
dy = yaml.safe_load((D / "dataset.yaml").read_text(encoding="utf-8"))
yn = dy.get("names") or {}
NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))

def load(split):
    lab = D / "labels" / split
    assert lab.is_dir(), lab
    rows = []
    for f in sorted(lab.iterdir()):
        if f.suffix != ".txt":
            continue
        stem = f.stem
        base = stem.rsplit("_aug", 1)[0] if "_aug" in stem else stem
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                rows.append((int(p[0]), float(p[3]), float(p[4]), base, stem))
    return rows

out = {"imgsz_reference": IMGSZ, "test_opened": False, "splits": {}}
for split in ("train", "val"):
    rows = load(split)
    uniq = sorted({r[3] for r in rows})
    for scope, sel in (("all_files", rows),
                       ("unique_source_photos", [r for r in rows if r[4] == r[3]])):
        if not sel:
            continue
        w = np.array([r[1] for r in sel]); h = np.array([r[2] for r in sel])
        cls = np.array([r[0] for r in sel])
        area_n = w * h
        wpx, hpx = w * IMGSZ, h * IMGSZ
        apx = wpx * hpx
        side = np.sqrt(apx)
        # COCO size strata at 512 px
        small = apx < 32 ** 2
        med = (apx >= 32 ** 2) & (apx < 96 ** 2)
        large = apx >= 96 ** 2
        q = lambda a: {k: round(float(np.percentile(a, p)), 4)
                       for k, p in (("p1", 1), ("p5", 5), ("p25", 25), ("p50", 50),
                                    ("p75", 75), ("p95", 95), ("p99", 99))}
        e = {"n_boxes": int(len(sel)), "n_files": int(len({r[4] for r in sel})),
             "n_source_photos": int(len({r[3] for r in sel})),
             "norm_w": {"mean": round(float(w.mean()), 4), "sd": round(float(w.std(ddof=1)), 4), **q(w)},
             "norm_h": {"mean": round(float(h.mean()), 4), "sd": round(float(h.std(ddof=1)), 4), **q(h)},
             "norm_area": {"mean": round(float(area_n.mean()), 5), "sd": round(float(area_n.std(ddof=1)), 5), **q(area_n)},
             "px_w_at_512": {"mean": round(float(wpx.mean()), 2), "sd": round(float(wpx.std(ddof=1)), 2), **q(wpx)},
             "px_h_at_512": {"mean": round(float(hpx.mean()), 2), "sd": round(float(hpx.std(ddof=1)), 2), **q(hpx)},
             "px_sqrt_area_at_512": {"mean": round(float(side.mean()), 2), "sd": round(float(side.std(ddof=1)), 2), **q(side)},
             "coco_strata_at_512": {
                 "small_pct": round(100 * float(small.mean()), 2),
                 "medium_pct": round(100 * float(med.mean()), 2),
                 "large_pct": round(100 * float(large.mean()), 2)},
             "fraction_of_image_area": {
                 "ge_10pct": round(100 * float((area_n >= 0.10).mean()), 2),
                 "ge_25pct": round(100 * float((area_n >= 0.25).mean()), 2),
                 "ge_50pct": round(100 * float((area_n >= 0.50).mean()), 2)},
             "by_stride_cell_at_512": {},
             "per_class": {}}
        # how many stride-s cells each object spans (descriptive; assignment is dynamic/TAL)
        for s in (8, 16, 32):
            cells = side / s
            e["by_stride_cell_at_512"][f"stride{s}"] = {
                "median_object_side_in_cells": round(float(np.median(cells)), 2),
                "pct_objects_under_1_cell": round(100 * float((cells < 1).mean()), 2),
                "pct_objects_under_4_cells": round(100 * float((cells < 4).mean()), 2)}
        for ci in sorted(set(cls.tolist())):
            m = cls == ci
            e["per_class"][NAMES.get(int(ci), f"class{ci}")] = {
                "n": int(m.sum()),
                "px_sqrt_area_median": round(float(np.median(side[m])), 2),
                "px_sqrt_area_mean": round(float(side[m].mean()), 2),
                "px_sqrt_area_sd": round(float(side[m].std(ddof=1)), 2) if m.sum() > 1 else None,
                "norm_area_median": round(float(np.median(area_n[m])), 5),
                "small_pct": round(100 * float(small[m].mean()), 2),
                "large_pct": round(100 * float(large[m].mean()), 2)}
        out["splits"].setdefault(split, {})[scope] = e

json.dump(out, open(REPO / "results" / "annotation_geometry.json", "w"), indent=1)
for sp in out["splits"]:
    for sc in out["splits"][sp]:
        e = out["splits"][sp][sc]
        k = e["coco_strata_at_512"]
        print(f"{sp}/{sc}: {e['n_boxes']} boxes, {e['n_source_photos']} photos | "
              f"median sqrt(area) {e['px_sqrt_area_at_512']['p50']:.1f} px | "
              f"small {k['small_pct']}% med {k['medium_pct']}% large {k['large_pct']}% | "
              f">=25% of image: {e['fraction_of_image_area']['ge_25pct']}%")
print("GEODONE")
