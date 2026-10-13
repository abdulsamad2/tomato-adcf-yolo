"""Phase 8 close: COMPLETE image-dimension audit. Train and validation only; TEST NEVER TOUCHED.

Two distinct populations, deliberately separated because conflating them would misattribute the
resolution ceiling:
  (A) the RELEASED files as distributed by the dataset authors   -> data/raw/.../Variant-c/<split>
  (B) our PREPARED leakage-free split (what training consumed)   -> data/tomato_village/images/<split>
Dimensions are read from the file header only (PIL lazy open); no pixels are decoded.
"""
import json, os
from collections import Counter
from pathlib import Path
from PIL import Image

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = Path(r"C:\Users\absamad\cs\phase8")
RAW = REPO / "data" / "raw" / "Tomato-Village" / "Variant-c(Object Detection)"
PREP = REPO / "data" / "tomato_village" / "images"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FORBIDDEN = "test"

def scan(root, label):
    if not root.exists():
        return {"root": str(root), "exists": False}
    dims, mode, fmt, files = Counter(), Counter(), Counter(), 0
    aug, uniq = Counter(), Counter()
    errs = []
    maxpix = (0, None)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        if FORBIDDEN in [q.lower() for q in p.parts]:
            raise SystemExit(f"REFUSING: test path reached: {p}")
        files += 1
        try:
            with Image.open(p) as im:
                w, h = im.size
                dims[(w, h)] += 1
                mode[im.mode] += 1
                fmt[(im.format or "?")] += 1
            if "_aug" in p.stem:
                aug[(w, h)] += 1
            else:
                uniq[(w, h)] += 1
            if w * h > maxpix[0]:
                maxpix = (w * h, (w, h, str(p.relative_to(root))))
        except Exception as e:
            errs.append(f"{p.name}: {type(e).__name__}")
    return {"root": str(root), "exists": True, "label": label, "files": files,
            "distinct_dims": len(dims),
            "dims": {f"{w}x{h}": n for (w, h), n in sorted(dims.items(), key=lambda x: -x[1])},
            "dims_source_photos_only": {f"{w}x{h}": n for (w, h), n in sorted(uniq.items(), key=lambda x: -x[1])},
            "dims_augmented_only": {f"{w}x{h}": n for (w, h), n in sorted(aug.items(), key=lambda x: -x[1])},
            "modes": dict(mode), "formats": {k: v for k, v in fmt.items()},
            "max_width": max((w for w, h in dims), default=0),
            "max_height": max((h for w, h in dims), default=0),
            "largest_file": maxpix[1], "unreadable": errs}

out = {"test_opened": False, "populations": {}}
for label, root in (("released_raw_train", RAW / "train"), ("released_raw_val", RAW / "val"),
                    ("prepared_train", PREP / "train"), ("prepared_val", PREP / "val")):
    out["populations"][label] = scan(root, label)

# roll-ups
def roll(keys):
    d = Counter(); n = 0
    for k in keys:
        p = out["populations"][k]
        if not p.get("exists"):
            continue
        n += p["files"]
        for s, v in p["dims"].items():
            d[s] += v
    return n, dict(sorted(d.items(), key=lambda x: -x[1]))

for name, keys in (("released_train_plus_val", ["released_raw_train", "released_raw_val"]),
                   ("prepared_train_plus_val", ["prepared_train", "prepared_val"])):
    n, d = roll(keys)
    mx = max([tuple(int(z) for z in s.split("x")) for s in d], default=(0, 0))
    out[name] = {"files": n, "distinct_dims": len(d), "dims": d,
                 "max_dim_any_axis": max([max(int(z) for z in s.split("x")) for s in d], default=0),
                 "all_640": set(d) == {"640x640"}}

# what the raw tree actually contains, for provenance
out["raw_tree"] = sorted(str(p.relative_to(RAW)) for p in RAW.iterdir()) if RAW.exists() else None
json.dump(out, open(OUT / "image_resolution_audit.json", "w"), indent=1)
for k in ("released_train_plus_val", "prepared_train_plus_val"):
    e = out[k]
    print(f"{k}: files={e['files']}  distinct_dims={e['distinct_dims']}  "
          f"max_any_axis={e['max_dim_any_axis']}  all_640x640={e['all_640']}")
    for s, v in list(e["dims"].items())[:8]:
        print(f"    {s}: {v}")
for k, p in out["populations"].items():
    if p.get("exists"):
        print(f"  [{k}] files={p['files']} distinct={p['distinct_dims']} "
              f"max={p['max_width']}x{p['max_height']} formats={p['formats']} "
              f"unreadable={len(p['unreadable'])}")
    else:
        print(f"  [{k}] MISSING: {p['root']}")
print("raw_tree:", out["raw_tree"])
print("AUDITDONE")
