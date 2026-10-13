"""Task 1: transformation provenance, from AUTHORITATIVE shipped artifacts only.

Uses the dataset's own label files (and image dimensions), never pixel inference, to decide whether
each released _augN variant differs from its family's base image GEOMETRICALLY or only
PHOTOMETRICALLY, and whether box correspondence is exact.

A geometric transform rewrites the boxes; a photometric one leaves them identical. Candidate
geometric maps are tested in CLOSED FORM against the base boxes:
  identity | hflip | vflip | rot180 | transpose | rot90cw | rot90ccw | flip-transpose
"""
import csv, json, os, re
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image

RAW = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\data\raw\Tomato-Village\Variant-c(Object Detection)")
AUG = re.compile(r"_aug(\d+)$")
TOL = 2e-3

def fam(stem): return AUG.sub("", stem)

def load(p):
    out = []
    for line in p.read_text().splitlines():
        t = line.split()
        if len(t) >= 5:
            out.append((int(float(t[0])), float(t[1]), float(t[2]), float(t[3]), float(t[4])))
    return out

MAPS = {
 "identity":       lambda c, x, y, w, h: (c, x, y, w, h),
 "hflip":          lambda c, x, y, w, h: (c, 1 - x, y, w, h),
 "vflip":          lambda c, x, y, w, h: (c, x, 1 - y, w, h),
 "rot180":         lambda c, x, y, w, h: (c, 1 - x, 1 - y, w, h),
 "transpose":      lambda c, x, y, w, h: (c, y, x, h, w),
 "rot90cw":        lambda c, x, y, w, h: (c, 1 - y, x, h, w),
 "rot90ccw":       lambda c, x, y, w, h: (c, y, 1 - x, h, w),
 "antitranspose":  lambda c, x, y, w, h: (c, 1 - y, 1 - x, h, w),
}

# index every released file by family
idx = defaultdict(dict)
dims = {}
for split in ("train", "val"):
    ld = RAW / split / "yolo"
    if not ld.is_dir(): continue
    for f in sorted(ld.iterdir()):
        if f.suffix != ".txt": continue
        m = AUG.search(f.stem)
        idx[fam(f.stem)][int(m.group(1)) if m else 0] = (split, f)

os.makedirs(r"C:\Users\absamad\cs\phase14", exist_ok=True)
fams = sorted(idx)
slotcount = Counter(len(v) for v in idx.values())
slots_seen = Counter(k for v in idx.values() for k in v)
rows, verdict = [], Counter()
amb = []
for fm in fams:
    v = idx[fm]
    if 0 not in v:
        verdict["no_base_variant"] += 1; continue
    base = load(v[0][1])
    bset = sorted(base)
    for slot in sorted(v):
        if slot == 0: continue
        cur = sorted(load(v[slot][1]))
        hit = []
        if len(cur) == len(bset):
            for nm, fn in MAPS.items():
                t = sorted(fn(*b) for b in bset)
                if all(a[0] == b2[0] and max(abs(a[i] - b2[i]) for i in range(1, 5)) < TOL
                       for a, b2 in zip(t, cur)):
                    hit.append(nm)
        rows.append({"family": fm, "slot": slot, "split_base": v[0][0], "split_variant": v[slot][0],
                     "n_boxes_base": len(bset), "n_boxes_variant": len(cur),
                     "box_count_equal": len(bset) == len(cur),
                     "exact_maps": ";".join(hit) if hit else ""})
        if not hit:
            verdict["NO_EXACT_MAP"] += 1
            if len(amb) < 12:
                amb.append({"family": fm, "slot": slot, "nb": len(bset), "nv": len(cur),
                            "base_first": bset[0] if bset else None, "var_first": cur[0] if cur else None})
        elif hit == ["identity"]:
            verdict["photometric_only"] += 1
        else:
            verdict["geometric:" + "|".join(hit)] += 1

with open(Path(r"C:\Users\absamad\cs\phase14") / "transform_provenance.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
out = {"families": len(fams), "variant_pairs_tested": len(rows),
       "variants_per_family_distribution": dict(sorted(slotcount.items())),
       "slot_occupancy": dict(sorted(slots_seen.items())),
       "verdict_counts": dict(verdict.most_common()),
       "unmapped_examples": amb}
json.dump(out, open(Path(r"C:\Users\absamad\cs\phase14") / "transform_provenance.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
print("AUDITDONE")
