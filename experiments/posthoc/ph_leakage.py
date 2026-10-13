"""Post-hoc analysis 12 of posthoc/plan.md: characterise the source-family leakage of the released split.
Uses released Variant-c file names and YOLO labels only (no image pixels). The clean test split is not used.
"""
import json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

B = Path(r"C:\Users\absamad\cs"); R = B / "tomato-adcf-yolo"; OUT = B / "posthoc"
RAW = R / "data" / "raw" / "Tomato-Village" / "Variant-c(Object Detection)"
AUG = re.compile(r"_aug(\d+)$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
fam = lambda s: norm(AUG.sub("", s))
TOL = 0.01

def read(p):
    a = [list(map(float, l.split()[:5])) for l in p.read_text().splitlines() if len(l.split()) >= 5]
    return np.array(a, dtype=np.float64).reshape(-1, 5)

TR = {"identity": lambda b: b, "hflip": lambda b: np.c_[b[:, 0], 1 - b[:, 1], b[:, 2], b[:, 3], b[:, 4]],
      "vflip": lambda b: np.c_[b[:, 0], b[:, 1], 1 - b[:, 2], b[:, 3], b[:, 4]],
      "rot180 (both flips)": lambda b: np.c_[b[:, 0], 1 - b[:, 1], 1 - b[:, 2], b[:, 3], b[:, 4]],
      "rot90 cw": lambda b: np.c_[b[:, 0], 1 - b[:, 2], b[:, 1], b[:, 4], b[:, 3]],
      "rot90 ccw": lambda b: np.c_[b[:, 0], b[:, 2], 1 - b[:, 1], b[:, 4], b[:, 3]]}

def same(a, b):
    if len(a) != len(b): return False
    if len(a) == 0: return True
    used = np.zeros(len(b), bool)
    for r in a:
        d = np.abs(b[:, 1:] - r[1:]).max(1); d[used | (b[:, 0] != r[0])] = 9
        j = int(np.argmin(d))
        if d[j] > TOL: return False
        used[j] = True
    return True

files = {sp: {p.stem: p for p in (RAW / sp / "yolo").iterdir() if p.suffix == ".txt"} for sp in ("train", "val")}
allf = {**{s: ("train", p) for s, p in files["train"].items()}, **{s: ("val", p) for s, p in files["val"].items()}}
byfam = defaultdict(list)
for s, (sp, p) in allf.items(): byfam[fam(s)].append((s, sp))
sizes = Counter(len(v) for v in byfam.values())
fam_tr = {fam(s) for s in files["train"]}; fam_va = {fam(s) for s in files["val"]}
val_with_sib = sum(1 for s in files["val"] if fam(s) in fam_tr)

ttype, ttab = {}, Counter()
for f, members in byfam.items():
    orig = [s for s, _ in members if not AUG.search(s)]
    base = read(allf[orig[0]][1]) if orig else None
    for s, sp in members:
        if not AUG.search(s):
            ttype[s] = "original"; continue
        if base is None:
            ttype[s] = "no original file"; ttab["no original file"] += 1; continue
        b = read(allf[s][1]); t = "other (crop/scale/shift or box count differs)"
        for name, fn in TR.items():
            if same(fn(base), b): t = name; break
        ttype[s] = t; ttab[t] += 1
n_aug = sum(ttab.values())

rep = {"source": str(RAW), "n_files": len(allf), "n_train_files": len(files["train"]), "n_val_files": len(files["val"]),
       "n_families": len(byfam), "family_size_distribution": dict(sorted(sizes.items())),
       "families_spanning_train_and_val": len(fam_tr & fam_va),
       "val_files_with_sibling_in_train": val_with_sib, "val_files_with_sibling_in_train_pct": 100 * val_with_sib / len(files["val"]),
       "val_families_with_sibling_in_train_pct": 100 * len(fam_va & fam_tr) / len(fam_va),
       "augmented_files": n_aug, "inferred_transform_counts": dict(ttab.most_common()),
       "inferred_transform_pct": {k: 100 * v / n_aug for k, v in ttab.most_common()},
       "note": "Transform is inferred from labels only; photometric changes (brightness, colour, blur) are invisible here."}

# exploratory: arm-A vs arm-B loc recall@0.50 by inferred transform of the evaluated file (w100, 5 seeds)
P9 = B / "phase9"
g = json.loads((P9 / "coco_gt_union.json").read_text()); arms = json.loads((P9 / "arm_image_ids.json").read_text())
stem_of = {im["id"]: Path(im["file_name"]).stem for im in g["images"]}
GT = defaultdict(list)
for a in g["annotations"]:
    x, y, w, h = a["bbox"]; GT[a["image_id"]].append((x, y, x + w, y + h))
def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / np.maximum(aa[:, None] + bb[None, :] - i, 1e-9)
by_t = {a: defaultdict(lambda: np.zeros(2)) for a in ("A", "B")}
for s in range(5):
    z = np.load(B / "phase15" / "dets" / ("w100__seed%d__step24570.npz" % s), allow_pickle=True)
    keep = z["conf"] >= 0.25; iid, xyxy, cf = z["image_id"][keep], z["xyxy"][keep], z["conf"][keep]
    byimg = defaultdict(list)
    for k in np.argsort(-cf): byimg[int(iid[k])].append(k)
    for arm in ("A", "B"):
        for i in arms["arm_image_ids"][arm]:
            G = np.array(GT.get(i, []), dtype=np.float32).reshape(-1, 4)
            if len(G) == 0: continue
            t = ttype.get(stem_of[i], ttype.get(norm(stem_of[i]), "unknown"))
            ks = byimg.get(i, []); hit = np.zeros(len(G), bool)
            if ks:
                M = iou(xyxy[ks].astype(np.float32), G)
                for p in range(len(ks)):
                    c = np.where((M[p] >= 0.5) & ~hit)[0]
                    if len(c): hit[c[np.argmax(M[p, c])]] = True
            by_t[arm][t] += (len(G), hit.sum())
rep["exploratory_recall50_by_transform_w100_5seeds"] = {a: {t: {"recall50": 100 * v[1] / v[0], "gt_boxes_x5seeds": int(v[0])}
                                                          for t, v in sorted(d.items())} for a, d in by_t.items()}
json.dump(rep, open(OUT / "leakage.json", "w"), indent=1)
print(json.dumps(rep, indent=1)); print("LEAKDONE")
