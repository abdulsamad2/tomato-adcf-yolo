"""Phase 9 step 2: build the evaluation dataset from the HASHED manifests and preflight the scorer.

Re-verifies both manifest hashes before building, then asserts the built dataset reproduces the
manifest exactly. Still no model loaded.
"""
import hashlib, json, os, re, shutil, sys
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RAW = REPO / "data" / "raw" / "Tomato-Village" / "Variant-c(Object Detection)" / "val"
CLEAN = REPO / "data" / "tomato_village"
OUT = Path(r"C:\Users\absamad\cs\phase9"); ARMS = OUT / "arms"
FR = json.loads((OUT / "arm_freeze_record.json").read_text())

def die(m): print("ABORT:", m); sys.exit(2)

# ---- re-verify hashes ----
man = {}
for a in ("A", "B"):
    p = ARMS / f"manifest_arm_{a}.tsv"
    txt = p.read_text(encoding="utf-8")
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    if h != FR["arms"][a]["manifest_sha256"]:
        die(f"manifest {a} hash drift: {h} != {FR['arms'][a]['manifest_sha256']}")
    recs = [l.split("\t") for l in txt.splitlines() if l and not l.startswith("#")]
    man[a] = [{"family": r[0], "label": r[1], "augmented": int(r[2]), "boxes": int(r[3])} for r in recs]
    print(f"manifest {a} hash VERIFIED, {len(man[a])} files")

# ---- build the union dataset (one copy; arms are scored as image-id subsets) ----
U = ARMS / "union"
for sub in ("images/val", "labels/val"):
    d = U / sub
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
idmap = {}
missing_img = []
nbox = 0
for a in ("A", "B"):
    for r in man[a]:
        stem = Path(r["label"]).stem
        src_i = RAW / "images" / (stem + ".jpg")
        src_l = RAW / "yolo" / r["label"]
        if not src_i.exists():
            missing_img.append(stem); continue
        dst_i = U / "images" / "val" / (stem + ".jpg")
        try: os.link(src_i, dst_i)
        except Exception: shutil.copy2(src_i, dst_i)
        # normalise class ids to int; geometry untouched
        lines = []
        for line in src_l.read_text().splitlines():
            q = line.split()
            if len(q) >= 5:
                lines.append(" ".join([str(int(float(q[0])))] + q[1:5]))
        (U / "labels" / "val" / (stem + ".txt")).write_text("\n".join(lines) + "\n")
        nbox += len(lines)
        idmap[stem] = {"arm": a, "family": r["family"], "augmented": r["augmented"], "boxes": len(lines)}
if missing_img: die(f"{len(missing_img)} manifest entries have no image: {missing_img[:5]}")

names = {int(k): v for k, v in FR["dataset_names"].items()}
yaml.safe_dump({"path": str(U), "train": "images/val", "val": "images/val", "names": names},
               open(U / "dataset.yaml", "w"), sort_keys=False, allow_unicode=True)

# ---- assert the build reproduces the manifest ----
for a in ("A", "B"):
    got_f = len([k for k, v in idmap.items() if v["arm"] == a])
    got_g = len({v["family"] for v in idmap.values() if v["arm"] == a})
    got_b = sum(v["boxes"] for v in idmap.values() if v["arm"] == a)
    exp = FR["arms"][a]
    if (got_f, got_g, got_b) != (exp["files"], exp["families"], exp["boxes"]):
        die(f"built arm {a} = {(got_f, got_g, got_b)} != manifest {(exp['files'], exp['families'], exp['boxes'])}")
    print(f"built arm {a} VERIFIED: {got_f} files / {got_g} families / {got_b} boxes")
if len(idmap) != FR["arms"]["A"]["files"] + FR["arms"]["B"]["files"]:
    die("union size mismatch")
json.dump(idmap, open(OUT / "image_index.json", "w"), indent=0)
print(f"union: {len(idmap)} images, {nbox} boxes, imgdir={U/'images'/'val'}")

# ---- COCO ground truth over the union (640x640 released images) ----
W = H = 640
imgs, anns = [], []
sid = {}
for i, (stem, v) in enumerate(sorted(idmap.items()), start=1):
    sid[stem] = i
    imgs.append({"id": i, "file_name": stem + ".jpg", "width": W, "height": H})
for stem, v in sorted(idmap.items()):
    for line in (U / "labels" / "val" / (stem + ".txt")).read_text().splitlines():
        q = line.split()
        if len(q) < 5: continue
        cid, xc, yc, w, h = int(q[0]), *map(float, q[1:5])
        bw, bh = w * W, h * H
        anns.append({"id": len(anns) + 1, "image_id": sid[stem], "category_id": cid,
                     "bbox": [(xc - w / 2) * W, (yc - h / 2) * H, bw, bh],
                     "area": bw * bh, "iscrowd": 0})
gt = {"images": imgs, "annotations": anns,
      "categories": [{"id": k, "name": v} for k, v in sorted(names.items())]}
json.dump(gt, open(OUT / "coco_gt_union.json", "w"))
json.dump(sid, open(OUT / "stem_to_cocoid.json", "w"))
print(f"coco gt: {len(imgs)} images, {len(anns)} anns, {len(gt['categories'])} categories")

# ---- scorer preflight ----
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    print("pycocotools: AVAILABLE")
except Exception as e:
    print("pycocotools: MISSING ->", type(e).__name__, e)
import numpy, ultralytics
print("numpy", numpy.__version__, "| ultralytics", ultralytics.__version__)
print("BUILDDONE")
