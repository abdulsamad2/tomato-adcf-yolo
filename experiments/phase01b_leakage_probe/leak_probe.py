import csv, json, re, shutil
from pathlib import Path
REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RAW = REPO/"data"/"raw"/"Tomato-Village"/"Variant-c(Object Detection)"
OUT = REPO/"data"/"leakage_probe"
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
gid = lambda p: norm(AUG.sub("", Path(p).stem))
NAMES = {0: "Early Blight", 1: "Healthy", 2: "Late Blight", 3: "Leaf Miner",
         4: "Magnesium Deficiency", 5: "Nitrogen Deficiency", 6: "Potassium Deficiency",
         7: "Spotted Wilt Virus"}

man = list(csv.DictReader((REPO/"data"/"tomato_village"/"split_manifest.csv").open(encoding="utf-8")))
clean_train = {norm(r["original_id"]) for r in man if r["split"] == "train"}

vimg, vlab = RAW/"val"/"images", RAW/"val"/"yolo"
files = [f for f in vimg.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
buckets = {"seen": [], "unseen": []}
for f in files:
    buckets["seen" if gid(f.name) in clean_train else "unseen"].append(f)

summary = {}
for name, fs in buckets.items():
    root = OUT/name
    for sub in ("images/val", "labels/val"):
        (root/sub).mkdir(parents=True, exist_ok=True)
    n_lab = 0
    for f in fs:
        tgt = root/"images"/"val"/f.name
        if not tgt.exists():
            shutil.copy2(f, tgt)
        lab = vlab/(f.stem + ".txt")
        if lab.exists():
            lt = root/"labels"/"val"/lab.name
            if not lt.exists():
                shutil.copy2(lab, lt)
            n_lab += 1
    (root/"dataset.yaml").write_text(
        f"path: {root}\ntrain: images/val\nval: images/val\ntest: images/val\nnames:\n"
        + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()))
    summary[name] = {"files": len(fs), "photos": len({gid(f.name) for f in fs}), "labels": n_lab}
print(json.dumps(summary, indent=1))
(REPO/"results"/"leakage_probe_summary.json").write_text(json.dumps(summary, indent=1))
