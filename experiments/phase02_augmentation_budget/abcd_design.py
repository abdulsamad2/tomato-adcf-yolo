"""Phase 0/2 preparation: dataset counts, EMPIRICAL steps/epoch, scheduler implications,
source-family integrity checks, and GPU-hour estimates for the A/B/C/D augmentation control.

Steps/epoch is measured by building the actual Ultralytics dataloader, not assumed from
ceil(N/batch), because drop_last and rect batching can change it.
"""
import csv, json, re, yaml, hashlib
from pathlib import Path
from collections import defaultdict, Counter

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
AUG = re.compile(r"_aug(\d+)$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
src_of = lambda p: norm(AUG.sub("", Path(p).stem))

CFG = yaml.safe_load((REPO/"configs"/"experiments.yaml").read_text(encoding="utf-8"))
TR = CFG["train"]
BATCH = TR["batch"]
out = {"train_block": {k: TR[k] for k in
       ("epochs","patience","imgsz","batch","optimizer","lr0","lrf","momentum",
        "weight_decay","warmup_epochs","cos_lr","close_mosaic") if k in TR}}

ARMS = {
 "A_1x_normal":      REPO/"data"/"tomato_village_noaug",
 "B_1x_matched":     REPO/"data"/"tomato_village_noaug",   # same data, more epochs
 "C_8x_duplicate":   REPO/"data"/"tomato_village_dup8",
 "D_8x_augmented":   REPO/"data"/"tomato_village",
}

# ---- dataset counts + source-family integrity ----
counts = {}
for arm, d in ARMS.items():
    if arm == "B_1x_matched":
        continue
    rec = {}
    fams = {}
    for split in ("train", "val", "test"):
        imgs = [f for f in (d/"images"/split).iterdir() if f.suffix.lower() in (".jpg",".jpeg",".png")]
        labs = list((d/"labels"/split).iterdir())
        s = {src_of(f) for f in imgs}
        fams[split] = s
        rec[split] = {"images": len(imgs), "labels": len(labs), "source_photos": len(s),
                      "aug_copies_present": sum(1 for f in imgs if AUG.search(f.stem))}
    rec["overlap_train_val"] = len(fams["train"] & fams["val"])
    rec["overlap_train_test"] = len(fams["train"] & fams["test"])
    rec["overlap_val_test"] = len(fams["val"] & fams["test"])
    counts[arm] = rec
counts["B_1x_matched"] = dict(counts["A_1x_normal"], note="identical data to A; only the epoch budget differs")

# ---- byte-identity proof for arm C (full check, not a sample) ----
dup = ARMS["C_8x_duplicate"]/"images"/"train"
groups = defaultdict(list)
for f in dup.iterdir():
    groups[src_of(f)].append(f)
mismatch = 0
sizes = Counter()
for s, fs in groups.items():
    sizes[len(fs)] += 1
    h = {hashlib.md5(f.read_bytes()).hexdigest() for f in fs}
    if len(h) != 1:
        mismatch += 1
out["arm_C_identity"] = {"source_photos": len(groups), "files_per_photo": dict(sizes),
                         "photos_with_differing_bytes": mismatch}

# ---- EMPIRICAL steps per epoch, via the real dataloader ----
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
steps = {}
for arm, d in ARMS.items():
    if arm == "B_1x_matched":
        continue
    args = get_cfg(DEFAULT_CFG)
    args.imgsz = TR["imgsz"]; args.batch = BATCH; args.cache = False; args.rect = False
    data = yaml.safe_load((d/"dataset.yaml").read_text(encoding="utf-8"))
    data["path"] = str(d); data["channels"] = data.get("channels", 3)
    ds = build_yolo_dataset(args, str(d/"images"/"train"), BATCH, data, mode="train", rect=False)
    dl = build_dataloader(ds, BATCH, workers=0, shuffle=True, rank=-1)
    steps[arm] = {"n_images": len(ds), "steps_per_epoch": len(dl)}
steps["B_1x_matched"] = dict(steps["A_1x_normal"])
out["steps"] = steps

# ---- epochs actually run, from finished runs ----
def epochs_of(run):
    rows = []
    p = REPO/"runs"/run
    if not p.exists():
        return rows
    for sd in sorted(p.iterdir()):
        c = sd/"results.csv"
        if c.exists():
            n = sum(1 for _ in c.open(encoding="utf-8")) - 1
            td = sd/"train_done.json"
            h = json.loads(td.read_text())["train_hours"] if td.exists() else None
            rows.append({"seed": sd.name, "epochs": n, "hours": h})
    return rows
out["observed"] = {"D_8x_augmented(yolo11s)": epochs_of("yolo11s"),
                   "A_1x_normal(yolo11s-noaug)": epochs_of("yolo11s-noaug")}

json.dump(out, open(r"C:\Users\absamad\cs\audit\abcd_design.json", "w"), indent=1)
json.dump(counts, open(r"C:\Users\absamad\cs\audit\abcd_counts.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:2600])
