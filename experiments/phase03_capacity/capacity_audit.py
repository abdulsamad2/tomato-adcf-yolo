"""Capacity audit: inventory every completed run whose base model is yolo11n/s/m,
reading the per-run args.yaml as the authoritative protocol record."""
import csv, json, re
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
TARGET = re.compile(r"yolo11[nsm]\b", re.I)

# train-image counts per dataset dir, to convert epochs -> optimizer updates
counts = {}
for d in (REPO / "data").iterdir():
    if d.is_dir():
        ti = d / "images" / "train"
        if ti.is_dir():
            counts[d.name] = len(list(ti.iterdir()))

PROTO = ["data", "imgsz", "epochs", "patience", "optimizer", "lr0", "lrf", "cos_lr",
         "warmup_epochs", "warmup_bias_lr", "warmup_momentum", "momentum", "weight_decay",
         "batch", "nbs", "close_mosaic", "mosaic", "mixup", "cutmix", "copy_paste",
         "scale", "translate", "fliplr", "flipud", "degrees", "shear", "perspective",
         "hsv_h", "hsv_s", "hsv_v", "erasing", "auto_augment", "multi_scale", "rect",
         "box", "cls", "dfl", "cls_pw", "seed", "deterministic", "pretrained", "model", "fraction"]

rows = []
for root in ("runs", "runs_cv", "runs_timing", "runs_bcd", "runs_p2b"):
    rd = REPO / root
    if not rd.is_dir():
        continue
    for a in sorted(x for x in rd.iterdir() if x.is_dir()):
        for s in sorted(x for x in a.iterdir() if x.is_dir()):
            ay = s / "args.yaml"
            if not ay.exists():
                continue
            cfg = yaml.safe_load(ay.read_text(encoding="utf-8")) or {}
            base = str(cfg.get("model", ""))
            if not TARGET.search(base) and not TARGET.search(a.name):
                continue
            dpath = str(cfg.get("data", ""))
            dname = Path(dpath).parent.name if dpath else ""
            ntrain = counts.get(dname)
            spe = (ntrain // int(cfg.get("batch", 16))) if ntrain else None
            rcsv = s / "results.csv"
            epochs_run = (sum(1 for _ in rcsv.open(encoding="utf-8")) - 1) if rcsv.exists() else None
            r = {"root": root, "run": a.name, "seed_dir": s.name,
                 "done": (s / "train_done.json").exists() or (s / "p2b_done.json").exists()
                         or (s / "bcd_done.json").exists(),
                 "dataset_dir": dname, "train_images": ntrain, "steps_per_epoch": spe,
                 "epochs_run": epochs_run,
                 "updates_run": (spe * epochs_run) if (spe and epochs_run) else None}
            for k in PROTO:
                r[k] = cfg.get(k)
            for tag, fn in (("val", "results_val.json"), ("test", "results_test.json")):
                p = s / fn
                if p.exists():
                    j = json.loads(p.read_text())
                    r[f"{tag}_mAP50_95"] = j.get("mAP50_95")
                    r[f"{tag}_mAP50"] = j.get("mAP50")
                    for kk in ("params_M", "GFLOPs"):
                        if j.get(kk) is not None:
                            r[kk] = j[kk]
                    co = j.get("coco") or {}
                    r[f"{tag}_AP_small"] = co.get("AP_small")
            for fn in ("train_done.json", "p2b_done.json", "bcd_done.json"):
                p = s / fn
                if p.exists():
                    j = json.loads(p.read_text())
                    r["train_hours"] = j.get("train_hours") or j.get("hours")
                    break
            rows.append(r)

keys = sorted({k for r in rows for k in r})
out = REPO / "results" / "capacity_audit.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(json.dumps({"rows": len(rows), "dataset_train_counts": counts,
                  "runs_seen": sorted({r["run"] for r in rows})}, indent=1))
