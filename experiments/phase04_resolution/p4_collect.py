"""Assemble Phase 4: arms 320/416/512 from runs_p4, arm 640 reused from runs_p3 (+ its size AP)."""
import csv, json
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RES = REPO / "results"
CK = [6300, 11970, 18270, 24570]
SIZE = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
s640 = json.loads((RES / "p4_size640.json").read_text())
yn = yaml.safe_load((REPO / "data" / "tomato_village" / "dataset.yaml").read_text(encoding="utf-8")).get("names") or {}
NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))

rows, curves, cls_rows = [], [], []
for imgsz in (320, 416, 512, 640):
    for seed in range(5):
        if imgsz == 640:
            d = REPO / "runs_p3" / "yolo11n" / f"seed{seed}"
            j = json.loads((d / "p3_done.json").read_text())
        else:
            d = REPO / "runs_p4" / f"r{imgsz}" / f"seed{seed}"
            j = json.loads((d / "p4_done.json").read_text())
        sch = json.loads((d / "schedule.json").read_text())
        cur = list(csv.DictReader((d / "curve.csv").open(encoding="utf-8")))
        for c in cur:
            c["imgsz"] = imgsz
            curves.append(c)
        bystep = {int(c["global_step"]): c for c in cur}
        for step in CK:
            m = j["checkpoints"][str(step)]
            cc = bystep.get(step, {})
            g = lambda k: (float(cc[k]) if cc.get(k) not in (None, "", "None") else None)
            co = m.get("coco") or s640.get(str(seed), {}).get(str(step)) or {}
            r = {"imgsz": imgsz, "seed": seed, "checkpoint_step": step,
                 "checkpoint_epoch": m["epoch"], "steps_per_epoch": sch["steps_per_epoch"],
                 "total_updates": sch["total_updates"], "warmup_updates": sch["warmup_updates"],
                 "val_mAP50_95": round(m["val_mAP50_95"], 6), "val_mAP50": round(m["val_mAP50"], 6),
                 "val_precision": round(m["val_precision"], 6), "val_recall": round(m["val_recall"], 6),
                 "val_box_loss": g("val_box_loss"), "val_cls_loss": g("val_cls_loss"),
                 "val_dfl_loss": g("val_dfl_loss"), "learning_rate": g("learning_rate"),
                 "train_hours": round(j["train_hours"], 4),
                 "params_M": j.get("params_M"), "GFLOPs": j.get("GFLOPs") or j.get("GFLOPs_640"),
                 "reused_from_phase3": imgsz == 640}
            for k in SIZE:
                r["coco_" + k] = co.get(k)
            rows.append(r)
            for ci, v in (m.get("per_class") or {}).items():
                cls_rows.append({"imgsz": imgsz, "seed": seed, "checkpoint_step": step,
                                 "class_id": int(ci), "class_name": NAMES.get(int(ci), f"class{ci}"),
                                 "ap50_95": round(v["ap50_95"], 6), "ap50": round(v["ap50"], 6)})

for name, data in (("p4_checkpoint_results.csv", rows), ("p4_per_class_ap.csv", cls_rows),
                   ("p4_learning_curves.csv", curves)):
    keys = sorted({k for r in data for k in r})
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(data)
missing = [(r["imgsz"], r["seed"], r["checkpoint_step"]) for r in rows if r["coco_AP_small"] is None]
print(json.dumps({"checkpoint_rows": len(rows), "per_class_rows": len(cls_rows),
                  "curve_rows": len(curves),
                  "budgets": sorted({r["total_updates"] for r in rows}),
                  "max_step": max(int(c["global_step"]) for c in curves),
                  "missing_size_ap": missing}, indent=1))
