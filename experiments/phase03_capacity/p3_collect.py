"""Assemble the Phase 3 capacity deliverables: n and m from runs_p3, s from Phase 2B arm D."""
import csv, json
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RES = REPO / "results"
SRC = {"yolo11n": (REPO / "runs_p3" / "yolo11n", "p3_done.json"),
       "yolo11m": (REPO / "runs_p3" / "yolo11m", "p3_done.json"),
       "yolo11s": (REPO / "runs_p2b" / "D", "p2b_done.json")}
yn = yaml.safe_load((REPO / "data" / "tomato_village" / "dataset.yaml").read_text(encoding="utf-8")).get("names") or {}
NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))

rows, curves, cls_rows = [], [], []
for model, (root, donefile) in SRC.items():
    for seed in range(5):
        d = root / f"seed{seed}"
        j = json.loads((d / donefile).read_text())
        sch = json.loads((d / "schedule.json").read_text())
        cur = list(csv.DictReader((d / "curve.csv").open(encoding="utf-8")))
        for c in cur:
            c["model"] = model
            curves.append(c)
        bystep = {int(c["global_step"]): c for c in cur}
        for step_s, m in j["checkpoints"].items():
            step = int(step_s)
            cc = bystep.get(step, {})
            g = lambda k: (float(cc[k]) if cc.get(k) not in (None, "", "None") else None)
            rows.append({
                "model": model, "seed": seed, "checkpoint_step": step, "checkpoint_epoch": m["epoch"],
                "steps_per_epoch": sch["steps_per_epoch"], "total_epochs": sch["epochs"],
                "total_updates": sch["total_updates"], "warmup_updates": sch["warmup_updates"],
                "val_mAP50_95": round(m["val_mAP50_95"], 6), "val_mAP50": round(m["val_mAP50"], 6),
                "val_precision": round(m["val_precision"], 6), "val_recall": round(m["val_recall"], 6),
                "val_box_loss": g("val_box_loss"), "val_cls_loss": g("val_cls_loss"),
                "val_dfl_loss": g("val_dfl_loss"), "learning_rate": g("learning_rate"),
                "train_hours": round(j["train_hours"], 4),
                "params_M": j.get("params_M"), "GFLOPs_640": j.get("GFLOPs_640")})
            for ci, v in (m.get("per_class") or {}).items():
                cls_rows.append({"model": model, "seed": seed, "checkpoint_step": step,
                                 "class_id": int(ci), "class_name": NAMES.get(int(ci), f"class{ci}"),
                                 "ap50_95": round(v["ap50_95"], 6), "ap50": round(v["ap50"], 6)})

for name, data in (("p3_checkpoint_results.csv", rows), ("p3_per_class_ap.csv", cls_rows),
                   ("p3_learning_curves.csv", curves)):
    keys = sorted({k for r in data for k in r})
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(data)
print(json.dumps({"checkpoint_rows": len(rows), "per_class_rows": len(cls_rows),
                  "curve_rows": len(curves),
                  "budgets": sorted({r["total_updates"] for r in rows}),
                  "max_step": max(int(c["global_step"]) for c in curves)}, indent=1))
