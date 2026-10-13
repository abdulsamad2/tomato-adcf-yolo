"""Assemble the Phase 2B deliverables: per-checkpoint metrics, per-class AP, curves."""
import csv, json
from pathlib import Path

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
R, RES = REPO / "runs_p2b", REPO / "results"
RES.mkdir(exist_ok=True)
NAMES = {}
try:
    import yaml
    d = yaml.safe_load((REPO / "data" / "tomato_village" / "dataset.yaml").read_text(encoding="utf-8"))
    NAMES = {int(k): v for k, v in (d.get("names") or {}).items()} if isinstance(d.get("names"), dict) \
            else {i: n for i, n in enumerate(d.get("names") or [])}
except Exception:
    pass

rows, curves, cls_rows = [], [], []
for arm in ("B", "C", "D"):
    for seed in range(5):
        d = R / arm / f"seed{seed}"
        j = json.loads((d / "p2b_done.json").read_text())
        sch = json.loads((d / "schedule.json").read_text())
        cur = list(csv.DictReader((d / "curve.csv").open(encoding="utf-8")))
        for c in cur:
            curves.append(c)
        bystep = {int(c["global_step"]): c for c in cur}
        for step_s, m in j["checkpoints"].items():
            step = int(step_s)
            if "error" in m:
                rows.append({"arm": arm, "seed": seed, "checkpoint_step": step, "error": m["error"]})
                continue
            cc = bystep.get(step, {})
            g = lambda k: (float(cc[k]) if cc.get(k) not in (None, "", "None") else None)
            rows.append({
                "arm": arm, "seed": seed, "checkpoint_step": step, "checkpoint_epoch": m["epoch"],
                "steps_per_epoch": sch["steps_per_epoch"], "total_epochs": sch["epochs"],
                "frac_budget": round(step / sch["total_updates"], 4),
                "val_mAP50_95": round(m["val_mAP50_95"], 6), "val_mAP50": round(m["val_mAP50"], 6),
                "val_precision": round(m["val_precision"], 6), "val_recall": round(m["val_recall"], 6),
                "val_box_loss": g("val_box_loss"), "val_cls_loss": g("val_cls_loss"),
                "val_dfl_loss": g("val_dfl_loss"), "learning_rate": g("learning_rate"),
                "train_hours": round(j["train_hours"], 4),
            })
            for ci, v in (m.get("per_class") or {}).items():
                cls_rows.append({"arm": arm, "seed": seed, "checkpoint_step": step,
                                 "class_id": int(ci), "class_name": NAMES.get(int(ci), f"class{ci}"),
                                 "ap50_95": round(v["ap50_95"], 6), "ap50": round(v["ap50"], 6)})

for name, data in (("p2b_checkpoint_results.csv", rows),
                   ("p2b_per_class_ap.csv", cls_rows),
                   ("p2b_learning_curves.csv", curves)):
    keys = sorted({k for r in data for k in r})
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(data)

print(json.dumps({"checkpoint_rows": len(rows), "per_class_rows": len(cls_rows),
                  "curve_rows": len(curves), "classes": sorted(NAMES.items())[:10],
                  "max_step": max(int(c["global_step"]) for c in curves),
                  "errors": [r for r in rows if "error" in r]}, indent=1))
