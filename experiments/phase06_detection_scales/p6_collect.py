"""Assemble Phase 6 deliverables: B and C from runs_p6, A reused from Phase 4 runs_p4/r512."""
import csv, json
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RES = REPO / "results"
SRC = {"A": (REPO / "runs_p4" / "r512", "p4_done.json"),
       "B": (REPO / "runs_p6" / "B", "p6_done.json"),
       "C": (REPO / "runs_p6" / "C", "p6_done.json")}
STEPS = [6300, 11970, 18270, 24570]
yn = yaml.safe_load((REPO / "data" / "tomato_village" / "dataset.yaml").read_text(encoding="utf-8")).get("names") or {}
NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))

rows, cls_rows, curves = [], [], []
for arm, (d, marker) in SRC.items():
    for seed in range(5):
        rd = d / f"seed{seed}"
        j = json.loads((rd / marker).read_text())
        init = j.get("init")
        ver = None
        if (rd / "init_verified.json").exists():
            ver = json.loads((rd / "init_verified.json").read_text())
        for step in STEPS:
            ck = j["checkpoints"].get(str(step))
            if ck is None or "error" in ck:
                continue
            co = ck.get("coco") or {}
            r = {"arm": arm, "seed": seed, "checkpoint_step": step, "epoch": ck["epoch"],
                 "total_updates": j.get("total_updates"), "train_hours": j.get("train_hours"),
                 "val_mAP50_95": ck["val_mAP50_95"], "val_mAP50": ck["val_mAP50"],
                 "val_precision": ck["val_precision"], "val_recall": ck["val_recall"],
                 "coco_AP_small": co.get("AP_small"), "coco_AP_medium": co.get("AP_medium"),
                 "coco_AP_large": co.get("AP_large"), "coco_AP50": co.get("AP50"),
                 "coco_AP75": co.get("AP75"),
                 "params_M": j.get("params_M"), "GFLOPs": j.get("GFLOPs") or j.get("GFLOPs_512"),
                 "init_transferred_pct": (init or {}).get("transferred_pct"),
                 "init_body_new_params": (init or {}).get("body_new_params"),
                 "init_detect_new_params": (init or {}).get("detect_new_params"),
                 "init_verified_tensors": (ver or {}).get("verified_tensors"),
                 "init_expected_tensors": (ver or {}).get("expected_tensors")}
            rows.append(r)
            for ci, v in (ck.get("per_class") or {}).items():
                cls_rows.append({"arm": arm, "seed": seed, "checkpoint_step": step,
                                 "class_id": int(ci), "class_name": NAMES.get(int(ci), f"class{ci}"),
                                 "ap50_95": v["ap50_95"], "ap50": v["ap50"]})
        cp = rd / "curve.csv"
        if cp.exists():
            for cr in csv.DictReader(cp.open(encoding="utf-8")):
                if int(cr["global_step"]) > int(j.get("total_updates") or 24570):
                    continue
                cr["arm"] = arm
                curves.append(cr)

for name, data in (("p6_checkpoint_results.csv", rows),
                   ("p6_per_class_ap.csv", cls_rows),
                   ("p6_learning_curves.csv", curves)):
    keys = sorted({k for r in data for k in r})
    with open(RES / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(data)
print(json.dumps({"checkpoint_rows": len(rows), "per_class_rows": len(cls_rows),
                  "curve_rows": len(curves),
                  "budgets": sorted({r["total_updates"] for r in rows}),
                  "max_step": max(int(c["global_step"]) for c in curves),
                  "arms": sorted({r["arm"] for r in rows}),
                  "missing_size_ap": [(r["arm"], r["seed"], r["checkpoint_step"])
                                      for r in rows if r["coco_AP_small"] is None]}, indent=1))
