"""Assemble the phase-2 deliverables from the nine completed B/C/D runs."""
import csv, json
from pathlib import Path

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
R = REPO / "runs_bcd"
RES = REPO / "results"
RES.mkdir(exist_ok=True)

ARM_DESC = {
    "B": "1x source photos, matched update budget",
    "C": "8x byte-identical storage, online augmentation only",
    "D": "8x shipped offline-augmented copies",
}

rows, curves = [], []
for arm in ("B", "C", "D"):
    for seed in (0, 1, 2):
        d = R / arm / f"seed{seed}"
        j = json.loads((d / "bcd_done.json").read_text())
        sch = json.loads((d / "schedule.json").read_text())
        rows.append({
            "arm": arm, "arm_description": ARM_DESC[arm], "seed": seed,
            "steps_per_epoch": sch["steps_per_epoch"], "epochs": sch["epochs"],
            "total_updates": sch["total_updates"], "warmup_updates": sch["warmup_updates"],
            "mosaic_free_updates": sch["mosaic_free_updates"],
            "train_hours": round(j["train_hours"], 4),
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in j.items() if k.startswith(("final_", "bestval_"))},
        })
        cr = list(csv.DictReader((d / "curve.csv").open(encoding="utf-8")))
        for c in cr:
            c["arm"] = arm
            curves.append(c)

with open(RES / "phase_2_controlled_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)

ck = sorted({k for c in curves for k in c})
with open(RES / "phase_2_learning_curves.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=ck, extrasaction="ignore")
    w.writeheader(); w.writerows(curves)

val_pts = {a: sum(1 for c in curves if c["arm"] == a and c["validated"] == "True") for a in "BCD"}
print(json.dumps({"result_rows": len(rows), "curve_rows": len(curves),
                  "validated_points_per_arm_total": val_pts,
                  "columns": [k for k in sorted({k for r in rows for k in r})]}, indent=1))
