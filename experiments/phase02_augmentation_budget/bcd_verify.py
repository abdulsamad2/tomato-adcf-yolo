import json
from pathlib import Path
B = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\runs_bcd")
out = {}
for arm in ("B", "C", "D"):
    d = B / arm / "smoke0"
    sch = json.loads((d / "schedule.json").read_text())
    w = json.loads((d / "warmup_lr.json").read_text())
    lrs = [x["lr"] for x in w]
    # warmup end = last step where LR is still rising
    end = 0
    for i in range(1, len(lrs)):
        if lrs[i] > lrs[i-1] + 1e-12:
            end = i + 1
    peak = max(lrs) if lrs else None
    rows = [l.split(",") for l in (d / "curve.csv").read_text().splitlines()]
    hdr = rows[0]
    gi, vi, mi, li = hdr.index("global_step"), hdr.index("validated"), hdr.index("mosaic_on"), hdr.index("learning_rate")
    mil = [(int(r[gi]), r[vi], float(r[li])) for r in rows[1:]]
    out[arm] = {
        "steps_per_epoch": sch["steps_per_epoch"], "epochs": sch["epochs"],
        "total_updates": sch["total_updates"], "warmup_updates_trainer": sch["warmup_updates"],
        "warmup_end_observed": end, "lr_peak_during_warmup": round(peak, 6),
        "mosaic_close_epoch": sch["mosaic_close_epoch"],
        "mosaic_free_updates": sch["mosaic_free_updates"],
        "validated_milestones": [m[0] for m in mil if m[1] == "True"][:12],
        "lr_at_milestones": [(m[0], round(m[2], 6)) for m in mil if m[1] == "True"][:8],
        "n_logged_epochs": len(mil),
    }
print(json.dumps(out, indent=1))
json.dump(out, open(r"C:\Users\absamad\cs\audit\bcd_smoke_verify.json", "w"), indent=1)
