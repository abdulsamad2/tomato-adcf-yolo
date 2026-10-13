import json
import numpy as np
from pathlib import Path
R = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\runs_bcd")
BUDGET, LR0, LRF, WARM = 49770, 0.01, 0.01, 1890

def analytic(u, steps, epochs, shift):
    e = np.maximum((u - 1) // steps - shift, 0)
    base = LR0 * ((1 - e / epochs) * (1 - LRF) + LRF)
    lr = base.copy()
    w = u <= WARM
    lr[w] = np.interp(u[w], [1, WARM], [0.0, 1.0]) * base[w]
    return lr

out = {}
for arm, steps, epochs in (("B", 79, 630), ("C", 630, 79), ("D", 630, 79)):
    rows = [l.split(",") for l in (R/arm/"smoke0"/"curve.csv").read_text().splitlines()]
    h = rows[0]; gi, li = h.index("global_step"), h.index("learning_rate")
    u = np.array([int(r[gi]) for r in rows[1:]])
    obs = np.array([float(r[li]) for r in rows[1:]])
    res = {}
    for shift in (0, 1):
        pred = analytic(u, steps, epochs, shift)
        rel = 100 * np.abs(pred - obs) / LR0        # as % of lr0, the meaningful scale
        post = u > WARM
        res[f"shift{shift}"] = {
            "n_points": int(len(u)),
            "max_err_pct_of_lr0": round(float(rel.max()), 3),
            "max_err_warmup": round(float(rel[~post].max()), 3) if (~post).any() else None,
            "max_err_postwarmup": round(float(rel[post].max()), 3) if post.any() else None,
        }
    out[arm] = res
print(json.dumps(out, indent=1))
json.dump(out, open(r"C:\Users\absamad\cs\audit\lr_model_error.json", "w"), indent=1)
