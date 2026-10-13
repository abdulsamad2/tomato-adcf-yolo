import csv, json, itertools
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
rows = list(csv.DictReader((RES / "p3_checkpoint_results.csv").open(encoding="utf-8")))
MODELS = ["yolo11n", "yolo11s", "yolo11m"]
STEPS = [6300, 11970, 18270, 24570]
PRIMARY, SECONDARY, MARGIN = 18270, 24570, 1.0

def vals(model, step, col="val_mAP50_95"):
    d = {int(r["seed"]): float(r[col]) * 100 for r in rows
         if r["model"] == model and int(r["checkpoint_step"]) == step}
    return np.array([d[s] for s in range(5)])

out = {"descriptives": {}, "contrasts": [], "rm_anova": {}, "peaks": {}}
for m in MODELS:
    out["descriptives"][m] = {}
    for st_ in STEPS:
        e = {}
        for col in ("val_mAP50_95", "val_mAP50", "val_precision", "val_recall",
                    "val_box_loss", "val_cls_loss", "val_dfl_loss"):
            x = vals(m, st_, col)
            e[col] = {"seeds": [round(z, 4) for z in x], "mean": round(x.mean(), 4),
                      "sd": round(x.std(ddof=1), 4)}
        out["descriptives"][m][str(st_)] = e

for st_ in STEPS:
    for hi, lo in (("yolo11s","yolo11n"), ("yolo11m","yolo11s"), ("yolo11m","yolo11n")):
        a, b = vals(hi, st_), vals(lo, st_)
        d = a - b
        t, p = stats.ttest_rel(a, b)
        se = stats.sem(d)
        ci = stats.t.interval(0.95, 4, loc=d.mean(), scale=se)
        tost = max(stats.t.sf((d.mean() + MARGIN) / se, 4), stats.t.cdf((d.mean() - MARGIN) / se, 4))
        w = stats.wilcoxon(a, b)
        out["contrasts"].append({
            "step": st_, "contrast": f"{hi[-1]}-{lo[-1]}", "primary": st_ == PRIMARY,
            "diffs": [round(z, 3) for z in d], "mean": round(d.mean(), 4),
            "sd": round(d.std(ddof=1), 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
            "t": round(float(t), 4), "p": float(p), "dz": round(d.mean() / d.std(ddof=1), 4),
            "wilcoxon_p": float(w.pvalue), "tost_p": float(tost),
            "equivalent": bool(tost < 0.05), "n_positive": int((d > 0).sum())})

# repeated-measures ANOVA: model (3) x checkpoint (4), subject = seed
try:
    import pandas as pd
    from statsmodels.stats.anova import AnovaRM
    long = pd.DataFrame([{"seed": s, "model": m, "step": st_, "y": vals(m, st_)[s]}
                         for m in MODELS for st_ in STEPS for s in range(5)])
    a = AnovaRM(long, "y", "seed", within=["model", "step"]).fit()
    tb = a.anova_table
    out["rm_anova"] = {str(i): {k: float(tb.loc[i, k]) for k in tb.columns} for i in tb.index}
except Exception as e:
    out["rm_anova"] = {"error": f"{type(e).__name__}: {e}"}

for m in MODELS:
    cur = list(csv.DictReader((RES / "p3_learning_curves.csv").open(encoding="utf-8")))
    pk, at = [], []
    for s in range(5):
        g = [c for c in cur if c["model"] == m and int(c["seed"]) == s
             and c.get("val_mAP50_95") not in ("", None)]
        best = max(g, key=lambda c: float(c["val_mAP50_95"]))
        pk.append(float(best["val_mAP50_95"]) * 100); at.append(int(best["global_step"]))
    out["peaks"][m] = {"peak_mean": round(float(np.mean(pk)), 3),
                       "peak_sd": round(float(np.std(pk, ddof=1)), 3),
                       "peak_seeds": [round(z, 3) for z in pk],
                       "at_updates": at, "at_mean": int(np.mean(at))}

pc = list(csv.DictReader((RES / "p3_per_class_ap.csv").open(encoding="utf-8")))
out["per_class_primary"] = {}
for cn in sorted({r["class_name"] for r in pc}):
    e = {}
    for m in MODELS:
        x = np.array([float(r["ap50_95"]) * 100 for r in pc
                      if r["class_name"] == cn and r["model"] == m
                      and int(r["checkpoint_step"]) == PRIMARY])
        e[m] = {"mean": round(x.mean(), 3), "sd": round(x.std(ddof=1), 3),
                "seeds": [round(z, 3) for z in x]}
    for hi, lo in (("yolo11s","yolo11n"), ("yolo11m","yolo11s"), ("yolo11m","yolo11n")):
        a_ = np.array(e[hi]["seeds"]); b_ = np.array(e[lo]["seeds"])
        t, p = stats.ttest_rel(a_, b_)
        e[f"{hi[-1]}-{lo[-1]}"] = {"mean": round(float((a_-b_).mean()), 3), "p": float(p)}
    out["per_class_primary"][cn] = e

json.dump(out, open(RES / "p3_stats.json", "w"), indent=1)
print(json.dumps({"contrasts": len(out["contrasts"]),
                  "rm_anova_keys": list(out["rm_anova"].keys()),
                  "primary": [c for c in out["contrasts"] if c["primary"]]}, indent=1))
