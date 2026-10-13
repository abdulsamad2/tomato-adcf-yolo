import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
rows = list(csv.DictReader((RES / "p4_checkpoint_results.csv").open(encoding="utf-8")))
ARMS = [320, 416, 512, 640]
STEPS = [6300, 11970, 18270, 24570]
PRIMARY, MARGIN = 18270, 1.0

def vals(imgsz, step, col="val_mAP50_95"):
    d = {int(r["seed"]): (float(r[col]) * 100 if r[col] not in ("", None) else float("nan"))
         for r in rows if int(r["imgsz"]) == imgsz and int(r["checkpoint_step"]) == step}
    return np.array([d[s] for s in range(5)])

def holm(ps_):
    order = np.argsort(ps_); adj = np.empty(len(ps_)); run = 0.0
    for i, k in enumerate(order):
        v = (len(ps_) - i) * ps_[k]
        run = max(run, v); adj[k] = min(run, 1.0)
    return adj

out = {"descriptives": {}, "primary": [], "adjacent": [], "all_contrasts": [],
       "rm_anova": {}, "peaks": {}, "size_ap": {}}

COLS = ["val_mAP50_95", "val_mAP50", "val_precision", "val_recall",
        "val_box_loss", "val_cls_loss", "val_dfl_loss",
        "coco_AP_small", "coco_AP_medium", "coco_AP_large", "coco_AP50", "coco_AP75"]
for a in ARMS:
    out["descriptives"][str(a)] = {}
    for st_ in STEPS:
        e = {}
        for col in COLS:
            x = vals(a, st_, col)
            e[col] = {"seeds": [round(float(z), 4) for z in x],
                      "mean": round(float(np.nanmean(x)), 4),
                      "sd": round(float(np.nanstd(x, ddof=1)), 4)}
        out["descriptives"][str(a)][str(st_)] = e

def contrast(hi, lo, step, col="val_mAP50_95"):
    a_, b_ = vals(hi, step, col), vals(lo, step, col)
    d = a_ - b_
    t, p = stats.ttest_rel(a_, b_)
    se = stats.sem(d)
    ci = stats.t.interval(0.95, 4, loc=d.mean(), scale=se)
    tost = max(stats.t.sf((d.mean() + MARGIN) / se, 4), stats.t.cdf((d.mean() - MARGIN) / se, 4))
    w = stats.wilcoxon(a_, b_)
    return {"step": step, "contrast": f"{hi}-{lo}", "metric": col,
            "diffs": [round(float(z), 3) for z in d], "mean": round(float(d.mean()), 4),
            "sd": round(float(d.std(ddof=1)), 4), "ci_lo": round(float(ci[0]), 4),
            "ci_hi": round(float(ci[1]), 4), "t": round(float(t), 4), "p": float(p),
            "dz": round(float(d.mean() / d.std(ddof=1)), 4), "wilcoxon_p": float(w.pvalue),
            "tost_p": float(tost), "equivalent": bool(tost < 0.05),
            "n_positive": int((d > 0).sum())}

prim = [contrast(a, 640, PRIMARY) for a in (320, 416, 512)]
hp = holm([c["p"] for c in prim]); ht = holm([c["tost_p"] for c in prim])
for c, a1, a2 in zip(prim, hp, ht):
    c["p_holm"] = float(a1); c["tost_p_holm"] = float(a2)
    c["equivalent_holm"] = bool(a2 < 0.05)
out["primary"] = prim
out["adjacent"] = [contrast(hi, lo, PRIMARY) for hi, lo in ((416, 320), (512, 416), (640, 512))]
for st_ in STEPS:
    for a in (320, 416, 512):
        out["all_contrasts"].append(contrast(a, 640, st_))

try:
    import pandas as pd
    from statsmodels.stats.anova import AnovaRM
    long = pd.DataFrame([{"seed": s, "imgsz": a, "step": st_, "y": vals(a, st_)[s]}
                         for a in ARMS for st_ in STEPS for s in range(5)])
    tb = AnovaRM(long, "y", "seed", within=["imgsz", "step"]).fit().anova_table
    out["rm_anova"] = {str(i): {k: float(tb.loc[i, k]) for k in tb.columns} for i in tb.index}
except Exception as e:
    out["rm_anova"] = {"error": f"{type(e).__name__}: {e}"}

cur = list(csv.DictReader((RES / "p4_learning_curves.csv").open(encoding="utf-8")))
for a in ARMS:
    pk, at = [], []
    for s in range(5):
        g = [c for c in cur if int(c["imgsz"]) == a and int(c["seed"]) == s
             and c.get("val_mAP50_95") not in ("", None)]
        b = max(g, key=lambda c: float(c["val_mAP50_95"]))
        pk.append(float(b["val_mAP50_95"]) * 100); at.append(int(b["global_step"]))
    out["peaks"][str(a)] = {"peak_mean": round(float(np.mean(pk)), 3),
                            "peak_sd": round(float(np.std(pk, ddof=1)), 3),
                            "at_updates": at, "at_mean": int(np.mean(at))}

for a in ARMS:
    e = out["descriptives"][str(a)][str(PRIMARY)]
    out["size_ap"][str(a)] = {k: e["coco_" + k] for k in ("AP_small", "AP_medium", "AP_large")}

pc = list(csv.DictReader((RES / "p4_per_class_ap.csv").open(encoding="utf-8")))
out["per_class_primary"] = {}
for cn in sorted({r["class_name"] for r in pc}):
    e = {}
    for a in ARMS:
        x = np.array([float(r["ap50_95"]) * 100 for r in pc if r["class_name"] == cn
                      and int(r["imgsz"]) == a and int(r["checkpoint_step"]) == PRIMARY])
        e[str(a)] = {"mean": round(float(x.mean()), 3), "sd": round(float(x.std(ddof=1)), 3),
                     "seeds": [round(float(z), 3) for z in x]}
    for a in (320, 416, 512):
        a_ = np.array(e[str(a)]["seeds"]); b_ = np.array(e["640"]["seeds"])
        t, p = stats.ttest_rel(a_, b_)
        e[f"{a}-640"] = {"mean": round(float((a_ - b_).mean()), 3), "p": float(p)}
    out["per_class_primary"][cn] = e

json.dump(out, open(RES / "p4_stats.json", "w"), indent=1)
print(json.dumps({"primary": [{k: c[k] for k in
                  ("contrast","mean","ci_lo","ci_hi","p","p_holm","tost_p","tost_p_holm",
                   "equivalent","equivalent_holm","dz","n_positive")} for c in out["primary"]],
                  "rm_anova": {k: round(v.get("Pr > F", -1), 6) for k, v in out["rm_anova"].items()}},
                 indent=1))
