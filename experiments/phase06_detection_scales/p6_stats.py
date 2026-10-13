import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
rows = list(csv.DictReader((RES / "p6_checkpoint_results.csv").open(encoding="utf-8")))
ARMS = ["A", "B", "C"]
STEPS = [6300, 11970, 18270, 24570]
PRIMARY, MARGIN = 18270, 1.0
COLS = ["val_mAP50_95", "val_mAP50", "val_precision", "val_recall",
        "coco_AP_small", "coco_AP_medium", "coco_AP_large"]

def vals(arm, step, col="val_mAP50_95"):
    d = {int(r["seed"]): (float(r[col]) * 100 if r[col] not in ("", None) else float("nan"))
         for r in rows if r["arm"] == arm and int(r["checkpoint_step"]) == step}
    return np.array([d[s] for s in range(5)])

def holm(ps_):
    order = np.argsort(ps_); adj = np.empty(len(ps_)); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (len(ps_) - i) * ps_[k]); adj[k] = min(run, 1.0)
    return adj

def contrast(hi, lo, step, primary=False):
    a, b = vals(hi, step), vals(lo, step)
    d = a - b; n = len(d)
    t, p = stats.ttest_rel(a, b)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    dz = d.mean() / d.std(ddof=1)
    w = stats.wilcoxon(a, b)
    se = stats.sem(d)
    p_lo = stats.t.sf((d.mean() + MARGIN) / se, n - 1)
    p_hi = stats.t.cdf((d.mean() - MARGIN) / se, n - 1)
    tost = max(p_lo, p_hi)
    return {"step": step, "contrast": f"{hi}-{lo}", "primary": primary,
            "diffs": [round(float(x), 3) for x in d],
            "mean": round(float(d.mean()), 3), "sd": round(float(d.std(ddof=1)), 3),
            "ci_lo": round(float(ci[0]), 3), "ci_hi": round(float(ci[1]), 3),
            "t": round(float(t), 3), "p": float(p), "dz": round(float(dz), 3),
            "wilcoxon_p": float(w.pvalue), "n_positive": int((d > 0).sum()),
            "tost_p": float(tost), "equivalent": bool(tost < 0.05)}

out = {"descriptives": {}, "primary": [], "secondary": [], "all_contrasts": [],
       "rm_anova": {}, "peaks": {}, "init": {}}
for a in ARMS:
    out["descriptives"][a] = {}
    for st in STEPS:
        e = {}
        for col in COLS:
            v = vals(a, st, col)
            e[col] = {"mean": round(float(v.mean()), 3), "sd": round(float(v.std(ddof=1)), 3),
                      "seeds": [round(float(x), 3) for x in v]}
        out["descriptives"][a][str(st)] = e

# primary: B-A and C-A at 18270, Holm over the two
prim = [contrast("B", "A", PRIMARY, True), contrast("C", "A", PRIMARY, True)]
for k, pa in (("p_holm", holm(np.array([c["p"] for c in prim]))),
              ("tost_p_holm", holm(np.array([c["tost_p"] for c in prim])))):
    for c, v in zip(prim, pa):
        c[k] = float(v)
for c in prim:
    c["equivalent_holm"] = bool(c["tost_p_holm"] < 0.05)
out["primary"] = prim
out["secondary"] = [contrast("B", "C", PRIMARY)]
for st in STEPS:
    for hi, lo in (("B", "A"), ("C", "A"), ("B", "C")):
        out["all_contrasts"].append(contrast(hi, lo, st))

try:
    import pandas as pd
    from statsmodels.stats.anova import AnovaRM
    long = []
    for a in ARMS:
        for st in STEPS:
            for s, v in enumerate(vals(a, st)):
                long.append({"seed": s, "arm": a, "step": st, "y": v})
    df = pd.DataFrame(long)
    res = AnovaRM(df, "y", "seed", within=["arm", "step"]).fit()
    tbl = res.anova_table
    out["rm_anova"] = {str(i): {k: float(tbl.loc[i, k]) for k in tbl.columns} for i in tbl.index}
except Exception as ex:
    out["rm_anova"] = {"error": f"{type(ex).__name__}: {ex}"}

cur = list(csv.DictReader((RES / "p6_learning_curves.csv").open(encoding="utf-8")))
for a in ARMS:
    pk, at = [], []
    for s in range(5):
        g = [(int(c["global_step"]), float(c["val_mAP50_95"]) * 100) for c in cur
             if c["arm"] == a and int(c["seed"]) == s and c.get("val_mAP50_95")]
        if not g:
            continue
        b = max(g, key=lambda x: x[1]); pk.append(b[1]); at.append(b[0])
    out["peaks"][a] = {"peak_mean": round(float(np.mean(pk)), 3),
                       "peak_sd": round(float(np.std(pk, ddof=1)), 3),
                       "at_mean": int(np.mean(at)), "at_seeds": at,
                       "frac_budget": round(float(np.mean(at)) / 24570, 3)}
for a in ARMS:
    rr = [r for r in rows if r["arm"] == a]
    out["init"][a] = {"transferred_pct": sorted({r["init_transferred_pct"] for r in rr}),
                      "body_new_params": sorted({r["init_body_new_params"] for r in rr}),
                      "verified": sorted({(r["init_verified_tensors"], r["init_expected_tensors"]) for r in rr})}

json.dump(out, open(RES / "p6_stats.json", "w"), indent=1)
print(json.dumps({"primary": [{k: c[k] for k in ("contrast","mean","sd","ci_lo","ci_hi","p","p_holm",
                                                 "dz","wilcoxon_p","n_positive","tost_p","tost_p_holm",
                                                 "equivalent","equivalent_holm")} for c in prim],
                  "secondary_BminusC": {k: out["secondary"][0][k] for k in ("mean","sd","ci_lo","ci_hi","p","tost_p")},
                  "rm_anova": {k: round(v.get("Pr > F", -1), 6) for k, v in out["rm_anova"].items()
                               if isinstance(v, dict) and "Pr > F" in v}}, indent=1))
