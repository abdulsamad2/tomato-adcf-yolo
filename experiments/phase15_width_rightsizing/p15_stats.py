"""Phase 15 statistics. Implements p15_prereg.json exactly; no rule is chosen here.

PRIMARY  six-class AP on arm B at 24570 updates. Delta(w) = AP_B(w) - AP_B(w100), paired over
         seeds 0-4. One-sided non-inferiority, margin -1.5 AP, paired t df=4, Holm step-down over
         {w075, w050, w035} at familywise alpha 0.05. NON-INFERIOR iff Holm-adjusted p < 0.05 AND
         the Holm-rank-level lower bound > -1.5 (Bonferroni alpha/3 bound also reported).
DECISION smallest-by-params width among those passing; else Pareto frontier, no preserved-accuracy claim.
SECONDARY TOST +/-1.5 (Holm), superiority screen (Holm), 18270 results, A/B interaction for every
         metric, source-family bootstrap (1000), convergence diagnostic, trajectories.

Uncertainty from seeds (paired t) and from the finite evaluation families (bootstrap) is reported
separately and never pooled. No model, no inference, TEST NOT TOUCHED.
"""
import csv, json, os, time
from pathlib import Path
import numpy as np
from scipy import stats as st

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
P9 = BASE / "phase9"
# P15_TEST_ROOT exists only for the pre-result synthetic self-test; unset for the real analysis.
_T = os.environ.get("P15_TEST_ROOT")
OUT = Path(_T) / "phase15" if _T else BASE / "phase15"
RES = Path(_T) / "results" if _T else REPO / "results"
RUNS = Path(_T) / "runs_p15" if _T else REPO / "runs_p15"
NBOOT = int(os.environ.get("P15_TEST_NBOOT", 1000)) if _T else 1000
PRE = json.loads((OUT / "p15_prereg.json").read_text())
MARGIN = PRE["primary_test"]["margin_AP_points"]
ALPHA = 0.05
SEEDS = PRE["seeds"]
PRIMARY, SECONDARY = PRE["primary_checkpoint"], PRE["secondary_checkpoint"]
STEPS = [6300, 11970, 18270, 24570]
RED = ["w075", "w050", "w035"]
WID = ["w100"] + RED
PARAMS = {k: v["params"] for k, v in PRE["architectures"].items()}
RNG = np.random.default_rng(20261012)
METRICS = ["AP", "AP50", "AP75", "APs", "AR100", "ca_AP", "ca_AP50", "ca_AP75",
           "loc_recall_iou50", "loc_recall_iou75", "cond_class_correct", "fp_per_family"]

rows = [r for r in json.loads((OUT / "eval_rows.json").read_text()) if r["width"] != "control"]
BY = {(r["width"], r["seed"], r["step"]): r for r in rows}
need = [(w, s, p) for w in WID for s in SEEDS for p in STEPS]
miss = [k for k in need if k not in BY]
if miss:
    raise SystemExit("INCOMPLETE: %d of %d checkpoints missing, e.g. %s" % (len(miss), len(need), miss[:3]))

def v(w, s, step, arm, m="AP"):
    return float(BY[(w, s, step)][arm][m])

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(p.size); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (p.size - i) * p[k]); adj[k] = min(1.0, run)
    return adj

def paired(d):
    d = np.asarray(d, float); n = d.size; sd = float(d.std(ddof=1)); se = sd / np.sqrt(n)
    tt = st.ttest_1samp(d, 0.0)
    ci = st.t.interval(0.95, n - 1, loc=d.mean(), scale=se) if sd > 0 else (np.nan, np.nan)
    return {"per_seed": [round(float(x), 4) for x in d], "mean": float(d.mean()), "sd": sd,
            "se": se, "ci95": [float(ci[0]), float(ci[1])], "t": float(tt.statistic),
            "p_two_sided": float(tt.pvalue), "n_positive": int((d > 0).sum()), "n": n}

out = {"prereg_sha256_note": "rules read from phase15/p15_prereg.json", "margin": MARGIN,
       "alpha": ALPHA, "primary_checkpoint": PRIMARY, "descriptives": {}, "primary": {},
       "secondary": {}, "interaction": {}, "convergence": {}, "bootstrap": {}}

# ---------------- descriptives / trajectories ----------------
for w in WID:
    out["descriptives"][w] = {}
    for step in STEPS:
        e = {}
        for arm in ("A", "B"):
            for m in METRICS:
                x = np.array([v(w, s, step, arm, m) for s in SEEDS])
                e["%s_%s" % (m, arm)] = {"seeds": [round(float(z), 4) for z in x],
                                         "mean": round(float(x.mean()), 4),
                                         "sd": round(float(x.std(ddof=1)), 4)}
        out["descriptives"][w][str(step)] = e

# ---------------- primary: Holm non-inferiority ----------------
def noninf(step):
    res = {}
    for w in RED:
        d = np.array([v(w, s, step, "B") - v("w100", s, step, "B") for s in SEEDS])
        P = paired(d)
        tni = (P["mean"] + MARGIN) / P["se"]
        P["t_noninf"] = float(tni); P["p_noninf_one_sided"] = float(st.t.sf(tni, len(d) - 1))
        P["lcb_one_sided_95_unadjusted"] = float(P["mean"] - st.t.ppf(1 - ALPHA, len(d) - 1) * P["se"])
        P["lcb_bonferroni_alpha_over_3"] = float(P["mean"] - st.t.ppf(1 - ALPHA / 3, len(d) - 1) * P["se"])
        res[w] = P
    ps = [res[w]["p_noninf_one_sided"] for w in RED]
    adj = holm(ps); order = np.argsort(ps); stopped = False
    for rank, k in enumerate(order, 1):
        w = RED[k]; P = res[w]
        lvl = ALPHA / (len(RED) - rank + 1)
        P["holm_rank"] = rank; P["holm_level"] = lvl; P["p_noninf_holm"] = float(adj[k])
        P["lcb_holm_rank_level"] = float(P["mean"] - st.t.ppf(1 - lvl, P["n"] - 1) * P["se"])
        reject = (not stopped) and adj[k] < ALPHA and P["lcb_holm_rank_level"] > -MARGIN
        if not reject: stopped = True
        P["non_inferior"] = bool(reject)
        P["verdict"] = "NON-INFERIOR" if reject else "non-inferiority not established"
    return res

prim = noninf(PRIMARY)
passing = [w for w in RED if prim[w]["non_inferior"]]
sel = min(passing, key=lambda w: PARAMS[w]) if passing else None
out["primary"] = {"step": PRIMARY, "endpoint": "six-class AP50-95, arm B", "tests": prim,
                  "passing": passing, "selected": sel,
                  "selected_params": PARAMS[sel] if sel else None,
                  "decision": (("select %s (%d params, %.3fx of w100)" %
                                (sel, PARAMS[sel], PARAMS[sel] / PARAMS["w100"])) if sel else
                               "no reduced width passed: report Pareto frontier, no preserved-accuracy claim")}
out["outcome"] = ("A" if "w035" in passing else "B" if "w050" in passing else "C")
out["secondary"]["noninf_at_%d" % SECONDARY] = noninf(SECONDARY)

# ---------------- secondary: TOST and superiority ----------------
for step in (PRIMARY, SECONDARY):
    tost, sup = {}, {}
    for w in RED:
        d = np.array([v(w, s, step, "B") - v("w100", s, step, "B") for s in SEEDS])
        m, se, df = d.mean(), d.std(ddof=1) / np.sqrt(d.size), d.size - 1
        tost[w] = {"mean": float(m), "p_tost": float(max(st.t.sf((m + MARGIN) / se, df),
                                                         st.t.cdf((m - MARGIN) / se, df))),
                   "ci90": [float(m - st.t.ppf(0.95, df) * se), float(m + st.t.ppf(0.95, df) * se)]}
        sup[w] = {"mean": float(m), "p_superior_one_sided": float(st.t.sf(m / se, df))}
    for dct, key in ((tost, "p_tost"), (sup, "p_superior_one_sided")):
        adj = holm([dct[w][key] for w in RED])
        for w, a in zip(RED, adj):
            dct[w][key + "_holm"] = float(a)
            dct[w]["significant_holm"] = bool(a < ALPHA)
    out["secondary"]["tost_%d" % step] = tost
    out["secondary"]["superiority_%d" % step] = sup
out["outcome_D"] = [w for w in RED if out["secondary"]["superiority_%d" % PRIMARY][w]["significant_holm"]]

# ---------------- exposure interaction for every metric ----------------
for step in (PRIMARY, SECONDARY):
    out["interaction"][str(step)] = {}
    for w in RED:
        e = {}
        for m in METRICS:
            dB = np.array([v(w, s, step, "B", m) - v("w100", s, step, "B", m) for s in SEEDS])
            dA = np.array([v(w, s, step, "A", m) - v("w100", s, step, "A", m) for s in SEEDS])
            e[m] = {"Delta_ind": paired(dB), "Delta_exp": paired(dA), "I": paired(dB - dA)}
        out["interaction"][str(step)][w] = e

# ---------------- convergence diagnostic ----------------
for w in WID:
    g = np.array([v(w, s, PRIMARY, "B") - v(w, s, SECONDARY, "B") for s in SEEDS])
    P = paired(g); P["p_one_sided_improving"] = float(st.t.sf(P["t"], g.size - 1))
    P["still_improving"] = bool(P["mean"] > 0.5 and P["p_one_sided_improving"] < 0.05)
    P["interpretation"] = ("fixed-budget performance, not an asymptotic capacity limit"
                           if P["still_improving"] else "plateau criterion met")
    out["convergence"][w] = P

# ---------------- clean-val (descriptive) ----------------
cv = {}
for w in WID:
    cv[w] = {}
    for step in STEPS:
        x = [json.loads((RUNS / w / ("seed%d" % s) / "p15_done.json").read_text())
             ["checkpoints"][str(step)]["val_mAP50_95"] * 100 for s in SEEDS]
        cv[w][str(step)] = {"seeds": [round(z, 4) for z in x], "mean": round(float(np.mean(x)), 4),
                            "sd": round(float(np.std(x, ddof=1)), 4)}
out["clean_val_descriptive"] = cv

# ---------------- source-family bootstrap (Phase 9 validated scorer) ----------------
AI = json.loads((P9 / "arm_image_ids.json").read_text())
FAMOF = {int(k): f for k, f in AI["family_of_image"].items()}
ALLIMG = np.array(sorted(FAMOF), dtype=np.int64); POS = {int(i): k for k, i in enumerate(ALLIMG)}
NIMG = ALLIMG.size
FAMS, FAMPOS = {}, {}
for a in ("A", "B"):
    d = {}
    for i in AI["arm_image_ids"][a]:
        d.setdefault(FAMOF[int(i)], []).append(POS[int(i)])
    FAMS[a] = sorted(d); FAMPOS[a] = [np.array(d[f], dtype=np.int64) for f in FAMS[a]]
RECT = np.linspace(0, 1, 101)
_CK = {}

def prep(tag):
    if tag in _CK: return _CK[tag]
    z = np.load(OUT / "match_cache" / (tag + ".npz"))
    meta, scores, tp, ig = z["meta"], z["scores"], z["tp"], z["ig"]
    o_ = {}
    for c in z["catIds"]:
        m = meta[meta[:, 0] == int(c)]
        npig = np.zeros(NIMG, dtype=np.int64); idx_l, img_l = [], []
        for _, iid, off, k, ng in m:
            p = POS.get(int(iid))
            if p is None: continue
            npig[p] += int(ng)
            if k > 0:
                idx_l.append(np.arange(off, off + k)); img_l.append(np.full(int(k), p))
        if idx_l:
            idx = np.concatenate(idx_l); img = np.concatenate(img_l)
            srt = np.argsort(-scores[idx], kind="mergesort"); idx, img = idx[srt], img[srt]
            o_[int(c)] = {"img": img, "tp": tp[:, idx].astype(np.float64), "notig": ig[:, idx] == 0, "npig": npig}
        else:
            o_[int(c)] = {"img": np.zeros(0, np.int64), "tp": np.zeros((10, 0)),
                          "notig": np.zeros((10, 0), bool), "npig": npig}
    _CK[tag] = o_
    return o_

def ap_w(tag, w):
    aps = []
    for c, d in prep(tag).items():
        npig = float((d["npig"] * w).sum())
        if npig <= 0: continue
        if d["img"].size == 0: aps.append(0.0); continue
        ww = w[d["img"]].astype(np.float64)
        per_t = np.empty(10)
        for t in range(10):
            keep = d["notig"][t] & (ww > 0)
            if not keep.any(): per_t[t] = 0.0; continue
            wk = ww[keep]; tk = d["tp"][t][keep]
            ctp = np.cumsum(tk * wk); cfp = np.cumsum((1.0 - tk) * wk)
            rc = ctp / npig; pr = ctp / (ctp + cfp + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            ins = np.searchsorted(rc, RECT, side="left"); q = np.zeros(RECT.size); ok = ins < pr.size
            q[ok] = pr[ins[ok]]; per_t[t] = q.mean()
        aps.append(float(per_t.mean()))
    return float(np.mean(aps)) * 100 if aps else 0.0

W1 = {a: np.isin(ALLIMG, AI["arm_image_ids"][a]).astype(np.int64) for a in ("A", "B")}
tag = lambda w, s, step: "%s__seed%d__step%d" % (w, s, step)
val = [abs(ap_w(tag(w, 0, PRIMARY), W1[a]) - v(w, 0, PRIMARY, a)) for w in WID for a in ("A", "B")]
out["bootstrap"]["scorer_validation_max_abs_diff"] = float(max(val))
assert max(val) < 0.05, "subset scorer does not reproduce pycocotools: %g" % max(val)
print("scorer validated, max |diff| %.2e AP points" % max(val), flush=True)

def loc_rate(w, s, step, arm, fidx, j):
    fc = BY[(w, s, step)][arm]["family_counts"]
    M = np.stack([fc[FAMS[arm][k]] for k in fidx]).sum(0)
    return 100.0 * M[j] / max(M[0], 1)

for step in (PRIMARY, SECONDARY):
    t0 = time.time()
    B = {(w, a): np.empty((NBOOT, len(SEEDS))) for w in WID for a in ("A", "B")}
    L = {(w, a, j): np.empty((NBOOT, len(SEEDS))) for w in WID for a in ("A", "B") for j in (1, 2)}
    for b in range(NBOOT):
        fi = {a: RNG.integers(0, len(FAMS[a]), len(FAMS[a])) for a in ("A", "B")}
        wv = {}
        for a in ("A", "B"):
            x = np.zeros(NIMG, dtype=np.int64)
            for k in fi[a]: np.add.at(x, FAMPOS[a][k], 1)
            wv[a] = x
        for w in WID:
            for si, s in enumerate(SEEDS):
                for a in ("A", "B"):
                    B[(w, a)][b, si] = ap_w(tag(w, s, step), wv[a])
                    for j in (1, 2):
                        L[(w, a, j)][b, si] = loc_rate(w, s, step, a, fi[a], j)
        if (b + 1) % 100 == 0:
            print("  step %d boot %d/%d  %.0fs" % (step, b + 1, NBOOT, time.time() - t0), flush=True)
    def summ(x):
        return {"mean": float(x.mean()), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
                "lcb_one_sided_95": float(np.percentile(x, 5))}
    e = {}
    for w in RED:
        dB = (B[(w, "B")] - B[("w100", "B")]).mean(1); dA = (B[(w, "A")] - B[("w100", "A")]).mean(1)
        e[w] = {"AP": {"Delta_ind": summ(dB), "Delta_exp": summ(dA), "I": summ(dB - dA)}}
        for j, nm in ((1, "loc_recall_iou50"), (2, "loc_recall_iou75")):
            lB = (L[(w, "B", j)] - L[("w100", "B", j)]).mean(1); lA = (L[(w, "A", j)] - L[("w100", "A", j)]).mean(1)
            e[w][nm] = {"Delta_ind": summ(lB), "Delta_exp": summ(lA), "I": summ(lB - lA)}
    out["bootstrap"][str(step)] = {"n_boot": NBOOT, "unit": "source family (all files together)",
                                   "families": {a: len(FAMS[a]) for a in ("A", "B")},
                                   "note": "checkpoints fixed; contains NO seed variance", "contrasts": e}

json.dump(out, open(RES / "p15_stats.json", "w"), indent=1)
with open(RES / "p15_checkpoint_results.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["width", "params", "seed", "step", "arm"] + METRICS)
    for (w, s, step), r in sorted(BY.items()):
        for a in ("A", "B"):
            wr.writerow([w, PARAMS[w], s, step, a] + [round(float(r[a][m]), 4) for m in METRICS])

print("PRIMARY (arm B AP @%d, margin -%.1f, Holm):" % (PRIMARY, MARGIN))
for w in RED:
    P = prim[w]
    print("  %s  mean d=%+.3f sd %.3f  LCB(holm-rank)=%+.3f  p_holm=%.4f  -> %s"
          % (w, P["mean"], P["sd"], P["lcb_holm_rank_level"], P["p_noninf_holm"], P["verdict"]))
print("DECISION:", out["primary"]["decision"], "| outcome", out["outcome"], "| D:", out["outcome_D"])
print("P15STATSDONE")
