"""Phase 16A statistics. Implements phase16/p16_prereg.json; no rule is chosen here.

SUCCESS iff  G1 mean Delta AP_B(lpwa50 - U50) @24570 >= +0.75 (paired 95% CI reported, no significance needed)
         and G2 mean Delta locR50_B > 0
         and G3 params <= 811,144 and GFLOPs < 2.707.
Secondary: non-inferiority vs w100 (margin -1.5, one comparison), TOST +/-1.5, superiority vs U50, 18270,
arm-A interaction, family bootstrap (1000, Phase 9 validated scorer), convergence. TEST NOT TOUCHED.
"""
import csv, json, time
from pathlib import Path
import numpy as np
from scipy import stats as st

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
P9, P15, P16 = BASE / "phase9", BASE / "phase15", BASE / "phase16"
RES = REPO / "results"
PRE = json.loads((P16 / "p16_prereg.json").read_text())
SEEDS, PRIMARY, SECONDARY = PRE["seeds"], 24570, 18270
STEPS = [6300, 11970, 18270, 24570]
MARGIN, GATE_AP, ALPHA = 1.5, 0.75, 0.05
NBOOT = 1000
RNG = np.random.default_rng(20261012)
METRICS = ["AP", "AP50", "AP75", "APs", "AR100", "ca_AP", "ca_AP50", "ca_AP75",
           "loc_recall_iou50", "loc_recall_iou75", "cond_class_correct", "fp_per_family"]

rows = [r for r in json.loads((P16 / "eval_rows.json").read_text()) if r["width"] == "lpwa50"]
rows += [r for r in json.loads((P15 / "eval_rows.json").read_text()) if r["width"] in ("w050", "w100")]
BY = {(r["width"], r["seed"], r["step"]): r for r in rows}
CACHEDIR = {"lpwa50": P16 / "match_cache", "w050": P15 / "match_cache", "w100": P15 / "match_cache"}
MODELS = ["lpwa50", "w050", "w100"]
miss = [(m, s, p) for m in MODELS for s in SEEDS for p in STEPS if (m, s, p) not in BY]
if miss:
    raise SystemExit("INCOMPLETE: %d missing, e.g. %s" % (len(miss), miss[:3]))
v = lambda m, s, step, arm, k="AP": float(BY[(m, s, step)][arm][k])

def paired(d):
    d = np.asarray(d, float); n = d.size; sd = float(d.std(ddof=1)); se = sd / np.sqrt(n)
    ci = st.t.interval(0.95, n - 1, loc=d.mean(), scale=se) if sd > 0 else (np.nan, np.nan)
    return {"per_seed": [round(float(x), 4) for x in d], "mean": float(d.mean()), "sd": sd, "se": se,
            "ci95": [float(ci[0]), float(ci[1])], "p_two_sided": float(st.ttest_1samp(d, 0).pvalue),
            "p_one_sided_greater": float(st.t.sf(d.mean() / se, n - 1)) if se > 0 else float("nan"),
            "n_positive": int((d > 0).sum())}

def delta(a, b, step, arm, k="AP"):
    return np.array([v(a, s, step, arm, k) - v(b, s, step, arm, k) for s in SEEDS])

out = {"rules": PRE["success_criteria"], "descriptives": {}, "gates": {}, "secondary": {},
       "interaction": {}, "convergence": {}, "bootstrap": {}}
for m in MODELS:
    out["descriptives"][m] = {str(step): {"%s_%s" % (k, a): {"mean": float(np.mean([v(m, s, step, a, k) for s in SEEDS])),
                                                             "sd": float(np.std([v(m, s, step, a, k) for s in SEEDS], ddof=1))}
                                          for a in ("A", "B") for k in METRICS} for step in STEPS}

# ---------------- gates ----------------
c = PRE["candidate"]
g1 = paired(delta("lpwa50", "w050", PRIMARY, "B")); g1["pass"] = bool(g1["mean"] >= GATE_AP)
g2 = paired(delta("lpwa50", "w050", PRIMARY, "B", "loc_recall_iou50")); g2["pass"] = bool(g2["mean"] > 0)
g3 = {"params": c["params"], "GFLOPs_512": c["GFLOPs_512"],
      "pass": bool(c["params"] <= 811144 and c["GFLOPs_512"] < 2.707)}
out["gates"] = {"G1_primary_AP": g1, "G2_mechanism_locR50": g2, "G3_resources": g3,
                "SUCCESS": bool(g1["pass"] and g2["pass"] and g3["pass"])}
out["decision"] = PRE["decision"]["success" if out["gates"]["SUCCESS"] else "failure"]

# ---------------- secondary ----------------
for step in (PRIMARY, SECONDARY):
    d = delta("lpwa50", "w100", step, "B"); P = paired(d); df = d.size - 1
    P["p_noninf_one_sided"] = float(st.t.sf((P["mean"] + MARGIN) / P["se"], df))
    P["lcb_one_sided_95"] = float(P["mean"] - st.t.ppf(0.95, df) * P["se"])
    P["non_inferior"] = bool(P["p_noninf_one_sided"] < ALPHA and P["lcb_one_sided_95"] > -MARGIN)
    P["p_tost"] = float(max(st.t.sf((P["mean"] + MARGIN) / P["se"], df), st.t.cdf((P["mean"] - MARGIN) / P["se"], df)))
    P["equivalent_tost"] = bool(P["p_tost"] < ALPHA)
    out["secondary"]["vs_w100_%d" % step] = P
    out["secondary"]["vs_U50_%d" % step] = paired(delta("lpwa50", "w050", step, "B"))
for step in (PRIMARY, SECONDARY):
    out["interaction"][str(step)] = {
        ref: {k: {"Delta_ind": paired(delta("lpwa50", ref, step, "B", k)),
                  "Delta_exp": paired(delta("lpwa50", ref, step, "A", k)),
                  "I": paired(delta("lpwa50", ref, step, "B", k) - delta("lpwa50", ref, step, "A", k))}
              for k in METRICS} for ref in ("w050", "w100")}
for m in MODELS:
    gg = np.array([v(m, s, PRIMARY, "B") - v(m, s, SECONDARY, "B") for s in SEEDS]); P = paired(gg)
    P["still_improving"] = bool(P["mean"] > 0.5 and P["p_one_sided_greater"] < 0.05)
    out["convergence"][m] = P

# ---------------- source-family bootstrap (Phase 9 scorer) ----------------
AI = json.loads((P9 / "arm_image_ids.json").read_text())
FAMOF = {int(k): f for k, f in AI["family_of_image"].items()}
ALLIMG = np.array(sorted(FAMOF), dtype=np.int64); POS = {int(i): k for k, i in enumerate(ALLIMG)}; NIMG = ALLIMG.size
FAMS, FAMPOS = {}, {}
for a in ("A", "B"):
    d = {}
    for i in AI["arm_image_ids"][a]:
        d.setdefault(FAMOF[int(i)], []).append(POS[int(i)])
    FAMS[a] = sorted(d); FAMPOS[a] = [np.array(d[f], dtype=np.int64) for f in FAMS[a]]
RECT = np.linspace(0, 1, 101); _CK = {}

def prep(m, tag):
    if tag in _CK: return _CK[tag]
    z = np.load(CACHEDIR[m] / (tag + ".npz")); meta, scores, tp, ig = z["meta"], z["scores"], z["tp"], z["ig"]
    o_ = {}
    for cat in z["catIds"]:
        mm = meta[meta[:, 0] == int(cat)]; npig = np.zeros(NIMG, dtype=np.int64); il, gl = [], []
        for _, iid, off, k, ng in mm:
            p = POS.get(int(iid))
            if p is None: continue
            npig[p] += int(ng)
            if k > 0: il.append(np.arange(off, off + k)); gl.append(np.full(int(k), p))
        if il:
            idx = np.concatenate(il); img = np.concatenate(gl); srt = np.argsort(-scores[idx], kind="mergesort")
            o_[int(cat)] = {"img": img[srt], "tp": tp[:, idx[srt]].astype(np.float64), "notig": ig[:, idx[srt]] == 0, "npig": npig}
        else:
            o_[int(cat)] = {"img": np.zeros(0, np.int64), "tp": np.zeros((10, 0)), "notig": np.zeros((10, 0), bool), "npig": npig}
    _CK[tag] = o_; return o_

def ap_w(m, tag, w):
    aps = []
    for _, d in prep(m, tag).items():
        npig = float((d["npig"] * w).sum())
        if npig <= 0: continue
        if d["img"].size == 0: aps.append(0.0); continue
        ww = w[d["img"]].astype(np.float64); per_t = np.empty(10)
        for t in range(10):
            keep = d["notig"][t] & (ww > 0)
            if not keep.any(): per_t[t] = 0.0; continue
            wk = ww[keep]; tk = d["tp"][t][keep]; ctp = np.cumsum(tk * wk); cfp = np.cumsum((1 - tk) * wk)
            rc = ctp / npig; pr = np.maximum.accumulate((ctp / (ctp + cfp + np.spacing(1)))[::-1])[::-1]
            ins = np.searchsorted(rc, RECT, side="left"); q = np.zeros(RECT.size); ok = ins < pr.size
            q[ok] = pr[ins[ok]]; per_t[t] = q.mean()
        aps.append(float(per_t.mean()))
    return float(np.mean(aps)) * 100 if aps else 0.0

W1 = {a: np.isin(ALLIMG, AI["arm_image_ids"][a]).astype(np.int64) for a in ("A", "B")}
tg = lambda m, s, step: "%s__seed%d__step%d" % (m, s, step)
val = [abs(ap_w(m, tg(m, 0, PRIMARY), W1[a]) - v(m, 0, PRIMARY, a)) for m in MODELS for a in ("A", "B")]
out["bootstrap"]["scorer_validation_max_abs_diff"] = float(max(val))
assert max(val) < 0.05, max(val)

for step in (PRIMARY, SECONDARY):
    t0 = time.time()
    Bm = {(m, a): np.empty((NBOOT, len(SEEDS))) for m in MODELS for a in ("A", "B")}
    Lm = {(m, a): np.empty((NBOOT, len(SEEDS))) for m in MODELS for a in ("A", "B")}
    for b in range(NBOOT):
        fi = {a: RNG.integers(0, len(FAMS[a]), len(FAMS[a])) for a in ("A", "B")}; wv = {}
        for a in ("A", "B"):
            x = np.zeros(NIMG, dtype=np.int64)
            for k in fi[a]: np.add.at(x, FAMPOS[a][k], 1)
            wv[a] = x
        for m in MODELS:
            for si, s in enumerate(SEEDS):
                for a in ("A", "B"):
                    Bm[(m, a)][b, si] = ap_w(m, tg(m, s, step), wv[a])
                    fc = BY[(m, s, step)][a]["family_counts"]
                    M = np.stack([fc[FAMS[a][k]] for k in fi[a]]).sum(0)
                    Lm[(m, a)][b, si] = 100.0 * M[1] / max(M[0], 1)
        if (b + 1) % 100 == 0:
            print("  step %d boot %d/%d %.0fs" % (step, b + 1, NBOOT, time.time() - t0), flush=True)
    summ = lambda x: {"mean": float(x.mean()), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]}
    e = {}
    for ref in ("w050", "w100"):
        dB = (Bm[("lpwa50", "B")] - Bm[(ref, "B")]).mean(1); dA = (Bm[("lpwa50", "A")] - Bm[(ref, "A")]).mean(1)
        lB = (Lm[("lpwa50", "B")] - Lm[(ref, "B")]).mean(1)
        e[ref] = {"AP_Delta_ind": summ(dB), "AP_Delta_exp": summ(dA), "AP_I": summ(dB - dA), "locR50_Delta_ind": summ(lB)}
    out["bootstrap"][str(step)] = {"n_boot": NBOOT, "unit": "source family", "contrasts": e}

json.dump(out, open(RES / "p16_stats.json", "w"), indent=1)
with open(RES / "p16_checkpoint_results.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["model", "seed", "step", "arm"] + METRICS)
    for (m, s, step), r in sorted(BY.items()):
        if m == "lpwa50":
            for a in ("A", "B"):
                wr.writerow([m, s, step, a] + [round(float(r[a][k]), 4) for k in METRICS])
G = out["gates"]
print("G1 dAP_B vs U50 = %+.3f %s -> %s" % (G["G1_primary_AP"]["mean"], [round(x, 2) for x in G["G1_primary_AP"]["ci95"]], G["G1_primary_AP"]["pass"]))
print("G2 dlocR50_B = %+.3f -> %s | G3 %s" % (G["G2_mechanism_locR50"]["mean"], G["G2_mechanism_locR50"]["pass"], G["G3_resources"]["pass"]))
print("SUCCESS:", G["SUCCESS"], "|", out["decision"])
print("P16STATSDONE")
