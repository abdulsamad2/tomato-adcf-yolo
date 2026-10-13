"""Exploratory pretraining-inflation statistics, exactly the contrasts in pretrain_audit/plan.json."""
import json, time
from pathlib import Path
import numpy as np
from scipy import stats as st

BASE = Path(r"C:\Users\absamad\cs"); REPO = BASE / "tomato-adcf-yolo"
P9, P15, PA = BASE / "phase9", BASE / "phase15", BASE / "pretrain_audit"
SEEDS, STEPS, NBOOT = [0, 1, 2, 3, 4], [24570, 18270, 11970, 6300], 1000
RNG = np.random.default_rng(20261013)
METRICS = ["AP", "AP50", "AP75", "APs", "ca_AP", "loc_recall_iou50", "loc_recall_iou75", "cond_class_correct", "fp_per_family"]
rows = json.loads((PA / "eval_rows.json").read_text())
ctrl = [r for r in rows if r["width"] == "control"]
rows = [r for r in rows if r["width"] == "r512"] + [r for r in json.loads((P15 / "eval_rows.json").read_text()) if r["width"] == "w100"]
BY = {(r["width"], r["seed"], r["step"]): r for r in rows}
miss = [(m, s, p) for m in ("r512", "w100") for s in SEEDS for p in STEPS if (m, s, p) not in BY]
assert not miss, miss
v = lambda m, s, p, a, k="AP": float(BY[(m, s, p)][a][k])
p9 = next(r for r in json.loads((P9 / "eval_rows.json").read_text()) if r["run"] == "runs_p4/r512" and r["seed"] == 0)
repro = {a: abs(v("r512", 0, 18270, a) - p9[a]["AP"] * 100) for a in ("A", "B")}
print("fresh r512 seed0@18270 vs Phase 9: |diff| A %.4f B %.4f" % (repro["A"], repro["B"]))

def paired(d):
    d = np.asarray(d, float); sd = d.std(ddof=1); se = sd / np.sqrt(d.size)
    ci = st.t.interval(0.95, d.size - 1, loc=d.mean(), scale=se)
    return {"per_seed": [round(float(x), 3) for x in d], "mean": float(d.mean()), "sd": float(sd),
            "ci95": [float(ci[0]), float(ci[1])], "p": float(st.ttest_1samp(d, 0).pvalue), "n_positive": int((d > 0).sum())}

out = {"plan": json.loads((PA / "plan.json").read_text()), "reproduction_vs_phase9": repro,
       "control": ctrl[0].get("control_abs_diffs") if ctrl else None, "steps": {}}
for p in STEPS:
    e = {"means": {m: {a: float(np.mean([v(m, s, p, a) for s in SEEDS])) for a in ("A", "B")} for m in ("r512", "w100")}}
    for k in METRICS:
        gA = np.array([v("r512", s, p, "A", k) - v("w100", s, p, "A", k) for s in SEEDS])
        gB = np.array([v("r512", s, p, "B", k) - v("w100", s, p, "B", k) for s in SEEDS])
        e[k] = {"gain_A": paired(gA), "gain_B": paired(gB), "I": paired(gA - gB),
                "ratio_A_over_B": float(gA.mean() / gB.mean()) if gB.mean() != 0 else None}
    out["steps"][str(p)] = e
    print("@%5d AP  pre A %.2f B %.2f | scratch A %.2f B %.2f | gain A %+.2f B %+.2f  I %+.2f %s p=%.1e ratio %.1f"
          % (p, e["means"]["r512"]["A"], e["means"]["r512"]["B"], e["means"]["w100"]["A"], e["means"]["w100"]["B"],
             e["AP"]["gain_A"]["mean"], e["AP"]["gain_B"]["mean"], e["AP"]["I"]["mean"],
             [round(x, 2) for x in e["AP"]["I"]["ci95"]], e["AP"]["I"]["p"], e["AP"]["ratio_A_over_B"]), flush=True)

# ---- family bootstrap (Phase 9 scorer) ----
AI = json.loads((P9 / "arm_image_ids.json").read_text())
FAMOF = {int(k): f for k, f in AI["family_of_image"].items()}
ALLIMG = np.array(sorted(FAMOF), dtype=np.int64); POS = {int(i): k for k, i in enumerate(ALLIMG)}; NIMG = ALLIMG.size
FAMPOS, NF = {}, {}
for a in ("A", "B"):
    d = {}
    for i in AI["arm_image_ids"][a]: d.setdefault(FAMOF[int(i)], []).append(POS[int(i)])
    FAMPOS[a] = [np.array(d[f], dtype=np.int64) for f in sorted(d)]; NF[a] = len(d)
RECT = np.linspace(0, 1, 101); _CK = {}
CD = {"r512": PA / "match_cache", "w100": P15 / "match_cache"}
def prep(m, tag):
    if tag in _CK: return _CK[tag]
    z = np.load(CD[m] / (tag + ".npz")); meta, sc, tp, ig = z["meta"], z["scores"], z["tp"], z["ig"]; o_ = {}
    for c in z["catIds"]:
        mm = meta[meta[:, 0] == int(c)]; npig = np.zeros(NIMG, np.int64); il, gl = [], []
        for _, iid, off, k, ng in mm:
            q = POS.get(int(iid))
            if q is None: continue
            npig[q] += int(ng)
            if k > 0: il.append(np.arange(off, off + k)); gl.append(np.full(int(k), q))
        if il:
            idx = np.concatenate(il); img = np.concatenate(gl); s_ = np.argsort(-sc[idx], kind="mergesort")
            o_[int(c)] = (img[s_], tp[:, idx[s_]].astype(np.float64), ig[:, idx[s_]] == 0, npig)
        else:
            o_[int(c)] = (np.zeros(0, np.int64), np.zeros((10, 0)), np.zeros((10, 0), bool), npig)
    _CK[tag] = o_; return o_
def ap_w(m, tag, w):
    aps = []
    for img, tpm, notig, npig in prep(m, tag).values():
        n_ = float((npig * w).sum())
        if n_ <= 0: continue
        if img.size == 0: aps.append(0.0); continue
        ww = w[img].astype(np.float64); pt = np.empty(10)
        for t in range(10):
            keep = notig[t] & (ww > 0)
            if not keep.any(): pt[t] = 0; continue
            wk, tk = ww[keep], tpm[t][keep]; ctp = np.cumsum(tk * wk); cfp = np.cumsum((1 - tk) * wk)
            pr = np.maximum.accumulate((ctp / (ctp + cfp + np.spacing(1)))[::-1])[::-1]
            ins = np.searchsorted(ctp / n_, RECT, side="left"); q = np.zeros(RECT.size); ok = ins < pr.size
            q[ok] = pr[ins[ok]]; pt[t] = q.mean()
        aps.append(float(pt.mean()))
    return float(np.mean(aps)) * 100 if aps else 0.0
tg = lambda m, s, p: "%s__seed%d__step%d" % (m, s, p)
W1 = {a: np.isin(ALLIMG, AI["arm_image_ids"][a]).astype(np.int64) for a in ("A", "B")}
val = max(abs(ap_w(m, tg(m, 0, 24570), W1[a]) - v(m, 0, 24570, a)) for m in ("r512", "w100") for a in ("A", "B"))
assert val < 0.05, val
out["bootstrap"] = {"scorer_validation_max_abs_diff": val}
for p in (24570, 18270):
    G = {a: np.empty((NBOOT, 5)) for a in ("A", "B")}; t0 = time.time()
    for b in range(NBOOT):
        for a in ("A", "B"):
            w = np.zeros(NIMG, np.int64)
            for k in RNG.integers(0, NF[a], NF[a]): np.add.at(w, FAMPOS[a][k], 1)
            for si, s in enumerate(SEEDS):
                G[a][b, si] = ap_w("r512", tg("r512", s, p), w) - ap_w("w100", tg("w100", s, p), w)
    gA, gB = G["A"].mean(1), G["B"].mean(1)
    sm = lambda x: {"mean": float(x.mean()), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]}
    out["bootstrap"][str(p)] = {"gain_A": sm(gA), "gain_B": sm(gB), "I": sm(gA - gB)}
    print("boot @%d gain A %s B %s I %s (%.0fs)" % (p, sm(gA), sm(gB), sm(gA - gB), time.time() - t0), flush=True)
json.dump(out, open(REPO / "results" / "pretrain_inflation.json", "w"), indent=1)
print("PXSTATSDONE")
