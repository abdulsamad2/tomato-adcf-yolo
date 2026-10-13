"""Phase 17 statistics. Implements phase17/p17_prereg.json; no rule is chosen here.

PRIMARY  Delta_i = AP_B(kd075, i) - AP_B(w075, i), six-class AP50-95, arm B, 24570 updates, seeds 0-4.
         PASS iff mean Delta >= +0.75 AP AND the resource/architecture-identity gate holds.
         Paired 95% t CI and per-seed deltas reported; significance is not required.
MECHANISM mean Delta locR50_B > 0 (reported; corroboration only, not part of PASS).
Also: AP50, AP75, AR100, locR50/75, conditional class correctness, FP/family, arms A and B, exposed-minus-
independent interaction, 18270 results, vs 1.00x (Phase 15 margin) and vs the pretrained teacher family
(context), family bootstrap 1000, convergence. TEST NOT TOUCHED.
"""
import csv, json, time
from pathlib import Path
import numpy as np
from scipy import stats as st

BASE = Path(r"C:\Users\absamad\cs"); REPO = BASE / "tomato-adcf-yolo"
P9, P15, P17, PA = BASE / "phase9", BASE / "phase15", BASE / "phase17", BASE / "pretrain_audit"
RES = REPO / "results"
PRE = json.loads((P17 / "p17_prereg.json").read_text())
SEEDS, PRIMARY, SECONDARY = PRE["seeds"], 24570, 18270
STEPS = [6300, 11970, 18270, 24570]
GATE, MARGIN, ALPHA, NBOOT = PRE["primary"]["gate_AP"], 1.5, 0.05, 1000
RNG = np.random.default_rng(20261014)
METRICS = ["AP", "AP50", "AP75", "APs", "AR100", "ca_AP", "ca_AP50", "ca_AP75",
           "loc_recall_iou50", "loc_recall_iou75", "cond_class_correct", "fp_per_family"]

rows = [r for r in json.loads((P17 / "eval_rows.json").read_text()) if r["width"] == "kd075"]
rows += [r for r in json.loads((P15 / "eval_rows.json").read_text()) if r["width"] in ("w075", "w100")]
rows += [r for r in json.loads((PA / "eval_rows.json").read_text()) if r["width"] == "r512"]
BY = {(r["width"], r["seed"], r["step"]): r for r in rows}
CD = {"kd075": P17 / "match_cache", "w075": P15 / "match_cache", "w100": P15 / "match_cache", "r512": PA / "match_cache"}
MODELS = ["kd075", "w075", "w100", "r512"]
miss = [(m, s, p) for m in MODELS for s in SEEDS for p in STEPS if (m, s, p) not in BY]
if miss: raise SystemExit("INCOMPLETE: %d missing, e.g. %s" % (len(miss), miss[:3]))
v = lambda m, s, p, a, k="AP": float(BY[(m, s, p)][a][k])

def paired(d):
    d = np.asarray(d, float); n = d.size; sd = float(d.std(ddof=1)); se = sd / np.sqrt(n)
    ci = st.t.interval(0.95, n - 1, loc=d.mean(), scale=se) if sd > 0 else (np.nan, np.nan)
    return {"per_seed": [round(float(x), 4) for x in d], "mean": float(d.mean()), "sd": sd, "se": se,
            "ci95": [float(ci[0]), float(ci[1])], "p_two_sided": float(st.ttest_1samp(d, 0).pvalue),
            "p_one_sided_greater": float(st.t.sf(d.mean() / se, n - 1)) if se > 0 else float("nan"),
            "n_positive": int((d > 0).sum())}
delta = lambda a, b, p, arm, k="AP": np.array([v(a, s, p, arm, k) - v(b, s, p, arm, k) for s in SEEDS])

out = {"rules": PRE["primary"], "descriptives": {}, "primary": {}, "mechanism": {}, "resource_gate": {},
       "secondary": {}, "interaction": {}, "convergence": {}, "bootstrap": {}}
for m in MODELS:
    out["descriptives"][m] = {str(p): {"%s_%s" % (k, a): {"mean": float(np.mean([v(m, s, p, a, k) for s in SEEDS])),
                                                          "sd": float(np.std([v(m, s, p, a, k) for s in SEEDS], ddof=1))}
                                       for a in ("A", "B") for k in METRICS} for p in STEPS}

# ---------- resource / architecture-identity gate (from the trained checkpoints) ----------
import torch, warnings; warnings.filterwarnings("ignore")
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops
rg = {"per_seed": []}
for s in SEEDS:
    wk = REPO / "runs_p17" / ("seed%d" % s) / "weights" / "ckpt_ep39_step24570.pt"
    wb = REPO / "runs_p15" / "w075" / ("seed%d" % s) / "weights" / "ckpt_ep39_step24570.pt"
    nk, nb = YOLO(str(wk)).model, YOLO(str(wb)).model
    sk = {k: tuple(t.shape) for k, t in nk.state_dict().items()}; sb = {k: tuple(t.shape) for k, t in nb.state_dict().items()}
    rg["per_seed"].append({"seed": s, "params_kd": get_num_params(nk), "params_base": get_num_params(nb),
                           "GFLOPs_kd": round(get_flops(nk, 512), 3), "GFLOPs_base": round(get_flops(nb, 512), 3),
                           "state_dict_identical_keys_shapes": sk == sb,
                           "file_MB_kd": round(wk.stat().st_size / 1e6, 4), "file_MB_base": round(wb.stat().st_size / 1e6, 4)})
rg["pass"] = all(r["params_kd"] == r["params_base"] == 1565628 and r["GFLOPs_kd"] == r["GFLOPs_base"]
                 and r["state_dict_identical_keys_shapes"] and abs(r["file_MB_kd"] / r["file_MB_base"] - 1) < 0.01
                 for r in rg["per_seed"])
out["resource_gate"] = rg

# ---------- primary + mechanism ----------
g1 = paired(delta("kd075", "w075", PRIMARY, "B")); g1["meets_gate"] = bool(g1["mean"] >= GATE)
g2 = paired(delta("kd075", "w075", PRIMARY, "B", "loc_recall_iou50")); g2["expected_direction"] = bool(g2["mean"] > 0)
out["primary"] = g1; out["mechanism"] = g2
out["PASS"] = bool(g1["meets_gate"] and rg["pass"])
out["decision"] = PRE["stop_rules"]["pass" if out["PASS"] else "fail"]

# ---------- secondary ----------
for p in (PRIMARY, SECONDARY):
    out["secondary"]["vs_w075_%d" % p] = {k: paired(delta("kd075", "w075", p, "B", k)) for k in METRICS}
    d = delta("kd075", "w100", p, "B"); P = paired(d); df = d.size - 1
    P["p_noninf_one_sided"] = float(st.t.sf((P["mean"] + MARGIN) / P["se"], df))
    P["lcb_one_sided_95"] = float(P["mean"] - st.t.ppf(0.95, df) * P["se"])
    P["non_inferior_margin_1p5"] = bool(P["p_noninf_one_sided"] < ALPHA and P["lcb_one_sided_95"] > -MARGIN)
    out["secondary"]["vs_w100_%d" % p] = P
    out["secondary"]["vs_pretrained_teacher_family_%d" % p] = paired(delta("kd075", "r512", p, "B"))
    gap = np.array([v("r512", s, p, "B") - v("w075", s, p, "B") for s in SEEDS])
    out["secondary"]["fraction_of_pretraining_gap_recovered_%d" % p] = float(g1["mean"] / gap.mean()) if p == PRIMARY else \
        float(delta("kd075", "w075", p, "B").mean() / gap.mean())
for p in (PRIMARY, SECONDARY):
    out["interaction"][str(p)] = {k: {"Delta_ind": paired(delta("kd075", "w075", p, "B", k)),
                                      "Delta_exp": paired(delta("kd075", "w075", p, "A", k)),
                                      "I": paired(delta("kd075", "w075", p, "B", k) - delta("kd075", "w075", p, "A", k))}
                                  for k in METRICS}
for m in ("kd075", "w075"):
    g = np.array([v(m, s, PRIMARY, "B") - v(m, s, SECONDARY, "B") for s in SEEDS]); P = paired(g)
    P["still_improving"] = bool(P["mean"] > 0.5 and P["p_one_sided_greater"] < 0.05); out["convergence"][m] = P
cv = {}
for s in SEEDS:
    dj = json.loads((REPO / "runs_p17" / ("seed%d" % s) / "p17_done.json").read_text())
    cv[s] = {k: x["val_mAP50_95"] * 100 for k, x in dj["checkpoints"].items() if "val_mAP50_95" in x}
out["clean_val_descriptive_kd075"] = cv

# ---------- source-family bootstrap (Phase 9 scorer) ----------
AI = json.loads((P9 / "arm_image_ids.json").read_text())
FAMOF = {int(k): f for k, f in AI["family_of_image"].items()}
ALLIMG = np.array(sorted(FAMOF), dtype=np.int64); POS = {int(i): k for k, i in enumerate(ALLIMG)}; NIMG = ALLIMG.size
FAMS, FAMPOS = {}, {}
for a in ("A", "B"):
    d = {}
    for i in AI["arm_image_ids"][a]: d.setdefault(FAMOF[int(i)], []).append(POS[int(i)])
    FAMS[a] = sorted(d); FAMPOS[a] = [np.array(d[f], dtype=np.int64) for f in FAMS[a]]
RECT = np.linspace(0, 1, 101); _CK = {}
def prep(m, tag):
    if (m, tag) in _CK: return _CK[(m, tag)]
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
    _CK[(m, tag)] = o_; return o_
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
            wk_, tk = ww[keep], tpm[t][keep]; ctp = np.cumsum(tk * wk_); cfp = np.cumsum((1 - tk) * wk_)
            pr = np.maximum.accumulate((ctp / (ctp + cfp + np.spacing(1)))[::-1])[::-1]
            ins = np.searchsorted(ctp / n_, RECT, side="left"); q = np.zeros(RECT.size); ok = ins < pr.size
            q[ok] = pr[ins[ok]]; pt[t] = q.mean()
        aps.append(float(pt.mean()))
    return float(np.mean(aps)) * 100 if aps else 0.0
tg = lambda m, s, p: "%s__seed%d__step%d" % (m, s, p)
W1 = {a: np.isin(ALLIMG, AI["arm_image_ids"][a]).astype(np.int64) for a in ("A", "B")}
val = max(abs(ap_w(m, tg(m, 0, PRIMARY), W1[a]) - v(m, 0, PRIMARY, a)) for m in ("kd075", "w075") for a in ("A", "B"))
assert val < 0.05, val
out["bootstrap"]["scorer_validation_max_abs_diff"] = float(val)
for p in (PRIMARY, SECONDARY):
    t0 = time.time(); D = {a: np.empty((NBOOT, len(SEEDS))) for a in ("A", "B")}; L = np.empty((NBOOT, len(SEEDS)))
    for b in range(NBOOT):
        fi = {a: RNG.integers(0, len(FAMS[a]), len(FAMS[a])) for a in ("A", "B")}
        for a in ("A", "B"):
            w = np.zeros(NIMG, np.int64)
            for k in fi[a]: np.add.at(w, FAMPOS[a][k], 1)
            for si, s in enumerate(SEEDS):
                D[a][b, si] = ap_w("kd075", tg("kd075", s, p), w) - ap_w("w075", tg("w075", s, p), w)
        for si, s in enumerate(SEEDS):
            rate = []
            for m in ("kd075", "w075"):
                fc = BY[(m, s, p)]["B"]["family_counts"]; M = np.stack([fc[FAMS["B"][k]] for k in fi["B"]]).sum(0)
                rate.append(100.0 * M[1] / max(M[0], 1))
            L[b, si] = rate[0] - rate[1]
        if (b + 1) % 100 == 0: print("  step %d boot %d/%d %.0fs" % (p, b + 1, NBOOT, time.time() - t0), flush=True)
    sm = lambda x: {"mean": float(x.mean()), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]}
    dA, dB = D["A"].mean(1), D["B"].mean(1)
    out["bootstrap"][str(p)] = {"AP_Delta_ind": sm(dB), "AP_Delta_exp": sm(dA), "AP_I": sm(dB - dA),
                                "locR50_Delta_ind": sm(L.mean(1)), "n_boot": NBOOT, "unit": "source family"}

json.dump(out, open(RES / "p17_stats.json", "w"), indent=1)
with open(RES / "p17_checkpoint_results.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["model", "seed", "step", "arm"] + METRICS)
    for (m, s, p), r in sorted(BY.items()):
        if m == "kd075":
            for a in ("A", "B"): wr.writerow([m, s, p, a] + [round(float(r[a][k]), 4) for k in METRICS])
print("PRIMARY dAP_B = %+.3f %s seeds %s -> gate %s | resource %s | PASS %s"
      % (g1["mean"], [round(x, 2) for x in g1["ci95"]], g1["per_seed"], g1["meets_gate"], rg["pass"], out["PASS"]))
print("MECHANISM dlocR50_B = %+.3f expected_direction %s" % (g2["mean"], g2["expected_direction"]))
print("DECISION:", out["decision"]); print("P17STATSDONE")
