"""Post-hoc analyses 1-11 of posthoc/plan.md. Development arms only (Phase 9 arms A/B); clean test NOT used.
Cached detections, step 24570, seeds 0-4. No training, no model selection. Writes posthoc/results.json + figures.
"""
import contextlib, io, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as st

B = Path(r"C:\Users\absamad\cs"); P9 = B / "phase9"; OUT = B / "posthoc"
STEP, SEEDS = 24570, [0, 1, 2, 3, 4]
SRC = {"w100": "phase15", "w075": "phase15", "w050": "phase15", "w035": "phase15",
       "lpwa50": "phase16", "kd075": "phase17", "r512": "pretrain_audit"}
MODELS = list(SRC)
PARAMS = {"w100": 2591400, "w075": 1565628, "w050": 811144, "w035": 537478, "lpwa50": 786882, "kd075": 1565628, "r512": 2591400}
GFLOPS = {"w100": 4.15, "w075": 2.707, "w050": 1.549, "w035": 1.16, "lpwa50": 1.965, "kd075": 2.707, "r512": 4.15}
INTERV = [("width 0.75x", "w075", "w100"), ("width 0.50x", "w050", "w100"), ("width 0.35x", "w035", "w100"),
          ("LPWA-50 vs uniform 0.50x", "lpwa50", "w050"), ("COCO pretraining", "r512", "w100"),
          ("distillation", "kd075", "w075")]
CONF, TAUS = 0.25, [round(0.5 + 0.05 * k, 2) for k in range(10)]
CONF_SWEEP = [round(0.05 * k, 2) for k in range(1, 20)]
NAMES = ["Early Blight", "Healthy", "Late Blight", "Leaf Miner", "Magnesium Deficiency", "Nitrogen Deficiency",
         "Potassium Deficiency", "Spotted Wilt Virus"]
t0 = time.time()


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(np.float32)


def top100(iid, xyxy, cf, cl):
    order = np.lexsort((-cf, iid)); iid2 = iid[order]; keep = np.zeros(len(order), bool); start = 0
    for end in list(np.flatnonzero(np.diff(iid2)) + 1) + [len(iid2)]:
        keep[start:min(start + 100, end)] = True; start = end
    s = order[keep]
    return iid[s], xyxy[s], cf[s], cl[s]


def greedy_agnostic(M, thr):
    """class-agnostic one-to-one greedy match, preds already in descending score. returns (gt_hit, pred_matched)."""
    ng, npd = M.shape[1], M.shape[0]; taken = np.zeros(ng, bool); pm = np.zeros(npd, bool)
    for p in range(npd):
        c = np.where((M[p] >= thr) & ~taken)[0]
        if len(c): taken[c[np.argmax(M[p, c])]] = True; pm[p] = True
    return taken, pm


def paired(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(d.size)
    ci = st.t.interval(0.95, d.size - 1, loc=d.mean(), scale=se) if se > 0 else (np.nan, np.nan)
    return {"per_seed": [round(float(x), 3) for x in d], "mean": float(d.mean()), "ci95": [float(ci[0]), float(ci[1])],
            "n_positive": int((d > 0).sum())}


# ---------------- ground truth, arms, families ----------------
g = json.loads((P9 / "coco_gt_union.json").read_text())
arms = json.loads((P9 / "arm_image_ids.json").read_text())
ARM = {k: sorted(arms["arm_image_ids"][k]) for k in ("A", "B")}
FAM = {int(k): v for k, v in arms["family_of_image"].items()}
GT = defaultdict(list)
for a in g["annotations"]:
    x, y, w, h = a["bbox"]; GT[a["image_id"]].append((x, y, x + w, y + h, a["category_id"]))
GT = {i: np.array(v, dtype=np.float32).reshape(-1, 5) for i, v in GT.items()}
sizeB = np.sqrt(np.concatenate([(GT[i][:, 2] - GT[i][:, 0]) * (GT[i][:, 3] - GT[i][:, 1]) for i in ARM["B"] if i in GT]))
TERT = [float(np.percentile(sizeB, 100 / 3)), float(np.percentile(sizeB, 200 / 3))]
def size_bin(side):
    return "small" if side < TERT[0] else ("medium" if side < TERT[1] else "large")
def coco_bin(area):
    return "coco_small" if area < 32 ** 2 else ("coco_medium" if area < 96 ** 2 else "coco_large")

# ---------------- evaluation rows (already-computed AP etc.) ----------------
ROWS = {}
for m, src in SRC.items():
    for r in json.loads((B / src / "eval_rows.json").read_text()):
        if r["width"] == m and r["step"] == STEP: ROWS[(m, r["seed"])] = r
assert len(ROWS) == len(MODELS) * 5, len(ROWS)
ap = lambda m, s, arm, k="AP": float(ROWS[(m, s)][arm][k])

out = {"plan_sha256": (OUT / "plan.sha256").read_text().split()[0], "data": "Phase 9 arms A/B only; clean test NOT used",
       "step": STEP, "size_tertiles_px_sqrt_area_armB": TERT}

# ---------------- 1. exposure distortion ----------------
dist = []
for name, a_, b_ in INTERV:
    dB = np.array([ap(a_, s, "B") - ap(b_, s, "B") for s in SEEDS]); dA = np.array([ap(a_, s, "A") - ap(b_, s, "A") for s in SEEDS])
    dist.append({"intervention": name, "contrast": "%s - %s" % (a_, b_), "delta_B": paired(dB), "delta_A": paired(dA),
                 "distortion_A_minus_B": paired(dA - dB),
                 "ratio_absA_over_absB": float(abs(dA.mean()) / abs(dB.mean())) if abs(dB.mean()) > 1e-9 else None})
out["1_exposure_distortion"] = dist

# ---------------- 7. protocol decision comparison (Phase 15 rule on each arm) ----------------
def phase15_rule(arm):
    res = {}
    for w in ("w075", "w050", "w035"):
        d = np.array([ap(w, s, arm) - ap("w100", s, arm) for s in SEEDS]); se = d.std(ddof=1) / np.sqrt(5)
        res[w] = {"mean": float(d.mean()), "p": float(st.t.sf((d.mean() + 1.5) / se, 4)), "se": float(se)}
    order = sorted(res, key=lambda w: res[w]["p"]); stopped = False; run = 0.0
    for k, w in enumerate(order, 1):
        lvl = 0.05 / (3 - k + 1); run = max(run, (3 - k + 1) * res[w]["p"])
        lcb = res[w]["mean"] - st.t.ppf(1 - lvl, 4) * res[w]["se"]
        ok = (not stopped) and min(1, run) < 0.05 and lcb > -1.5
        if not ok: stopped = True
        res[w].update(p_holm=min(1.0, run), lcb=float(lcb), non_inferior=bool(ok))
    passing = [w for w in res if res[w]["non_inferior"]]
    return {"tests": res, "selected": min(passing, key=lambda w: PARAMS[w]) if passing else "w100 (no reduced width passes)"}
out["7_protocol_decision"] = {"arm_B_independent": phase15_rule("B"), "arm_A_exposed": phase15_rule("A")}

# ---------------- per-detection analyses 2,3,4,5,10,11 ----------------
TYPES = ["tp", "duplicate", "cls_err", "loc_err", "cls_loc_err", "background"]
per = {}
for m in MODELS:
    for s in SEEDS:
        z = np.load(B / SRC[m] / "dets" / ("%s__seed%d__step%d.npz" % (m, s, STEP)), allow_pickle=True)
        iid, xyxy, cf, cl = top100(z["image_id"], z["xyxy"], z["conf"], z["cls"].astype(np.int64))
        byimg = defaultdict(list)
        for k in range(len(iid)): byimg[int(iid[k])].append(k)
        R = {}
        for arm in ("A", "B"):
            dec = {t: 0 for t in TYPES}; ngt = 0; missed_near = missed_far = 0
            rec_tau = np.zeros(len(TAUS)); size_hit = defaultdict(lambda: np.zeros(2)); cls_hit = defaultdict(lambda: np.zeros(2))
            sweep = np.zeros((len(CONF_SWEEP), 2)); fams = set()
            calib = []  # (conf, is_tp) for conf>=0.05
            for i in ARM[arm]:
                fams.add(FAM[i]); G = GT.get(i, np.zeros((0, 5), np.float32)); ngt += len(G)
                ks = sorted(byimg.get(i, []), key=lambda k: -cf[k])
                Pall = xyxy[ks]; Call = cf[ks]; Lall = cl[ks]
                Mall = iou_mat(Pall, G[:, :4]) if len(G) and len(ks) else np.zeros((len(ks), len(G)), np.float32)
                # --- calibration (arm B only): class-aware one-to-one, conf >= 0.05
                if arm == "B":
                    sel = Call >= 0.05; taken = np.zeros(len(G), bool)
                    for p in np.where(sel)[0]:
                        ok = False
                        if len(G):
                            c = np.where((Mall[p] >= 0.5) & ~taken & (G[:, 4] == Lall[p]))[0]
                            if len(c): taken[c[np.argmax(Mall[p, c])]] = True; ok = True
                        calib.append((float(Call[p]), ok))
                    # --- operating curve (class-agnostic, Phase 12 matching)
                    for j, t in enumerate(CONF_SWEEP):
                        q = Call >= t
                        if len(G) == 0:
                            sweep[j, 1] += q.sum(); continue
                        hit, pm = greedy_agnostic(Mall[q], 0.5)
                        sweep[j, 0] += hit.sum(); sweep[j, 1] += (~pm).sum()
                # --- operating point conf >= 0.25
                q = Call >= CONF; P, Cc, M = Pall[q], Lall[q], Mall[q]
                if len(G) == 0:
                    dec["background"] += len(P); continue
                for j, tau in enumerate(TAUS):
                    hit, _ = greedy_agnostic(M, tau); rec_tau[j] += hit.sum()
                hit50, _ = greedy_agnostic(M, 0.5)
                for gi in range(len(G)):
                    side = float(np.sqrt((G[gi, 2] - G[gi, 0]) * (G[gi, 3] - G[gi, 1])))
                    for key in (size_bin(side), coco_bin(side ** 2)):
                        size_hit[key] += (1, hit50[gi])
                    cls_hit[int(G[gi, 4])] += (1, hit50[gi])
                taken = np.zeros(len(G), bool)
                for p in range(len(P)):
                    same = G[:, 4] == Cc[p]
                    c = np.where((M[p] >= 0.5) & same & ~taken)[0]
                    if len(c): taken[c[np.argmax(M[p, c])]] = True; dec["tp"] += 1
                    elif (M[p][same] >= 0.5).any(): dec["duplicate"] += 1
                    elif (M[p][~same] >= 0.5).any(): dec["cls_err"] += 1
                    elif same.any() and M[p][same].max() >= 0.1: dec["loc_err"] += 1
                    elif (~same).any() and M[p][~same].max() >= 0.1: dec["cls_loc_err"] += 1
                    else: dec["background"] += 1
                near = (M >= 0.1).any(0) if len(P) else np.zeros(len(G), bool)
                missed_near += int((~taken & near).sum()); missed_far += int((~taken & ~near).sum())
            R[arm] = {"n_gt": ngt, "n_families": len(fams), "decomp_counts": dec,
                      "decomp_per100gt": {k: 100 * v / ngt for k, v in dec.items()},
                      "missed_near_pct_gt": 100 * missed_near / ngt, "missed_far_pct_gt": 100 * missed_far / ngt,
                      "recall_vs_iou": {str(t): 100 * rec_tau[j] / ngt for j, t in enumerate(TAUS)},
                      "recall50_by_size": {k: 100 * v[1] / max(v[0], 1) for k, v in size_hit.items()},
                      "n_by_size": {k: int(v[0]) for k, v in size_hit.items()},
                      "recall50_by_class": {NAMES[k]: 100 * v[1] / max(v[0], 1) for k, v in cls_hit.items()},
                      "n_by_class": {NAMES[k]: int(v[0]) for k, v in cls_hit.items()}}
            if arm == "B":
                R[arm]["operating_curve"] = [{"conf": t, "recall50": 100 * sweep[j, 0] / ngt, "fp_per_family": sweep[j, 1] / len(fams)}
                                             for j, t in enumerate(CONF_SWEEP)]
                cc = np.array(calib); bins = np.linspace(0.05, 1.0, 11); tab = []; ece = 0.0
                for lo, hi in zip(bins[:-1], bins[1:]):
                    msk = (cc[:, 0] >= lo) & (cc[:, 0] < hi if hi < 1 else cc[:, 0] <= hi)
                    if msk.sum():
                        mc, pr = cc[msk, 0].mean(), cc[msk, 1].mean(); ece += msk.sum() / len(cc) * abs(mc - pr)
                        tab.append({"bin": [round(lo, 3), round(hi, 3)], "n": int(msk.sum()), "mean_conf": float(mc), "precision": float(pr)})
                R[arm]["calibration"] = {"D_ECE": float(ece), "table": tab, "n_dets": int(len(cc))}
        per[(m, s)] = R
    print("  %-7s done %.0fs" % (m, time.time() - t0), flush=True)

# ---------------- 3b. AP per IoU threshold (pycocotools precision array) ----------------
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
with contextlib.redirect_stdout(io.StringIO()):
    cocoGt = COCO(str(P9 / "coco_gt_union.json"))
for m in MODELS:
    for s in SEEDS:
        z = np.load(B / SRC[m] / "dets" / ("%s__seed%d__step%d.npz" % (m, s, STEP)), allow_pickle=True)
        iid, xyxy, cf, cl = top100(z["image_id"], z["xyxy"], z["conf"], z["cls"].astype(np.int64))
        dt = [{"image_id": int(i), "category_id": int(c), "bbox": [float(a), float(b), float(c2 - a), float(e - b)], "score": float(sc)}
              for i, (a, b, c2, e), sc, c in zip(iid, xyxy, cf, cl)]
        with contextlib.redirect_stdout(io.StringIO()):
            cocoDt = cocoGt.loadRes(dt)
            for arm in ("A", "B"):
                E = COCOeval(cocoGt, cocoDt, "bbox"); E.params.maxDets = [1, 10, 100]; E.params.imgIds = ARM[arm]
                E.evaluate(); E.accumulate()
                pr = E.eval["precision"][:, :, :, 0, 2]
                per[(m, s)][arm]["AP_vs_iou"] = {str(t): float(np.mean(pr[j][pr[j] > -1])) * 100 for j, t in enumerate(TAUS)}
    print("  AP-vs-IoU %-7s done %.0fs" % (m, time.time() - t0), flush=True)

# ---------------- aggregate ----------------
def mean_over(m, f):
    v = [f(per[(m, s)]) for s in SEEDS]; return float(np.mean(v)), float(np.std(v, ddof=1))
agg = {}
for m in MODELS:
    e = {}
    for arm in ("A", "B"):
        r0 = per[(m, 0)][arm]
        e[arm] = {"decomp_per100gt": {t: mean_over(m, lambda r: r[arm]["decomp_per100gt"][t]) for t in TYPES},
                  "missed_near_pct_gt": mean_over(m, lambda r: r[arm]["missed_near_pct_gt"]),
                  "missed_far_pct_gt": mean_over(m, lambda r: r[arm]["missed_far_pct_gt"]),
                  "recall_vs_iou": {t: mean_over(m, lambda r: r[arm]["recall_vs_iou"][t]) for t in r0["recall_vs_iou"]},
                  "AP_vs_iou": {t: mean_over(m, lambda r: r[arm]["AP_vs_iou"][t]) for t in r0["AP_vs_iou"]},
                  "recall50_by_size": {k: mean_over(m, lambda r: r[arm]["recall50_by_size"][k]) for k in r0["recall50_by_size"]},
                  "n_by_size": r0["n_by_size"],
                  "recall50_by_class": {k: mean_over(m, lambda r: r[arm]["recall50_by_class"][k]) for k in r0["recall50_by_class"]},
                  "n_by_class": r0["n_by_class"], "AP_by_class": {}}
        for c in range(8):
            vals = [ROWS[(m, s)][arm]["per_class_AP"].get(str(c)) for s in SEEDS]
            if all(v is not None for v in vals):
                e[arm]["AP_by_class"][NAMES[c]] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    oc = [per[(m, s)]["B"]["operating_curve"] for s in SEEDS]
    e["B"]["operating_curve"] = [{"conf": oc[0][j]["conf"], "recall50": float(np.mean([o[j]["recall50"] for o in oc])),
                                  "fp_per_family": float(np.mean([o[j]["fp_per_family"] for o in oc]))} for j in range(len(CONF_SWEEP))]
    def at_fp(target):
        vals = []
        for o in oc:
            fp = np.array([p["fp_per_family"] for p in o])[::-1]; rc = np.array([p["recall50"] for p in o])[::-1]
            vals.append(float(np.interp(target, fp, rc)) if fp.min() <= target <= fp.max() else np.nan)
        return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else None
    e["B"]["recall50_at_fp_per_family"] = {str(t): at_fp(t) for t in (2, 3, 4)}
    e["B"]["D_ECE"] = mean_over(m, lambda r: r["B"]["calibration"]["D_ECE"])
    e["B"]["reliability_seed0"] = per[(m, 0)]["B"]["calibration"]["table"]
    agg[m] = e
out["per_model"] = agg

# intervention deltas for mechanism endpoints (paired over seeds)
mech = {}
for name, a_, b_ in INTERV:
    d = {}
    for arm in ("A", "B"):
        d[arm] = {"decomp_per100gt": {t: paired([per[(a_, s)][arm]["decomp_per100gt"][t] - per[(b_, s)][arm]["decomp_per100gt"][t] for s in SEEDS]) for t in TYPES},
                  "missed_near_pct_gt": paired([per[(a_, s)][arm]["missed_near_pct_gt"] - per[(b_, s)][arm]["missed_near_pct_gt"] for s in SEEDS]),
                  "missed_far_pct_gt": paired([per[(a_, s)][arm]["missed_far_pct_gt"] - per[(b_, s)][arm]["missed_far_pct_gt"] for s in SEEDS]),
                  "recall_vs_iou": {t: paired([per[(a_, s)][arm]["recall_vs_iou"][t] - per[(b_, s)][arm]["recall_vs_iou"][t] for s in SEEDS]) for t in per[(a_, 0)][arm]["recall_vs_iou"]},
                  "AP_vs_iou": {t: paired([per[(a_, s)][arm]["AP_vs_iou"][t] - per[(b_, s)][arm]["AP_vs_iou"][t] for s in SEEDS]) for t in per[(a_, 0)][arm]["AP_vs_iou"]},
                  "recall50_by_size": {k: paired([per[(a_, s)][arm]["recall50_by_size"][k] - per[(b_, s)][arm]["recall50_by_size"][k] for s in SEEDS]) for k in per[(a_, 0)][arm]["recall50_by_size"]},
                  "recall50_by_class": {k: paired([per[(a_, s)][arm]["recall50_by_class"][k] - per[(b_, s)][arm]["recall50_by_class"][k] for s in SEEDS]) for k in per[(a_, 0)][arm]["recall50_by_class"]}}
        pc = {}
        for c in range(8):
            va = [ROWS[(a_, s)][arm]["per_class_AP"].get(str(c)) for s in SEEDS]; vb = [ROWS[(b_, s)][arm]["per_class_AP"].get(str(c)) for s in SEEDS]
            if all(v is not None for v in va + vb): pc[NAMES[c]] = paired(np.array(va) - np.array(vb))
        d[arm]["AP_by_class"] = pc
    d["B"]["D_ECE"] = paired([per[(a_, s)]["B"]["calibration"]["D_ECE"] - per[(b_, s)]["B"]["calibration"]["D_ECE"] for s in SEEDS])
    mech[name] = d
out["intervention_mechanism_deltas"] = mech
out["seed_level"] = {m: {arm: {k: [round(ap(m, s, arm, k), 3) for s in SEEDS] for k in ("AP", "AP75", "loc_recall_iou50")} for arm in ("A", "B")} for m in MODELS}
json.dump(out, open(OUT / "results.json", "w"), indent=1)
print("results written %.0fs" % (time.time() - t0), flush=True)

# ---------------- figures ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LBL = {"w100": "1.00x scratch", "w075": "0.75x", "w050": "0.50x", "w035": "0.35x", "lpwa50": "LPWA-50",
       "kd075": "0.75x distilled", "r512": "1.00x pretrained"}
COL = {"w100": "#1f4e79", "w075": "#4a7fb0", "w050": "#7aa6cf", "w035": "#a9c8e6", "lpwa50": "#b5651d", "kd075": "#2e8b57", "r512": "#7b3f99"}
mAP = lambda m, arm: np.mean([ap(m, s, arm) for s in SEEDS])
# fig 1 distortion
fig, ax = plt.subplots(figsize=(8, 4)); y = np.arange(len(dist))
ax.barh(y - 0.2, [d["delta_A"]["mean"] for d in dist], 0.4, label="exposed (arm A)", color="#c0504d")
ax.barh(y + 0.2, [d["delta_B"]["mean"] for d in dist], 0.4, label="independent (arm B)", color="#1f4e79")
ax.set_yticks(y); ax.set_yticklabels([d["intervention"] for d in dist]); ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Effect on AP50-95 (points)"); ax.legend(); ax.set_title("Same intervention, two evaluation protocols")
fig.tight_layout(); fig.savefig(OUT / "fig_distortion.png", dpi=150); plt.close(fig)
# fig 6 pareto
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, xkey, xl in ((axs[0], PARAMS, "Parameters (M)"), (axs[1], GFLOPS, "GFLOPs @512")):
    for arm, ls, mk in (("B", "-", "o"), ("A", "--", "s")):
        ws = ["w035", "w050", "w075", "w100"]
        ax.plot([xkey[w] / (1e6 if xkey is PARAMS else 1) for w in ws], [mAP(w, arm) for w in ws], ls, marker=mk, color="#1f4e79" if arm == "B" else "#c0504d",
                label="uniform width, arm %s (%s)" % (arm, "independent" if arm == "B" else "exposed"))
        for mm in ("lpwa50", "kd075", "r512"):
            ax.scatter(xkey[mm] / (1e6 if xkey is PARAMS else 1), mAP(mm, arm), color=COL[mm], marker=mk, zorder=3,
                       label=LBL[mm] if arm == "B" else None)
    ax.set_xlabel(xl); ax.set_ylabel("AP50-95"); ax.grid(alpha=0.3)
axs[0].legend(fontsize=7); fig.suptitle("Accuracy vs size under exposed and independent evaluation"); fig.tight_layout()
fig.savefig(OUT / "fig_pareto.png", dpi=150); plt.close(fig)
# fig 8 paired seeds
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, k, t in zip(axs, ("AP", "loc_recall_iou50", "AP75"), ("AP50-95", "Loc. recall @0.50", "AP75")):
    for s in SEEDS:
        ax.plot([0, 1], [ap("w075", s, "B", k), ap("kd075", s, "B", k)], "-o", color="#2e8b57", alpha=0.8)
        ax.plot([2, 3], [ap("w050", s, "B", k), ap("lpwa50", s, "B", k)], "-o", color="#b5651d", alpha=0.8)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["0.75x", "0.75x KD", "0.50x", "LPWA"], fontsize=8); ax.set_title(t + " (arm B)"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_seeds.png", dpi=150); plt.close(fig)
# fig 3 IoU sweep
fig, axs = plt.subplots(1, 2, figsize=(10, 4)); xs = [float(t) for t in TAUS]
for m in ("w100", "w075", "kd075", "r512"):
    axs[0].plot(xs, [agg[m]["B"]["AP_vs_iou"][str(t)][0] for t in TAUS], "-o", color=COL[m], label=LBL[m])
    axs[1].plot(xs, [agg[m]["B"]["recall_vs_iou"][str(t)][0] for t in TAUS], "-o", color=COL[m], label=LBL[m])
axs[0].set_title("Six-class AP at each IoU threshold (arm B)"); axs[1].set_title("Loc. recall vs IoU threshold, conf>=0.25 (arm B)")
for ax in axs: ax.set_xlabel("IoU threshold"); ax.grid(alpha=0.3)
axs[0].legend(); fig.tight_layout(); fig.savefig(OUT / "fig_iou.png", dpi=150); plt.close(fig)
# fig 4 size
fig, ax = plt.subplots(figsize=(7, 4)); bins = ["small", "medium", "large"]; wd = 0.12
for j, m in enumerate(["w035", "w050", "w075", "w100", "kd075", "r512"]):
    ax.bar(np.arange(3) + (j - 2.5) * wd, [agg[m]["B"]["recall50_by_size"][b][0] for b in bins], wd, color=COL[m], label=LBL[m])
ax.set_xticks(range(3)); ax.set_xticklabels(["small\n(<%.0f px)" % TERT[0], "medium", "large\n(>=%.0f px)" % TERT[1]])
ax.set_ylabel("Loc. recall @0.50 (%)"); ax.set_title("Recall by lesion size, arm B (tertiles of sqrt(area))"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_size.png", dpi=150); plt.close(fig)
# fig 10 operating curves
fig, ax = plt.subplots(figsize=(6, 4))
for m in ("w100", "w075", "kd075", "r512"):
    c = agg[m]["B"]["operating_curve"]; ax.plot([p["fp_per_family"] for p in c], [p["recall50"] for p in c], "-", color=COL[m], label=LBL[m])
ax.set_xlabel("False positives per family"); ax.set_ylabel("Loc. recall @0.50 (%)"); ax.set_title("Operating curves (arm B, conf 0.05-0.95)")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_operating.png", dpi=150); plt.close(fig)
# fig 11 calibration
fig, ax = plt.subplots(figsize=(5, 4)); ax.plot([0, 1], [0, 1], "k:", lw=0.8)
for m in ("w100", "w075", "kd075", "r512"):
    t = agg[m]["B"]["reliability_seed0"]; ax.plot([b["mean_conf"] for b in t], [b["precision"] for b in t], "-o", color=COL[m], label="%s (D-ECE %.3f)" % (LBL[m], agg[m]["B"]["D_ECE"][0]))
ax.set_xlabel("Confidence"); ax.set_ylabel("Precision (TP share)"); ax.set_title("Reliability, arm B, seed 0"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_calibration.png", dpi=150); plt.close(fig)
print("PHDONE %.0fs" % (time.time() - t0))
