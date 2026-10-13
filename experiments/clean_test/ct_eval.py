"""ONE-TIME clean-test evaluation. Frozen by clean_test/prereg.json; writes the full table in one pass.

Rows (all 5 seeds each, fixed 24,570-update checkpoint, imgsz 512):
  kd075  distilled 0.75x (PRIMARY)      runs_p17/seed{s}
  w075   ordinary 0.75x                 runs_p15/w075/seed{s}
  w100   from-scratch 1.00x             runs_p15/w100/seed{s}
  r512   pretrained 1.00x               runs_p4/r512/seed{s}
Settings identical to Phases 9-17: conf 0.001, NMS IoU 0.7, max_det 300, top-100 per image, no TTA
(augment=False), pycocotools AP@[.50:.95] maxDets 100 over 8 categories; decomposition at conf 0.25, greedy
one-to-one class-agnostic match at IoU 0.50 / strict 0.75; family = split-manifest cluster.
NO threshold search, NO class-specific setting, NO checkpoint or model selection: every row is computed with the
same constants and reported. Nothing is printed or written until all 20 checkpoints are scored.
Refuses to run if results exist, if the run lock exists, or if any checkpoint / test-set hash differs from prereg.
"""
import contextlib, copy, csv, hashlib, io, json, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as st

B = Path(r"C:\Users\absamad\cs"); R = B / "tomato-adcf-yolo"; OUT = B / "clean_test"; D = R / "data" / "tomato_village"
CONF_AP, IOU_NMS, MAXDET, IMGSZ, TOPK = 0.001, 0.7, 300, 512, 100
CONF, IOU_MATCH, IOU_STRICT = 0.25, 0.50, 0.75
SEEDS, STEP, NBOOT = [0, 1, 2, 3, 4], 24570, 1000
MODELS = ["kd075", "w075", "w100", "r512"]
REFS = ["w075", "w100", "r512"]
METRICS = ["AP", "AP50", "AP75", "AR100", "loc_recall_iou50", "loc_recall_iou75", "cond_class_correct", "fp_per_family"]
RNG = np.random.default_rng(20261013)
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(np.float32)


def main():
    pre = json.loads((OUT / "prereg.json").read_text())
    if (OUT / "results.json").exists(): sys.exit("REFUSING: clean-test results already exist")
    if (OUT / "RUN_LOCK").exists(): sys.exit("REFUSING: run lock exists (a run was started before)")
    vr = json.loads((OUT / "verify_report.json").read_text())
    assert vr["all_pass"] and vr["test_set_sha256"] == pre["test_set"]["sha256"], "test set changed or verification failed"
    for m in MODELS:
        for s in SEEDS:
            p = Path(pre["checkpoints"][m][str(s)]["path"])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == pre["checkpoints"][m][str(s)]["sha256"], "hash " + str(p)
    (OUT / "RUN_LOCK").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    t_start = time.time()

    from PIL import Image
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = {0: "Early Blight", 1: "Healthy", 2: "Late Blight", 3: "Leaf Miner", 4: "Magnesium Deficiency",
             5: "Nitrogen Deficiency", 6: "Potassium Deficiency", 7: "Spotted Wilt Virus"}
    clus = {norm(r["original_id"]): r["cluster"] for r in csv.DictReader(open(R / "splits" / "split_manifest.csv", encoding="utf-8"))
            if r["split"] == "test"}
    imgs = sorted(p for p in (D / "images" / "test").iterdir() if p.is_file())
    g = {"images": [], "annotations": [], "categories": [{"id": k, "name": v} for k, v in names.items()]}
    FAM, gt_by_img, aid = {}, defaultdict(list), 1
    for i, p in enumerate(imgs, 1):
        w, h = Image.open(p).size                      # header only (dimensions for box de-normalisation)
        g["images"].append({"id": i, "file_name": p.name, "width": w, "height": h}); FAM[i] = clus[norm(p.stem)]
        for ln in (D / "labels" / "test" / (p.stem + ".txt")).read_text().splitlines():
            q = ln.split()
            if len(q) < 5: continue
            c, xc, yc, bw, bh = int(float(q[0])), *map(float, q[1:5])
            x, y, ww, hh = (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h
            g["annotations"].append({"id": aid, "image_id": i, "category_id": c, "bbox": [x, y, ww, hh],
                                     "area": ww * hh, "iscrowd": 0}); aid += 1
            gt_by_img[i].append((x, y, x + ww, y + hh, c))
    cocoGt = COCO(); cocoGt.dataset = g; cocoGt.createIndex()
    IDS = [im["id"] for im in g["images"]]; idof = {im["file_name"]: im["id"] for im in g["images"]}
    paths = [str(p) for p in imgs]

    rows, cache = {}, {}
    for m in MODELS:
        for s in SEEDS:
            wpath = pre["checkpoints"][m][str(s)]["path"]
            model = YOLO(wpath); dets = []
            for k in range(0, len(paths), 32):
                res = model.predict(paths[k:k + 32], imgsz=IMGSZ, conf=CONF_AP, iou=IOU_NMS, max_det=MAXDET,
                                    augment=False, device=0, verbose=False)
                for pth, rr in zip(paths[k:k + 32], res):
                    b = rr.boxes
                    if b is None or len(b) == 0: continue
                    xy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
                    for j in np.argsort(-cf)[:TOPK]:
                        dets.append((idof[Path(pth).name], xy[j], float(cf[j]), int(cl[j])))
            dt = [{"image_id": i, "category_id": c, "bbox": [float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])],
                   "score": sc} for i, x, sc, c in dets]
            with contextlib.redirect_stdout(io.StringIO()):
                cocoDt = cocoGt.loadRes(dt)
            E = COCOeval(cocoGt, cocoDt, "bbox"); E.params.maxDets = [1, 10, 100]; E.params.imgIds = IDS
            with contextlib.redirect_stdout(io.StringIO()):     # pycocotools prints per-run AP; keep silent
                E.evaluate(); E.accumulate(); E.summarize()
            prec = E.eval["precision"]; pc = {}
            for ci, c in enumerate(E.params.catIds):
                pp = prec[:, :, ci, 0, 2]; pp = pp[pp > -1]; pc[names[c]] = float(pp.mean()) * 100 if pp.size else None
            rec = {"AP": E.stats[0] * 100, "AP50": E.stats[1] * 100, "AP75": E.stats[2] * 100, "AR100": E.stats[8] * 100,
                   "APs": E.stats[3] * 100, "per_class_AP": pc, "n_dets": len(dt)}
            # per-detection match state for the family bootstrap (Phase 9 layout, in memory)
            nI, nA = len(E.params.imgIds), len(E.params.areaRng); mc = []
            for ci, c in enumerate(E.params.catIds):
                for k_, im in enumerate(E.params.imgIds):
                    e = E.evalImgs[ci * nA * nI + k_]
                    if e is None: continue
                    sc = np.asarray(e["dtScores"], dtype=np.float32).ravel()[:100]; kk = sc.shape[0]
                    mc.append((c, im, sc, (np.asarray(e["dtMatches"]).reshape(10, -1)[:, :kk] > 0),
                               np.asarray(e["dtIgnore"]).reshape(10, -1)[:, :kk] == 0,
                               int((np.asarray(e["gtIgnore"]).ravel() == 0).sum())))
            cache[(m, s)] = mc
            by_img = defaultdict(list)
            for i, x, sc, c in dets:
                if sc >= CONF: by_img[i].append((sc, x, c))
            pf = defaultdict(lambda: np.zeros(5, dtype=np.int64))   # gt, hit50, hit75, class-correct, fp
            for i in IDS:
                G = np.array(gt_by_img[i], dtype=np.float32).reshape(-1, 5); P = sorted(by_img.get(i, []), key=lambda t: -t[0])
                v = pf[FAM[i]]; v[0] += len(G)
                if len(G) == 0: v[4] += len(P); continue
                if len(P) == 0: continue
                M = iou_mat(np.array([p[1] for p in P], dtype=np.float32), G[:, :4]); taken = np.zeros(len(G), bool)
                for pi in range(len(P)):
                    cand = np.where((M[pi] >= IOU_MATCH) & (~taken))[0]
                    if len(cand) == 0: v[4] += 1; continue
                    gi = cand[np.argmax(M[pi, cand])]; taken[gi] = True; v[1] += 1
                    if M[pi, gi] >= IOU_STRICT: v[2] += 1
                    if P[pi][2] == int(G[gi, 4]): v[3] += 1
            tot = np.stack(list(pf.values())).sum(0)
            rec.update(loc_recall_iou50=100 * tot[1] / max(tot[0], 1), loc_recall_iou75=100 * tot[2] / max(tot[0], 1),
                       cond_class_correct=100 * tot[3] / max(tot[1], 1), fp_per_family=float(tot[4] / len(pf)),
                       family_counts={k: v.tolist() for k, v in pf.items()})
            rows[(m, s)] = rec

    # ---------------- aggregate (no selection: every row reported) ----------------
    def paired(d):
        d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(d.size)
        ci = st.t.interval(0.95, d.size - 1, loc=d.mean(), scale=se)
        return {"per_seed": [round(float(x), 4) for x in d], "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                "ci95": [float(ci[0]), float(ci[1])], "n_positive": int((d > 0).sum())}
    table = {m: {k: {"seeds": [round(rows[(m, s)][k], 4) for s in SEEDS], "mean": float(np.mean([rows[(m, s)][k] for s in SEEDS])),
                     "sd": float(np.std([rows[(m, s)][k] for s in SEEDS], ddof=1))} for k in METRICS + ["APs"]} for m in MODELS}
    deltas = {r: {k: paired([rows[("kd075", s)][k] - rows[(r, s)][k] for s in SEEDS]) for k in METRICS} for r in REFS}

    # ---------------- family (cluster) bootstrap of paired seed-mean deltas ----------------
    fams = sorted(set(FAM.values())); fpos = defaultdict(list)
    for j, i in enumerate(IDS): fpos[FAM[i]].append(j)
    fpos = [np.array(fpos[f]) for f in fams]; POS = {i: j for j, i in enumerate(IDS)}; RECT = np.linspace(0, 1, 101)
    prepd = {}
    for key, mc in cache.items():
        per = defaultdict(lambda: [[], [], [], [], np.zeros(len(IDS), np.int64)])
        for c, im, sc, tp, ni, ng in mc:
            e = per[c]; e[4][POS[im]] += ng
            if sc.size: e[0].append(sc); e[1].append(np.full(sc.size, POS[im])); e[2].append(tp); e[3].append(ni)
        out_ = {}
        for c, (scl, il, tpl, nil, npig) in per.items():
            if scl:
                sc = np.concatenate(scl); o = np.argsort(-sc, kind="mergesort")
                out_[c] = (np.concatenate(il)[o], np.concatenate(tpl, 1)[:, o].astype(float), np.concatenate(nil, 1)[:, o], npig)
            else:
                out_[c] = (np.zeros(0, int), np.zeros((10, 0)), np.zeros((10, 0), bool), npig)
        prepd[key] = out_
    def ap_w(key, w):
        aps = []
        for img, tpm, notig, npig in prepd[key].values():
            n_ = float((npig * w).sum())
            if n_ <= 0: continue
            if img.size == 0: aps.append(0.0); continue
            ww = w[img].astype(float); pt = np.empty(10)
            for t in range(10):
                kp = notig[t] & (ww > 0)
                if not kp.any(): pt[t] = 0; continue
                ctp = np.cumsum(tpm[t][kp] * ww[kp]); cfp = np.cumsum((1 - tpm[t][kp]) * ww[kp])
                pr = np.maximum.accumulate((ctp / (ctp + cfp + np.spacing(1)))[::-1])[::-1]
                ins = np.searchsorted(ctp / n_, RECT, side="left"); q = np.zeros(RECT.size); ok = ins < pr.size
                q[ok] = pr[ins[ok]]; pt[t] = q.mean()
            aps.append(float(pt.mean()))
        return float(np.mean(aps)) * 100 if aps else 0.0
    w1 = np.ones(len(IDS), np.int64)
    val = max(abs(ap_w(k, w1) - rows[k]["AP"]) for k in prepd)
    assert val < 0.05, "bootstrap scorer does not reproduce pycocotools: %g" % val
    BS = {r: np.empty(NBOOT) for r in REFS}; BL = {r: np.empty(NBOOT) for r in REFS}
    for b in range(NBOOT):
        fi = RNG.integers(0, len(fams), len(fams)); w = np.zeros(len(IDS), np.int64)
        for k in fi: np.add.at(w, fpos[k], 1)
        apv = {key: ap_w(key, w) for key in prepd}
        lr = {}
        for key in prepd:
            fc = rows[key]["family_counts"]; M = np.stack([fc[fams[k]] for k in fi]).sum(0); lr[key] = 100 * M[1] / max(M[0], 1)
        for r in REFS:
            BS[r][b] = np.mean([apv[("kd075", s)] - apv[(r, s)] for s in SEEDS])
            BL[r][b] = np.mean([lr[("kd075", s)] - lr[(r, s)] for s in SEEDS])
    sm = lambda x: {"mean": float(x.mean()), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]}
    boot = {r: {"AP": sm(BS[r]), "loc_recall_iou50": sm(BL[r])} for r in REFS}

    res = {"prereg_sha256": hashlib.sha256((OUT / "prereg.json").read_bytes()).hexdigest(),
           "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - t_start) / 60, 1),
           "n_images": len(IDS), "n_boxes": len(g["annotations"]), "n_families": len(fams),
           "settings": {"conf": CONF_AP, "nms_iou": IOU_NMS, "max_det": MAXDET, "imgsz": IMGSZ, "top_per_image": TOPK,
                        "tta": False, "decomp_conf": CONF, "match_iou": IOU_MATCH, "strict_iou": IOU_STRICT, "step": STEP},
           "table": table, "paired_deltas_kd075_minus_ref": deltas, "family_bootstrap": boot,
           "bootstrap_scorer_validation_max_abs_diff": val,
           "per_run": {"%s_seed%d" % k: {kk: vv for kk, vv in v.items() if kk != "family_counts"} for k, v in rows.items()}}
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    lab = {"kd075": "Distilled 0.75x (primary)", "w075": "Ordinary 0.75x", "w100": "From-scratch 1.00x", "r512": "Pretrained 1.00x"}
    with open(R / "results" / "clean_test_table.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["model", "seed"] + METRICS)
        for (m, s), v in rows.items(): wr.writerow([m, s] + [round(v[k], 4) for k in METRICS])
    md = ["# Clean-test results (one-time, %s)" % res["evaluated_at"], "", "| model | " + " | ".join(METRICS) + " |",
          "|---|" + "---|" * len(METRICS)]
    md += ["| %s | " % lab[m] + " | ".join("%.2f ± %.2f" % (table[m][k]["mean"], table[m][k]["sd"]) for k in METRICS) + " |" for m in MODELS]
    md += ["", "Paired Δ (distilled − reference), mean [95% CI over seeds]; family bootstrap for AP:", ""]
    md += ["- vs %s: AP %+.2f [%.2f, %.2f] (bootstrap [%.2f, %.2f]); locR50 %+.2f" % (lab[r], deltas[r]["AP"]["mean"], *deltas[r]["AP"]["ci95"],
           *boot[r]["AP"]["ci95"], deltas[r]["loc_recall_iou50"]["mean"]) for r in REFS]
    (R / "results" / "clean_test_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md)); print("CLEANTESTDONE")


if __name__ == "__main__":
    main()
