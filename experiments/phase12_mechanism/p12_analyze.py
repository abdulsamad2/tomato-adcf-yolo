"""Phase 12 Stage 0/1 + capacity decomposition. Applies the rules frozen in
p12_matching_rules_frozen.md. No training. No threshold or rule changed here.
"""
import copy, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as sstats

P9 = Path(r"C:\Users\absamad\cs\phase9")
P12 = Path(r"C:\Users\absamad\cs\phase12")
IOU_MATCH, IOU_STRICT, CONF = 0.50, 0.75, 0.25
NBOOT = 1000
RNG = np.random.default_rng(20261008)
TRIO = [("runs_p3__yolo11n", "n"), ("runs_p2b__D", "s"), ("runs_p3__yolo11m", "m")]
SEEDS = [0, 1, 2, 3, 4]


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(np.float32)


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    g = json.loads((P9 / "coco_gt_union.json").read_text())
    arms = json.loads((P9 / "arm_image_ids.json").read_text())
    ARM = {k: set(arms["arm_image_ids"][k]) for k in ("A", "B")}
    FAM = {int(k): v for k, v in arms["family_of_image"].items()}
    assert set(FAM) >= (ARM["A"] | ARM["B"]), "family map incomplete"

    gt_by_img = defaultdict(list)
    for a in g["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img[a["image_id"]].append((x, y, x + w, y + h, a["category_id"]))
    imgs_of = {k: sorted(ARM[k]) for k in ("A", "B")}

    # AMBIG flag: GT overlapping another GT of a DIFFERENT class at IoU >= IOU_MATCH
    ambig = {}
    for arm in ("A", "B"):
        n = 0
        for i in imgs_of[arm]:
            G = np.array(gt_by_img[i], dtype=np.float32)
            if len(G) < 2: continue
            M = iou_mat(G[:, :4], G[:, :4]); np.fill_diagonal(M, 0)
            diff = G[:, 4][:, None] != G[:, 4][None, :]
            n += int(((M >= IOU_MATCH) & diff).any(1).sum())
        ambig[arm] = n

    cocoGt = COCO(str(P9 / "coco_gt_union.json"))
    ga = copy.deepcopy(g)
    for a in ga["annotations"]: a["category_id"] = 1
    ga["categories"] = [{"id": 1, "name": "labelled foliar region"}]
    cocoGtA = COCO(); cocoGtA.dataset = ga; cocoGtA.createIndex()

    out = {"rules": {"IOU_MATCH": IOU_MATCH, "IOU_STRICT": IOU_STRICT, "CONF": CONF,
                     "n_boot": NBOOT, "unit": "source family"},
           "ambiguous_gt_boxes": ambig, "checkpoints": {}}

    REF = json.loads((P12 / "p9_reference.json").read_text())
    TOL = 0.05  # AP points; generous for GPU/batch nondeterminism, tight enough to catch a settings mismatch
    ctrl, trunc_stats = [], []

    def top100(iid, xyxy, cf, cl):
        """Phase 9 rule: keep only the 100 highest-scoring detections per image."""
        order = np.lexsort((-cf, iid))
        iid2, cf2 = iid[order], cf[order]
        keep = np.zeros(len(order), dtype=bool)
        start = 0
        bounds = np.flatnonzero(np.diff(iid2)) + 1
        for end in list(bounds) + [len(iid2)]:
            keep[start:min(start + 100, end)] = True
            start = end
        sel = order[keep]
        return iid[sel], xyxy[sel], cf[sel], cl[sel]

    for f in sorted((P12 / "dets").glob("*.npz")):
        tag = f.stem
        d = np.load(f, allow_pickle=True)
        iid, xyxy, cf, cl = d["image_id"], d["xyxy"], d["conf"], d["cls"]
        n_raw = len(iid)
        iid, xyxy, cf, cl = top100(iid, xyxy, cf, cl)
        trunc_stats.append((tag, n_raw, int(len(iid))))
        rec = {"n_dets": int(len(iid)), "imgsz": int(d["imgsz"])}

        # ---- class-agnostic AP (all detections, no conf threshold) ----
        dt = [{"image_id": int(i), "category_id": 1,
               "bbox": [float(a), float(b), float(c - a), float(e - b)], "score": float(s)}
              for i, (a, b, c, e), s in zip(iid, xyxy, cf)]
        cocoDtA = cocoGtA.loadRes(dt)
        for arm in ("A", "B"):
            E = COCOeval(cocoGtA, cocoDtA, "bbox"); E.params.maxDets = [1, 10, 100]
            E.params.imgIds = imgs_of[arm]
            E.evaluate(); E.accumulate(); E.summarize()
            rec["ca_AP_" + arm] = float(E.stats[0]); rec["ca_AP50_" + arm] = float(E.stats[1])
            rec["ca_AP75_" + arm] = float(E.stats[2]); rec["ca_AR100_" + arm] = float(E.stats[8])

        dt6 = [{"image_id": int(i), "category_id": int(c),
                "bbox": [float(a), float(b), float(c2 - a), float(e - b)], "score": float(sc)}
               for i, (a, b, c2, e), sc, c in zip(iid, xyxy, cf, cl)]
        cocoDt6 = cocoGt.loadRes(dt6)
        for arm in ("A", "B"):
            E = COCOeval(cocoGt, cocoDt6, "bbox"); E.params.maxDets = [1, 10, 100]
            E.params.imgIds = imgs_of[arm]
            E.evaluate(); E.accumulate(); E.summarize()
            rec["sc_AP_" + arm] = float(E.stats[0]) * 100
            rec["sc_AP50_" + arm] = float(E.stats[1]) * 100
            k = "%s|%s" % (tag, arm)
            ctrl.append((k, REF[k]["AP"], rec["sc_AP_" + arm],
                         abs(REF[k]["AP"] - rec["sc_AP_" + arm]), REF[k]["n_dets"]))

        # ---- GT-centric decomposition at CONF, greedy one-to-one, class-agnostic match ----
        keep = cf >= CONF
        by_img = defaultdict(list)
        for i, bx, s, c in zip(iid[keep], xyxy[keep], cf[keep], cl[keep]):
            by_img[int(i)].append((float(s), bx, int(c)))
        for arm in ("A", "B"):
            per_fam = defaultdict(lambda: np.zeros(7, dtype=np.int64))  # gt,miss,cls_err,loc_q,tp_strict,fp,imgs
            for i in imgs_of[arm]:
                G = np.array(gt_by_img[i], dtype=np.float32)
                P = sorted(by_img.get(i, []), key=lambda t: -t[0])
                fam = FAM[i]; v = per_fam[fam]; v[6] += 1; v[0] += len(G)
                if len(G) == 0:
                    v[5] += len(P); continue
                if len(P) == 0:
                    v[1] += len(G); continue
                Pb = np.array([p[1] for p in P], dtype=np.float32)
                Pc = np.array([p[2] for p in P], dtype=np.int64)
                M = iou_mat(Pb, G[:, :4])
                taken = np.zeros(len(G), dtype=bool)
                gt_out = np.full(len(G), -1, dtype=np.int64)   # 0 miss 1 cls_err 2 loc_q 3 tp_strict
                for pi in range(len(P)):
                    cand = np.where((M[pi] >= IOU_MATCH) & (~taken))[0]
                    if len(cand) == 0:
                        v[5] += 1; continue
                    gi = cand[np.argmax(M[pi, cand])]
                    taken[gi] = True
                    if Pc[pi] != int(G[gi, 4]):
                        gt_out[gi] = 1
                    else:
                        gt_out[gi] = 3 if M[pi, gi] >= IOU_STRICT else 2
                gt_out[gt_out == -1] = 0
                for code, slot in ((0, 1), (1, 2), (2, 3), (3, 4)):
                    v[slot] += int((gt_out == code).sum())
            fams = sorted(per_fam)
            Ms = np.stack([per_fam[k] for k in fams])
            tot = Ms.sum(0)
            def rates(m):
                gt, miss, cerr, locq, tps, fp, nim = m
                loc = gt - miss
                return (100.0 * loc / max(gt, 1),
                        100.0 * (locq + tps) / max(loc, 1),
                        100.0 * (locq + tps) / max(gt, 1),
                        100.0 * tps / max(gt, 1),
                        fp / max(nim, 1))
            names = ["loc_recall", "cond_class_correct", "full_correct_iou50", "full_correct_iou75", "fp_per_image"]
            pt = rates(tot)
            bs = np.empty((NBOOT, len(names)))
            for b in range(NBOOT):
                s_ = RNG.integers(0, len(fams), len(fams))
                bs[b] = rates(Ms[s_].sum(0))
            for j, nm in enumerate(names):
                rec["%s_%s" % (nm, arm)] = float(pt[j])
                rec["%s_%s_ci" % (nm, arm)] = [float(np.percentile(bs[:, j], 2.5)),
                                               float(np.percentile(bs[:, j], 97.5))]
            rec["counts_" + arm] = {"families": len(fams), "images": int(tot[6]), "gt": int(tot[0]),
                                    "miss": int(tot[1]), "class_err": int(tot[2]),
                                    "loc_quality": int(tot[3]), "tp_strict": int(tot[4]),
                                    "fp": int(tot[5])}
            rec["fp_per_family_" + arm] = float(tot[5] / max(len(fams), 1))
        out["checkpoints"][tag] = rec
        print("  %-28s locR A=%.1f B=%.1f | condC A=%.1f B=%.1f | caAP50 A=%.3f B=%.3f"
              % (tag, rec["loc_recall_A"], rec["loc_recall_B"], rec["cond_class_correct_A"],
                 rec["cond_class_correct_B"], rec["ca_AP50_A"], rec["ca_AP50_B"]), flush=True)

    md = max(c[3] for c in ctrl)
    out["reproduction_control"] = {"pairs": len(ctrl), "max_abs_diff_AP_points": md,
        "mean_abs_diff_AP_points": float(np.mean([c[3] for c in ctrl])), "tolerance": TOL,
        "passed": bool(md <= TOL),
        "worst": sorted(ctrl, key=lambda z: -z[3])[:5],
        "truncation": {"rule": "top 100 detections per image (Phase 9)",
            "n_raw_total": int(sum(t[1] for t in trunc_stats)),
            "n_kept_total": int(sum(t[2] for t in trunc_stats))}}
    print("REPRO max|diff| = %.4f AP pts (tol %.2f) over %d pairs -> %s"
          % (md, TOL, len(ctrl), "PASS" if md <= TOL else "FAIL"), flush=True)
    if md > TOL:
        json.dump(out, open(P12 / "p12_results.json", "w"), indent=1)
        print("STOP_REPRODUCTION_FAILED"); return

    # ---- capacity x exposure, paired across the five seeds, 640 px trio only ----
    COMP = ["ca_AP50", "ca_AP", "loc_recall", "cond_class_correct", "full_correct_iou50",
            "full_correct_iou75", "fp_per_image"]
    cap = {}
    for (k1, n1), (k2, n2) in (((TRIO[0]), (TRIO[1])), ((TRIO[1]), (TRIO[2]))):
        lbl = "%s-%s" % (n2, n1)
        cap[lbl] = {}
        for comp in COMP:
            dA, dB = [], []
            for s in SEEDS:
                r1 = out["checkpoints"]["%s__seed%d" % (k1, s)]
                r2 = out["checkpoints"]["%s__seed%d" % (k2, s)]
                dA.append(r2[comp + "_A"] - r1[comp + "_A"])
                dB.append(r2[comp + "_B"] - r1[comp + "_B"])
            dA, dB = np.array(dA), np.array(dB)
            I = dB - dA
            t, p = sstats.ttest_rel(dB, dA)
            ci = sstats.t.interval(0.95, len(I) - 1, loc=I.mean(), scale=sstats.sem(I)) if I.std(ddof=1) > 0 else (np.nan, np.nan)
            cap[lbl][comp] = {"delta_A": dA.tolist(), "delta_B": dB.tolist(),
                              "delta_A_mean": float(dA.mean()), "delta_A_sd": float(dA.std(ddof=1)),
                              "delta_B_mean": float(dB.mean()), "delta_B_sd": float(dB.std(ddof=1)),
                              "I": I.tolist(), "I_mean": float(I.mean()), "I_sd": float(I.std(ddof=1)),
                              "t": float(t), "p": float(p),
                              "ci95": [float(ci[0]), float(ci[1])],
                              "n_positive": int((I > 0).sum()),
                              "sign_consistent": bool(np.all(I > 0) or np.all(I < 0))}
    out["capacity"] = cap
    json.dump(out, open(P12 / "p12_results.json", "w"), indent=1)
    print("ANALYZEDONE")

if __name__ == "__main__":
    main()
