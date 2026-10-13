"""Phase 13 evaluation: single-class-trained localization vs six-class-trained scored
class-agnostically, under IDENTICAL rules (p12_matching_rules_frozen.md).

Comparator: runs_p4/r512 seed 0 (matched seed) as primary, with the 5-seed spread as context.
Six-class detections are REUSED from phase12/dets (no re-inference). Only the new single-class
checkpoint is run, at the same checkpoint step (18270), imgsz 512, conf 0.001, NMS 0.7,
max_det 300, top-100-per-image truncation.

All AP values on the 0-100 scale. Clean test is not touched.
"""
import copy, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
P9 = Path(r"C:\Users\absamad\cs\phase9")
P12 = Path(r"C:\Users\absamad\cs\phase12")
P13 = Path(r"C:\Users\absamad\cs\phase13"); P13.mkdir(parents=True, exist_ok=True)
CONF_AP, IOU_NMS, MAXDET = 0.001, 0.7, 300
CONF, IOU_MATCH, IOU_STRICT, STEP = 0.25, 0.50, 0.75, 18270
NBOOT = 1000
RNG = np.random.default_rng(20261008)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(np.float32)


def top100(iid, xyxy, cf):
    order = np.lexsort((-cf, iid))
    iid2 = iid[order]
    keep = np.zeros(len(order), dtype=bool)
    start = 0
    for end in list(np.flatnonzero(np.diff(iid2)) + 1) + [len(iid2)]:
        keep[start:min(start + 100, end)] = True
        start = end
    sel = order[keep]
    return iid[sel], xyxy[sel], cf[sel]


def main():
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    g = json.loads((P9 / "coco_gt_union.json").read_text())
    arms = json.loads((P9 / "arm_image_ids.json").read_text())
    ARM = {k: sorted(arms["arm_image_ids"][k]) for k in ("A", "B")}
    FAM = {int(k): v for k, v in arms["family_of_image"].items()}
    gt_by_img = defaultdict(list)
    for a in g["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img[a["image_id"]].append((x, y, x + w, y + h))

    ga = copy.deepcopy(g)
    for a in ga["annotations"]: a["category_id"] = 1
    ga["categories"] = [{"id": 1, "name": "labelled foliar region"}]
    cocoGtA = COCO(); cocoGtA.dataset = ga; cocoGtA.createIndex()

    # ---- single-class detections (the only new inference) ----
    sc_npz = P13 / "dets_single_cls.npz"
    if not sc_npz.exists():
        w = REPO / "runs_p13" / "single_cls" / "seed0" / "weights" / ("ckpt_ep29_step%d.pt" % STEP)
        assert w.exists(), w
        order = [im["file_name"] for im in sorted(g["images"], key=lambda x: x["id"])]
        idof = {im["file_name"]: im["id"] for im in g["images"]}
        IMGDIR = P9 / "arms" / "union" / "images" / "val"
        paths = [str(IMGDIR / f) for f in order]
        model = YOLO(str(w))
        ii, bb, cc = [], [], []
        for i in range(0, len(paths), 32):
            batch = paths[i:i+32]
            res = model.predict(batch, imgsz=512, conf=CONF_AP, iou=IOU_NMS, max_det=MAXDET,
                                device=0, verbose=False)
            for p, rr in zip(batch, res):
                b = rr.boxes
                if b is None or len(b) == 0: continue
                n = len(b)
                ii.append(np.full(n, idof[Path(p).name], dtype=np.int64))
                bb.append(b.xyxy.cpu().numpy().astype(np.float32))
                cc.append(b.conf.cpu().numpy().astype(np.float32))
        np.savez_compressed(sc_npz, image_id=np.concatenate(ii), xyxy=np.concatenate(bb),
                            conf=np.concatenate(cc), weights=str(w))
        print("single-class detections written", flush=True)

    def metrics(iid, xyxy, cf, label):
        iid, xyxy, cf = top100(iid, xyxy, cf)
        rec = {"n_dets": int(len(iid))}
        dt = [{"image_id": int(i), "category_id": 1,
               "bbox": [float(a), float(b), float(c - a), float(e - b)], "score": float(s)}
              for i, (a, b, c, e), s in zip(iid, xyxy, cf)]
        cocoDt = cocoGtA.loadRes(dt)
        for arm in ("A", "B"):
            E = COCOeval(cocoGtA, cocoDt, "bbox"); E.params.maxDets = [1, 10, 100]
            E.params.imgIds = ARM[arm]
            E.evaluate(); E.accumulate(); E.summarize()
            rec["ca_AP_" + arm] = float(E.stats[0]) * 100
            rec["ca_AP50_" + arm] = float(E.stats[1]) * 100
            rec["ca_AP75_" + arm] = float(E.stats[2]) * 100
        keep = cf >= CONF
        by_img = defaultdict(list)
        for i, bx, s in zip(iid[keep], xyxy[keep], cf[keep]):
            by_img[int(i)].append((float(s), bx))
        for arm in ("A", "B"):
            per_fam = defaultdict(lambda: np.zeros(4, dtype=np.int64))  # gt, hit50, hit75, fp
            for i in ARM[arm]:
                G = np.array(gt_by_img[i], dtype=np.float32)
                P = sorted(by_img.get(i, []), key=lambda t: -t[0])
                v = per_fam[FAM[i]]; v[0] += len(G)
                if len(G) == 0:
                    v[3] += len(P); continue
                if len(P) == 0: continue
                M = iou_mat(np.array([p[1] for p in P], dtype=np.float32), G)
                taken = np.zeros(len(G), dtype=bool)
                for pi in range(len(P)):
                    cand = np.where((M[pi] >= IOU_MATCH) & (~taken))[0]
                    if len(cand) == 0:
                        v[3] += 1; continue
                    gi = cand[np.argmax(M[pi, cand])]
                    taken[gi] = True; v[1] += 1
                    if M[pi, gi] >= IOU_STRICT: v[2] += 1
            fams = sorted(per_fam); Ms = np.stack([per_fam[k] for k in fams]); tot = Ms.sum(0)
            def rates(m):
                gt, h50, h75, fp = m
                return (100.0 * h50 / max(gt, 1), 100.0 * h75 / max(gt, 1), fp / max(len(fams), 1))
            pt = rates(tot)
            bs = np.array([rates(Ms[RNG.integers(0, len(fams), len(fams))].sum(0)) for _ in range(NBOOT)])
            for j, nm in enumerate(["loc_recall_iou50", "loc_recall_iou75", "fp_per_family"]):
                rec["%s_%s" % (nm, arm)] = float(pt[j])
                rec["%s_%s_ci" % (nm, arm)] = [float(np.percentile(bs[:, j], 2.5)),
                                               float(np.percentile(bs[:, j], 97.5))]
            rec["counts_" + arm] = {"families": len(fams), "gt": int(tot[0]),
                                    "hit50": int(tot[1]), "hit75": int(tot[2]), "fp": int(tot[3])}
        print("  %-22s caAP A=%.2f B=%.2f | locR50 A=%.2f B=%.2f | locR75 A=%.2f B=%.2f"
              % (label, rec["ca_AP_A"], rec["ca_AP_B"], rec["loc_recall_iou50_A"],
                 rec["loc_recall_iou50_B"], rec["loc_recall_iou75_A"], rec["loc_recall_iou75_B"]), flush=True)
        return rec

    out = {"rules": {"CONF_AP": CONF_AP, "CONF": CONF, "IOU_MATCH": IOU_MATCH,
                     "IOU_STRICT": IOU_STRICT, "top100_per_image": True, "step": STEP,
                     "scale": "all AP values x100"}, "runs": {}}
    d = np.load(sc_npz, allow_pickle=True)
    out["runs"]["single_cls_seed0"] = metrics(d["image_id"], d["xyxy"], d["conf"], "single_cls seed0")
    for s in range(5):
        d = np.load(P12 / "dets" / ("runs_p4__r512__seed%d.npz" % s), allow_pickle=True)
        out["runs"]["six_cls_seed%d" % s] = metrics(d["image_id"], d["xyxy"], d["conf"], "six_cls seed%d" % s)

    SC = out["runs"]["single_cls_seed0"]; SIX = out["runs"]["six_cls_seed0"]
    gains = {}
    for m in ("ca_AP", "ca_AP50", "ca_AP75", "loc_recall_iou50", "loc_recall_iou75", "fp_per_family"):
        five = {arm: [out["runs"]["six_cls_seed%d" % s]["%s_%s" % (m, arm)] for s in range(5)]
                for arm in ("A", "B")}
        gains[m] = {arm: {"single": SC["%s_%s" % (m, arm)], "six_matched_seed0": SIX["%s_%s" % (m, arm)],
                          "gain_vs_matched": SC["%s_%s" % (m, arm)] - SIX["%s_%s" % (m, arm)],
                          "six_5seed_mean": float(np.mean(five[arm])),
                          "six_5seed_sd": float(np.std(five[arm], ddof=1)),
                          "gain_vs_5seed_mean": SC["%s_%s" % (m, arm)] - float(np.mean(five[arm]))}
                    for arm in ("A", "B")}
    out["gains"] = gains
    json.dump(out, open(P13 / "p13_results.json", "w"), indent=1)
    print("P13EVALDONE")

if __name__ == "__main__":
    main()
