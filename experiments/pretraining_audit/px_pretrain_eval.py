"""Exploratory pretraining audit: pretrained runs_p4/r512 checkpoints scored with the UNCHANGED Phase 15 code path.

Rules are those of p15_prereg.json, which reuse Phase 9 / Phase 12 exactly:
  ONE inference pass per checkpoint over the A+B union (2439 images), imgsz 512, conf 0.001,
  NMS IoU 0.7, max_det 300, top-100 per image; both arms scored from the SAME detections.
  Six-class AP: pycocotools, maxDets 100, 8 categories.
  Class-agnostic AP: all GT/detections mapped to one category.
  Decomposition (Phase 12 rules): conf 0.25, greedy one-to-one class-agnostic match at IoU 0.50,
  strict 0.75; per-family counts so rates can be family-bootstrapped later.
  Per-detection six-class COCO match state is cached (Phase 9 layout) for the family bootstrap.

Resumable (phase15/eval_rows.json). No training. No checkpoint selection. TEST NOT TOUCHED.
"""
import copy, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
P9 = BASE / "phase9"
OUT = BASE / "pretrain_audit"
DETS = OUT / "dets"; DETS.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "match_cache"; CACHE.mkdir(exist_ok=True)
CONF_AP, IOU_NMS, MAXDET, IMGSZ = 0.001, 0.7, 300, 512
CONF, IOU_MATCH, IOU_STRICT = 0.25, 0.50, 0.75
WIDTHS = ("r512",)
STEPS = {10: 6300, 19: 11970, 29: 18270, 39: 24570}


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(np.float32)


def top100(iid, xyxy, cf, cl):
    order = np.lexsort((-cf, iid))
    iid2 = iid[order]
    keep = np.zeros(len(order), dtype=bool)
    start = 0
    for end in list(np.flatnonzero(np.diff(iid2)) + 1) + [len(iid2)]:
        keep[start:min(start + 100, end)] = True
        start = end
    sel = order[keep]
    return iid[sel], xyxy[sel], cf[sel], cl[sel]


def main():
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    g = json.loads((P9 / "coco_gt_union.json").read_text())
    arms = json.loads((P9 / "arm_image_ids.json").read_text())
    ARM = {k: sorted(arms["arm_image_ids"][k]) for k in ("A", "B")}
    FAM = {int(k): v for k, v in arms["family_of_image"].items()}
    assert len(ARM["A"]) == 2028 and len(ARM["B"]) == 411, {k: len(v) for k, v in ARM.items()}
    order = [im["file_name"] for im in sorted(g["images"], key=lambda x: x["id"])]
    idof = {im["file_name"]: im["id"] for im in g["images"]}
    IMGDIR = P9 / "arms" / "union" / "images" / "val"
    paths = [str(IMGDIR / f) for f in order]
    assert len(paths) == 2439 and all(Path(p).exists() for p in paths), "union images missing"

    gt_by_img = defaultdict(list)
    for a in g["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img[a["image_id"]].append((x, y, x + w, y + h, a["category_id"]))
    cocoGt = COCO(str(P9 / "coco_gt_union.json"))
    ga = copy.deepcopy(g)
    for a in ga["annotations"]: a["category_id"] = 1
    ga["categories"] = [{"id": 1, "name": "labelled foliar region"}]
    cocoGtA = COCO(); cocoGtA.dataset = ga; cocoGtA.createIndex()

    rp = OUT / "eval_rows.json"
    rows = json.loads(rp.read_text()) if rp.exists() else []
    done = {r["tag"] for r in rows}

    todo = []
    for s in (0, 1, 2, 3, 4):
        for wd in WIDTHS:
            for e, step in STEPS.items():
                w = REPO / "runs_p4" / wd / ("seed%d" % s) / "weights" / ("ckpt_ep%d_step%d.pt" % (e, step))
                todo.append(("%s__seed%d__step%d" % (wd, s, step), wd, s, step, w))
    missing = [t[0] for t in todo if not t[4].exists()]
    print("checkpoints planned %d, present %d, missing %d" % (len(todo), len(todo) - len(missing),
                                                              len(missing)), flush=True)

    # ---- reproduction control: score cached Phase 12 r512 seed0 dets with THIS code path ----
    ctrl_tag = "CONTROL_runs_p4__r512__seed0__step18270"
    if ctrl_tag not in done:
        todo.insert(0, (ctrl_tag, "control", 0, 18270, BASE / "phase12" / "dets" / "runs_p4__r512__seed0.npz"))

    import sys
    control_only = "--control-only" in sys.argv
    for n, (tag, wd, seed, step, w) in enumerate(todo, 1):
        if tag in done or not w.exists() or (control_only and wd != "control"):
            continue
        t0 = time.time()
        fo = w if wd == "control" else DETS / (tag + ".npz")
        if not fo.exists():
            model = YOLO(str(w))
            ii, bb, cc, kk = [], [], [], []
            for i in range(0, len(paths), 32):
                batch = paths[i:i + 32]
                res = model.predict(batch, imgsz=IMGSZ, conf=CONF_AP, iou=IOU_NMS, max_det=MAXDET,
                                    device=0, verbose=False)
                for p, rr in zip(batch, res):
                    b = rr.boxes
                    if b is None or len(b) == 0: continue
                    k = len(b)
                    ii.append(np.full(k, idof[Path(p).name], dtype=np.int64))
                    bb.append(b.xyxy.cpu().numpy().astype(np.float32))
                    cc.append(b.conf.cpu().numpy().astype(np.float32))
                    kk.append(b.cls.cpu().numpy().astype(np.int16))
            np.savez_compressed(fo, image_id=np.concatenate(ii), xyxy=np.concatenate(bb),
                                conf=np.concatenate(cc), cls=np.concatenate(kk), weights=str(w),
                                imgsz=IMGSZ)
        t_inf = time.time() - t0
        d = np.load(fo, allow_pickle=True)
        iid, xyxy, cf, cl = top100(d["image_id"], d["xyxy"], d["conf"], d["cls"])
        rec = {"tag": tag, "width": wd, "seed": seed, "step": step, "weights": str(w),
               "n_dets": int(len(iid)), "inference_s": round(t_inf, 1)}

        # ---- six-class AP (authoritative, pycocotools) ----
        dt6 = [{"image_id": int(i), "category_id": int(c),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s_)}
               for i, (x1, y1, x2, y2), s_, c in zip(iid, xyxy, cf, cl)]
        cocoDt = cocoGt.loadRes(dt6)
        for arm in ("A", "B"):
            E = COCOeval(cocoGt, cocoDt, "bbox"); E.params.maxDets = [1, 10, 100]
            E.params.imgIds = ARM[arm]
            E.evaluate(); E.accumulate(); E.summarize()
            st = E.stats; prec = E.eval["precision"]
            pc = {}
            for ci, c in enumerate(E.params.catIds):
                pp = prec[:, :, ci, 0, 2]; pp = pp[pp > -1]
                pc[int(c)] = float(pp.mean()) * 100 if pp.size else None
            rec[arm] = {"AP": st[0] * 100, "AP50": st[1] * 100, "AP75": st[2] * 100,
                        "APs": st[3] * 100, "APm": st[4] * 100, "APl": st[5] * 100,
                        "AR100": st[8] * 100, "per_class_AP": pc}

        # ---- six-class match cache over the full union (Phase 9 layout) ----
        E = COCOeval(cocoGt, cocoDt, "bbox"); E.params.maxDets = [1, 10, 100]
        E.evaluate()
        cats = list(E.params.catIds); nI = len(E.params.imgIds); nA = len(E.params.areaRng)
        sc_l, tp_l, ig_l, meta, off = [], [], [], [], 0
        for ci, c in enumerate(cats):
            for k_, im in enumerate(E.params.imgIds):
                e = E.evalImgs[ci * nA * nI + k_]
                if e is None: continue
                sc = np.asarray(e["dtScores"], dtype=np.float32).ravel()[:100]
                k = int(sc.shape[0])
                m = np.asarray(e["dtMatches"], dtype=np.float64).reshape(10, -1)[:, :k]
                ig = np.asarray(e["dtIgnore"], dtype=np.float64).reshape(10, -1)[:, :k]
                gi = np.asarray(e["gtIgnore"], dtype=np.float64).ravel()
                sc_l.append(sc); tp_l.append((m > 0).astype(np.uint8)); ig_l.append(ig.astype(np.uint8))
                meta.append((int(c), int(im), off, k, int((gi == 0).sum()))); off += k
        np.savez_compressed(CACHE / (tag + ".npz"),
            scores=np.concatenate(sc_l) if sc_l else np.zeros(0, np.float32),
            tp=np.concatenate(tp_l, axis=1) if tp_l else np.zeros((10, 0), np.uint8),
            ig=np.concatenate(ig_l, axis=1) if ig_l else np.zeros((10, 0), np.uint8),
            meta=np.array(meta, dtype=np.int64), iouThrs=np.asarray(E.params.iouThrs),
            catIds=np.array(cats, dtype=np.int64))

        # ---- class-agnostic AP ----
        dta = [dict(x, category_id=1) for x in dt6]
        cocoDtA = cocoGtA.loadRes(dta)
        for arm in ("A", "B"):
            E = COCOeval(cocoGtA, cocoDtA, "bbox"); E.params.maxDets = [1, 10, 100]
            E.params.imgIds = ARM[arm]
            E.evaluate(); E.accumulate(); E.summarize()
            rec[arm].update({"ca_AP": E.stats[0] * 100, "ca_AP50": E.stats[1] * 100,
                             "ca_AP75": E.stats[2] * 100})

        # ---- GT-centric decomposition at CONF (Phase 12 rules), per-family counts ----
        keep = cf >= CONF
        by_img = defaultdict(list)
        for i, bx, s_, c in zip(iid[keep], xyxy[keep], cf[keep], cl[keep]):
            by_img[int(i)].append((float(s_), bx, int(c)))
        for arm in ("A", "B"):
            # gt, hit50 (class-agnostic), hit75 (class-agnostic), class-correct among hit50, fp
            per_fam = defaultdict(lambda: np.zeros(5, dtype=np.int64))
            for i in ARM[arm]:
                G = np.array(gt_by_img[i], dtype=np.float32).reshape(-1, 5)
                P = sorted(by_img.get(i, []), key=lambda t: -t[0])
                v = per_fam[FAM[i]]; v[0] += len(G)
                if len(G) == 0:
                    v[4] += len(P); continue
                if len(P) == 0: continue
                M = iou_mat(np.array([p[1] for p in P], dtype=np.float32), G[:, :4])
                taken = np.zeros(len(G), dtype=bool)
                for pi in range(len(P)):
                    cand = np.where((M[pi] >= IOU_MATCH) & (~taken))[0]
                    if len(cand) == 0:
                        v[4] += 1; continue
                    gi = cand[np.argmax(M[pi, cand])]
                    taken[gi] = True; v[1] += 1
                    if M[pi, gi] >= IOU_STRICT: v[2] += 1
                    if P[pi][2] == int(G[gi, 4]): v[3] += 1
            fams = sorted(per_fam); tot = np.stack([per_fam[k] for k in fams]).sum(0)
            rec[arm].update({"loc_recall_iou50": 100.0 * tot[1] / max(tot[0], 1),
                             "loc_recall_iou75": 100.0 * tot[2] / max(tot[0], 1),
                             "cond_class_correct": 100.0 * tot[3] / max(tot[1], 1),
                             "fp_per_family": float(tot[4] / max(len(fams), 1)),
                             "family_counts": {f: per_fam[f].tolist() for f in fams}})

        if wd == "control":
            p9 = next(r for r in json.loads((P9 / "eval_rows.json").read_text())
                      if r["run"] == "runs_p4/r512" and r["seed"] == 0)
            p12 = json.loads((BASE / "phase12" / "p12_results.json").read_text())["checkpoints"]["runs_p4__r512__seed0"]
            diffs = {"AP_A": abs(rec["A"]["AP"] - p9["A"]["AP"] * 100),
                     "AP_B": abs(rec["B"]["AP"] - p9["B"]["AP"] * 100),
                     "ca_AP_B": abs(rec["B"]["ca_AP"] - p12["ca_AP_B"] * 100),
                     "loc_recall_iou50_B": abs(rec["B"]["loc_recall_iou50"] - p12["loc_recall_B"]),
                     "cond_class_correct_B": abs(rec["B"]["cond_class_correct"] - p12["cond_class_correct_B"])}
            rec["control_abs_diffs"] = diffs
            ok = max(diffs.values()) <= 0.05
            print("CONTROL " + json.dumps({k: round(v, 4) for k, v in diffs.items()}) +
                  (" PASS" if ok else " FAIL"), flush=True)
            if not ok:
                (OUT / "control_failed.json").write_text(json.dumps(rec, indent=1))
                raise SystemExit("STOP_CONTROL_FAILED")
        rows.append(rec)
        rp.write_text(json.dumps(rows))
        print("ROW %d/%d %-26s AP A=%.2f B=%.2f | caAP B=%.2f | locR50 B=%.2f | %.0fs"
              % (n, len(todo), tag, rec["A"]["AP"], rec["B"]["AP"], rec["B"]["ca_AP"],
                 rec["B"]["loc_recall_iou50"], time.time() - t0), flush=True)
    print("P15EVALDONE rows=%d" % len(rows), flush=True)


if __name__ == "__main__":
    main()
