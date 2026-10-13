"""Phase 9 step 3: evaluate every authorised checkpoint on arms A and B.

ONE inference pass per checkpoint over the A+B union; both arms are then scored from the SAME
detections as image-id subsets - identical weights, preprocessing, NMS and scorer, differing only
in which image ids enter the metric.

No training. No checkpoint reselection. No per-arm thresholds or hyperparameters.
Per-detection COCO match state is cached in a flat (ragged-safe) layout so the source-family
bootstrap needs no re-inference.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = Path(r"C:\Users\absamad\cs\phase9")
CACHE = OUT / "match_cache"; CACHE.mkdir(exist_ok=True)
U = OUT / "arms" / "union"
CONF, IOU_NMS, MAXDET = 0.001, 0.7, 300

PLAN = [
 ("runs_p3/yolo11n",           None, [0,1,2,3,4], 640, ("ckpt", 18270)),
 ("runs_p3/yolo11m",           None, [0,1,2,3,4], 640, ("ckpt", 18270)),
 ("runs_p2b/D",                None, [0,1,2,3,4], 640, ("ckpt", 18270)),
 ("runs_p2b/B",                None, [0,1,2,3,4], 640, ("ckpt", 18170)),
 ("runs_p2b/C",                None, [0,1,2,3,4], 640, ("ckpt", 18270)),
 ("runs_p4/r512",              None, [0,1,2,3,4], 512, ("ckpt", 18270)),
 ("runs_p4/r320",              None, [0,1,2,3,4], 320, ("ckpt", 18270)),
 ("runs_p4/r416",              None, [0,1,2,3,4], 416, ("ckpt", 18270)),
 ("runs_p6/B",                 None, [0,1,2,3,4], 512, ("ckpt", 18270)),
 ("runs_p6/C",                 None, [0,1,2,3,4], 512, ("ckpt", 18270)),
 ("runs/yolo11s",              50,   [0,1,2],     640, ("last", None)),
 ("runs/yolo11s-adcf",         50,   [0,1,2],     640, ("last", None)),
 ("runs/yolo11s-adcf-replace", 50,   [0,1,2],     640, ("last", None)),
 ("runs/yolo11s",              25,   [3,4],       640, ("last", None)),
 ("runs/yolo11s-cbam",         25,   [0,1,2],     640, ("last", None)),
 ("runs/yolo11s-960",          25,   [0,1,2],     960, ("last", None)),
 ("runs/yolo11s-p2",           25,   [0,1,2],     640, ("last", None)),
]

def weights_for(key, seed, rule):
    root, run = key.split("/", 1)
    d = REPO / root / run / ("seed" + str(seed))
    w = d / "weights"
    if rule[0] == "last":
        return d, (w / "last.pt")
    cands = sorted(w.glob("ckpt_ep*_step" + str(rule[1]) + ".pt"))
    return d, (cands[0] if cands else None)

def main():
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    idx = json.loads((OUT / "image_index.json").read_text())
    sid = json.loads((OUT / "stem_to_cocoid.json").read_text())
    cocoGt = COCO(str(OUT / "coco_gt_union.json"))
    armof = {sid[s]: v["arm"] for s, v in idx.items()}
    famof = {sid[s]: v["family"] for s, v in idx.items()}
    IDS = {a: sorted(i for i, x in armof.items() if x == a) for a in ("A", "B")}
    json.dump({"arm_image_ids": IDS, "family_of_image": famof},
              open(OUT / "arm_image_ids.json", "w"))
    print("arm image ids: A=" + str(len(IDS["A"])) + " B=" + str(len(IDS["B"])), flush=True)

    imgdir = U / "images" / "val"
    files = [str(imgdir / (s + ".jpg")) for s in sorted(idx)]
    assert len(files) == 2439, len(files)

    todo = []
    for key, pat, seeds, imgsz, rule in PLAN:
        for s in seeds:
            d, w = weights_for(key, s, rule)
            tag = key.replace("/", "__") + "__p" + str(pat) + "__seed" + str(s)
            if w is None or not w.exists():
                print("MISSINGWEIGHTS " + tag + " -> " + str(w), flush=True); continue
            todo.append((tag, key, pat, s, imgsz, str(w)))
    print("checkpoints to evaluate: " + str(len(todo)), flush=True)

    rp = OUT / "eval_rows.json"
    rows = json.loads(rp.read_text()) if rp.exists() else []
    done = {r["tag"] for r in rows}

    for n, (tag, key, pat, seed, imgsz, w) in enumerate(todo, 1):
        if tag in done:
            print("SKIP " + str(n) + "/" + str(len(todo)) + " " + tag, flush=True); continue
        t0 = time.time()
        model = YOLO(w)
        dets = []
        for i in range(0, len(files), 64):
            batch = files[i:i+64]
            res = model.predict(batch, imgsz=imgsz, conf=CONF, iou=IOU_NMS, max_det=MAXDET,
                                device=0, verbose=False)
            for f, r in zip(batch, res):
                iid = sid[Path(f).stem]
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy()
                for j in np.argsort(-cf)[:100]:
                    x1, y1, x2, y2 = xyxy[j]
                    dets.append({"image_id": int(iid), "category_id": int(cl[j]),
                                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                 "score": float(cf[j])})
        t_inf = time.time() - t0
        if not dets:
            print("NODETS " + tag, flush=True); continue
        cocoDt = cocoGt.loadRes(dets)

        # authoritative per-arm metrics from pycocotools, computed FIRST
        ref = {}
        for a in ("A", "B"):
            E2 = COCOeval(cocoGt, cocoDt, "bbox")
            E2.params.imgIds = IDS[a]
            E2.params.maxDets = [1, 10, 100]
            E2.evaluate(); E2.accumulate(); E2.summarize()
            st = E2.stats
            prec = E2.eval["precision"]
            pc = {}
            for ci, c in enumerate(E2.params.catIds):
                pp = prec[:, :, ci, 0, 2]
                pp = pp[pp > -1]
                pc[int(c)] = float(pp.mean()) if pp.size else None
            ref[a] = {"AP": float(st[0]), "AP50": float(st[1]), "AP75": float(st[2]),
                      "APs": float(st[3]), "APm": float(st[4]), "APl": float(st[5]),
                      "AR100": float(st[8]), "per_class_AP": pc}

        # flat, ragged-safe cache of per-(cat,img) match state over the full union
        E = COCOeval(cocoGt, cocoDt, "bbox")
        E.params.maxDets = [1, 10, 100]
        E.evaluate()
        cats = list(E.params.catIds); nI = len(E.params.imgIds); nA = len(E.params.areaRng)
        sc_l, tp_l, ig_l, meta = [], [], [], []
        off = 0
        for ci, c in enumerate(cats):
            for ii, iid in enumerate(E.params.imgIds):
                e = E.evalImgs[ci * nA * nI + 0 * nI + ii]
                if e is None:
                    continue
                sc = np.asarray(e["dtScores"], dtype=np.float32).ravel()[:100]
                k = int(sc.shape[0])
                m = np.asarray(e["dtMatches"], dtype=np.float64).reshape(10, -1)[:, :k]
                ig = np.asarray(e["dtIgnore"], dtype=np.float64).reshape(10, -1)[:, :k]
                gi = np.asarray(e["gtIgnore"], dtype=np.float64).ravel()
                sc_l.append(sc)
                tp_l.append((m > 0).astype(np.uint8))
                ig_l.append(ig.astype(np.uint8))
                meta.append((int(c), int(iid), off, k, int((gi == 0).sum())))
                off += k
        np.savez_compressed(
            CACHE / (tag + ".npz"),
            scores=np.concatenate(sc_l) if sc_l else np.zeros(0, np.float32),
            tp=np.concatenate(tp_l, axis=1) if tp_l else np.zeros((10, 0), np.uint8),
            ig=np.concatenate(ig_l, axis=1) if ig_l else np.zeros((10, 0), np.uint8),
            meta=np.array(meta, dtype=np.int64),
            iouThrs=np.asarray(E.params.iouThrs), catIds=np.array(cats, dtype=np.int64))

        rows.append({"tag": tag, "run": key, "patience": pat, "seed": seed, "imgsz": imgsz,
                     "weights": w, "n_dets": len(dets), "inference_s": round(t_inf, 1),
                     "A": ref["A"], "B": ref["B"]})
        rp.write_text(json.dumps(rows, indent=1))
        print("DONEROW " + str(n) + "/" + str(len(todo)) + " " + tag +
              "  A_AP=" + format(ref["A"]["AP"], ".4f") + " B_AP=" + format(ref["B"]["AP"], ".4f") +
              " infer=" + format(t_inf, ".0f") + "s dets=" + str(len(dets)), flush=True)
    print("EVALDONE", flush=True)

if __name__ == "__main__":
    main()
