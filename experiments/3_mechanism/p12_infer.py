"""Phase 12 Stage 0: re-run inference to obtain RAW detections (boxes + predicted classes).

Identical to Phase 9 in every setting: same union image set, same frozen checkpoints (step 18270),
same per-run imgsz, same conf/NMS/max_det, same COCO scorer. The ONLY difference is that raw box
geometry and predicted class are written out, which the Phase 9 cache does not contain.

Six-class per-arm AP is recomputed here as a CONTROL: it must reproduce the Phase 9 table.
No training. No checkpoint reselection. No per-arm thresholds.
"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
P9 = Path(r"C:\Users\absamad\cs\phase9")
OUT = Path(r"C:\Users\absamad\cs\phase12"); OUT.mkdir(parents=True, exist_ok=True)
DETS = OUT / "dets"; DETS.mkdir(exist_ok=True)
CONF, IOU_NMS, MAXDET = 0.001, 0.7, 300
PLAN = [("runs_p4/r512", [0,1,2,3,4], 512, 18270),
        ("runs_p3/yolo11n", [0,1,2,3,4], 640, 18270),
        ("runs_p2b/D",      [0,1,2,3,4], 640, 18270),
        ("runs_p3/yolo11m", [0,1,2,3,4], 640, 18270)]

def main():
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = json.loads((P9 / "coco_gt_union.json").read_text())
    order = [im["file_name"] for im in sorted(gt["images"], key=lambda x: x["id"])]
    idof = {im["file_name"]: im["id"] for im in gt["images"]}
    IMGDIR = P9 / "arms" / "union" / "images" / "val"
    paths = [str(IMGDIR / f) for f in order]
    assert all(Path(p).exists() for p in paths), "union images missing"

    arms = json.loads((P9 / "arm_image_ids.json").read_text())
    armids = {k: set(arms["arm_image_ids"][k]) for k in ("A", "B")}
    print("union %d images | arm A %d | arm B %d" % (len(paths), len(armids["A"]), len(armids["B"])), flush=True)

    cocoGt = COCO(str(P9 / "coco_gt_union.json"))
    rows = []
    for key, seeds, imgsz, step in PLAN:
        root, run = key.split("/")
        for s in seeds:
            tag = "%s__%s__seed%d" % (root, run, s)
            fo = DETS / (tag + ".npz")
            wdir = REPO / root / run / ("seed%d" % s) / "weights"
            cands = sorted(wdir.glob("ckpt_ep*_step%d.pt" % step))
            assert len(cands) == 1, (wdir, cands)
            w = cands[0]
            if fo.exists():
                print("  %s cached" % tag, flush=True); continue
            t0 = time.time()
            model = YOLO(str(w))
            img_id, xyxy, cf, cl = [], [], [], []
            B = 32
            for i in range(0, len(paths), B):
                batch = paths[i:i+B]
                res = model.predict(batch, imgsz=imgsz, conf=CONF, iou=IOU_NMS, max_det=MAXDET,
                                    device=0, verbose=False, stream=False)
                for p, rr in zip(batch, res):
                    b = rr.boxes
                    if b is None or len(b) == 0: continue
                    n = len(b)
                    img_id.append(np.full(n, idof[Path(p).name], dtype=np.int64))
                    xyxy.append(b.xyxy.cpu().numpy().astype(np.float32))
                    cf.append(b.conf.cpu().numpy().astype(np.float32))
                    cl.append(b.cls.cpu().numpy().astype(np.int16))
            img_id = np.concatenate(img_id); xyxy = np.concatenate(xyxy)
            cf = np.concatenate(cf); cl = np.concatenate(cl)
            np.savez_compressed(fo, image_id=img_id, xyxy=xyxy, conf=cf, cls=cl,
                                weights=str(w), imgsz=imgsz)
            # CONTROL: six-class AP per arm from these detections
            dt = [{"image_id": int(i), "category_id": int(c),
                   "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(sc)}
                  for i, (x1, y1, x2, y2), sc, c in zip(img_id, xyxy, cf, cl)]
            rec = {"tag": tag, "run": key, "seed": s, "imgsz": imgsz, "step": step,
                   "n_dets": int(len(dt)), "secs": round(time.time()-t0, 1)}
            cocoDt = cocoGt.loadRes(dt)
            for arm in ("A", "B"):
                E = COCOeval(cocoGt, cocoDt, "bbox")
                E.params.maxDets = [1, 10, 100]
                E.params.imgIds = sorted(armids[arm])
                E.evaluate(); E.accumulate(); E.summarize()
                rec["AP_" + arm] = float(E.stats[0]); rec["AP50_" + arm] = float(E.stats[1])
                rec["AP75_" + arm] = float(E.stats[2]); rec["AR100_" + arm] = float(E.stats[8])
            rows.append(rec)
            print("  %s dets=%d  6cls AP A=%.4f B=%.4f  %.0fs"
                  % (tag, rec["n_dets"], rec["AP_A"], rec["AP_B"], rec["secs"]), flush=True)
    json.dump(rows, open(OUT / "sixclass_control.json", "w"), indent=1)
    print("INFERDONE %d checkpoints" % len(rows))

if __name__ == "__main__":
    main()
