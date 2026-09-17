"""COCO-style AP by object size (AP_small / AP_medium / AP_large).

Ultralytics' val reports mAP50 and mAP50-95 but not per-size AP. Our claim is that
ADCF helps *small* lesions, so we need AP_small as direct evidence.

LEARN: COCO size buckets are absolute pixel areas. Photos here come in different
resolutions, so boxes are rescaled to a 640 px long side (the network's input
size) before bucketing. The numbers are therefore "size as the network sees it".
COCO uses 101-point interpolation, so AP here differs slightly from Ultralytics'
mAP50-95. Report Ultralytics mAP as the headline and these as a breakdown.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.utils import exif_size

from adcf_yolo.data.stats import REF_SIZE, load_dataset, read_boxes, split_files

STAT_NAMES = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def coco_size_eval(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    batch: int = 16,
    device: str | None = None,
) -> dict[str, float | None]:
    root, cfg = load_dataset(Path(data_yaml))
    files = split_files(root, cfg, split)
    gt = {"images": [], "annotations": [], "categories": [{"id": int(k) + 1, "name": v} for k, v in cfg["names"].items()]}
    scales = {}
    for img_id, (img, lbl) in enumerate(files, 1):
        with Image.open(img) as im:
            w, h = exif_size(im)
        s = REF_SIZE / max(w, h)
        scales[img_id] = (s, w, h)
        gt["images"].append({"id": img_id, "file_name": img.name, "width": w * s, "height": h * s})
        for c, cx, cy, bw, bh in read_boxes(lbl):
            x, y, bw_px, bh_px = (cx - bw / 2) * w * s, (cy - bh / 2) * h * s, bw * w * s, bh * h * s
            gt["annotations"].append(
                {
                    "id": len(gt["annotations"]) + 1,
                    "image_id": img_id,
                    "category_id": int(c) + 1,
                    "bbox": [x, y, bw_px, bh_px],
                    "area": bw_px * bh_px,
                    "iscrowd": 0,
                }
            )

    model = YOLO(str(weights))
    dets = []
    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        results = model.predict(
            [str(p) for p, _ in chunk], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False
        )
        for offset, r in enumerate(results):
            img_id = start + offset + 1
            s, w, h = scales[img_id]
            if tuple(r.orig_shape) != (h, w):
                raise RuntimeError(f"{chunk[offset][0]}: loader shape {r.orig_shape} != EXIF size {(h, w)}")
            for (x1, y1, x2, y2), score, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                dets.append(
                    {
                        "image_id": img_id,
                        "category_id": int(c) + 1,
                        "bbox": [x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s],
                        "score": score,
                    }
                )
    if not dets:
        return dict.fromkeys(STAT_NAMES, 0.0)

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools is chatty
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, max_det]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # -1 means "no ground truth in this bucket"
    return {k: (float(v) if v >= 0 else None) for k, v in zip(STAT_NAMES, ev.stats)}
