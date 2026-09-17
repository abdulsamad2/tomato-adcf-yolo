"""COCO-style AP by object size (AP_small / AP_medium / AP_large).

Ultralytics' val reports mAP50 and mAP50-95 but not per-size AP. Our claim is that
ADCF helps *small* lesions, so we need AP_small as direct evidence.

LEARN: COCO size buckets are absolute pixel areas. Photos here come in different
resolutions, so every image is rescaled to a 640 px long side before bucketing,
*regardless of the network's input size*. A lesion is therefore "small" in every
run, including 960 px runs, and the buckets are comparable across models.
COCO uses 101-point interpolation, so AP here differs slightly from Ultralytics'
mAP50-95. Report Ultralytics mAP as the headline and these as a breakdown.
"""

from __future__ import annotations

import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from adcf_yolo.data.stats import REF_SIZE
from adcf_yolo.eval.predictions import ImagePredictions

STAT_NAMES = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def coco_size_eval(images: list[ImagePredictions], names: dict[int, str], max_det: int = 300) -> dict[str, float | None]:
    gt = {"images": [], "annotations": [], "categories": [{"id": int(k) + 1, "name": v} for k, v in names.items()]}
    dets = []
    for img_id, im in enumerate(images, 1):
        s = REF_SIZE / max(im.height, im.width)
        gt["images"].append({"id": img_id, "file_name": im.name, "width": im.width * s, "height": im.height * s})
        for c, (x1, y1, x2, y2) in zip(im.gt_cls, im.gt_xyxy * s):
            gt["annotations"].append(
                {
                    "id": len(gt["annotations"]) + 1,
                    "image_id": img_id,
                    "category_id": int(c) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                }
            )
        for c, score, (x1, y1, x2, y2) in zip(im.cls, im.conf, im.xyxy * s):
            dets.append(
                {
                    "image_id": img_id,
                    "category_id": int(c) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    if not dets or not gt["annotations"]:
        return dict.fromkeys(STAT_NAMES, 0.0)

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools is chatty
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
        ev.params.maxDets = [1, 10, max_det]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # -1 means "no ground truth in this bucket"
    return {k: (float(v) if v >= 0 else None) for k, v in zip(STAT_NAMES, ev.stats)}
