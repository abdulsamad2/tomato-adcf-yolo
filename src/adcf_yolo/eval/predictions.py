"""Per-image predictions + ground truth for a split, cached next to the checkpoint.

Shared by the COCO size breakdown and the paired bootstrap, so each model predicts
the split once. Matching and AP reuse Ultralytics' own logic, so mAP computed here
tracks `yolo val` closely (small differences come from val's rectangular batching).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import exif_size
from ultralytics.utils.metrics import ap_per_class

from adcf_yolo.data.stats import load_dataset, read_boxes, split_files

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class ImagePredictions:
    name: str
    height: int
    width: int
    gt_cls: np.ndarray  # (M,)
    gt_xyxy: np.ndarray  # (M, 4) pixels, original image
    cls: np.ndarray  # (N,)
    conf: np.ndarray  # (N,)
    xyxy: np.ndarray  # (N, 4) pixels, original image


def predict_split(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    batch: int = 16,
    device: str | None = None,
    cache: Path | None = None,
) -> list[ImagePredictions]:
    settings = {"weights": str(Path(weights).resolve()), "mtime": Path(weights).stat().st_mtime, "data": str(data_yaml),
                "split": split, "imgsz": imgsz, "conf": conf, "iou": iou, "max_det": max_det}
    if cache and cache.exists():
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved["settings"] == settings:
            return saved["images"]

    root, cfg = load_dataset(Path(data_yaml))
    files = split_files(root, cfg, split)
    model = YOLO(str(weights))
    images = []
    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        results = model.predict(
            [str(p) for p, _ in chunk], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False
        )
        for (img, lbl), r in zip(chunk, results):
            with Image.open(img) as im:
                w, h = exif_size(im)
            if tuple(r.orig_shape) != (h, w):
                raise RuntimeError(f"{img}: loader shape {r.orig_shape} != EXIF-corrected size {(h, w)}")
            gt = read_boxes(lbl)
            cx, cy, bw, bh = gt[:, 1] * w, gt[:, 2] * h, gt[:, 3] * w, gt[:, 4] * h
            images.append(
                ImagePredictions(
                    name=img.name,
                    height=h,
                    width=w,
                    gt_cls=gt[:, 0].astype(int),
                    gt_xyxy=np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1),
                    cls=r.boxes.cls.cpu().numpy().astype(int),
                    conf=r.boxes.conf.cpu().numpy(),
                    xyxy=r.boxes.xyxy.cpu().numpy(),
                )
            )
    if cache:
        with open(cache, "wb") as f:
            pickle.dump({"settings": settings, "images": images}, f)
    return images


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(p: ImagePredictions) -> np.ndarray:
    """(N, 10) true-positive matrix, same greedy matching as Ultralytics' validator."""
    correct = np.zeros((len(p.cls), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(p.cls) or not len(p.gt_cls):
        return correct
    iou = box_iou(p.gt_xyxy, p.xyxy) * (p.gt_cls[:, None] == p.cls[None, :])  # (M, N)
    for t, thr in enumerate(IOU_THRESHOLDS):
        m = np.argwhere(iou >= thr)  # rows: [gt, pred]
        if len(m) > 1:
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        correct[m[:, 1], t] = True
    return correct


class MapEvaluator:
    """Precomputes per-image matches so mAP over any subset of images is cheap."""

    def __init__(self, images: list[ImagePredictions]):
        self.names = [im.name for im in images]
        self.tp = [match(im) for im in images]
        self.conf = [im.conf for im in images]
        self.cls = [im.cls for im in images]
        self.gt = [im.gt_cls for im in images]

    def __call__(self, idx: np.ndarray | None = None) -> tuple[float, float]:
        """(mAP50, mAP50-95) over the images at `idx` (all images if None; repeats allowed)."""
        idx = range(len(self.tp)) if idx is None else idx
        target = np.concatenate([self.gt[i] for i in idx])
        if not len(target):
            return 0.0, 0.0
        ap = ap_per_class(
            np.concatenate([self.tp[i] for i in idx]),
            np.concatenate([self.conf[i] for i in idx]),
            np.concatenate([self.cls[i] for i in idx]),
            target,
        )[5]
        return float(ap[:, 0].mean()), float(ap.mean())
