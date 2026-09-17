"""Visualise and quantify the ADCF gate (alpha): where does the model trust detail?

Outputs:
  * <out>/<image>.png      image + detections, then one alpha heat-map per ADCF slot
                           (alpha -> 1: detail branch, alpha -> 0: context branch)
  * <out>/gate_stats.json  mean alpha inside vs outside ground-truth boxes, per slot
                           and per class, over the whole split

LEARN: this is interpretability evidence. If the gate learned something meaningful,
alpha should differ between lesion boxes and background. If it is ~0.5 everywhere,
the gate is not doing much and the ablation (add vs gated) should show that too.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from adcf_yolo.build import adcf_layers
from adcf_yolo.data.stats import load_dataset, read_boxes, split_files

MAP_SIZE = 160  # long side of the alpha maps used for statistics


def unletterbox(
    alpha: torch.Tensor, input_hw: tuple[int, int], orig_hw: tuple[int, int], imgsz: int, out_hw: tuple[int, int]
) -> np.ndarray:
    """Map a gate tensor (1, C, h, w) from the letterboxed network input back onto the original image."""
    a = alpha.float().mean(1, keepdim=True)  # channel gate -> average over channels
    a = F.interpolate(a, size=input_hw, mode="bilinear", align_corners=False)
    h0, w0 = orig_hw
    r = min(imgsz / h0, imgsz / w0)
    new_h, new_w = round(h0 * r), round(w0 * r)
    # Ultralytics LetterBox centres the image and splits the padding this way.
    top, left = round((input_hw[0] - new_h) / 2 - 0.1), round((input_hw[1] - new_w) / 2 - 0.1)
    a = a[..., top : top + new_h, left : left + new_w]
    return F.interpolate(a, size=out_hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


class GateProbe:
    """Runs prediction and captures every gated ADCF layer's alpha map."""

    def __init__(self, weights: str | Path, imgsz: int = 640, device: str | None = None):
        self.model = YOLO(str(weights))
        self.imgsz, self.device = imgsz, device
        self.slots = [slot for slot, m in adcf_layers(self.model.model) if m.gate is not None]
        if not self.slots:
            raise ValueError("checkpoint has no gated ADCF layers")
        self.input_hw: tuple[int, int] | None = None
        self._hooked = None

    def _prepare(self) -> list:
        # The predictor may wrap/fuse the model, so hook the network it actually runs.
        net = self.model.predictor.model.model
        if self._hooked is not net:

            def record_input(_module, inputs):
                self.input_hw = tuple(inputs[0].shape[-2:])

            net.model[0].register_forward_pre_hook(record_input)
            self._hooked = net
        layers = [(slot, m) for slot, m in adcf_layers(net) if m.gate is not None]
        for _, m in layers:
            m.store_alpha = True
        return layers

    def run(self, image: Path, out_long_side: int | None = None, conf: float = 0.25):
        kwargs = dict(imgsz=self.imgsz, conf=conf, device=self.device, verbose=False)
        if self.model.predictor is None:
            self.model.predict(str(image), **kwargs)  # first call builds the predictor
        layers = self._prepare()
        result = self.model.predict(str(image), **kwargs)[0]
        h0, w0 = result.orig_shape
        if out_long_side:
            s = out_long_side / max(h0, w0)
            out_hw = (max(round(h0 * s), 1), max(round(w0 * s), 1))
        else:
            out_hw = (h0, w0)
        maps = {slot: unletterbox(m.last_alpha, self.input_hw, (h0, w0), self.imgsz, out_hw) for slot, m in layers}
        return result, maps


def gate_statistics(probe: GateProbe, data_yaml: Path, split: str) -> dict:
    root, cfg = load_dataset(data_yaml)
    names = cfg["names"]
    inside, outside = defaultdict(list), defaultdict(list)
    per_class = defaultdict(lambda: defaultdict(list))
    for img, lbl in split_files(root, cfg, split):
        _, maps = probe.run(img, out_long_side=MAP_SIZE)
        boxes = read_boxes(lbl)
        for slot, a in maps.items():
            h, w = a.shape
            mask = np.zeros_like(a, dtype=bool)
            for c, cx, cy, bw, bh in boxes:
                x1, x2 = max(int((cx - bw / 2) * w), 0), int(np.ceil((cx + bw / 2) * w))
                y1, y2 = max(int((cy - bh / 2) * h), 0), int(np.ceil((cy + bh / 2) * h))
                region = a[y1:y2, x1:x2]
                if region.size:
                    per_class[slot][names[int(c)]].append(float(region.mean()))
                mask[y1:y2, x1:x2] = True
            if mask.any():
                inside[slot].append(float(a[mask].mean()))
            if (~mask).any():
                outside[slot].append(float(a[~mask].mean()))
    return {
        slot: {
            "alpha_inside_boxes": float(np.mean(inside[slot])) if inside[slot] else None,
            "alpha_outside_boxes": float(np.mean(outside[slot])) if outside[slot] else None,
            "alpha_inside_by_class": {c: float(np.mean(v)) for c, v in per_class[slot].items()},
        }
        for slot in probe.slots
    }


def figure(probe: GateProbe, image: Path, out: Path) -> None:
    result, maps = probe.run(image, out_long_side=640)
    plotted = result.plot()[..., ::-1]  # BGR -> RGB, with predicted boxes drawn
    fig, axes = plt.subplots(1, 1 + len(maps), figsize=(4 * (1 + len(maps)), 4.2))
    axes[0].imshow(plotted)
    axes[0].set_title("detections")
    extent = (0, plotted.shape[1], plotted.shape[0], 0)
    for ax, (slot, a) in zip(axes[1:], maps.items()):
        ax.imshow(plotted, alpha=0.35)
        im = ax.imshow(a, cmap="coolwarm", vmin=0, vmax=1, alpha=0.75, extent=extent)
        ax.set_title(f"α at {slot}")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, label="α  (1 = detail, 0 = context)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/tomato_village/dataset.yaml"))
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=12, help="number of example figures")
    p.add_argument("--out", type=Path, default=Path("results/gates"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device")
    p.add_argument("--no-stats", action="store_true", help="skip the (slower) whole-split statistics")
    args = p.parse_args()

    probe = GateProbe(args.weights, args.imgsz, args.device)
    root, cfg = load_dataset(args.data)
    files = split_files(root, cfg, args.split)
    picks = np.random.default_rng(0).choice(len(files), size=min(args.n, len(files)), replace=False)
    for k in picks:
        img = files[k][0]
        figure(probe, img, args.out / f"{img.stem}.png")
    print(f"[gates] {len(picks)} figures -> {args.out}")
    if not args.no_stats:
        s = gate_statistics(probe, args.data, args.split)
        (args.out / "gate_stats.json").write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
