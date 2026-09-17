"""Insert ADCF into an Ultralytics YOLO neck and train it with the stock trainer.

Rather than forking Ultralytics or its YAML parser, we build the normal model and
modify the neck fusion blocks (C2f in YOLOv8, C3k2 in YOLO11/YOLO26).

Standard neck (4 fusion nodes):           P2 neck (6 fusion nodes, extra stride-4 head):
    p4_td   after Concat(P5_up, P4)           p4_td, p3_td, p2_out  (top-down)
    p3_out  after Concat(P4_up, P3)           p3_out, p4_out, p5_out (bottom-up)
    p4_out  after Concat(P3_down, P4)
    p5_out  after Concat(P4_down, P5)

Two ways to insert (`mode`):
    replace   the fusion block becomes ADCF (fresh weights, fewer params)
    residual  the pretrained block is kept; out = block(x) + gamma * refiner(block(x))
"""

from __future__ import annotations

import copy
from typing import Any

import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import CBAM
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import initialize_weights

from adcf_yolo.modules import ADCF, ResidualRefine

NECK_SLOTS = {
    4: ("p4_td", "p3_out", "p4_out", "p5_out"),
    6: ("p4_td", "p3_td", "p2_out", "p3_out", "p4_out", "p5_out"),
}
PLACEMENTS: dict[str, tuple[str, ...] | None] = {
    "all": None,  # every fusion node
    "p3": ("p3_out",),  # only the small-object path
    "top_down": ("p4_td", "p3_out"),
    "bottom_up": ("p4_out", "p5_out"),
    "fine": ("p3_td", "p2_out", "p3_out"),  # P2 necks: the high-resolution nodes
}
MODES = ("replace", "residual")
MODULES = ("adcf", "cbam")  # cbam = generic-attention control in the same slots
ADCF_KEYS = ("e", "fusion", "gate", "dilations", "highpass")


def neck_fusion_indices(model: DetectionModel) -> list[int]:
    """Layer indices of the neck fusion blocks (original or already patched)."""
    n_backbone = len(model.yaml["backbone"])
    idx = [i for i, m in enumerate(model.model) if i >= n_backbone and isinstance(m, (C2f, ADCF, ResidualRefine))]
    if len(idx) not in NECK_SLOTS:
        raise RuntimeError(f"expected 4 or 6 neck fusion blocks, found {len(idx)} at {idx}; unsupported architecture")
    return idx


def neck_slots(model: DetectionModel) -> dict[str, int]:
    """Slot name -> layer index."""
    idx = neck_fusion_indices(model)
    return dict(zip(NECK_SLOTS[len(idx)], idx))


def _resolve_placement(placement: str | list[str], slots: dict[str, int]) -> list[str]:
    if isinstance(placement, str):
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {list(PLACEMENTS)} or a list of slot names, got {placement!r}")
        names = PLACEMENTS[placement]
        names = list(slots) if names is None else list(names)
    else:
        names = list(placement)
    missing = [n for n in names if n not in slots]
    if missing:
        raise ValueError(f"slots {missing} don't exist in this neck (available: {list(slots)})")
    return names


def apply_adcf(
    model: DetectionModel,
    placement: str | list[str] = "all",
    mode: str = "replace",
    module: str = "adcf",
    gamma_init: float = 0.01,
    **adcf_kwargs: Any,
) -> DetectionModel:
    """Patch neck fusion blocks in place and record the spec in model.yaml["adcf"]."""
    unknown = set(adcf_kwargs) - set(ADCF_KEYS)
    if unknown:
        raise ValueError(f"unknown ADCF options: {sorted(unknown)}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if module not in MODULES:
        raise ValueError(f"module must be one of {MODULES}, got {module!r}")
    if module == "cbam" and (mode != "residual" or adcf_kwargs):
        raise ValueError("module='cbam' is only a residual control and takes no ADCF options")

    slots = neck_slots(model)
    for name in _resolve_placement(placement, slots):
        i = slots[name]
        old = model.model[i]
        if not isinstance(old, C2f):
            raise RuntimeError(f"slot {name} (layer {i}) is already patched")
        c1, c2 = old.cv1.conv.in_channels, old.cv2.conv.out_channels
        if mode == "replace":
            new = ADCF(c1, c2, **adcf_kwargs)
            initialize_weights(new)  # same BatchNorm eps/momentum Ultralytics uses everywhere else
        else:
            refiner = ADCF(c2, c2, **adcf_kwargs) if module == "adcf" else CBAM(c2)
            initialize_weights(refiner)
            new = ResidualRefine(old, refiner, c2, gamma_init)
        # Ultralytics' forward loop relies on these attributes to route tensors.
        new.i, new.f, new.type = old.i, old.f, type(new).__name__
        new.np = sum(p.numel() for p in new.parameters())
        model.model[i] = new.to(next(old.parameters()).device)

    model.yaml["adcf"] = {
        "placement": placement,
        "mode": mode,
        "module": module,
        **({"gamma_init": gamma_init} if mode == "residual" else {}),
        **adcf_kwargs,
    }
    return model


def adcf_layers(model: nn.Module) -> list[tuple[str, ADCF]]:
    """(slot name, ADCF module) for every ADCF block in a DetectionModel, in either mode."""
    out = []
    for name, i in neck_slots(model).items():
        m = model.model[i]
        m = m.refiner if isinstance(m, ResidualRefine) else m
        if isinstance(m, ADCF):
            out.append((name, m))
    return out


def make_trainer(adcf: dict[str, Any] | None) -> type[DetectionTrainer]:
    """Return a DetectionTrainer subclass that builds the ADCF variant.

    `adcf=None` gives the unmodified baseline, unless the model cfg already carries an
    "adcf" spec (e.g. resuming from an ADCF checkpoint).
    """

    class ADCFDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            cfg = copy.deepcopy(cfg) if isinstance(cfg, dict) else cfg
            spec = adcf if adcf is not None else (cfg.get("adcf") if isinstance(cfg, dict) else None)
            model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
            model = self.set_model_names_for_load(model)
            # LEARN: weights are matched by parameter *name* and shape, so we load twice.
            #  1. before patching: pretrained C2f/C3k2 weights land in the original blocks
            #     (which residual mode then keeps inside the wrapper);
            #  2. after patching: restores patched layers when resuming an ADCF checkpoint.
            #     For COCO weights this second pass matches nothing new and changes nothing.
            if weights:
                model.load(weights, verbose=False)
            if spec:
                apply_adcf(model, **spec)
            if weights:
                model.load(weights, verbose=verbose and RANK in {-1, 0})
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"ADCF spec: {spec}")
                model.info()
            return model

    return ADCFDetectionTrainer
