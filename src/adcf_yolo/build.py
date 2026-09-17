"""Insert ADCF into an Ultralytics YOLO neck and train it with the stock trainer.

Rather than forking Ultralytics or its YAML parser, we build the normal model and
swap the four neck fusion blocks (C2f in YOLOv8, C3k2 in YOLO11/YOLO26):

    slot 0  p4_td   top-down,  after Concat(P5_up, P4)
    slot 1  p3_out  top-down,  after Concat(P4_up, P3)  -> small-object head
    slot 2  p4_out  bottom-up, after Concat(P3_down, P4) -> medium-object head
    slot 3  p5_out  bottom-up, after Concat(P4_down, P5) -> large-object head
"""

from __future__ import annotations

import copy
from typing import Any

import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import initialize_weights

from adcf_yolo.modules import ADCF

NECK_SLOTS = ("p4_td", "p3_out", "p4_out", "p5_out")
PLACEMENTS: dict[str, tuple[int, ...]] = {
    "all": (0, 1, 2, 3),
    "p3": (1,),  # only the small-object path
    "top_down": (0, 1),
    "bottom_up": (2, 3),
}
ADCF_KEYS = ("placement", "e", "fusion", "gate", "dilations")


def neck_fusion_indices(model: DetectionModel) -> list[int]:
    """Layer indices of the 4 neck fusion blocks (C2f/C3k2, or ADCF if already patched)."""
    n_backbone = len(model.yaml["backbone"])
    idx = [i for i, m in enumerate(model.model) if i >= n_backbone and isinstance(m, (C2f, ADCF))]
    if len(idx) != 4:
        raise RuntimeError(f"expected 4 neck fusion blocks, found {len(idx)} at {idx}; unsupported architecture")
    return idx


def apply_adcf(model: DetectionModel, placement: str = "all", **adcf_kwargs: Any) -> DetectionModel:
    """Replace neck fusion blocks with ADCF in place. Records the spec in model.yaml["adcf"]."""
    unknown = set(adcf_kwargs) - set(ADCF_KEYS)
    if unknown:
        raise ValueError(f"unknown ADCF options: {sorted(unknown)}")
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {list(PLACEMENTS)}, got {placement!r}")

    indices = neck_fusion_indices(model)
    for slot in PLACEMENTS[placement]:
        i = indices[slot]
        old = model.model[i]
        if isinstance(old, ADCF):
            raise RuntimeError(f"layer {i} is already ADCF")
        c1, c2 = old.cv1.conv.in_channels, old.cv2.conv.out_channels
        new = ADCF(c1, c2, **adcf_kwargs)
        initialize_weights(new)  # same BatchNorm eps/momentum Ultralytics uses everywhere else
        # Ultralytics' forward loop relies on these attributes to route tensors.
        new.i, new.f, new.type = old.i, old.f, "ADCF"
        new.np = sum(p.numel() for p in new.parameters())
        model.model[i] = new.to(next(old.parameters()).device)

    model.yaml["adcf"] = {"placement": placement, **adcf_kwargs}
    return model


def adcf_layers(model: nn.Module) -> list[tuple[str, ADCF]]:
    """(slot name, module) for every ADCF block in a DetectionModel."""
    indices = neck_fusion_indices(model)
    return [(NECK_SLOTS[s], model.model[i]) for s, i in enumerate(indices) if isinstance(model.model[i], ADCF)]


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
            if spec:
                apply_adcf(model, **spec)
            # LEARN: load *after* patching. Weights are matched by name and shape, so the
            # COCO-pretrained backbone/head load, while the new ADCF layers (different
            # names) keep their fresh init. Resuming an ADCF checkpoint loads everything.
            if weights:
                model.load(weights)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"ADCF spec: {spec}")
                model.info()
            return model

    return ADCFDetectionTrainer
