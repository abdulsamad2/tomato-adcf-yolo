"""Adaptive Detail–Context Fusion (ADCF) block.

ADCF replaces a C2f/C3k2 fusion block in the YOLO neck. It keeps the same
input/output channel contract (c1 -> c2, same H x W), so it is a drop-in swap.

    x ─ 1x1 reduce ─┬─ DetailBranch ──┐
                    │                 ├─ fusion (gated / add / concat) ─┐
                    ├─ ContextBranch ─┘                                  ├─ concat ─ 1x1 proj ─ out
                    └────────────────────── shortcut ────────────────────┘

LEARN: every paper-level claim about this block should be backed by an ablation
row in configs/experiments.yaml. See docs/notes/03-adcf-module.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

FUSION_MODES = ("gated", "add", "concat", "detail", "context")
GATE_TYPES = ("spatial", "channel")


class DetailBranch(nn.Module):
    """Local texture: 3x3 depthwise conv plus an explicit high-frequency residual.

    LEARN: `x - avgpool(x)` is a cheap high-pass filter. Smooth regions (uniform
    leaf colour) cancel out; abrupt changes (lesion borders, speckles) remain.
    """

    def __init__(self, c: int):
        super().__init__()
        self.dw = Conv(c, c, 3, g=c)  # depthwise: one 3x3 filter per channel, ~9*c params
        self.pw = Conv(c, c, 1)  # pointwise: mixes channels
        self.blur = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_freq = x - self.blur(x)
        return self.pw(self.dw(x) + high_freq)


class ContextBranch(nn.Module):
    """Surrounding context: cascaded dilated depthwise convs plus global channel attention.

    LEARN: a 3x3 conv with dilation d covers (2d+1)x(2d+1) pixels using only 9
    weights. Cascading d=2 then d=3 gives an 11x11 receptive field on the feature
    map, i.e. 88 px at P3 (stride 8) and 352 px at P5 (stride 32). The global
    average pool summarises the whole image (squeeze-and-excitation style).
    """

    def __init__(self, c: int, dilations: tuple[int, ...] = (2, 3), reduction: int = 4):
        super().__init__()
        self.dws = nn.Sequential(*(Conv(c, c, 3, g=c, d=d) for d in dilations))
        self.pw = Conv(c, c, 1)
        hidden = max(c // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.dws(x))
        return y * self.channel_att(y)


class FusionGate(nn.Module):
    """Predicts alpha in (0, 1): how much to trust detail (1) versus context (0).

    `spatial` gives one alpha per pixel (easy to visualise, the default).
    `channel` gives one alpha per pixel *and* channel (more flexible, ablation).
    """

    def __init__(self, c: int, gate: str = "spatial", reduction: int = 4):
        super().__init__()
        if gate not in GATE_TYPES:
            raise ValueError(f"gate must be one of {GATE_TYPES}, got {gate!r}")
        hidden = max(c // reduction, 8)
        self.net = nn.Sequential(
            Conv(2 * c, hidden, 1),
            Conv(hidden, hidden, 3, g=hidden),  # lets the gate look at a 3x3 neighbourhood
            nn.Conv2d(hidden, 1 if gate == "spatial" else c, 1),
        )
        # LEARN: zero-init the last conv so sigmoid(0) = 0.5 at step 0. Training starts
        # from an unbiased 50/50 blend and the network learns where to deviate.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, detail: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat((detail, context), 1)))


class ADCF(nn.Module):
    """Adaptive Detail–Context Fusion block (drop-in for C2f/C3k2 in the YOLO neck).

    Args:
        c1: input channels (the concatenated neck features).
        c2: output channels.
        e: bottleneck expansion; branches run at int(c2 * e) channels.
        fusion: "gated" (proposed), or ablations "add", "concat", "detail", "context".
        gate: "spatial" or "channel" (only used when fusion == "gated").
        dilations: dilation rates of the cascaded context convs.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 0.5,
        fusion: str = "gated",
        gate: str = "spatial",
        dilations: tuple[int, ...] = (2, 3),
    ):
        super().__init__()
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got {fusion!r}")
        c_ = int(c2 * e)
        self.fusion = fusion
        self.reduce = Conv(c1, c_, 1)
        self.detail = DetailBranch(c_) if fusion != "context" else None
        self.context = ContextBranch(c_, tuple(dilations)) if fusion != "detail" else None
        self.gate = FusionGate(c_, gate) if fusion == "gated" else None
        self.mix = Conv(2 * c_, c_, 1) if fusion == "concat" else None
        self.proj = Conv(2 * c_, c2, 1)  # sees fused features + shortcut (C2f-style)

        # Set store_alpha=True to keep the last gate map for visualisation (adcf-gates).
        self.store_alpha = False
        self.last_alpha: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        if self.fusion == "detail":
            fused = self.detail(x)
        elif self.fusion == "context":
            fused = self.context(x)
        else:
            d, c = self.detail(x), self.context(x)
            if self.fusion == "gated":
                alpha = self.gate(d, c)
                if self.store_alpha:
                    self.last_alpha = alpha.detach()
                fused = alpha * d + (1 - alpha) * c
            elif self.fusion == "add":
                fused = d + c
            else:  # concat
                fused = self.mix(torch.cat((d, c), 1))
        return self.proj(torch.cat((x, fused), 1))
