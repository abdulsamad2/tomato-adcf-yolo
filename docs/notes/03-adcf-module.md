# 03 · The ADCF module, layer by layer

Code: `src/adcf_yolo/modules.py` (about 150 lines, worth reading alongside this note).

## The idea in one sentence

Compute a **detail** view and a **context** view of the same features, then let a
small **gate** decide, pixel by pixel, how much of each to keep.

## Walkthrough with real shapes (YOLO11s, slot `p3_out`, 640 × 640 input)

```
input  X            (B, 512, 80, 80)   concat of upsampled P4 features and backbone P3
│
├─ reduce: 1×1 Conv-BN-SiLU            → x (B, 64, 80, 80)     c_ = int(c2 · e) = int(128 · 0.5)
│
├─ DetailBranch(x)                     → D (B, 64, 80, 80)
│     dw   = 3×3 depthwise conv(x)          local texture
│     hf   = x − AvgPool3×3(x)              high-pass: edges and speckles; flat colour cancels out
│     D    = 1×1 conv(dw + hf)
│
├─ ContextBranch(x)                    → C (B, 64, 80, 80)
│     y    = 3×3 depthwise, dilation 2  →  3×3 depthwise, dilation 3     receptive field 11×11 cells
│     y    = 1×1 conv(y)
│     s    = sigmoid(MLP(GlobalAvgPool(y)))  (B, 64, 1, 1)              "what's in the whole image"
│     C    = y · s
│
├─ FusionGate(D, C)                    → α (B, 1, 80, 80)   values in (0, 1)
│     α    = sigmoid( 1×1 conv( 3×3 dw conv( 1×1 conv( concat(D, C) ))))
│
├─ F = α · D + (1 − α) · C             → (B, 64, 80, 80)
│
└─ out = 1×1 conv( concat(x, F) )      → (B, 128, 80, 80)  same output shape as the C3k2 it replaces
```

The receptive field of 11×11 cells is 88 px at P3 (stride 8) and 352 px at P5
(stride 32). Context grows with the level, as it should.

## Why each piece is there

| Piece | Purpose | What would break without it | Ablation run |
|---|---|---|---|
| 1×1 reduce | Makes the branches cheap (fewer channels) | cost grows ~4× | `-wide` (e=0.9) |
| Depthwise 3×3 | Local texture at ~9 params/channel | detail branch can't see neighbours | `-context` (no detail) |
| High-pass `x − blur(x)` | Explicitly amplifies small spots and edges, which are easily washed out by the upsample + concat | small lesions blend into leaf colour | — (candidate extra ablation) |
| Dilated cascade | Wide receptive field at depthwise cost | context branch no wider than detail | `-detail` (no context) |
| Global channel attention | Image-level cue ("this leaf is generally yellow"), which helps separate deficiency from disease | only local context | — |
| Gate α | Location-dependent choice | fixed 50/50 mix everywhere | `-add`, `-concat` |
| Spatial vs channel gate | 1 value per pixel (interpretable) vs per pixel·channel (flexible) | — | `-chgate` |
| Zero-init last gate conv | α = 0.5 exactly at start → no initial bias; stable training | random early preference for one branch | — |
| Shortcut concat with x | Keeps the original features, eases gradient flow (like C2f) | harder optimisation | — |

**Reading the gate:** α → 1 means "trust the fine detail here", α → 0 means "trust
the surroundings". `adcf-gates` plots α and measures its mean inside vs outside
lesion boxes. That lets you *show* the mechanism instead of just asserting it.

## Cost, measured (YOLO11s, nc = 8)

| Slot | In → out | C3k2 params | ADCF params |
|---|---|---|---|
| p4_td | 768 → 256 @ 40² | 443.8 k | 219.1 k |
| p3_out | 512 → 128 @ 80² | 127.7 k | 64.5 k |
| p4_out | 384 → 256 @ 40² | 345.5 k | 170.0 k |
| p5_out | 768 → 512 @ 20² | 1511.4 k | 667.6 k |
| **Whole model** | | **9.42 M, 21.4 GFLOPs** | **8.11 M, 18.6 GFLOPs** |

Most ADCF parameters sit in the 1×1 `reduce` and `proj` convs. The branches are
nearly free because they are depthwise.

**Two consequences to understand before writing:**

1. ADCF is *smaller* than what it replaces. If accuracy improves, capacity isn't the
   reason. If it doesn't, capacity might be, which is why `yolo11s-adcf-wide`
   (e = 0.9, ≈ baseline size) exists.
2. **Fewer FLOPs does not guarantee faster.** Depthwise convs and element-wise ops
   (multiply, sigmoid, subtraction) are *memory-bound*: GPUs spend their time moving
   data, not multiplying. FLOP counters (thop) also ignore most element-wise ops. So
   measure latency (note 05) and report both. Never claim "faster" from GFLOPs alone.

## Related designs you must cite and distinguish (reviewers will ask)

| Prior work | Similarity | How ADCF differs (the claim you must support) |
|---|---|---|
| SENet (Hu et al., 2018) | channel attention | used only inside the context branch |
| CBAM (Woo et al., 2018) | spatial + channel attention | CBAM re-weights *one* feature map; ADCF chooses *between two* purpose-built views |
| SKNet (Li et al., 2019) | softmax selection between kernel sizes | SK selects per channel, globally pooled; ADCF's gate is per pixel and the branches are designed (high-pass vs dilated) |
| ASFF (Liu et al., 2019) | learned per-pixel weights when fusing pyramid levels | ASFF weights *different scales*; ADCF weights *detail vs context views within a fusion node* |
| BiFPN / EfficientDet (Tan et al., 2020) | weighted feature fusion | BiFPN learns one scalar per input, the same at every location |

Verify each citation yourself before using it. The "differs" column is your
positioning, and experiments (fusion ablation + gate maps) must back it up.

## Ideas if results disappoint (make these decisions on **val**, not test)

- Put ADCF only where it helps (see the placement ablation).
- The high-pass term may be too strong at P5; try it only at P3/P4.
- Longer schedule: new layers start from scratch while the rest is pretrained.
