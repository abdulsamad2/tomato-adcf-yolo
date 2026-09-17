# 02 · YOLO primer: just enough to understand where ADCF goes

## One-stage detection in one paragraph

A one-stage detector like YOLO looks at the image once. It predicts, for every cell
of a few grids, "is there an object here, what class, and where is its box". It then
removes overlapping duplicates with NMS (non-maximum suppression). YOLO26 is trained
to skip NMS ("NMS-free"), which matters when you compare speed (note 05).

## The three parts

```
image ─► BACKBONE ─► P3, P4, P5 ─► NECK ─► 3 fused maps ─► HEAD ─► boxes, classes, scores
```

### Backbone: extracts features

A stack of convolutions that repeatedly halves spatial size and adds channels.
We tap it at three depths:

| Level | Stride | Size at 640 input | Sees | Good for |
|---|---|---|---|---|
| P3 | 8 | 80 × 80 | small areas, fine texture | small lesions, speckles |
| P4 | 16 | 40 × 40 | medium areas | medium lesions |
| P5 | 32 | 20 × 20 | large areas, but blurry | whole-leaf symptoms, context |

**The core tension:** shallow maps (P3) are sharp but "don't know what they're
looking at"; deep maps (P5) understand the scene but have lost the fine detail.

### Neck: fuses the levels (FPN + PAN)

- **FPN (top-down, Lin et al. 2017):** upsample P5, concatenate with P4, fuse; then
  upsample that and concatenate with P3, fuse. Semantic knowledge flows *down* to
  the sharp maps.
- **PAN (bottom-up, Liu et al. 2018):** downsample the fused P3, concatenate with
  the P4 result, fuse; again for P5. Precise localisation flows back *up*.

Each "fuse" in Ultralytics is a **C2f** block (YOLOv8) or **C3k2** block
(YOLO11/YOLO26): a stack of 3×3 convolutions with shortcut connections. They mix
the concatenated features the same way everywhere in the image. They have no
explicit notion of "here, trust the sharp detail; there, trust the context".

```
            P5 ──────────────┬──────────────────────────────► concat ─► [fuse 3: p5_out] ─► head (large)
             │ upsample      │                                  ▲ downsample
             ▼               │                                  │
P4 ─► concat ─► [fuse 0: p4_td] ──┬─────────────► concat ─► [fuse 2: p4_out] ─► head (medium)
                   │ upsample     │                  ▲ downsample
                   ▼              │                  │
P3 ─────► concat ─► [fuse 1: p3_out] ────────────────┴──────────────────────────► head (small)
```

### Head: predicts

For each of the three fused maps, a decoupled head predicts box coordinates and
class scores at every cell.

## Where ADCF goes, and why there

ADCF replaces the four `[fuse]` blocks (named slots `p4_td, p3_out, p4_out, p5_out`
in the code). The neck is the only place where features at **different scales meet**,
so it's where "detail vs context" is actually decided. The backbone and head stay
untouched, which:

- keeps COCO-pretrained weights for most of the network, so the comparison with
  the baseline is fair;
- makes ADCF a plug-in for any YOLO with this neck (we test v8, 11, 26).

## How the code does it (`src/adcf_yolo/build.py`)

Ultralytics builds models from YAML files. Adding a new block to the YAML parser
means patching library internals that change between versions. Instead we:

1. build the normal model (`DetectionModel("yolo11s.yaml")`);
2. find the 4 neck fusion blocks and read their in/out channels;
3. replace each with `ADCF(c_in, c_out)`, copying the routing attributes
   (`.f` = which layer's output to take, `.i` = own index);
4. **then** load pretrained weights. They match by name and shape, so everything
   except the new ADCF layers loads.

`make_trainer()` wraps this in Ultralytics' `DetectionTrainer`, so training,
augmentation, loss, EMA and checkpointing are all stock and identical for baseline
and ADCF.

## Terms you'll meet

- **Depthwise conv:** one filter per channel (no channel mixing), about 9× fewer
  parameters than a normal 3×3 conv with the same channels.
- **Pointwise (1×1) conv:** mixes channels, doesn't look at neighbours.
- **Dilated conv:** 3×3 weights spread apart, so it sees wider with the same cost.
- **Receptive field:** the input area that influences one output pixel.
- **EMA:** a moving average of the weights, which is what's actually evaluated.
- **Mosaic:** Ultralytics' augmentation that stitches 4 images; turned off for the
  last 10 epochs (`close_mosaic`).
