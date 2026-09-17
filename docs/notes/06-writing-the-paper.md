# 06 · Writing the paper

## Pick one title

The draft has two. Use one throughout:

- *Adaptive Detail–Context Fusion for Efficient Tomato Leaf Disease Detection in Field Environments*
- *Detail-Preserving Context-Gated YOLO for Tomato Leaf Disease Detection in Field Environments*

Pick after tier 2. Only keep "Efficient" if `adcf-bench` shows latency is equal or
better, not just GFLOPs.

## Structure, and where every number comes from

| Section | Content | Source in this repo |
|---|---|---|
| Abstract | problem → method → key numbers (mAP50-95 gain ± std, AP_small gain, params/latency) → one sentence on the fair split | `results/tables_test.md` |
| 1 Introduction | Field images are hard (clutter, shadows, veins, tiny lesions). Existing YOLO necks fuse uniformly. Contributions as 3–4 bullets | notes 02, 03 |
| 2 Related work | (a) plant disease detection, lab vs field data; (b) YOLO family; (c) feature pyramids and fusion; (d) attention/gating | note 03 table |
| 3 Dataset | Tomato-Village Variant-c, **the leakage analysis**, re-split protocol, class and size stats | `prepare_report.json`, `results/dataset/dataset_table.md` |
| 4 Method | overall figure; ADCF equations; placement; complexity | `modules.py`, note 03 |
| 5 Experimental setup | hardware, software versions, hyperparameters, seeds, metrics definitions | `configs/experiments.yaml`, `train_done.json` (env) |
| 6 Results | Table: main comparison. Table: per-class AP | `adcf-collect` |
| 7 Ablations | Tables: fusion (incl. capacity control), placement, scale | `adcf-collect` |
| 8 Analysis | gate maps figure + inside/outside α; failure cases; AP_small | `adcf-gates`, `runs/*/eval_test/` |
| 9 Limitations | one dataset, one region, 3 seeds, boxes at leaf level… | — |
| 10 Conclusion | | |

## Equations for the method section

With reduced input $x$:

$$D = \phi_{1\times1}\big(\mathrm{DW}_{3\times3}(x) + (x - \mathrm{AvgPool}_{3\times3}(x))\big)$$

$$\tilde C = \phi_{1\times1}\big(\mathrm{DW}^{d=3}_{3\times3}(\mathrm{DW}^{d=2}_{3\times3}(x))\big), \quad C = \tilde C \odot \sigma\big(\mathrm{MLP}(\mathrm{GAP}(\tilde C))\big)$$

$$\alpha = \sigma\big(g([D, C])\big) \in (0,1)^{1\times H\times W}, \qquad F = \alpha \odot D + (1-\alpha)\odot C$$

$$Y = \phi_{1\times1}([x, F])$$

where $\phi$ is Conv-BN-SiLU, DW is depthwise, $[\cdot]$ is concatenation, and $g$ is
1×1 → 3×3 DW → 1×1 conv.

## Figures to make

1. **Architecture overview** (backbone → neck with the 4 ADCF slots highlighted → head).
2. **ADCF block diagram** (the ASCII diagram in note 03, drawn properly; draw.io works).
3. **Dataset:** sample images per class + class and size histograms.
4. **Qualitative comparison:** same test images, YOLO11s vs ADCF detections; choose
   *both* successes and failures, since honest figures earn trust.
5. **Gate maps** from `adcf-gates`, with the inside/outside α numbers in the caption.
6. Optional: PR curves (`runs/*/seed0/eval_test/BoxPR_curve.png`).

## Reproducibility statement (include it)

- dataset commit (`SOURCE.txt`) and the published `split_manifest.csv`;
- code repository and commit hash;
- library versions (`train_done.json` → `env`);
- seeds, hardware, all hyperparameters (the config file).

## Phrases to avoid unless the data supports them

- "significantly" → only with a significance test you actually ran; otherwise "consistently"
- "real-time" → only with FPS on stated hardware
- "state-of-the-art" → only if you compared against the actual SOTA on this split,
  which nobody has, since the split is new. Say "outperforms the evaluated baselines".
- "lightweight/efficient" → back it with measured latency, not just params

## Where to submit (discuss with your supervisor)

Agriculture + computer vision venues fit well, e.g. *Computers and Electronics in
Agriculture*, *Smart Agricultural Technology*, *Plant Methods*, *Frontiers in Plant
Science*, or CV workshop tracks on agriculture (e.g. CVPR's Agriculture-Vision
workshop). Check each venue's scope and page limit before formatting.
