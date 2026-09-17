# 07 · Critical review of the project (and what was changed because of it)

A reviewer's job is to find the weakest link. Better that we find it first. Issues
are ranked by how likely they are to sink the paper.

## A. Scientific design

### A1. The first ADCF design was set up to lose — **fixed**
*Problem.* "Replace" mode swaps the COCO-pretrained C3k2 neck blocks for randomly
initialised ADCF blocks, and those blocks are ~14% smaller. The baseline therefore
starts with a pretrained neck and more capacity. If ADCF doesn't win, we can't tell
whether the *idea* failed or the *setup* did.

*Fix.* `mode: residual` (`ResidualRefine`): keep the pretrained block and add ADCF on
top, `out = block(x) + γ·ADCF(block(x))`, with γ per channel starting at 0.01. At step
0 the model behaves almost exactly like the pretrained baseline, so any gain comes
from ADCF. Both modes are in the ladder (`yolo11s-adcf-replace`, `yolo11s-adcf`); the
val results decide.

*New risk this creates.* Residual mode **adds** ~0.9 M params (9.4 → 10.4 M). "It's just
more parameters" becomes the objection → answered by the CBAM control (same slots,
same residual form) and the fusion ablations (`add`/`concat` have almost the same
parameter count as `gated`).

### A2. No answer to "why not just use a bigger image or a P2 head?" — **fixed**
*Problem.* The paper's story is "preserve small-lesion detail". The two standard,
simple ways to do that are (1) a higher input resolution and (2) an extra stride-4
(P2) detection head. A reviewer will ask about both immediately, and if a 960 px
YOLO11s beats ADCF at 640, the contribution looks pointless.

*Fix.* The `resolution` table: YOLO11s at 640/960, with and without P2, with and
without ADCF. The honest claim becomes one of:
- ADCF adds accuracy **on top of** P2 and/or 960 (complementary): strongest story;
- ADCF at 640 matches the 960 baseline at lower latency: an efficiency story;
- neither: the method isn't worth publishing as-is; see section D.

A `yolo11-p2.yaml` was added because Ultralytics has none for YOLO11.

### A3. "Any attention module would do" — **fixed**
*Problem.* Attention plug-ins in YOLO necks are extremely common in agriculture
papers. Without a control, a gain can't be credited to ADCF's specific design.

*Fix.* `yolo11s-cbam`: CBAM in the same slots with the same residual form. The
fusion ablation then shows which ADCF ingredient matters, with `nohighpass`
added to test the high-pass idea directly.

### A4. The test set is small, so small gains are unreliable — **fixed**
*Problem.* 15% of 1,796 photos ≈ 270 test photos. Three seeds measure *training*
noise, not *test-sampling* noise. A +0.8 mAP gain on 270 photos can easily be luck.
A Welch t-test on 3 seeds has very low power.

*Fix (two tools):*
- `adcf-compare --a yolo11s --b yolo11s-adcf`: **paired bootstrap** over test photos
  → 95% CI of Δ mAP. Cheap, uses existing runs.
- `tv-prepare --cv-folds 5` + `adcf-run --cv` + `adcf-collect --cv`: **5-fold grouped
  cross-validation**. Every photo is tested once (≈1,800 test photos in total) and
  the paired t-test across folds is far more powerful. Use it for the headline
  "baseline vs final model" claim only (10 trainings).

### A5. Peeking at test while developing — **process fixed**
*Problem.* The first runner evaluated only on test. Choosing replace vs residual,
P2 or not, or the placement by looking at test numbers turns test into a second
val set, and the reported gain becomes optimistic.

*Fix.* Every run now writes `results_val.json` **and** `results_test.json`, and
`adcf-collect` writes `tables_val.md` (labelled "use for decisions") separately
from `tables_test.md`. The rule: **freeze `adcf_final` in the config from val, then
open the test tables.** Cross-validation is the final confirmation.

### A6. Evaluation at the wrong resolution — **fixed (bug)**
*Problem.* Eval `imgsz` was global. A model trained at 960 would have been tested at
640 and look bad. *Fix:* each run is evaluated and timed at its own training size.

### A7. Novelty is incremental — **not fixable in code; manage it**
Depthwise and dilated convs, SE attention, high-pass residuals and gating are all
known. ADCF is a sensible *combination*. Reviewers at strong venues will say so.
Make the paper's value come from:
1. a **clean, leakage-free benchmark** of Tomato-Village detection (a contribution by itself);
2. **rigorous evidence**: controls, resolution/P2 comparison, CV, bootstrap CIs;
3. **mechanistic analysis**: gate maps and α inside vs outside lesions, per-class and per-size AP;
4. **generality**: gains on YOLOv8/11/26 (and ideally a second dataset, see D3).

Target an applied venue (note 06) and don't oversell novelty.

## B. Data

### B1. Official split leakage — handled (note 01). **Also quantify it** (D1).
### B2. Label quality unknown
Nobody has looked at the boxes yet. Label noise caps achievable mAP and can hide
real gains. *Action:* inspect `label_preview.png`, and after tier 1 look at the
worst false positives and negatives (`runs/*/eval_val/` confusion matrix and
predictions). Some "errors" may be missing labels; mention this under limitations.
### B3. Class imbalance unknown
Check `dataset_table.md`. If a class has very few test boxes, its AP is noise and
the mean is unstable. Options applied **to every model equally**: `cls_pw`
(Ultralytics class weighting) in `train:`, or report macro and micro averages.
### B4. 8× offline augmentation copies
Each epoch sees every training photo 8 times, which is roughly 8× compute and a real risk
of overfitting to those specific transforms. The `data` table (tier 3) tests
training without them. If it's as good, switch **all** runs to no-copies and train
more epochs; it's much cheaper.
### B5. EXIF orientation not yet verified
The pipeline checks sizes consistently, but whether labels were drawn on rotated
images can only be seen in the preview.

## C. Engineering

| Issue | Status |
|---|---|
| Resume after crash between "training done" and marker write → Ultralytics refuses to resume a finished checkpoint | **fixed**: detected via stripped optimizer |
| Only 4-node necks supported (no P2) | **fixed**: 4- and 6-node necks, named slots |
| Weight loading for residual mode (pretrained names vs wrapped names) | **fixed**: load before and after patching; tested for pretrained and resume |
| Ablation Δ was vs the baseline instead of vs full ADCF | **fixed**: per-table reference |
| Tests only on CPU, synthetic data; never on CUDA or the real dataset | **open**: first lab run is the real test. Run `pytest` there first |
| Multi-GPU (DDP) untested | **open**: use a single GPU per job (`--device 0`); run several jobs in parallel on different GPUs with `--only` |
| FPS measures the network only (no NMS, no preprocessing) | documented; report end-to-end from val's `speed_ms` as well |
| GFLOPs from thop miss element-wise ops (the gate multiply, high-pass subtract) | documented; latency is the ground truth |

## D. What's still missing for a strong paper (not implemented, decide with supervisor)

1. **Leakage quantified:** train YOLO11s on the *official* split, report its val
   mAP next to the clean test mAP. One extra training; strong motivating result.
2. **Cross-site generalisation:** train on Jaipur, test on Jodhpur. Needs site labels
   for the `IMG*` files (ask the dataset authors).
3. **Second dataset:** a public field plant-disease *detection* dataset (PlantDoc is
   the usual choice; verify its licence and annotation format) with the same ladder.
   This does more for the "generality" claim than any extra ablation.
4. **Robustness:** evaluate test photos under blur, low light and JPEG compression
   (field conditions). Cheap: evaluation only.
5. **Deployment:** export to ONNX/TensorRT and time on an edge device (Jetson) if
   "efficient" stays in the title.
