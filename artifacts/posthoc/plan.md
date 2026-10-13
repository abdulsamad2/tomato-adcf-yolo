# Post-hoc analysis plan (written before any of these analyses is computed)

Status: **post-hoc, descriptive.** No training, no new model, no model selection.
**Data: development arms only.** Arm A = source-exposed (1,047 families, 2,028 files) and arm B = source-independent
(213 families, 411 files), Phase 9 frozen manifests. The clean test split is NOT used. Its single locked pass stands.
Detections are the cached Phase 15/16/17/pretraining-audit inference passes (imgsz 512, conf 0.001, NMS 0.7,
max_det 300), truncated to the top 100 per image. Checkpoint: 24,570 updates. Seeds 0–4.

## Models
w100, w075, w050, w035 (Phase 15, from scratch); lpwa50 (Phase 16A); kd075 (Phase 17); r512 (pretrained 1.00×).

## Interventions (paired by seed)
| name | contrast |
|---|---|
| width 0.75× | w075 − w100 |
| width 0.50× | w050 − w100 |
| width 0.35× | w035 − w100 |
| LPWA-50 | lpwa50 − w050 |
| pretraining | r512 − w100 |
| distillation | kd075 − w075 |

## Analyses
1. **Exposure distortion.** For each intervention: Δ_B, Δ_A and distortion D = Δ_A − Δ_B for six-class AP,
   with 95% paired-t CIs over seeds. Ratio |Δ_A| / |Δ_B| reported where Δ_B ≠ 0.
2. **Error decomposition (TIDE-style counts).** At conf ≥ 0.25 (the Phase 12 operating point), each detection in
   descending score is assigned exactly one type. A GT can be taken by only one TP.
   - **TP:** IoU ≥ 0.5 with an untaken same-class GT.
   - **Duplicate:** IoU ≥ 0.5 with a same-class GT already taken.
   - **Class error:** IoU ≥ 0.5 with a GT of a different class only.
   - **Localization error:** 0.1 ≤ max same-class IoU < 0.5.
   - **Class + localization error:** 0.1 ≤ max IoU < 0.5 with different-class GT only.
   - **Background:** max IoU < 0.1 with every GT.
   - **Missed GT:** GTs not taken by any TP. These split into "near-missed" (some detection overlaps at IoU ≥ 0.1) and
     "fully missed".

   Reported per model on arms A and B as counts per 100 GT boxes (detection-side types) and as a share of GTs
   (missed types).
3. **IoU sweep.** Six-class AP at each of the 10 COCO IoU thresholds 0.50–0.95 (pycocotools precision array), and
   class-agnostic localization recall at conf ≥ 0.25 for IoU 0.50, 0.55, …, 0.95.
4. **Lesion size (exploratory thresholds).** GT boxes are binned by sqrt(area) into tertiles computed on arm-B GT only,
   with no model involved. COCO small/medium/large (32², 96² px on the 640 px images) is reported alongside.
   Metric: class-agnostic localization recall @0.50 per bin, plus the per-bin deltas for each intervention.
5. **Per class (exploratory).** Per class: n GT, six-class AP, localization recall @0.50. Deltas for width 0.75×
   and distillation, with seed-paired CIs. Classes with < 30 arm-B GT boxes are flagged. No significance test per class.
6. **Accuracy vs size.** Params and GFLOPs vs AP on arms A and B for the four widths, with LPWA-50, distilled 0.75×
   and pretrained 1.00× as markers.
7. **Protocol decision comparison.** Apply the Phase 15 rule (smallest reduced width whose Holm-adjusted one-sided
   lower bound of Δ vs 1.00× exceeds −1.5 AP) once with arm-B AP (the frozen result) and once with arm-A AP, and
   report which width each protocol would select.
8. **Paired seed plots.** Seed-level lines for AP, localization recall @0.50 and AP75: distillation (w075 → kd075),
   LPWA (w050 → lpwa50) and width (w100 → w075 → w050 → w035).
9. *(Included in 1/8.)*
10. **Operating curves.** Arm B. For conf thresholds 0.05, 0.10, …, 0.95: class-agnostic localization recall @0.50
    and false positives per family (Phase 12 matching). Recall at matched FP/family of 2, 3 and 4 is read by linear
    interpolation. Models: w100, w075, kd075, r512.
11. **Calibration (supplementary).** Arm B. Detections with conf ≥ 0.05 are put in 10 equal-width confidence bins.
    Precision per bin = share of detections that are TP (class-correct, IoU ≥ 0.5, one-to-one).
    D-ECE = Σ_b (n_b / N) · |mean conf_b − precision_b|. Reliability table per model, mean over seeds.
12. **Leakage characterisation (metadata/labels only).** From the released Variant-c files:
    - files per source family and the family-size distribution;
    - share of official-val files and families with a sibling in official train;
    - the augmentation type of each `_augN` file, inferred by comparing its YOLO labels with its original's labels
      (identity / horizontal flip / vertical flip / both flips / 90° rotation / other geometric, with box-matching
      tolerance 0.01). Photometric changes cannot be seen in labels.

    Exploratory: arm-A localization recall @0.50 (w100, 5 seeds) split by the inferred transform type of the arm-A
    file vs its being an original.

## Outputs
`cs/posthoc/results.json`, figures in `cs/posthoc/fig_*.png`, and a new section in `docs/results_summary.md`.
