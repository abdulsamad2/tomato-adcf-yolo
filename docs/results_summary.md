# Tomato leaf disease detection: what was done and what we found

Summary for review, 13 Oct 2026. All AP values are COCO AP50-95 × 100 unless stated otherwise.

## 1. Setup

**Dataset.** Tomato-Village, Variant-c (object detection). These are field photos of tomato leaves with lesion boxes in 8 classes. Two of the classes, Early Blight and Healthy, have no boxes at all.

**The leakage problem.** The authors made 7–8 augmented copies of each photo *before* splitting the files at random.
- The 14,368 files come from only 1,796 original photos.
- All 1,484 photos behind the official validation files also have copies in the training set.
- There is no official test split.

**The fix: a leakage-free re-split** (`src/adcf_yolo/data/prepare.py`, manifest in `splits/split_manifest.csv`).
- Files are grouped by original photo, and near-duplicate photos (dHash, Hamming distance ≤ 4) are merged into clusters.
- The photos are split 70/15/15, stratified by class, giving **1,260 train / 268 val / 268 test photos**.
- Augmented copies stay in train only (10,080 train files).
- The test set has 268 images, 3,205 boxes and 261 clusters.

**Two evaluation arms**, built from the official validation files and used in Phase 9 and later:

| Arm | Contents | Size |
|---|---|---|
| **B: source-independent** | Photos whose source never appears in our train split. **Primary development metric.** | 213 photo families, 411 files |
| **A: source-exposed** | Photos whose source photo is in our train split. Shows how leaky evaluation behaves. | 1,047 families, 2,028 files |

**The standard protocol** from Phase 3 onwards, unless a phase says otherwise:

| Setting | Value |
|---|---|
| Model | Ultralytics YOLO11n (8.4.155), 512 px input |
| Optimiser | SGD, lr0 0.01, lrf 0.01, momentum 0.937, weight decay 0.0005 |
| Batch | 16 (nbs 64) |
| Budget | 39 epochs × 630 steps = **24,570 updates**; 933 warm-up updates; mosaic off for the last 5 epochs |
| Other | Early stopping off, AMP, deterministic, default Ultralytics augmentation |
| Checkpoints | Fixed grid at 6,300 / 11,970 / 18,270 / 24,570 updates. No best-validation checkpoint selection. |
| Seeds | 5 paired seeds (0–4) per arm |
| Evaluation | conf 0.001, NMS IoU 0.7, max 300 detections, top 100 per image, pycocotools, no TTA |
| Error breakdown | conf 0.25, one-to-one box matching at IoU 0.50 and 0.75. Reports *localization recall* (share of GT boxes found) and *conditional class correctness* (class right, given the box was found). |
| Statistics | Paired-seed t-tests or intervals, plus a bootstrap that resamples whole source-photo families (1,000 resamples) |
| Hardware | NVIDIA RTX 5000 Ada (32 GB) |

Every phase from 15 on was **pre-registered**: rules, margins and script hashes were frozen before training, under `artifacts/phase15`, `phase16`, `phase17` and `clean_test`.

## 2. Phase-by-phase results

### Early phases (Sep 21 – Oct 7): mostly negative or protocol-setting
- **ADCF module (the original idea).** ADCF is a gated detail/context fusion block in the YOLO neck. **No benefit.**
  - YOLO11s + ADCF: −0.4 AP (p = 0.75). Replace mode: +0.6 (p = 0.71).
  - 5-fold grouped cross-validation: 20.1 vs 19.9.
  - The learned gate stayed at α ≈ 0.50 everywhere, both inside and outside boxes.
  - CBAM, P2 head and 960 px controls were also null.
  - These early runs were also scored on the clean test split. That is disclosed below.
- **Leakage probe.** Seen-source photos scored 42–56 AP, against 15.5–17 for unseen photos, so the official split inflates AP about **2.7×**.
- **Phase 2B, augmentation at an equal update budget.** The shipped augmented copies help a little. Validation AP at 18k updates: originals only 20.6, 8× exact duplicates 20.2, shipped augmented copies 23.1.
- **Phase 3, model size** (validation AP at 18,270 updates): n 22.4, s 23.1, m 22.7. No meaningful gain from size, so **YOLO11n** was used from here on.
- **Phase 4, input size:** 320 → 20.0, 416 → 21.7, **512 → 23.0**, 640 → 22.4. 512 was chosen (equivalent to 640 at ~64% of the GFLOPs).
- **Phase 6, removing detection heads:** keeping only P3+P4 or only P4+P5 cost about 1.1–1.3 AP, so the full head was kept.
- **Phases 9–14, exposure and mechanism analyses.**
  - Ranking models on arm A vs arm B reorders them.
  - Capacity gains seen on arm A are mostly a **localization** effect.
  - Training a single-class detector (Phase 13) did not change localization: class-agnostic AP on arm B was 20.75 vs 20.79 ± 0.15 for the six-class model, and localization recall @0.50 was 41.9 vs 41.5.

### Phase 15: width right-sizing, all widths trained from scratch (main result)
Uniform channel width at 1.00× / 0.75× / 0.50× / 0.35× of YOLO11n.
- **Design:** 5 seeds each, all trained from scratch, because pretrained weights would only fit the 1.00× model.
- **Primary test:** non-inferiority on arm B at 24,570 updates. Margin −1.5 AP, Holm-corrected across the three narrower widths.

| Width | Params | GFLOPs @512 | Arm B AP | Δ vs 1.00× | Adjusted lower bound | Verdict | Arm A AP |
|---|---|---|---|---|---|---|---|
| 1.00× | 2.59 M | 4.15 | 13.65 ± 0.39 | — | — | reference | 29.76 |
| **0.75×** | **1.57 M** | **2.71** | 13.34 ± 0.54 | **−0.31** | −1.32 | **non-inferior** (also equivalent by TOST ±1.5) | 23.96 |
| 0.50× | 0.81 M | 1.55 | 11.85 ± 0.52 | −1.80 | −2.80 | not established | 17.54 |
| 0.35× | 0.54 M | 1.16 | 10.45 ± 0.89 | −3.20 | −3.95 | not established | 13.21 |

- **Selected model: 0.75×**, with 60% of the params and 65% of the GFLOPs.
- **Leaky evaluation exaggerates the cost of shrinking 5–19×.** Shrinking to 0.75× / 0.50× / 0.35× costs −5.8 / −12.2 / −16.5 AP on arm A, against −0.3 / −1.8 / −3.2 on arm B (seed means). The family-bootstrap interaction is +5.5 [4.4, 6.5], +10.2 [8.2, 11.9] and +13.2 [10.9, 15.3].
- **The loss is localization, not classification.**

  | | 1.00× | 0.75× | 0.50× | 0.35× |
  |---|---|---|---|---|
  | Localization recall @0.50 on B | 36.9 | 34.6 | 29.9 | 27.2 |
  | Class correctness on found boxes | ~96–97% for every width | | | |

- The narrower widths were still improving at 24,570 updates, so these numbers are **fixed-budget** results.
- **Speed:** no GPU gain; batch-16 latency is 6.6–7.1 ms for every width. Batch-1 CPU latency falls from 27 ms to 17 ms.

### Phase 16A: localization-motivated architecture (failed, so architecture work stopped)
**LPWA-50** is the same topology with channels moved from the P5 path to the P3/P4 paths, at the uniform-0.50× parameter budget: 0.787 M params, 1.965 GFLOPs (GFLOPs target ≤ 2.0).

| Gate (vs uniform 0.50×, arm B, 5 seeds) | Result |
|---|---|
| ΔAP ≥ +0.75 | **+0.26** [−0.51, +1.03] ❌ |
| Δ localization recall @0.50 > 0 | +1.31 [+0.33, +2.28] ✅ |
| Budget | ✅ |

- Under the frozen rule, architecture work stopped.
- Most of its gain appeared on arm A (+1.6 AP) rather than arm B (+0.26); the interaction is −1.2 [−2.1, −0.5]. Under leaky evaluation it would have looked like a win.

### Exploratory: how much is pretraining worth?
The COCO-pretrained YOLO11n was compared with the from-scratch one. Everything else was identical; the plan was written before scoring.

| | Arm A | Arm B |
|---|---|---|
| Pretrained | 44.82 | 17.59 |
| From scratch | 29.76 | 13.65 |
| **Gain** | **+15.1** | **+3.9** |

- Exposure exaggerates the value of pretraining about **4×** (interaction +11.1 [9.6, 12.7]).
- The arm-B gain is real: +3.9 [3.3, 4.6], 5 of 5 seeds positive.

### Phase 17: localization-focused distillation into the selected 0.75× (passed)
- **Teacher:** pretrained 1.00× (`runs_p4/r512` seed 0, final checkpoint), kept frozen.
- **Student:** the Phase 15 0.75×, trained from scratch, with no architecture change and 0 added parameters.
- **Loss:**
  `L = detection loss + 1.0 · 1.5 · KL(teacher ‖ student over the box-distribution bins, on foreground anchors, T = 1) + 1.25 · P3/P4 spatial-attention matching`.
  The 1.25 weight came from a rule fixed in advance: both distillation terms start equal in size on training batches. There was no tuning.
- **Checks before training:** with distillation turned off, the loss matched the ordinary trainer exactly. The teacher was confirmed frozen.

| Model (arm B, 24,570 updates) | Params | Arm B AP | Arm A AP | Loc. recall @0.50 |
|---|---|---|---|---|
| Pretrained 1.00× | 2.59 M | 17.59 | 44.82 | 43.7 |
| **Distilled 0.75×** | **1.57 M** | **14.30 ± 0.59** | 26.78 | 36.3 |
| From-scratch 1.00× | 2.59 M | 13.65 | 29.76 | 36.9 |
| Ordinary 0.75× | 1.57 M | 13.34 | 23.96 | 34.6 |

- **Primary result:** ΔAP vs ordinary 0.75× = **+0.95** [+0.21, +1.69], positive in 5 of 5 seeds (+0.96, +0.62, +0.47, +1.97, +0.74). Gate ≥ +0.75, so it **passes**.
- **Mechanism:** localization recall @0.50 +1.75; AP75 +1.70; class correctness unchanged.
- **Size and speed:** identical to the ordinary 0.75× (same params, GFLOPs and file size).
- **Caveats:**
  - The family bootstrap of ΔAP is wider, [−0.09, +1.72].
  - Arm A again exaggerates the gain (+2.8 vs +0.95).
  - It recovers only 22% of the gap to the pretrained model.

### Clean test: run once, after every selection was locked (13 Oct)
268 photos, 5 seeds per row, checkpoint at 24,570 updates, the same settings as development, no tuning.

| Model | Params | **Test AP** | AP50 | AP75 | AR100 | Loc. recall @0.50 | Loc. recall @0.75 | Class correct | FP / family |
|---|---|---|---|---|---|---|---|---|---|
| **Distilled 0.75× (selected)** | 1.57 M | **16.08 ± 0.96** | 33.07 | 13.71 | 31.29 | 39.64 | 22.05 | 96.5 | 2.63 |
| Ordinary 0.75× | 1.57 M | 14.95 ± 0.98 | 31.23 | 12.35 | 29.58 | 37.91 | 20.79 | 96.9 | 2.47 |
| From-scratch 1.00× | 2.59 M | 16.19 ± 0.69 | 33.69 | 13.46 | 31.55 | 40.27 | 22.12 | 96.6 | 2.68 |
| Pretrained 1.00× | 2.59 M | **17.27 ± 0.39** | 34.66 | 15.04 | 30.11 | 46.42 | 25.09 | 96.5 | 3.43 |

Distilled 0.75× minus each reference, with 95% CIs over seeds and from the family bootstrap:

| Reference | ΔAP | Seed CI | Bootstrap CI | Δ Loc. recall @0.50 |
|---|---|---|---|---|
| Ordinary 0.75× | **+1.14** | [−0.42, +2.70] | [−0.21, +2.45] | +1.73 |
| From-scratch 1.00× | **−0.11** | [−1.75, +1.53] | [−1.53, +1.17] | −0.63 |
| Pretrained 1.00× | **−1.19** | [−2.64, +0.26] | [−2.35, −0.08] | −6.78 |

The test reproduces the development ranking: pretrained > distilled 0.75× ≈ from-scratch 1.00× > ordinary 0.75×.

## 3. Main conclusions
1. **Leaky evaluation distorts design decisions, not just scores.** Exposed evaluation exaggerated the value of **width** (5–19×), a **localization architecture** (+1.6 vs +0.26), **pretraining** (~4×) and **distillation** (+2.8 vs +0.95).
2. **Honest right-sizing selects 0.75×** of YOLO11n: 1.57 M params, equivalent to 1.00× within ±1.5 AP on independent images.
3. **The bottleneck is localization.** Class accuracy on found lesions stays around 97% in every condition, while lesion-finding recall carries every gain and loss.
4. **Distillation brings the 0.75× level with the full-width from-scratch model** (test −0.11 AP) at 60% of the params. That gain is consistent but modest; on the test set alone its CI includes zero.
5. **The COCO-pretrained full-width model is still the most accurate** (test 17.3 AP).

## 4. Limitations and disclosures
- **Test-set history.** The clean test split was scored earlier for models outside this plan: the original ADCF pipeline runs (Sep 21 – Oct 3) and one Phase 2B evaluation (Oct 5). It is **"held out from all Phase 4–17 models and decisions"**, not "never opened". The 20 checkpoints in the final table had never been scored on it.
- **One dataset.** All results are on Tomato-Village only, with no external dataset yet.
- **Small, imbalanced test set.** It has 268 photos (261 clusters). Several classes are tiny (for example, 13 Potassium boxes), and two classes have no boxes.
- **Fixed budget.** All results are at 24,570 updates, and the smaller and from-scratch models were still improving at the end.
- **Speed.** All timing is on a desktop GPU and CPU. There are no edge-device measurements, and no speed-up is claimed (GPU latency is flat across widths).
- **Exploratory, not pre-registered.** The pretraining analysis was exploratory: its plan was written before scoring, but not pre-registered. The Phase 17 attention-term weight was set by a fixed rule on training batches.
- **Small gains vs uncertainty.** With 5 seeds and roughly 200–260 evaluation families, effects under about 1 AP have wide intervals. The family bootstrap is reported separately from seed variation.

## 5. Post-hoc analyses (development arms only, 13 Oct)

- **Status:** descriptive and post-hoc. There was no training, no new model and no selection, and the clean test was **not** reused.
- **Plan:** written and hashed before any computation (`artifacts/posthoc/plan.md`, sha256 `8d092e9a…`).
- **Data:** the cached arm A/B detections at 24,570 updates, 5 seeds.
- **Check:** the new code reproduces the frozen AP and localization recall of all 7 models exactly.

Figures: `artifacts/posthoc/fig_distortion.png`, `fig_pareto.png`, `fig_seeds.png`, `fig_iou.png`, `fig_size.png`, `fig_operating.png`, `fig_calibration.png`. Numbers: `artifacts/posthoc/results.json` and `leakage.json`.

### 5.1 Exposure distorts the measured effect of every intervention
Δ is the six-class AP effect of the intervention; D = Δ_A − Δ_B. CIs are 95% paired-t over seeds.

| Intervention | Δ independent (B) | Δ exposed (A) | Distortion D | \|Δ_A\| / \|Δ_B\| |
|---|---|---|---|---|
| Width 0.75× (vs 1.00×) | −0.31 [−1.19, +0.57] | −5.79 | −5.48 [−6.58, −4.39] | 18.9× |
| Width 0.50× | −1.80 [−2.80, −0.80] | −12.22 | −10.42 [−12.56, −8.27] | 6.8× |
| Width 0.35× | −3.20 [−4.17, −2.24] | −16.54 | −13.34 [−15.03, −11.66] | 5.2× |
| LPWA-50 (vs uniform 0.50×) | +0.26 [−0.51, +1.03] | +1.58 | +1.32 [−1.08, +3.73] | 6.2× |
| COCO pretraining (vs scratch 1.00×) | +3.93 [+3.25, +4.61] | +15.07 | +11.13 [+9.60, +12.66] | 3.8× |
| Distillation (vs ordinary 0.75×) | +0.95 [+0.21, +1.69] | +2.81 | +1.86 [−0.60, +4.32] | 3.0× |

In all six cases the exposed protocol overstates the effect, by 3–19×. For width and pretraining, the distortion CI excludes zero.

### 5.2 The model a practitioner would pick depends on the protocol
The frozen Phase 15 rule was applied unchanged: the smallest reduced width whose Holm-adjusted lower bound vs 1.00× is above −1.5 AP.

| Validation used | 0.75×: Δ, LCB | 0.50× | 0.35× | **Selected** |
|---|---|---|---|---|
| Independent (arm B) | −0.31, −1.32 ✅ | −1.80, −2.80 ✗ | −3.20, −3.95 ✗ | **0.75× (1.57 M)** |
| Exposed (arm A) | −5.79, −7.95 ✗ | −12.22, −14.06 ✗ | −16.54, −18.13 ✗ | **1.00× (2.59 M)**, no reduction passes |

Exposed validation would have rejected every compression and kept the 2.59 M model. Independent validation keeps a 40% smaller one. This is a **decision distortion**, not only a score inflation.

### 5.3 Error decomposition: the dominant error is missing lesions entirely
This is a TIDE-style count decomposition on arm B at conf ≥ 0.25, as events per 100 GT boxes. "Fully missed" is the share of GT boxes with no detection overlapping them at IoU ≥ 0.1.

| Model | TP | Localization err. | Class err. | Class+loc err. | Duplicate | Background FP | Near-missed GT % | **Fully missed GT %** |
|---|---|---|---|---|---|---|---|---|
| 1.00× scratch | 35.6 | 9.7 | 1.5 | 0.8 | 2.2 | 8.6 | 13.1 | **51.2** |
| 0.75× | 33.5 | 8.5 | 1.3 | 0.8 | 2.3 | 7.6 | 12.1 | **54.5** |
| 0.50× | 29.0 | 7.3 | 1.0 | 0.6 | 2.3 | 5.9 | 10.6 | **60.4** |
| 0.35× | 26.5 | 6.6 | 0.8 | 0.5 | 2.3 | 5.5 | 9.5 | **64.0** |
| LPWA-50 | 30.3 | 7.4 | 1.0 | 0.6 | 2.2 | 6.4 | 10.9 | 58.8 |
| 0.75× distilled | 35.2 | 8.9 | 1.3 | 1.0 | 2.3 | 8.4 | 13.0 | **51.8** |
| 1.00× pretrained | 42.3 | 12.5 | 1.8 | 1.4 | 1.7 | 12.8 | 16.3 | **41.4** |

- Classification errors are rare (≤ 1.8 per 100 GT) in every model.
- **Narrowing mainly raises the number of lesions that are fully missed:** +3.2, +9.2 and +12.7 points for 0.75× / 0.50× / 0.35×.
- **Distillation mainly lowers it:** −2.7 points, so it recovers 84% of the 0.75× loss; TP rises by +1.8. Class errors are unchanged (+0.0).
- Pretraining cuts fully missed lesions by 9.8 points.
- So the compression → localization chain is really **compression → more lesions never detected**, and distillation partly reverses exactly that.

### 5.4 IoU sweep
Arm B, at IoU thresholds 0.50 / 0.60 / 0.70 / 0.80 / 0.90.

| | AP at IoU | Loc. recall at IoU |
|---|---|---|
| Distillation, Δ | +0.78 / +0.71 / **+1.70 / +1.32** / +0.16 | +1.75 / +1.53 / +1.50 / +1.01 / +0.47 |
| Width 0.75×, Δ | −0.36 / −0.65 / −0.38 / −0.28 / −0.01 | −2.31 / −2.25 / −1.67 / −0.96 / −0.33 |

- The distillation AP gain peaks at IoU 0.70–0.80, so it improves box tightness as well as the number of lesions found.
- The distilled 0.75× has slightly higher AP at IoU ≥ 0.70 than the 1.00× scratch model (16.4 vs 15.1 at 0.70; 7.1 vs 6.0 at 0.80). This is exploratory.

### 5.5 Lesion size (exploratory thresholds: tertiles of √area on arm-B GT, 42.7 px and 77.1 px)

| Loc. recall @0.50 (arm B) | Small (< 43 px) | Medium | Large (≥ 77 px) |
|---|---|---|---|
| 1.00× scratch | 9.6 | 37.5 | 63.5 |
| 0.75× | 7.4 | 34.1 | 62.0 |
| 0.50× | 4.5 | 27.1 | 58.1 |
| 0.35× | 3.5 | 23.3 | 54.6 |
| 0.75× distilled | 8.7 | 36.6 | 63.5 |
| 1.00× pretrained | 14.3 | 47.1 | 69.6 |

- **Narrowing hurts small lesions most in relative terms:** 0.75× loses 23% of small-lesion recall vs 2% of large; 0.50× loses 53% vs 8%. The largest absolute losses fall on medium lesions.
- **Distillation recovers +1.3 [0.7, 1.9] small, +2.5 [0.9, 4.1] medium and +1.5 [−0.1, 3.0] large.** That is 59%, 75% and >100% of the 0.75× loss.
- Even the best model finds only 14% of small lesions.

### 5.6 Per class (exploratory, arm B; no per-class significance tests)
GT counts: Leaf Miner 3,138, Spotted Wilt Virus 899, Magnesium 497, Late Blight 78, Potassium 39, Nitrogen 32. **The last three are flagged: under 100 boxes.**

| Class | AP 1.00× | AP 0.75× | AP 0.75× distilled | Δ recall, width 0.75× | Δ recall, distillation |
|---|---|---|---|---|---|
| Leaf Miner | 18.0 | 17.1 | 17.9 | −2.8 | +1.9 |
| Spotted Wilt Virus | 19.5 | 19.4 | 20.2 | −0.1 | +0.0 |
| Magnesium Def. | 22.1 | 20.9 | 23.1 | −3.4 | +3.3 |
| Late Blight ⚑ | 16.9 | 17.2 | 21.1 | −0.8 | +3.8 |
| Potassium Def. ⚑ | 3.3 | 3.7 | 2.4 | −5.6 | +4.1 |
| Nitrogen Def. ⚑ | 2.1 | 1.6 | 1.1 | −3.1 | +5.0 |

- The recall loss from narrowing, and the recall gain from distillation, show up in almost every class.
- The two deficiency classes with fewest boxes are nearly undetectable (AP < 4) for every model.

### 5.7 Operating curves (arm B, confidence 0.05–0.95)

| Loc. recall @0.50 at matched FP/family | 2 | 3 | 4 |
|---|---|---|---|
| 1.00× scratch | 28.9 | 32.4 | 35.0 |
| 0.75× | 27.8 | 31.3 | 33.8 |
| 0.75× distilled | 29.1 | 32.6 | 35.0 |
| 1.00× pretrained | 32.6 | 36.5 | 39.3 |

- At equal false-alarm rates, the distilled 0.75× matches the 1.00× scratch model and beats the ordinary 0.75× by about 1.2 points.
- The pretrained model is best at every operating point.
- The distilled model's lower FP/family on the test set at conf 0.25 reflects where that threshold falls on its curve. It is not a better trade-off.

### 5.8 Calibration (supplementary)
- Detection-ECE (10 bins, conf ≥ 0.05, arm B) is 0.026–0.030 for every model.
- Distillation changes it by +0.0004 [−0.0031, +0.0038], and narrowing to 0.75× by +0.0009.
- Neither compression nor distillation measurably affects calibration.

### 5.9 Leakage characterisation (released Variant-c files, from labels only)
- **Structure:** 14,368 files from 1,796 source photos, and **every** family has exactly 8 files (1 original + 7 augmented).
- **Leakage:** 1,484 families span official train and val, and **100% of the 2,875 official-val files have a sibling in train.**
- **Augmentation types,** inferred from labels (12,572 copies):

  | Type | Share |
  |---|---|
  | Label-identical (photometric only) | 32.7% |
  | Crop / scale / shift | 28.3% |
  | Horizontal flip | 14.2% |
  | Vertical flip | 14.2% |
  | 90° rotations | 7.2% |
  | 180° rotation | 3.5% |

- **Exploratory, 1.00× scratch, loc. recall @0.50:**

  | File evaluated | Arm A (exposed) | Arm B (independent) | Exposure advantage |
  |---|---|---|---|
  | Original photo | 53.9 | 40.3 | +13.6 |
  | Photometric copy | 50.8 | 35.2 | +15.6 |
  | Flipped copy | 49.8–53.3 | 35.3–38.8 | +14.5 |
  | Crop/scale copy | 45.9 | 39.4 | **+6.4** |

  Exposure helps most when the evaluated image is a near-pixel copy of a training image, and least when it is cropped or rescaled.

## 6. Where the files are
- **Code:** `src/adcf_yolo/` (data split, original ADCF). Every phase script is in `experiments/` (byte-identical to what was run; hashes in `experiments/MANIFEST.json`).
- **Pre-registrations:** `artifacts/phase15/p15_prereg.md`, `artifacts/phase16/p16_prereg.md`, `artifacts/phase17/p17_prereg.md`, `artifacts/clean_test/prereg.md` (each with script hashes).
- **Results:**
  - `tomato-adcf-yolo/results/p15_stats.json`, `p16_stats.json`, `p17_stats.json`
  - `pretrain_inflation.json`
  - `clean_test_table.md` and `.csv`
  - `artifacts/clean_test/results.json`
- **Post-hoc analyses:** `artifacts/posthoc/` (plan, scripts, `results.json`, `leakage.json`, figures).
- **Trained models:** `runs_p15/` (widths), `runs_p16/` (LPWA-50), `runs_p17/` (distilled), `runs_p4/r512/` (pretrained 1.00×).
