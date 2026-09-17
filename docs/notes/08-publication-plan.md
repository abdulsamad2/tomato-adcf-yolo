# 08 · Publication plan: how we show real improvement

## What "improvement" has to mean

We need to show the final model is **better than strong baselines under identical
conditions**, and **why**. Not "higher number on one run". Concretely, all of:

1. Higher mAP50-95 than YOLO11s, **beyond seed noise** (mean ± std) and with a
   bootstrap CI above 0 (`adcf-compare`), confirmed by 5-fold CV (`adcf-collect --cv`).
2. The gain survives the fair comparisons: same resolution, same head type, CBAM
   control, other YOLO versions.
3. The gain is **where the method says it should be**: AP_small, lesion classes
   with fine texture, α maps favouring detail on lesions.
4. The cost is stated honestly: params, GFLOPs, **measured** latency.

## The improvement ladder (the paper's central table)

Each row adds one change on top of the previous. Every row is a real run (3 seeds).
The "step" column shows what *that* change bought, and the cost columns show what it
cost. This format (used by YOLOX, ConvNeXt and many others) makes improvements
transparent.

| Step | Run | Change | Question it answers |
|---|---|---|---|
| 0 | `yolo11s` | baseline, 640 px | reference |
| 1 | `yolo11s-adcf-replace` | ADCF replaces neck blocks | does the raw idea help? |
| 2 | `yolo11s-adcf` | ADCF as residual refinement (pretrained neck kept) | was step 1 held back by losing pretrained weights? |
| 3 | `yolo11s-p2-adcf` | + stride-4 detection head | do fine lesions need a finer head as well? |
| 4 | `yolo11s-p2-adcf-960` | + 960 px input | does detail from resolution add up with ADCF? |

**Decision rule, on VAL only:** keep a step if its mean gain exceeds the larger of
the two seed stds *and* its latency cost is acceptable for the story you want. The
kept steps define **ADCF-YOLO**. Write that choice into `adcf_final` (and
`cv.runs`) *before* opening `tables_test.md`.

Note that steps 1→2 are a design choice between alternatives (report both), while 3
and 4 are additions. If step 3 or 4 is kept, the fair baseline must have the same
additions, which is why the `resolution` table has `yolo11s-p2`, `yolo11s-960` and
`yolo11s-p2-960`. Claim only the gain over the **matched** baseline.

## Model adjustments, and why each might improve results

| Adjustment | Mechanism | Expected effect | Cost | Where tested |
|---|---|---|---|---|
| Residual insertion (`mode: residual`) | keeps COCO-pretrained neck; ADCF learns a correction | most likely source of a *clean* gain over baseline | +~0.9 M params | ladder 1→2 |
| P2 head (`yolo11-p2.yaml`) | stride-4 predictions for tiny lesions | AP_small ↑ | GFLOPs ↑ (large feature maps) | ladder 3, resolution |
| 960 px input | 2.25× pixels, lesions no longer sub-pixel | AP_small ↑ for all models | ~2.25× compute | ladder 4, resolution |
| Placement (`p3`, `top_down`, `bottom_up`) | ADCF only where detail/context conflict most | similar accuracy, lower cost | — | placement |
| Gate type (`channel`) | per-channel choice | small ↑ or none | ≈0 | fusion |
| High-pass on/off | explicit edge/speckle emphasis | tests the "detail" premise directly | ≈0 | fusion |
| Offline copies off (`noaug`) | less redundancy per epoch | same accuracy at ~1/8 compute, or less overfitting | — | data |
| Model scale (n / m) | ADCF on smaller/bigger models | gains often largest on small models | — | scale |

**If results disappoint** (on val), in order of expected value:
1. Longer schedule for *all* models (new layers may be undertrained at 200 epochs).
2. ADCF only at `p3`/`fine` slots (less disruption, same small-object benefit).
3. `gamma_init` 0.1 (the refinement may be learning too slowly), or 0 (pure ReZero).
4. Class weighting `cls_pw: 0.5` for all models, if imbalance is severe.

Each attempt is a new named run in the config, so nothing is silently tuned.

## What we publish

**Title (pick after tier 2):** keep "Efficient" only if latency supports it.

**Contributions (write the version the data supports):**
1. A leakage-free, reproducible benchmark protocol for Tomato-Village detection
   (grouped, deduplicated, stratified split + 5-fold CV), with a quantified leakage effect.
2. ADCF, a lightweight neck module that adaptively fuses detail and context per
   pixel, pluggable into YOLOv8/11/26.
3. A controlled evaluation: improvement ladder, matched resolution/P2 baselines,
   attention control, ablations, CV and bootstrap confidence intervals.
4. Analysis of what the gate learns (α maps, lesion vs background) and where the
   gains come from (object size, class).

**Tables:** (1) dataset stats + leakage; (2) ladder; (3) main comparison with
baselines; (4) resolution/P2 fairness; (5) fusion ablation; (6) placement;
(7) per-class AP; (8) 5-fold CV with paired test.
**Figures:** architecture; ADCF block; qualitative detections; α maps; PR curves;
accuracy-vs-latency scatter (all models: a good one-glance summary).

**Released with the paper:** code (this repo), `split_manifest.csv` and CV manifests,
configs, trained weights for the final model and baseline.

## Execution order and GPU budget

| Stage | Command | Trainings |
|---|---|---|
| Data | `tv-download`, `tv-prepare`, `tv-prepare --cv-folds 5`, `tv-stats` | 0 |
| 1. Ladder, one seed | `adcf-run --tier 1 --seeds 0` | 4 |
| 2. Ladder, all seeds → **decide `adcf_final` on val** | `adcf-run --tier 1`, `adcf-collect` | +8 |
| 3. Baselines, controls, resolution, ablations | `adcf-run --tier 2` | 57 |
| 4. Freeze → open test tables, bootstrap | `adcf-collect`, `adcf-compare` | 0 |
| 5. Cross-validation (baseline vs final) | `adcf-run --cv`, `adcf-collect --cv` | 10 |
| 6. Speed + gates | `adcf-bench`, `adcf-gates` | 0 |
| 7. Tier 3 (optional) | `adcf-run --tier 3` | 18 |

After stage 1, multiply the measured `train_hours` to get the real budget (960 px
runs take roughly 2–2.5× longer). If it's too much, cut seeds on placement and
scale first, never on the ladder, main table or CV.

## Honesty clause

We cannot know in advance that ADCF will win; nobody can. What this plan guarantees
is that **whatever we report is real and defensible**. If ADCF doesn't beat the
matched baselines, the paper can still stand on contribution 1 plus a careful
negative/neutral result ("attention-style neck fusion does not help once resolution
is matched"). That is less exciting but publishable, and far better than a gain
that disappears when a reviewer reruns it.
