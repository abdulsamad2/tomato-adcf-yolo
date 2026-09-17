# 00 · Roadmap: from code to paper

Read the notes in order. Each one explains *why* before *how*.

| # | Note | You will understand |
|---|------|---------------------|
| 01 | [Dataset and leakage](01-dataset-and-leakage.md) | Why we re-split Tomato-Village, and why that matters for the paper |
| 02 | [YOLO primer](02-yolo-primer.md) | Backbone / neck / head, feature pyramids, where ADCF goes |
| 03 | [The ADCF module](03-adcf-module.md) | Every layer of ADCF, tensor shapes, parameter cost |
| 04 | [Experiment design](04-experiment-design.md) | Baselines, ablations, seeds, fairness rules |
| 05 | [Metrics](05-metrics.md) | IoU, AP, mAP50 vs mAP50-95, AP_small, FPS done right, significance |
| 06 | [Writing the paper](06-writing-the-paper.md) | Structure, which result goes in which table/figure, related work |
| 07 | [Critical review](07-critical-review.md) | Weak points a reviewer would attack, and what was changed because of them |
| 08 | [Publication plan](08-publication-plan.md) | The improvement ladder, model adjustments, decision rules, what we publish |

## The checklist

Tick these off in order. Don't skip ahead: later steps rely on earlier ones being right.

**Phase A — data (≈ half a day)**
- [ ] `tv-download` on the lab machine; note the commit in `data/raw/Tomato-Village/SOURCE.txt`
- [ ] `tv-prepare`; read `data/tomato_village/prepare_report.json` (leak count, problems, duplicates)
- [ ] Look at every near-duplicate pair listed in the report. Are they really the same scene?
- [ ] `tv-stats`; open `results/dataset/label_preview.png`. Do boxes sit on lesions? (If boxes look rotated or shifted, stop and read note 01, "EXIF")
- [ ] Copy `data/tomato_village/split_manifest.csv` to `splits/` and commit it

**Phase B — tier 1: the improvement ladder (note 08)**
- [ ] `tv-prepare --cv-folds 5` too (needed later; cheap)
- [ ] `adcf-run --tier 1 --seeds 0`. Check training curves look sane
- [ ] Record hours per run (`train_done.json`) → estimate the GPU budget for everything else
- [ ] Finish seeds 1 and 2, `adcf-collect`, read **`tables_val.md`** only
- [ ] Decide replace vs residual (and P2) → set `adcf_final` and `cv.runs` in the config, commit

**Phase C — tier 2: baselines, controls, resolution, ablations**
- [ ] `adcf-run --tier 2`
- [ ] If adding 960 px or P2 to the final model: the claim is vs the *matched* baseline
- [ ] **Freeze the design.** Only now open `tables_test.md`; run `adcf-compare --a yolo11s --b <final>`

**Phase D — confirmation, efficiency, analysis**
- [ ] `adcf-run --cv` + `adcf-collect --cv` (baseline vs final, 5 folds)
- [ ] `adcf-bench` for all runs **in one session** on an idle GPU
- [ ] `adcf-gates` on the final model → figures + `gate_stats.json`
- [ ] Tier 3 if time allows

**Phase E — write** (note 06)

## Be honest with yourself about outcomes

A negative or mixed result is still publishable *if the analysis is good*, but it
changes the story. Decide what you'll claim only after Phase B, not before. The
leakage finding (note 01) and the fair re-split are a contribution on their own.
