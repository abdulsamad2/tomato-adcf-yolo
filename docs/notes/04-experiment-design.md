# 04 · Experiment design: what to run and what each run proves

The whole plan lives in `configs/experiments.yaml`. This note explains it.

## The question each table answers

A paper is a chain of claims. Each claim needs an experiment that could have
*disproved* it.

| Claim | Evidence | Runs (table) |
|---|---|---|
| C1. ADCF improves detection on field images | ADCF vs YOLO11s, same everything else; bootstrap CI; 5-fold CV | `yolo11s`, `yolo11s-adcf` (ladder, main, cv) |
| C2. The improvements add up | each ladder step's gain on val | ladder |
| C3. The result is competitive with current YOLOs | YOLOv8s, YOLO26s baselines | `yolov8s`, `yolo26s` (main) |
| C4. ADCF is a general plug-in | the gain also appears on other YOLO versions | `yolov8s-adcf`, `yolo26s-adcf` (main) |
| C5. It isn't just "any attention module" or "more parameters" | beats CBAM in the same slots, same residual form | `yolo11s-cbam` (control) |
| C6. It isn't replaced by resolution or a P2 head | gain over the **matched** 960 / P2 baselines | resolution |
| C7. Both branches are needed | detail-only and context-only are worse | `-detail`, `-context` (fusion) |
| C8. *Adaptive* fusion beats fixed fusion | gated > add and concat (≈ same params) | `-add`, `-concat` (fusion) |
| C9. The high-pass detail cue matters | full > `-nohighpass` | fusion |
| C10. Placement choice is justified | all slots vs P3 only / top-down / bottom-up | placement |
| C11. It helps small lesions specifically | AP_small improves more than AP_large | every run's COCO breakdown |
| C12. The gate learns something meaningful | α differs inside vs outside lesions, figures | `adcf-gates` |
| C13. It stays efficient | params, GFLOPs, **measured** latency | `adcf-bench` |
| C14. It works across model sizes | n and m variants | tier 3 (scale) |

If an experiment contradicts a claim, **drop or rewrite the claim**. Don't drop
the experiment.

## Fairness rules (the code enforces most of these)

1. **Same data, same split** for every run (one `dataset.yaml`, one manifest).
2. **Same hyperparameters.** `train:` is shared. We pin `optimizer: SGD` because
   Ultralytics' `auto` can choose differently per model. No per-model tuning, since
   tuning only your own model is a classic unfair comparison.
3. **Same initialisation policy.** Everything starts from COCO weights; only the
   new ADCF layers start from scratch.
4. **Same budget.** Same epochs and patience. (ADCF has fresh layers and might want
   longer. If you test that, give the baseline the same longer schedule.)
5. **Model selection on val, reporting on test.** `best.pt` is picked by val
   fitness. Design decisions (replace vs residual, P2, resolution) are made from
   `tables_val.md`; `tables_test.md` is opened only after `adcf_final` is frozen.
6. **Multiple seeds.** Training is noisy. Two runs of the *same* model can differ
   by several tenths of mAP. We use 3 seeds and report mean ± std.

## Why seeds matter (a concrete way to think about it)

Suppose YOLO11s scores 62.1, 62.9 and 61.8 mAP50-95 over three seeds, and ADCF
scores 62.8, 63.4 and 62.2. The means differ by +0.5, but the spread inside each
model is about ±0.5. That gain is inside the noise. With a single seed you might
have reported +1.6 (61.8 → 63.4) or −0.7 (62.9 → 62.2). Reviewers know this. *(These
numbers are made up to illustrate the reasoning.)*

## Budget: estimate before you commit

Counts in the config: tier 1 = 4 runs, tier 2 = 19 runs, tier 3 = 6 runs; × 3 seeds =
**87 trainings**, plus **10** for 5-fold CV (`adcf-run --dry-run` prints exact counts).
Note 08 has the recommended order.

Run `adcf-run --tier 1 --seeds 0` first, read `train_hours` in
`runs/*/seed0/train_done.json`, and multiply. If it's too much:

- keep 3 seeds for tier 1 and the fusion ablation (the central claims);
- use 1 seed for placement and scale, and say so in the paper;
- lower `epochs` for **all** runs (never for only some), e.g. 100 with patience 30;
- try `cache: ram` if the machine has the memory (faster data loading).

## Order of work (details and decision rules in note 08)

1. `adcf-run --tier 1 --seeds 0`. **Look at the curves** (`runs/*/seed0/results.png`):
   losses should fall, val mAP should rise and plateau. If the ADCF curve is much
   worse early on, that's expected (new layers); a much worse final result is a
   bug signal.
2. Finish tier 1. Run `adcf-collect`. Decide whether the story holds.
3. Tier 2. Tier 3 if budget allows.
4. `adcf-bench` once, at the end, for everything.
5. `adcf-gates` on `runs/yolo11s-adcf/seed0/weights/best.pt`.

## Useful commands

```bash
adcf-run --dry-run                           # see what's left
adcf-run --tier 1 --seeds 0 --device 0       # first pass
adcf-run --only yolo11s-adcf-add             # a single run, all seeds
adcf-run --cv                                # 5-fold CV jobs from the cv: section
adcf-compare --a yolo11s --b yolo11s-adcf    # paired bootstrap CI on test
adcf-run --reeval                            # recompute test metrics (e.g. after fixing eval code)
```

Interrupted? Run the same command again: it resumes from `last.pt`.

## Experiments worth adding later (only if they fit the story)

- **Official (leaky) split vs clean split** for YOLO11s: quantifies the leakage.
- **High-pass term off** (`x − blur(x)` removed): isolates that design choice.
- **Cross-location test:** train on one site, test on the other. Only the
  `Jaipur_Pots*` / `Jodhpur_Land*` files name their site; the `IMG*` files would need
  the authors' metadata. A strong generalisation experiment for "field environments".
- **Robustness:** evaluate on test images with blur, brightness or JPEG changes.
