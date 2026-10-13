# Phase 15 pre-registration: leakage-aware width right-sizing

Frozen 2026-10-12 03:00:05, before any Phase 15 training. Machine-readable copy: `p15_prereg.json`.

## Question
How much detector width is required when model selection uses source-INDEPENDENT rather than source-exposed performance?

Separately authorised deployment/right-sizing study. The Phase 12-14 mechanism search stays closed. No new module of any kind in this phase.

## Architectures (from scratch, identical depth/topology)

| key | width | params | GFLOPs@512 | yaml sha256 |
|---|---|---|---|---|
| w100 | 1.00x | 2.5914 M | 4.15 | `5629466b8f2c8134` |
| w075 | 0.75x | 1.5656 M | 2.707 | `157eb1175501af4f` |
| w050 | 0.50x | 0.8111 M | 1.549 | `06fded4e046bfca5` |
| w035 | 0.35x | 0.5375 M | 1.16 | `2d33b2ec8bad9bd7` |

## Training
Random initialisation for ALL four widths, including 1.00x. No pretrained tensor is loaded (yaml-built model, pretrained=False). Rationale: pretrained transfer is 98.21% / 7.94% / 15.22% / 22.93% of params for 1.00x / 0.75x / 0.50x / 0.35x and would confound width with initialisation quality.

source=configs/experiments.yaml [train], identical to runs_p4/r512 except init; data=data/tomato_village (leakage-free source-family split, 10080 train files); imgsz=512; batch=16; nbs=64; optimizer=SGD; lr0=0.01; lrf=0.01; cos_lr=False; epochs=39; steps_per_epoch=630; total_updates=24570; warmup_updates=933; close_mosaic_epochs=5; patience=1e6 (disabled); multi_scale=False; rect=False; deterministic=True; augmentation=unchanged Ultralytics defaults (as runs_p4/r512)

Seeds [0, 1, 2, 3, 4]. Same five seeds for every width. No seed added post hoc to any width.

Before step 1 of every run the trainer's computed schedule (steps/epoch, epochs, total updates, warmup updates, mosaic-close epoch) must equal runs_p4/r512/seed0/schedule.json, else the run aborts (exit 3).

Before training, every run checks (a) built params == audited params and (b) for w100, the first conv weight differs from yolo11n.pt (no silent pretrained load).

Checkpoints {'epoch10': 6300, 'epoch19': 11970, 'epoch29': 18270, 'epoch39': 24570}. **Primary 24570**, secondary 18270. NONE. The clean test split is not read by any Phase 15 script.

## Primary test
- type: non-inferiority, paired across the 5 seeds
- contrast: Delta(w) = AP_B(w) - AP_B(w100), per seed
- margin_AP_points: 1.5
- hypotheses: H0(w): mean Delta(w) <= -1.5  vs  H1(w): mean Delta(w) > -1.5, w in {w075,w050,w035}
- test: one-sided paired t, df=4: t = (mean Delta + 1.5) / SE
- multiplicity: Holm step-down across the three reduced-width comparisons, familywise alpha 0.05 (one-sided).
- rule: w is NON-INFERIOR iff its Holm-adjusted one-sided p < 0.05 AND its multiplicity-adjusted one-sided lower confidence bound > -1.5, where the bound for the hypothesis at Holm rank k (k=1 smallest p) uses level alpha/(3-k+1); a Bonferroni alpha/3 simultaneous bound is also reported as the conservative check. Testing stops at the first non-rejected rank (standard Holm).
- interpretation_guard: Failure to establish non-inferiority is NOT evidence of inferiority and is reported as 'non-inferiority not established'.
- margin_rationale: Median paired 5-seed SD of arm-B AP differences across the 45 existing Phase 9 model pairs is 0.84 AP; at +/-1.0 equivalence would only be attainable for |mean diff| < 0.20. Margin -1.5 specified by the user before any Phase 15 run.
- not_evidence: A non-significant difference is never reported as equivalence.

## Secondary
- tost_equivalence: two-sided TOST at +/-1.5 AP, paired t df=4, per width vs w100, Holm over 3 widths
- trajectories: all four checkpoints + per-epoch clean-val curves, descriptive
- superiority: one-sided paired t that w > w100 on arm B, Holm over 3 widths (Outcome D screen)
- family_bootstrap: 1000 resamples of source families (arms A and B independently, all files of a family together); paired seed-mean Delta and interaction per width
- reported_at: [18270, 24570]
- endpoints: six-class AP50, AP75, APs, AR100 (A and B); class-agnostic AP50-95, AP50, AP75 (A and B); localization recall @IoU .50 and .75 (A and B); conditional class correctness (A and B); false positives per family (A and B); per-class AP (A and B)
- exposure interaction: {"Delta_ind": "metric(w, B) - metric(w100, B)", "Delta_exp": "metric(w, A) - metric(w100, A)", "I": "Delta_ind - Delta_exp, per seed; paired t + family bootstrap"}

## Convergence
- statistic: g(w) = AP_B(w, 24570) - AP_B(w, 18270), per seed
- flag: w is 'still improving' iff mean g(w) > 0.5 AP AND one-sided paired t p < 0.05
- consequence: A flagged width is reported as FIXED-BUDGET performance, never as an asymptotic capacity limit.
- extension_rule: No schedule extension inside Phase 15. Any extension is a new, separately frozen phase that must apply the same longer schedule to ALL four widths. No rescue training of any single width.

## Resources
- trainable params; GFLOPs @512; serialized fp16 weight size; peak inference VRAM (batch 1 and 16); GPU batch-1 latency; GPU batch-16 throughput; CPU batch-1 latency
- GPU speed-up is claimed only if measured. Untrained timing 8.90 ms (0.35x) vs 9.13 ms (1.00x) at batch 16 is treated as no meaningful latency gain.

## Decision rule
Among reduced widths that pass corrected non-inferiority, select the SMALLEST by parameter count. If none pass, report the observed accuracy-size Pareto frontier without claiming preserved accuracy.

## Integrity rules
- The clean test split is not inspected during development.
- The margin and all rules are not changed after any result is seen.
- Best-validation checkpoints are never the primary outcome; the primary is the fixed 24570-update checkpoint.
- No edge-device speed claim is made from desktop GPU timings.

## Interpretation of outcomes
- **A** w035 non-inferior -> select 0.35x; STOP architecture experimentation.
- **B** w050 non-inferior, w035 not -> frontier lies between 0.35x and 0.50x; authorise design of ONE matched-budget localization-preserving asymmetric-width candidate (separate frozen phase).
- **C** w050 not non-inferior -> report the Pareto frontier; architecture work only with a mechanism-based budget-redistribution hypothesis; no module invention.
- **D** any narrower width superior to w100 on arm B (Holm one-sided p < 0.05) -> report as a capacity-generalization result and test whether arm A gives the opposite sign. D can co-occur with A/B/C.

No knowledge distillation in Phase 15.
