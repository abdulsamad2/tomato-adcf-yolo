# Phase 16A pre-registration: LPWA-50

Frozen 2026-10-12 18:04:53. Machine-readable: `p16_prereg.json`.

## Motivation
Phase 15: uniform narrowing to 0.50x costs -1.80 AP on arm B; class correctness given a found box stays ~97% while localization recall@0.50 falls 7.0 points. Budget audit: the class head is <2% of params and the box branch is already floored at 64 ch, so the reallocatable budget is P5 (32.8%) -> P3/P4 (9.2%). Hypothesis: at equal params, moving width from the P5 path to the P3/P4 paths recovers localization and AP.

## Candidate (one, frozen)
params 786882, GFLOPs@512 1.965, P3/P4 x1.37, P5 x0.65 vs uniform 0.50x; head unchanged.

| layer | group | uniform 0.50x | LPWA-50 |
|---|---|---|---|
| 0 | stem | 8 | 8 |
| 1 | stem | 16 | 16 |
| 2 | stem | 32 | 32 |
| 3 | P3/P4 | 32 | 48 |
| 4 | P3/P4 | 64 | 88 |
| 5 | P3/P4 | 64 | 88 |
| 6 | P3/P4 | 64 | 88 |
| 7 | P5 | 128 | 88 |
| 8 | P5 | 128 | 88 |
| 9 | P5 | 128 | 88 |
| 10 | P5 | 128 | 88 |
| 13 | P3/P4 | 64 | 88 |
| 16 | P3/P4 | 32 | 48 |
| 17 | P3/P4 | 32 | 48 |
| 19 | P3/P4 | 64 | 88 |
| 20 | P5 | 64 | 48 |
| 22 | P5 | 128 | 88 |

Detect head: {"in_channels": [48, 88, 88], "box_branch_mid": [64, 64, 64], "box_branch_out": [64, 64, 64], "cls_branch_out": [8, 8, 8], "reg_max": 16, "nc": 8, "strides": [8, 16, 32]}

## Training
random (pretrained=False), as Phase 15; identical to Phase 15 (runner p16_run.py derived from p15_run.py)
Seeds 0-4. All five seeds run directly; no screening subset. Paired with U50/w100 seeds of the same index.

## Success (all three)
- **rule** LPWA-50 SUCCEEDS iff all three gates pass
- **G1_primary_AP** mean over seeds of Delta_i = AP_B(lpwa50, i) - AP_B(U50, i) at 24570 is >= +0.75 AP. The paired 95% t CI (df=4) is reported; statistical significance is NOT required.
- **G2_mechanism** mean over seeds of Delta locR50_B (localization recall @IoU .50, arm B, Phase 12 rules) > 0. Mechanistic corroboration; no separate significance test.
- **G3_resources** params <= 811,144 AND GFLOPs@512 < 2.707 (target <= 2.0, met at design: 1.965)

## Secondary
- noninferiority_vs_w100: Phase 15 rule: one-sided paired t, margin -1.5 AP, arm B @24570 (single comparison, no Holm)
- tost_vs_w100: +/-1.5 AP
- superiority_vs_U50: one-sided paired t p reported (not a gate)
- at_18270: all of the above repeated at 18270
- exposure: Delta on arm A and interaction I = Delta_B - Delta_A vs U50 and vs w100; family bootstrap 1000
- decomposition: class-agnostic AP, AP50/75, locR50/75, conditional class correctness, FP/family
- convergence: Phase 15 rule: still improving iff mean gain 18270->24570 > 0.5 AP and one-sided p < 0.05
- resources: params, GFLOPs, fp16 size, peak VRAM, GPU b1/b16 latency, CPU b1 latency vs U50 and w100

## Decision
- success: proceed to Phase 17: localization-focused distillation into LPWA-50 (separately frozen)
- failure: STOP architecture work; no second allocation, no module search

## Integrity
- The clean test split is not inspected during development.
- The margin and all rules are not changed after any result is seen.
- Best-validation checkpoints are never the primary outcome; the primary is the fixed 24570-update checkpoint.
- No edge-device speed claim is made from desktop GPU timings.
- No change to the candidate after any seed is trained.
