# Clean-test results (one-time, 2026-10-13 09:08:19)

| model | AP | AP50 | AP75 | AR100 | loc_recall_iou50 | loc_recall_iou75 | cond_class_correct | fp_per_family |
|---|---|---|---|---|---|---|---|---|
| Distilled 0.75x (primary) | 16.08 ± 0.96 | 33.07 ± 1.40 | 13.71 ± 1.04 | 31.29 ± 1.73 | 39.64 ± 0.49 | 22.05 ± 0.12 | 96.52 ± 0.38 | 2.63 ± 0.13 |
| Ordinary 0.75x | 14.95 ± 0.98 | 31.23 ± 1.59 | 12.35 ± 1.35 | 29.58 ± 2.02 | 37.91 ± 0.89 | 20.79 ± 0.47 | 96.90 ± 0.30 | 2.47 ± 0.12 |
| From-scratch 1.00x | 16.19 ± 0.69 | 33.69 ± 1.39 | 13.46 ± 0.82 | 31.55 ± 1.73 | 40.27 ± 0.48 | 22.12 ± 0.60 | 96.64 ± 0.34 | 2.68 ± 0.16 |
| Pretrained 1.00x | 17.27 ± 0.39 | 34.66 ± 1.00 | 15.04 ± 0.46 | 30.11 ± 0.50 | 46.42 ± 0.59 | 25.09 ± 0.29 | 96.51 ± 0.32 | 3.43 ± 0.16 |

Paired Δ (distilled − reference), mean [95% CI over seeds]; family bootstrap for AP:

- vs Ordinary 0.75x: AP +1.14 [-0.42, 2.70] (bootstrap [-0.21, 2.45]); locR50 +1.73
- vs From-scratch 1.00x: AP -0.11 [-1.75, 1.53] (bootstrap [-1.53, 1.17]); locR50 -0.63
- vs Pretrained 1.00x: AP -1.19 [-2.64, 0.26] (bootstrap [-2.35, -0.08]); locR50 -6.78
