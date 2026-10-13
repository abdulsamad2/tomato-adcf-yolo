# Experiment scripts (frozen copies)

Every script here is a **byte-identical copy** of the script that produced the reported results.
- `MANIFEST.json` lists the SHA-256 of each file.
- `MANIFEST.json` also checks the 26 scripts whose hashes were frozen in the Phase 15, 16A, 17 and locked-test
  pre-registrations. All 26 match.

The scripts hard-code the original root `C:\Users\absamad\cs`, the parent of this repo. To run them elsewhere, use
`python tools/relocate_paths.py --root <ROOT> --out <ROOT>`; this makes a relocated copy and leaves these files
unchanged. Outputs go to sibling folders of the repo, such as `<ROOT>/phase15/`. The frozen records they produced are
in `../artifacts/`.

## Phases in the manuscript

The order below is the order of execution. Each phase has its own subfolder.

| Folder | What it does | Key scripts | Outputs (in this repo) |
|---|---|---|---|
| `phase01_adcf_and_protocol` | Original ADCF study and early protocol, using the `src/adcf_yolo` pipeline (`adcf-run`). **Note:** this pipeline also scored the test split, for these early models only. | `queue*.ps1`, `tier1.ps1`, `test_eval.ps1` | `results/tables_*.md`, `results/all_runs_record.csv` |
| `phase01b_leakage_probe` | Seen- vs unseen-source probe; grouped-CV audit | `leak_probe.py`, `leak_audit.py`, `cv_audit*.py` | `results/leakage_*.json`, `results/cv_fold_audit.json` |
| `phase02_augmentation_budget` | Originals vs duplicates vs shipped augmentations at an equal update budget (2 / 2B) | `bcd_run.py`, `p2b_run.py`, `p2b_*.py` | `results/phase_2_*`, `results/p2b_*` |
| `phase03_capacity` | YOLO11 n / s / m at an equal budget | `p3_run.py`, `p3_stats.py` | `results/p3_*` |
| `phase04_resolution` | 320 / 416 / 512 / 640 px. **The 512 px runs are the pretrained 1.00× reference and the Phase 17 teacher family.** | `p4_run.py`, `p4_stats.py` | `results/p4_*` |
| `phase05_width_audit` | Builds and audits the width variants (no training) | `width_audit.py`, `width_transfer.py` | `artifacts/widths/` |
| `phase06_detection_scales` | Removing detection heads (P3+P4 or P4+P5) | `scale_*.py`, `p6_run.py` | `results/p6_*`, `artifacts/scales/` |
| `phase07`–`phase08`, `phase10` | Decomposition and sampling diagnostics, power analysis | `p7_*`, `p8_*`, `p10_power.py` | — |
| `phase09_exposure_arms` | Freezes the source-exposed (A) and source-independent (B) arms; evaluates and bootstraps | `p9_freeze.py`, `p9_eval.py`, `p9_stats.py` | `artifacts/phase9/` |
| `phase12_mechanism` | Localization vs classification decomposition | `p12_infer.py`, `p12_analyze.py` | `artifacts/phase12/` |
| `phase13_single_class` | Single-class control | `p13_run.py`, `p13_eval.py` | `artifacts/phase13/` |
| `phase14_provenance` | Transform provenance of the evaluation files | `p14_prov.py` | `artifacts/phase14/` |
| `phase15_width_rightsizing` | **Pre-registered.** Four widths from scratch × 5 seeds, non-inferiority | `p15_freeze.py`, `p15_run.py`, `p15_eval.py`, `p15_stats.py`, `p15_bench.py` | `artifacts/phase15/`, `results/p15_*` |
| `phase16_lpwa` | **Pre-registered.** One width-reallocation candidate (LPWA-50) | `p16_budget.py`, `p16_design.py`, `p16_freeze.py`, `p16_run.py`, `p16_stats.py` | `artifacts/phase16/`, `results/p16_*` |
| `phase17_distillation` | **Pre-registered.** Localization-focused distillation into 0.75× | `p17_kd.py` (loss), `p17_dryrun.py`, `p17_freeze.py`, `p17_run.py`, `p17_stats.py` | `artifacts/phase17/`, `results/p17_*` |
| `pretraining_audit` | Exploratory: pretrained vs scratch 1.00× (plan written before scoring) | `px_pretrain_*.py` | `artifacts/pretrain_audit/`, `results/pretrain_inflation.json` |
| `clean_test` | **Locked test, run once.** Verification, freeze, single-pass evaluation | `ct_verify.py`, `ct_freeze.py`, `ct_eval.py` | `artifacts/clean_test/`, `results/clean_test_table.*` |
| `posthoc` | Post-hoc analyses (plan hashed before computing): error decomposition, IoU sweep, size, operating curves, calibration, leakage | `ph_analyses.py`, `ph_leakage.py` | `artifacts/posthoc/` |

## Typical sequence for the main experiments

```
python p15_freeze.py                       # writes phase15/p15_prereg.json (refuses if any run exists)
powershell -File p15_queue.ps1             # 20 training runs, one at a time
powershell -File p15_post.ps1              # eval -> bench -> stats  (results/p15_stats.json)
```

Phases 16A and 17 follow the same pattern: budget/design or dry run, then freeze, queue and post.
`clean_test/ct_eval.py` refuses to run if `RUN_LOCK` or `results.json` already exists.

## Not included

These helpers were left out; they produced no reported results:
- watchdog and CUDA/environment setup scripts;
- a side experiment on an unrelated skin-lesion dataset (`ham_*`, `isic_*`, `dup_*`, `nb_*`, `nd_*`).

The full list is in `MANIFEST.json` under `omitted_helpers`.
