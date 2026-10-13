# Experiments

There is one folder per step of the paper, run in this order. Every script is **byte-identical** to the version that
produced the reported results:
- `MANIFEST.json` lists the SHA-256 of each file;
- it also confirms that the 26 scripts frozen in the pre-registrations (Steps 4, 5, 6 and 8) match their recorded hashes.

| Step | What it does | Main scripts | Records |
|---|---|---|---|
| `1_protocol` | Model size (YOLO11 n/s/m) and input size (320–640 px) at a fixed 24,570-update budget. The 512 px runs are the **COCO-pretrained 1.00× reference** and the distillation teacher. | `p3_run.py`, `p4_run.py`, `*_stats.py` | `results/p3_*`, `results/p4_*` |
| `2_evaluation_arms` | Builds and freezes the source-exposed (A) and source-independent (B) arms; scores and bootstraps | `p9_freeze.py`, `p9_eval.py`, `p9_stats.py` | `artifacts/phase9/` |
| `3_mechanism` | Localization vs classification decomposition; single-class control | `p12_analyze.py`, `p13_run.py`, `p13_eval.py` | `artifacts/phase12/`, `phase13/` |
| `4_width_rightsizing` | **Pre-registered.** 1.00/0.75/0.50/0.35× from scratch, 5 seeds, non-inferiority (margin −1.5 AP) | `p15_freeze.py` → `p15_queue.ps1` → `p15_post.ps1` | `artifacts/phase15/`, `results/p15_*` |
| `5_lpwa` | **Pre-registered.** One width-reallocation candidate at the 0.50× parameter budget | `p16_budget.py`, `p16_design.py`, `p16_freeze.py` → `p16_queue.ps1` | `artifacts/phase16/`, `results/p16_*` |
| `6_distillation` | **Pre-registered.** One localization-focused distillation configuration into 0.75× | `p17_kd.py` (loss), `p17_dryrun.py`, `p17_freeze.py` → `p17_queue.ps1` → `p17_post.ps1` | `artifacts/phase17/`, `results/p17_*` |
| `7_pretraining_audit` | Exploratory: pretrained vs from-scratch 1.00× (plan written before scoring) | `px_pretrain_*.py` | `artifacts/pretrain_audit/`, `results/pretrain_inflation.json` |
| `8_locked_test` | **Run once.** Test-set verification, freeze, single-pass evaluation of 4 models × 5 seeds | `ct_verify.py`, `ct_freeze.py`, `ct_eval.py` | `artifacts/clean_test/`, `results/clean_test_table.*` |
| `9_posthoc` | Plan hashed before computing: error decomposition, IoU sweep, lesion size, operating curves, calibration, leakage structure | `ph_analyses.py`, `ph_leakage.py` | `artifacts/posthoc/` |

## Running on your machine

The scripts were run from the parent folder of this repository and hard-code that path. Leave them untouched (their
hashes matter) and generate a runnable copy:

```bash
python tools/relocate_paths.py --root /path/to/parent --out /path/to/parent
```

Each pre-registered step follows the same pattern:
1. a freeze script writes the rules and refuses to run if any result already exists;
2. a queue script trains the seeds one at a time;
3. a post script scores the checkpoints and writes the statistics.

To re-score the locked test from the released weights without retraining, use `tools/rescore_locked_test.py`.
