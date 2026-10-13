# Beyond Metric Inflation: source-family exposure in tomato disease detection

Code, frozen pre-registrations and results for the study *"Beyond Metric Inflation: Source-Family Exposure Distorts
Compression, Pretraining and Distillation Conclusions in Tomato Disease Detection"* (manuscript in
[`docs/manuscript.docx`](docs/manuscript.docx); plain-language summary in [`docs/results_summary.md`](docs/results_summary.md)).

**The problem.** The public [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village) detection data
ships each source photograph as 8 files: the original plus 7 augmented copies. The official split was made at file
level, so **every validation file has a sibling in training.**

**What we did.** We built a source-family-aware split and two development evaluation arms:

| Arm | Contents |
|---|---|
| **Exposed (A)** | Images whose source photograph was seen in training |
| **Independent (B)** | Images whose source photograph was never seen in training |

We then measured how this exposure changes the conclusions drawn from detector experiments.

## Key results

**1. Exposure distorts every design decision.** Effect on six-class AP50-95, YOLO11n, 5 paired seeds:

| Intervention | Independent (arm B) | Exposed (arm A) |
|---|---|---|
| Shrink width to 0.75× | −0.31 | −5.79 |
| Shrink width to 0.50× | −1.80 | −12.22 |
| Width reallocation (LPWA-50) | +0.26 | +1.58 |
| COCO pretraining | +3.93 | +15.07 |
| Localization distillation | +0.95 | +2.81 |

**2. It changes the model you would deploy.** The same frozen selection rule picks:
- a **1.57 M**-parameter model (0.75×) with independent validation;
- the **2.59 M** model with exposed validation, because it rejects every reduced width.

**3. The detectors miss lesions; they rarely misname them.**
- About half of all ground-truth lesions receive no detection near them.
- Disease misclassification is under 2 per 100 lesions.
- Compression mainly increases missed lesions, and small lesions suffer most.

**4. Distillation partly repairs the compact model.** Distilling box distributions and P3/P4 attention from a
pretrained teacher raises the 0.75× model by +0.95 AP [95% CI 0.21, 1.69], with no change to its architecture.

**5. Locked test** (268 photographs, evaluated once after every selection was frozen):

| Model | Params | Test AP50-95 (mean ± SD, 5 seeds) |
|---|---|---|
| Pretrained YOLO11n 1.00× | 2.59 M | **17.27 ± 0.39** |
| **Distilled 0.75× (selected)** | **1.57 M** | **16.08 ± 0.96** |
| From-scratch 1.00× | 2.59 M | 16.19 ± 0.69 |
| Ordinary 0.75× | 1.57 M | 14.95 ± 0.98 |

## Dataset: Tomato-Village (third-party, not redistributed here)

The data come from Tomato-Village, created and published by Gehlot, Saxena and Gandhi:

> Gehlot, M., Saxena, R.K. & Gandhi, G.C. "Tomato-Village": a dataset for end-to-end tomato disease detection in a
> real-world environment. *Multimedia Systems* (2023). https://doi.org/10.1007/s00530-023-01158-y
> Data: https://github.com/mamta-joshi-gehlot/Tomato-Village (Variant-c, Object Detection)

**What this repository contains and doesn't contain:**
- The Tomato-Village repository is public but declares **no licence**.
- We therefore **do not redistribute any of its images or labels.** This repo contains only our code, our split
  manifest (file names and split assignments), and results derived from our own runs.
- Please obtain the data from the authors, follow their terms, and cite their paper.
- Our experiments used commit **`dfc246014558293987b381c1edb49398d9f570ac`**.

See [`DATASET.md`](DATASET.md) for the leakage structure we found and how the split is rebuilt.

## Reproduce

**Environment.** Python 3.12, PyTorch 2.14 + CUDA, Ultralytics 8.4.155. We trained on one NVIDIA RTX 5000 Ada (32 GB).

```bash
git clone https://github.com/abdulsamad2/tomato-adcf-yolo && cd tomato-adcf-yolo
uv sync                                   # exact versions from uv.lock
uv pip install statsmodels pandas         # used by some phase statistics scripts
uv run pytest -q                          # unit tests + end-to-end smoke test on synthetic data
```

**1. Get the data at the pinned commit and rebuild the exact split from the committed manifest:**

```bash
uv run tv-download --commit dfc246014558293987b381c1edb49398d9f570ac
uv run tv-prepare --manifest splits/split_manifest.csv      # writes data/tomato_village/ (1,260 / 268 / 268 photos)
```

**2. Re-run the experiments.** These scripts were run from a parent directory: they assume `<ROOT>/tomato-adcf-yolo`
plus sibling output folders, and they hard-code the original root path. Keep the frozen originals untouched (their
SHA-256 hashes are recorded in the pre-registrations) and make a runnable copy with your own root:

```bash
python tools/relocate_paths.py --root /path/to/ROOT --out /path/to/ROOT    # copies experiments/* with paths rewritten
```

Then follow [`experiments/README.md`](experiments/README.md) phase by phase. Approximate GPU time on our hardware:

| Phase | Runs | Time |
|---|---|---|
| Phase 15 (width) | 20 | ~12 h |
| Phase 16A (LPWA-50) | 5 | ~3 h |
| Phase 17 (distillation) | 5 | ~4 h |
| Pretrained reference (Phase 4, 512 px) | 5 | ~3 h |

The evaluation and statistics scripts reproduce every number in the manuscript from the trained checkpoints.

**Trained weights** for the 20 locked-test checkpoints will be attached to a GitHub Release.

## Repository map

| Path | Contents |
|---|---|
| `src/adcf_yolo/` | Data download, family-aware split (`data/prepare.py`), and the original ADCF pipeline (see [`docs/ADCF_PIPELINE.md`](docs/ADCF_PIPELINE.md)) |
| `splits/split_manifest.csv` | Our family-aware split: photograph id, split, near-duplicate cluster, stratum |
| `configs/` | Training configuration shared by all phases |
| `experiments/` | Every phase script exactly as run (Phases 1–17, pretraining audit, locked test, post-hoc analyses), with `MANIFEST.json` (hashes) |
| `artifacts/` | Frozen records: pre-registrations, freeze/hash files, dry-run reports, evaluation rows, statistics, figures |
| `results/` | Per-phase result tables (CSV/JSON) |
| `docs/` | Manuscript (`.docx` and its generator), results summary, development notes |
| `tools/` | Helper scripts (path relocation) |
| `tests/` | Unit and end-to-end tests |

## Evidence standards

| Category | What it covers |
|---|---|
| **Pre-registered** | Phases 15, 16A and 17: rules, margins and script hashes frozen before training (`artifacts/phase15–17/`). |
| **Locked test** | Run once after model selection was locked (`artifacts/clean_test/`). The test split was scored earlier for models outside this study, so we describe it as *held out from all models and decisions reported in the paper*, not as never opened. |
| **Post-hoc, plan hashed** | Error decomposition and the other analyses in manuscript Section 4.6–4.8 used saved development-set detections only (`artifacts/posthoc/plan.md`). |
| **Exploratory** | Labelled as such in the manuscript. |

## Licence and citation

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **Dataset:** remains under its owners' terms; it is not covered by this licence.
- **Citing:** if you use this work, cite it via [`CITATION.cff`](CITATION.cff) and also cite Tomato-Village.
