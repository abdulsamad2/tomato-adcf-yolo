# Frozen records

These are small records copied from the original run folders. Things left out on purpose:
- **Dataset content** (images, labels, and `coco_gt_union.json`, which is derived from labels) is not redistributed;
  see `../DATASET.md`.
- **Detection caches** (`dets/`, `match_cache/`, about 2.5 GB) and **model weights** are not included. The 20
  locked-test checkpoints will be published as a GitHub Release.

| Folder | Contents |
|---|---|
| `phase9/` | Arm freeze record and manifests (file names only), arm image ids, evaluation rows, interaction statistics (`p9_results.json`) |
| `phase12/`, `phase13/`, `phase14/` | Mechanism decomposition, single-class control, transform provenance |
| `phase15/` | `p15_prereg.json` and `.md`, `p15_analysis_freeze.json` (+ `_v1`), `eval_rows.json` (every checkpoint × arm) |
| `phase16/` | Parameter-budget audit, LPWA-50 design solve and YAML, pre-registration, analysis freeze, evaluation rows |
| `phase17/` | Pre-registration, analysis freeze, dry-run report (loss reproduction, shapes, λ_AT calibration), evaluation rows |
| `pretrain_audit/` | Analysis plan (written before scoring) and evaluation rows |
| `clean_test/` | Verification report, pre-registration, addendum, analysis freeze, single-pass `results.json` |
| `posthoc/` | Analysis plan + SHA-256, `results.json`, `leakage.json`, figures used in the manuscript |
| `widths/`, `scales/` | Width and detection-scale model YAMLs and audits |
| `audit/` | Early-phase audits and phase figures |

Each pre-registration lists the SHA-256 of its scripts. `../experiments/MANIFEST.json` confirms that the copies in
`../experiments/` match them.
