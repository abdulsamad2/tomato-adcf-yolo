# Frozen records

These are the records written during the study. Dataset content (images, labels, label-derived files), detection
caches and weights are not included. Weights are on the GitHub Release; see `weights_manifest.json`.

| Folder | Contents |
|---|---|
| `phase9/` | Evaluation-arm freeze record and manifests (file names only), arm image ids, evaluation rows, interaction statistics |
| `phase12/`, `phase13/` | Localization vs classification decomposition; single-class control |
| `widths/` | YAMLs and audits of the four width variants |
| `phase15/` | Width study: pre-registration (`.json` and `.md`), analysis-code freeze, evaluation rows |
| `phase16/` | LPWA-50: parameter-budget audit, design solve and YAML, pre-registration, analysis freeze, evaluation rows |
| `phase17/` | Distillation: pre-registration, analysis freeze, dry-run report (loss reproduction, shapes, λ_AT rule), evaluation rows |
| `pretrain_audit/` | Analysis plan (written before scoring) and evaluation rows |
| `clean_test/` | Test-set verification, pre-registration, addendum, analysis freeze, single-pass `results.json` |
| `posthoc/` | Analysis plan + SHA-256, `results.json`, `leakage.json`, figures |
| `weights_manifest.json` | Released checkpoints with SHA-256, identical to the locked-test pre-registration |
