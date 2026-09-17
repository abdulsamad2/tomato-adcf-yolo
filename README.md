# ADCF-YOLO: Adaptive Detail–Context Fusion for Tomato Leaf Disease Detection

A YOLO neck module that fuses fine lesion **detail** with surrounding leaf **context**
through a learned per-pixel gate, evaluated on
[Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village) (Variant-c,
object detection) with a **leakage-free re-split**.

> **New here? Read [`docs/notes/00-roadmap.md`](docs/notes/00-roadmap.md) first.**
> The notes explain the dataset issue, YOLO, the module, the experiment design,
> the metrics and how to write the paper, in that order.

```
image → Backbone → P3/P4/P5 → Neck (PAN-FPN, 4 × ADCF) → Detect head → box + class + score

ADCF:  x ─ 1×1 ─┬─ Detail (DW 3×3 + high-pass) ──┐
                ├─ Context (dilated DW + SE) ────┼─ α·D + (1−α)·C ─┐
                │         Gate α = σ(g[D, C]) ───┘                  ├─ concat ─ 1×1 → out
                └──────────────────── shortcut ────────────────────┘
```

## Run it on the lab GPU

```bash
git clone <this repo> && cd tomato-adcf-yolo
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv isn't installed
uv sync                                             # exact versions from uv.lock

# If the lab needs a specific CUDA build of torch, install it into the env afterwards, e.g.
#   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX  (match the lab's CUDA)

uv run pytest -q                    # ~1 min: unit tests + end-to-end smoke test on synthetic data

# 1. Data (downloads only Variant-c)
uv run tv-download
uv run tv-prepare                   # leakage-free split → data/tomato_village/
uv run tv-stats                     # tables + results/dataset/label_preview.png (look at it!)
mkdir -p splits && cp data/tomato_village/split_manifest.csv splits/   # commit this

# 2. Experiments (resumable; re-run the same command after any interruption)
uv run adcf-run --dry-run
uv run adcf-run --tier 1 --seeds 0 --device 0      # first pass, check curves + timing
uv run adcf-run --tier 1 --device 0
uv run adcf-run --tier 2 --device 0

# 3. Analysis
uv run adcf-bench                                   # all models, one session, idle GPU
uv run adcf-gates --weights runs/yolo11s-adcf/seed0/weights/best.pt
uv run adcf-collect                                 # → results/tables_test.md
```

Long runs: use `tmux` or `nohup uv run adcf-run --tier 2 --device 0 > tier2.log 2>&1 &`
so a dropped SSH session doesn't kill training.

## Repository map

| Path | What |
|---|---|
| `src/adcf_yolo/modules.py` | the ADCF block |
| `src/adcf_yolo/build.py` | inserts ADCF into YOLOv8/11/26 necks; custom trainer |
| `src/adcf_yolo/data/` | `download`, `prepare` (grouping, dedup, stratified split), `stats` |
| `src/adcf_yolo/experiments.py` | train → test-eval runner (`adcf-run`) |
| `src/adcf_yolo/eval/` | COCO size-AP, params/GFLOPs/latency |
| `src/adcf_yolo/gates.py` | gate α maps + inside/outside-box statistics |
| `src/adcf_yolo/collect.py` | paper tables, mean ± std, Welch t-test |
| `configs/experiments.yaml` | every run in the paper, with tiers |
| `docs/notes/` | learning notes |
| `tests/` | unit tests + end-to-end pipeline test on synthetic data |

## Dataset citation

Gehlot, M., Saxena, R.K. & Gandhi, G.C. "Tomato-Village": a dataset for end-to-end
tomato disease detection in a real-world environment. *Multimedia Systems* (2023).
https://doi.org/10.1007/s00530-023-01158-y
