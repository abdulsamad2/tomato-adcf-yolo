# 05 · Metrics: what the numbers mean and how to report them honestly

## IoU: the building block

**Intersection over Union** = area of overlap ÷ area of union of a predicted box and
a true box. 1.0 is a perfect match, 0 is no overlap. A prediction counts as a
**true positive (TP)** if IoU ≥ threshold, the class is correct, and that true box
hasn't already been matched. Otherwise it's a **false positive (FP)**. True boxes
nobody matched are **false negatives (FN)**.

## Precision, recall, and why neither is enough

- **Precision** = TP / (TP + FP): "of what I flagged, how much was right".
- **Recall** = TP / (TP + FN): "of what exists, how much did I find".

Both depend on the confidence threshold. A low threshold gives high recall and
low precision; a high threshold gives the reverse. The P and R Ultralytics prints
are at the threshold that maximises F1, so they're a snapshot, not the full picture.

## AP: the area under the precision–recall curve

Sort all predictions of one class by confidence, walk down the list, and plot
precision against recall. **AP** is the area under that curve, which summarises
all thresholds at once. **mAP** is the mean over classes.

- **mAP50:** IoU threshold 0.5. "Did you find the lesion, roughly?"
- **mAP50-95:** average over IoU 0.50, 0.55, …, 0.95. Also rewards *tight* boxes.
  This is the main metric in modern detection papers; report both.

Why evaluation uses `conf=0.001`: AP needs the whole curve, including
low-confidence predictions. Using 0.25 (the prediction default) cuts the curve
and understates AP.

## AP by object size (`coco.AP_small` etc.)

COCO defines small < 32² px, medium 32²–96², large > 96² (box area). Our images
vary in resolution, so `coco_eval.py` rescales every image to a 640 px long side
first (the size the network sees). **AP_small is the most direct evidence for the
"detail" claim.** It is also the noisiest bucket if there are few small boxes;
check the counts in `results/dataset/dataset_table.md`.

COCO's AP uses 101-point interpolation, Ultralytics uses a slightly different
method, so COCO AP ≠ Ultralytics mAP50-95 exactly. Say in the paper which is which.

## Per-class AP

Averages hide things. ADCF might help Leaf Miner (tiny trails) and do nothing for
Healthy. That's a *good* story if it matches the mechanism. The per-class table
from `adcf-collect` is where you find it.

## Efficiency: params, GFLOPs, latency, FPS

| Metric | Measures | Pitfall |
|---|---|---|
| Params (M) | model size / memory | says nothing about speed |
| GFLOPs | arithmetic operations (estimate) | ignores memory access; misses element-wise ops |
| Latency (ms) | real time per image on given hardware | depends on GPU, precision, batch, drivers, load |
| FPS | 1000 / latency (batch 1) | same as latency |

**How `adcf-bench` measures latency, and why:**
- **warm-up** (50 iterations): the first calls include cuDNN algorithm search and
  memory allocation, so they're far slower than steady state;
- **synchronise** before and after each timed call, because CUDA runs asynchronously
  and without it you time the *launch* of the work, not the work;
- **many iterations** (500), reporting mean, std, p50 and p95;
- **batch 1, FP16, 640 px, network only.** Say exactly this in the paper, plus the
  GPU model;
- **all models in one session** on an idle GPU. Check `nvidia-smi` first.

NMS time isn't included (it depends on how many boxes are found). Ultralytics'
val prints `postprocess` ms per image; include it for an end-to-end comparison.
That matters for YOLO26, which is NMS-free.

## Mean ± std and significance

- Report **mean ± sample std** over seeds (the code uses `ddof=1`).
- `adcf-collect` also prints Welch's t-test p-values vs the reference. With 3 seeds
  the test has low power: p > 0.05 does **not** prove "no difference", and p < 0.05
  on 3 seeds is suggestive, not conclusive.
- Rule of thumb for writing: call a gain "consistent" only if every seed of ADCF
  beats the baseline mean, and the gain is larger than the std.
- A stronger alternative if a reviewer pushes: a paired bootstrap over *test
  images* (resample images, recompute mAP for both models, count how often ADCF
  wins).

## The table format reviewers expect

| Model | Params (M) | GFLOPs | FPS | P | R | mAP50 | mAP50-95 | AP_S |
|---|---|---|---|---|---|---|---|---|
| YOLO11s | 9.4 | 21.4 | … | … | … | … ± … | … ± … | … |
| **YOLO11s + ADCF (ours)** | 8.1 | 18.6 | … | … | … | **… ± …** | **… ± …** | … |

Bold the best value per column only when it beats the others beyond the noise.
