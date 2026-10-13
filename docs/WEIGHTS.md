# Trained weights

These are the 20 YOLO11n checkpoints scored once on the locked test set in the paper: 4 models × 5 seeds, each at
24,570 updates with 512 px input. Every file is **byte-identical** to the checkpoint hashed in
`artifacts/clean_test/prereg.json`; see `SHA256SUMS.txt`.

| Files | Model | Params | Test AP50-95 (mean ± SD) |
|---|---|---|---|
| `yolo11n-w075-distilled_seed{0-4}.pt` | **Distilled 0.75× (selected model)** | 1.57 M | 16.08 ± 0.96 |
| `yolo11n-w075-scratch_seed{0-4}.pt` | 0.75×, trained from scratch | 1.57 M | 14.95 ± 0.98 |
| `yolo11n-w100-scratch_seed{0-4}.pt` | 1.00×, trained from scratch | 2.59 M | 16.19 ± 0.69 |
| `yolo11n-w100-cocopretrained_seed{0-4}.pt` | 1.00×, COCO-pretrained (Phase 17 teacher family) | 2.59 M | 17.27 ± 0.39 |

**Download and verify:**

```bash
python tools/download_weights.py              # all files -> weights/, SHA-256 checked
python tools/download_weights.py --restore    # also place them where the evaluation scripts expect them
```

**Use** (classes: 8 Tomato-Village categories):

```python
from ultralytics import YOLO
model = YOLO("weights/yolo11n-w075-distilled_seed0.pt")
results = model.predict("leaf.jpg", imgsz=512, conf=0.25)
```

**Licence and data**
- **Licence:** these weights are released under **AGPL-3.0**. They were trained with Ultralytics YOLO11 (AGPL-3.0),
  and the pretrained models start from Ultralytics' COCO weights. The repository's code is MIT; the weights are not.
- **Training data:** the Tomato-Village dataset (Gehlot, Saxena & Gandhi, *Multimedia Systems* 2023,
  doi:10.1007/s00530-023-01158-y), with our family-aware split. They are provided for research reproducibility.
  Please cite the dataset and respect its authors' terms.
- **Not production models:** they detect about 40% of lesions at the conf 0.25 operating point, and miss most small
  lesions.
