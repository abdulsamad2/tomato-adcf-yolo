"""Size-stratified COCO AP for the REUSED Phase 3 640 px checkpoints. EVALUATION ONLY.

No training, no retraining, no test evaluation. Reads the already-saved Phase 3 YOLO11n
checkpoints and adds AP_small / AP_medium / AP_large at 640 px, so the 640 arm carries the
same secondary metrics as the new Phase 4 arms.
"""
import json
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = REPO / "results" / "p4_size640.json"
CKPT = {10: 6300, 19: 11970, 29: 18270, 39: 24570}


def main():
    from adcf_yolo.eval.predictions import predict_split
    from adcf_yolo.eval.coco_eval import coco_size_eval
    if OUT.exists():
        raise SystemExit(f"EXISTS {OUT}")
    data = str(REPO / "data" / "tomato_village" / "dataset.yaml")
    yn = yaml.safe_load(Path(data).read_text(encoding="utf-8")).get("names") or {}
    NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))
    out = {}
    for seed in range(5):
        rd = REPO / "runs_p3" / "yolo11n" / f"seed{seed}"
        out[str(seed)] = {}
        for e, step in CKPT.items():
            w = rd / "weights" / f"ckpt_ep{e}_step{step}.pt"
            assert w.exists(), w
            images = predict_split(w, data, split="val", imgsz=640, device="0")
            out[str(seed)][str(step)] = coco_size_eval(images, NAMES)
            print(f"seed{seed} step{step} AP_small={out[str(seed)][str(step)]['AP_small']}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("SIZE640DONE")


if __name__ == "__main__":
    main()
