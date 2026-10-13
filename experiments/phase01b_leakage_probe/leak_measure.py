import json
from pathlib import Path

def main():
    from ultralytics import YOLO
    REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
    res = {}
    for arm in ("seen", "unseen"):
        out = {}
        for seed in ("seed0", "seed1", "seed2"):
            w = REPO/"runs"/"yolo11s"/seed/"weights"/"best.pt"
            if not w.exists():
                continue
            r = YOLO(str(w)).val(
                data=str(REPO/"data"/"leakage_probe"/arm/"dataset.yaml"), split="val",
                imgsz=640, batch=16, conf=0.001, iou=0.7, device="cuda:0",
                workers=0, verbose=False, plots=False, save_json=False)
            out[seed] = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map)}
            print(arm, seed, out[seed], flush=True)
        res[arm] = out
    (REPO/"results"/"leakage_inflation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
