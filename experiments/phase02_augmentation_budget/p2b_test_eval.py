"""SINGLE test evaluation of the fifteen prospectively-frozen ~18k checkpoints.

The checkpoint GRID was frozen before Phase 2B training. The SELECTION of ~18k for the final test
comparison is validation-informed, made by the reviewer after the validation analysis was accepted.

Evaluates test exactly once. The 6k / 12k / final checkpoints are NOT evaluated on test.
No training, no tuning, no checkpoint reselection.

Windows notes: the __main__ guard is required because Ultralytics' dataloader uses spawn, which
re-executes this module in each worker; workers=0 keeps evaluation single-process (268 images).
"""
import json, time
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = REPO / "results" / "p2b_test_eval.json"
CK = {"B": ("ckpt_ep230_step18170.pt", 18170, 230), "C": ("ckpt_ep29_step18270.pt", 18270, 29),
      "D": ("ckpt_ep29_step18270.pt", 18270, 29)}
DATA = {"B": "data/tomato_village_noaug/dataset.yaml", "C": "data/tomato_village_dup8/dataset.yaml",
        "D": "data/tomato_village/dataset.yaml"}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists -- test is evaluated exactly once")
    from ultralytics import YOLO
    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))["train"]
    yn = yaml.safe_load((REPO / DATA["D"]).read_text(encoding="utf-8")).get("names") or {}
    NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))

    res, t0 = [], time.time()
    for arm, (fn, step, ep) in CK.items():
        for seed in range(5):
            w = REPO / "runs_p2b" / arm / f"seed{seed}" / "weights" / fn
            r = YOLO(str(w)).val(data=str(REPO / DATA[arm]), split="test", imgsz=cfg["imgsz"],
                                 batch=cfg["batch"], device=0, workers=0, plots=False, verbose=False)
            pc = {NAMES.get(int(ci), f"class{int(ci)}"): {
                      "ap50_95": float(r.box.maps[int(ci)]), "ap50": float(r.box.ap50[i])}
                  for i, ci in enumerate(r.box.ap_class_index)}
            row = {"arm": arm, "seed": seed, "checkpoint_step": step, "checkpoint_epoch": ep,
                   "weights": fn,
                   "test_mAP50_95": float(r.box.map), "test_mAP50": float(r.box.map50),
                   "test_precision": float(r.box.mp), "test_recall": float(r.box.mr),
                   "per_class": pc,
                   "speed_ms": {k: round(float(v), 3) for k, v in (getattr(r, "speed", {}) or {}).items()}}
            res.append(row)
            print(f"{arm} seed{seed} test mAP50-95={row['test_mAP50_95']:.4f} "
                  f"mAP50={row['test_mAP50']:.4f}", flush=True)
    json.dump({"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_s": round(time.time() - t0, 1),
               "checkpoint_grid": "frozen before Phase 2B training",
               "checkpoint_selection": "~18k; validation-informed; chosen by reviewer after the validation analysis",
               "workers": 0, "evaluations": res}, open(OUT, "w"), indent=1)
    print("TESTEVALDONE " + str(len(res)))


if __name__ == "__main__":
    main()
