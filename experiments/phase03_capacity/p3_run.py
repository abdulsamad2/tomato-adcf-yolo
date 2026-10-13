"""PHASE 3 -- capacity scaling. FROZEN; do not modify after any seed completes.

Phase 2B arm-D protocol, identical in every field except the model architecture:

  data            data/tomato_village (leakage-free split, 10080 train files = 1260 photos x 8)
  imgsz/batch/nbs 640 / 16 / 64        steps_per_epoch 630 (identical for every model)
  epochs          39                    total updates 24570 EXACTLY, all arms
  patience        1e6 (early stopping disabled)
  warmup_epochs   1.480952 -> round(1.480952 * 630) = 933 updates, all arms
  close_mosaic    5 -> 3150 mosaic-free updates, all arms
  optimizer/LR    SGD, lr0 0.01, lrf 0.01, cos_lr false  (from configs/experiments.yaml)
  augmentation    unchanged Ultralytics defaults
  init            COCO-pretrained
  seeds           0-4, paired with Phase 2B arm D

  FROZEN CHECKPOINTS (same grid as arm D, fixed before training):
      epoch 10 -> 6300    epoch 19 -> 11970    epoch 29 -> 18270    epoch 39 -> 24570
  PRIMARY validation endpoint prospectively fixed at 18270 updates (reviewer, before any result).
  24570 is reported as a SECONDARY fixed-budget endpoint.

  YOLO11s is NOT retrained -- Phase 2B runs_p2b/D/seed{0..4} is the s arm.
  TEST IS NOT TOUCHED. Checkpoint weights are saved so a reviewer-selected comparison can be
  scored later without retraining.

  Windows: the __main__ guard is required because Ultralytics' dataloader uses spawn.
"""
import argparse, csv, json, shutil, time
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = REPO / "runs_p3"
MODELS = {"n": "yolo11n.pt", "m": "yolo11m.pt"}
CKPT = {10: 6300, 19: 11970, 29: 18270, 39: 24570}
EPOCHS, WARMUP_EPOCHS, CLOSE_MOSAIC = 39, 1.480952, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    a = dict(cfg["train"])
    a.update(data=str(REPO / "data" / "tomato_village" / "dataset.yaml"),
             epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, close_mosaic=CLOSE_MOSAIC,
             patience=10 ** 6, seed=args.seed, deterministic=True, val=True, plots=False,
             project=str(OUT / f"yolo11{args.size}"), name=f"seed{args.seed}", exist_ok=True)
    a.pop("cv", None)

    rd = Path(a["project"]) / a["name"]
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "p3_done.json").exists():
        print(f"SKIP yolo11{args.size} seed{args.seed}"); return

    rows, st = [], {"t0": time.time()}
    model = YOLO(MODELS[args.size])

    def on_train_start(tr):
        st["nb"] = len(tr.train_loader)
        st["nw"] = tr._get_warmup_iterations(st["nb"])
        st["mos"] = tr.epochs - tr.args.close_mosaic
        json.dump({"size": args.size, "model": MODELS[args.size], "seed": args.seed,
                   "steps_per_epoch": st["nb"], "epochs": tr.epochs,
                   "total_updates": st["nb"] * tr.epochs, "warmup_updates": st["nw"],
                   "mosaic_close_epoch": st["mos"],
                   "mosaic_free_updates": tr.args.close_mosaic * st["nb"],
                   "ckpt_epochs": CKPT}, open(rd / "schedule.json", "w"), indent=1)

    def on_fit_epoch_end(tr):
        if tr.epoch + 1 > tr.epochs:          # Ultralytics fires this again from final_eval
            return
        e, nb = tr.epoch + 1, st["nb"]
        gs = e * nb
        m = tr.metrics or {}
        row = {"size": args.size, "seed": args.seed, "epoch": e, "global_step": gs,
               "frac_budget": round(gs / (nb * tr.epochs), 6),
               "learning_rate": tr.optimizer.param_groups[0]["lr"],
               "validated": bool(tr.args.val), "is_checkpoint": e in CKPT,
               "mosaic_on": tr.epoch < st["mos"],
               "val_mAP50": m.get("metrics/mAP50(B)"), "val_mAP50_95": m.get("metrics/mAP50-95(B)"),
               "precision": m.get("metrics/precision(B)"), "recall": m.get("metrics/recall(B)"),
               "val_box_loss": m.get("val/box_loss"), "val_cls_loss": m.get("val/cls_loss"),
               "val_dfl_loss": m.get("val/dfl_loss"),
               "wall_s": round(time.time() - st["t0"], 1)}
        try:
            row.update({k: float(v) for k, v in tr.label_loss_items(tr.tloss, prefix="train").items()})
        except Exception:
            pass
        rows.append(row)
        keys = sorted({k for r in rows for k in r})
        with open(rd / "curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        if e in CKPT:
            src = rd / "weights" / "last.pt"
            if src.exists():
                shutil.copy2(src, rd / "weights" / f"ckpt_ep{e}_step{CKPT[e]}.pt")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    t0 = time.time()
    model.train(**a)
    hours = (time.time() - t0) / 3600

    from ultralytics.utils.torch_utils import get_num_params, get_flops
    out = {"size": args.size, "model": MODELS[args.size], "seed": args.seed,
           "train_hours": round(hours, 4), "total_updates": st["nb"] * EPOCHS,
           "checkpoints": {}}
    for e, step in CKPT.items():
        w = rd / "weights" / f"ckpt_ep{e}_step{step}.pt"
        if not w.exists():
            out["checkpoints"][str(step)] = {"error": "missing"}; continue
        mm = YOLO(str(w))
        r = mm.val(data=a["data"], split="val", imgsz=a["imgsz"], batch=a["batch"],
                   device=0, workers=0, plots=False, verbose=False)
        pc = {}
        try:
            for i, ci in enumerate(r.box.ap_class_index):
                pc[str(int(ci))] = {"ap50_95": float(r.box.maps[int(ci)]),
                                    "ap50": float(r.box.ap50[i])}
        except Exception:
            pass
        out["checkpoints"][str(step)] = {
            "epoch": e, "val_mAP50": float(r.box.map50), "val_mAP50_95": float(r.box.map),
            "val_precision": float(r.box.mp), "val_recall": float(r.box.mr), "per_class": pc}
        if e == EPOCHS:
            try:
                out["params_M"] = round(get_num_params(mm.model) / 1e6, 4)
                out["GFLOPs_640"] = round(get_flops(mm.model, 640), 3)
            except Exception:
                pass
    json.dump(out, open(rd / "p3_done.json", "w"), indent=1)
    print("P3_DONE " + json.dumps({k: v for k, v in out.items() if k != "checkpoints"}))


if __name__ == "__main__":
    main()
