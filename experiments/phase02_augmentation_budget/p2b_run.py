"""PHASE 2B -- final augmentation experiment. FROZEN; do not modify after any seed completes.

Horizon ~24.5k updates, 5 paired seeds, no early stopping, four checkpoints fixed BEFORE training.

  arm  data                      imgs   steps/ep  epochs  updates  warmup_epochs  close_mosaic
  B    tomato_village_noaug      1260      79       311    24569    11.810127         39
  C    tomato_village_dup8      10080     630        39    24570     1.480952          5
  D    tomato_village           10080     630        39    24570     1.480952          5

  warmup = 933 updates in every arm (Phase 2's 3.797% fraction preserved)
  mosaic-free tail: B 3081 updates, C/D 3150 (integer-epoch granularity, 69-update mismatch)

  FROZEN CHECKPOINTS (approved; inter-arm alignment prioritised over the nominal labels)
      ~6k   B epoch  80 -> step  6320     C/D epoch 10 -> step  6300   (mismatch 20)
      ~12k  B epoch 152 -> step 12008     C/D epoch 19 -> step 11970   (mismatch 38)
      ~18k  B epoch 230 -> step 18170     C/D epoch 29 -> step 18270   (mismatch 100)
      final B epoch 311 -> step 24569     C/D epoch 39 -> step 24570   (mismatch  1)

  TEST SET IS NOT TOUCHED. Validation only. Checkpoint weights are saved so the prospectively
  selected final comparison can be evaluated on test later, after the validation analysis is frozen.
"""
import argparse, csv, json, shutil, time
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = REPO / "runs_p2b"

ARMS = {
    "B": dict(data="data/tomato_village_noaug/dataset.yaml", epochs=311, steps=79,
              warmup_epochs=11.810127, close_mosaic=39, val_every=8,
              ckpt={80: 6320, 152: 12008, 230: 18170, 311: 24569}),
    "C": dict(data="data/tomato_village_dup8/dataset.yaml", epochs=39, steps=630,
              warmup_epochs=1.480952, close_mosaic=5, val_every=1,
              ckpt={10: 6300, 19: 11970, 29: 18270, 39: 24570}),
    "D": dict(data="data/tomato_village/dataset.yaml", epochs=39, steps=630,
              warmup_epochs=1.480952, close_mosaic=5, val_every=1,
              ckpt={10: 6300, 19: 11970, 29: 18270, 39: 24570}),
}


def build(arm, seed):
    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    a = dict(cfg["train"])
    sp = ARMS[arm]
    a.update(data=str(REPO / sp["data"]), epochs=sp["epochs"],
             warmup_epochs=sp["warmup_epochs"], close_mosaic=sp["close_mosaic"],
             patience=10 ** 6, seed=seed, deterministic=True, val=True, plots=False,
             project=str(OUT / arm), name=f"seed{seed}", exist_ok=True)
    a.pop("cv", None)
    return a, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    a, sp = build(args.arm, args.seed)
    rd = Path(a["project"]) / a["name"]
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "p2b_done.json").exists():
        print(f"SKIP {args.arm} seed{args.seed}"); return

    rows, st = [], {"t0": time.time()}
    model = YOLO("yolo11s.pt")

    def on_train_start(tr):
        st["nb"] = len(tr.train_loader)
        st["nw"] = tr._get_warmup_iterations(st["nb"])
        st["mos"] = tr.epochs - tr.args.close_mosaic
        json.dump({"arm": args.arm, "seed": args.seed, "steps_per_epoch": st["nb"],
                   "epochs": tr.epochs, "total_updates": st["nb"] * tr.epochs,
                   "warmup_updates": st["nw"], "mosaic_close_epoch": st["mos"],
                   "mosaic_free_updates": tr.args.close_mosaic * st["nb"],
                   "ckpt_epochs": sp["ckpt"]}, open(rd / "schedule.json", "w"), indent=1)

    def on_train_epoch_end(tr):
        e = tr.epoch + 1
        tr.args.val = (e % sp["val_every"] == 0) or e in sp["ckpt"] or e == 1 or e >= tr.epochs

    def on_fit_epoch_end(tr):
        if tr.epoch + 1 > tr.epochs:       # Ultralytics fires this again from final_eval -- ignore
            return
        e, nb = tr.epoch + 1, st["nb"]
        gs = e * nb
        m = tr.metrics or {}
        row = {"arm": args.arm, "seed": args.seed, "epoch": e, "global_step": gs,
               "frac_budget": round(gs / (nb * tr.epochs), 6),
               "learning_rate": tr.optimizer.param_groups[0]["lr"],
               "validated": bool(tr.args.val), "is_checkpoint": e in sp["ckpt"],
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
        if e in sp["ckpt"]:
            src = rd / "weights" / "last.pt"
            if src.exists():
                shutil.copy2(src, rd / "weights" / f"ckpt_ep{e}_step{sp['ckpt'][e]}.pt")

    for ev, fn in (("on_train_start", on_train_start), ("on_train_epoch_end", on_train_epoch_end),
                   ("on_fit_epoch_end", on_fit_epoch_end)):
        model.add_callback(ev, fn)

    t0 = time.time()
    model.train(**a)
    hours = (time.time() - t0) / 3600

    # full VALIDATION metrics at each frozen checkpoint. Test set deliberately untouched.
    out = {"arm": args.arm, "seed": args.seed, "train_hours": round(hours, 4),
           "total_updates": st["nb"] * sp["epochs"], "checkpoints": {}}
    for e, step in sp["ckpt"].items():
        w = rd / "weights" / f"ckpt_ep{e}_step{step}.pt"
        if not w.exists():
            out["checkpoints"][str(step)] = {"error": "missing"}
            continue
        r = YOLO(str(w)).val(data=a["data"], split="val", imgsz=a["imgsz"],
                             batch=a["batch"], device=0, plots=False, verbose=False)
        pc = {}
        try:
            for i, ci in enumerate(r.box.ap_class_index):
                pc[str(int(ci))] = {"ap50_95": float(r.box.maps[int(ci)]),
                                    "ap50": float(r.box.ap50[i])}
        except Exception:
            pass
        out["checkpoints"][str(step)] = {
            "epoch": e, "val_mAP50": float(r.box.map50), "val_mAP50_95": float(r.box.map),
            "val_precision": float(r.box.mp), "val_recall": float(r.box.mr),
            "per_class": pc}
    json.dump(out, open(rd / "p2b_done.json", "w"), indent=1)
    print("P2B_DONE " + json.dumps({k: v for k, v in out.items() if k != "checkpoints"}))


if __name__ == "__main__":
    main()
