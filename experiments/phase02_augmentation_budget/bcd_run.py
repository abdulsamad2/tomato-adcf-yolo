"""Controlled augmentation-budget experiment: arms B / C / D at an exactly-equal update budget.

FROZEN CONFIGURATION -- do not modify after the first seed completes.

  budget   = LCM(79, 630) = 49770 optimizer updates, identical in all three arms
  B  1x source photos        1260 imgs,   79 steps/epoch, 630 epochs
  C  8x byte-identical       10080 imgs, 630 steps/epoch,  79 epochs
  D  8x shipped augmented    10080 imgs, 630 steps/epoch,  79 epochs

  warmup   1890 updates in every arm (B: warmup_epochs=23.924051 -> round(23.924051*79)=1890)
  mosaic   closed for the final ~6300 updates (B: 80 ep = 6320; C/D: 10 ep = 6300)
  no early stopping; PRIMARY endpoint is the state at update 49770 (last.pt)
  best.pt is retained as a SECONDARY exploratory result only
  validation by update milestone: B every 8 epochs (632 upd), C/D every epoch (630 upd)

All hyperparameters other than epochs / warmup_epochs / close_mosaic / patience / seed / data are
inherited verbatim from configs/experiments.yaml `train:` so the arms stay comparable with A0/D0.
"""
import argparse, csv, json, time, sys
from pathlib import Path
import yaml, torch

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = REPO / "runs_bcd"
BUDGET = 49770

ARMS = {
    "B": dict(data="data/tomato_village_noaug/dataset.yaml", epochs=630,
              warmup_epochs=23.924051, close_mosaic=80, val_every=8, steps=79),
    "C": dict(data="data/tomato_village_dup8/dataset.yaml", epochs=79,
              warmup_epochs=3.0, close_mosaic=10, val_every=1, steps=630),
    "D": dict(data="data/tomato_village/dataset.yaml", epochs=79,
              warmup_epochs=3.0, close_mosaic=10, val_every=1, steps=630),
}


def build_args(arm, seed, max_epochs=None):
    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    a = dict(cfg["train"])                      # inherit every shared hyperparameter
    spec = ARMS[arm]
    a.update(
        data=str(REPO / spec["data"]),
        epochs=spec["epochs"],                  # full budget -- LR progress == global_step/BUDGET
        warmup_epochs=spec["warmup_epochs"],
        close_mosaic=spec["close_mosaic"],
        patience=10 ** 6,                       # early stopping disabled
        seed=seed,
        deterministic=True,
        val=True,
        plots=False,
        project=str(OUT / arm),
        name=f"seed{seed}" if max_epochs is None else f"smoke{seed}",
        exist_ok=True,
    )
    a.pop("cv", None)
    return a, spec


def attach(model, arm, spec, run_dir, max_epochs):
    """Callbacks: update-space logging, milestone validation, optional smoke truncation."""
    state = {"rows": [], "iters": [], "t0": time.time(), "nb": None, "val_this_epoch": True}

    def on_train_start(tr):
        state["nb"] = len(tr.train_loader)
        state["nw"] = tr._get_warmup_iterations(state["nb"])
        state["mosaic_close_epoch"] = tr.epochs - tr.args.close_mosaic
        json.dump({"arm": arm, "steps_per_epoch": state["nb"], "epochs": tr.epochs,
                   "total_updates": state["nb"] * tr.epochs, "warmup_updates": state["nw"],
                   "mosaic_close_epoch": state["mosaic_close_epoch"],
                   "mosaic_free_updates": tr.args.close_mosaic * state["nb"],
                   "lrf": tr.args.lrf, "lr0": tr.args.lr0, "seed": tr.args.seed},
                  open(run_dir / "schedule.json", "w"), indent=1)

    def on_train_batch_end(tr):
        # per-iteration LR for the warmup window only (verification of warmup alignment).
        # The trainer exposes no batch index, so count iterations here.
        state["ni"] = state.get("ni", 0) + 1
        if state["ni"] <= 2200:
            state["iters"].append({"global_step": state["ni"],
                                   "lr": tr.optimizer.param_groups[0]["lr"],
                                   "momentum": tr.optimizer.param_groups[0].get("momentum")})

    def on_train_epoch_end(tr):
        # decide whether THIS epoch validates, by update milestone
        e = tr.epoch
        ve = spec["val_every"]
        state["val_this_epoch"] = ((e + 1) % ve == 0) or e == 0 or (e + 1 >= tr.epochs)
        tr.args.val = state["val_this_epoch"]

    def on_fit_epoch_end(tr):
        nb = state["nb"]
        gs = (tr.epoch + 1) * nb
        lr = tr.optimizer.param_groups[0]["lr"]
        m = tr.metrics or {}
        losses = {}
        try:
            losses = {k: float(v) for k, v in
                      tr.label_loss_items(tr.tloss, prefix="train").items()}
        except Exception:
            pass
        row = {"arm": arm, "seed": tr.args.seed, "epoch": tr.epoch + 1, "global_step": gs,
               "frac_budget": round(gs / (nb * tr.epochs), 6), "learning_rate": lr,
               "validated": bool(state["val_this_epoch"]),
               "mosaic_on": tr.epoch < state["mosaic_close_epoch"],
               "val_mAP50": m.get("metrics/mAP50(B)"), "val_mAP50_95": m.get("metrics/mAP50-95(B)"),
               "precision": m.get("metrics/precision(B)"), "recall": m.get("metrics/recall(B)"),
               "val_loss_box": m.get("val/box_loss"), "val_loss_cls": m.get("val/cls_loss"),
               "wall_s": round(time.time() - state["t0"], 1)}
        row.update(losses)
        state["rows"].append(row)
        if state["rows"]:
            keys = sorted({k for r in state["rows"] for k in r})
            with open(run_dir / "curve.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(state["rows"])
        if max_epochs and tr.epoch + 1 >= max_epochs:
            tr.stop = True

    def on_train_end(tr):
        json.dump(state["iters"], open(run_dir / "warmup_lr.json", "w"))

    for ev, fn in (("on_train_start", on_train_start), ("on_train_batch_end", on_train_batch_end),
                   ("on_train_epoch_end", on_train_epoch_end), ("on_fit_epoch_end", on_fit_epoch_end),
                   ("on_train_end", on_train_end)):
        model.add_callback(ev, fn)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=None, help="truncate after N epochs (verification only)")
    args = ap.parse_args()

    from ultralytics import YOLO
    a, spec = build_args(args.arm, args.seed, args.smoke)
    run_dir = Path(a["project"]) / a["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    done = run_dir / "bcd_done.json"
    if done.exists() and not args.smoke:
        print(f"SKIP {args.arm} seed{args.seed} already complete"); return

    t0 = time.time()
    model = YOLO("yolo11s.pt")
    st = attach(model, args.arm, spec, run_dir, args.smoke)
    model.train(**a)
    hours = (time.time() - t0) / 3600

    if args.smoke:
        print(f"SMOKE {args.arm} seed{args.seed} ok, {hours*60:.1f} min")
        print(json.dumps(json.load(open(run_dir / "schedule.json")), indent=1))
        return

    # PRIMARY endpoint: the state at the final update, i.e. last.pt -- not best.pt
    res = {"arm": args.arm, "seed": args.seed, "train_hours": round(hours, 4),
           "total_updates": st["nb"] * spec["epochs"]}
    for tag, wt in (("final", "last.pt"), ("bestval", "best.pt")):
        w = run_dir / "weights" / wt
        if not w.exists():
            continue
        m = YOLO(str(w))
        for split in ("val", "test"):
            r = m.val(data=a["data"], split=split, imgsz=a["imgsz"], batch=a["batch"],
                      device=0, plots=False, verbose=False)
            res[f"{tag}_{split}_mAP50"] = float(r.box.map50)
            res[f"{tag}_{split}_mAP50_95"] = float(r.box.map)
            res[f"{tag}_{split}_precision"] = float(r.box.mp)
            res[f"{tag}_{split}_recall"] = float(r.box.mr)
    json.dump(res, open(done, "w"), indent=1)
    print("BCD_RUN_DONE " + json.dumps(res))


if __name__ == "__main__":
    main()
