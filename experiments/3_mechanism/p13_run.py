"""Phase 13: ONE single-class localization diagnostic run. Matched to runs_p4/r512 seed 0.

Every field is taken from the same configs/experiments.yaml the baseline used. The ONLY
differences are (a) data -> the derived single-class view and (b) the output directory.
imgsz 512, seed 0, SGD lr0 0.01 lrf 0.01, 39 epochs, 630 steps/epoch, 24570 updates,
warmup 933, close_mosaic 5, patience 1e6 (no early stopping), frozen checkpoint grid.

ABORTS if the trainer's own computed schedule differs from the baseline's schedule.json.
This is a mechanism diagnostic, not a superiority claim. One seed. No test access.
"""
import csv, json, shutil, time
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = REPO / "runs_p13" / "single_cls"
BASE = REPO / "runs_p4" / "r512" / "seed0"
CKPT = {10: 6300, 19: 11970, 29: 18270, 39: 24570}
EPOCHS, WARMUP_EPOCHS, CLOSE_MOSAIC, IMGSZ, SEED = 39, 1.480952, 5, 512, 0


def main():
    from ultralytics import YOLO

    want = json.loads((BASE / "schedule.json").read_text())
    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    a = dict(cfg["train"])
    data = str(REPO / "data" / "tomato_village_1cls" / "dataset.yaml")
    a.update(data=data, imgsz=IMGSZ, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
             close_mosaic=CLOSE_MOSAIC, patience=10 ** 6, seed=SEED, deterministic=True,
             val=True, plots=False, multi_scale=False, rect=False,
             project=str(OUT), name="seed%d" % SEED, exist_ok=True)
    a.pop("cv", None)

    rd = OUT / ("seed%d" % SEED)
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "p13_done.json").exists():
        print("ALREADY_DONE"); return

    rows, st = [], {"t0": time.time()}
    model = YOLO("yolo11n.pt")

    def on_train_start(tr):
        st["nb"] = len(tr.train_loader)
        st["nw"] = tr._get_warmup_iterations(st["nb"])
        st["mos"] = tr.epochs - tr.args.close_mosaic
        got = {"steps_per_epoch": st["nb"], "epochs": tr.epochs,
               "total_updates": st["nb"] * tr.epochs, "warmup_updates": st["nw"],
               "mosaic_close_epoch": st["mos"]}
        bad = {k: (want[k], got[k]) for k in got if want[k] != got[k]}
        json.dump({"imgsz": IMGSZ, "seed": SEED, **got, "ckpt_epochs": CKPT,
                   "baseline": {k: want[k] for k in got}, "mismatch": bad},
                  open(rd / "schedule.json", "w"), indent=1)
        if bad:
            print("SCHEDULE_MISMATCH " + json.dumps(bad))
            raise SystemExit(3)
        print("schedule matches baseline exactly: " + json.dumps(got), flush=True)

    def on_fit_epoch_end(tr):
        if tr.epoch + 1 > tr.epochs:
            return
        e, nb = tr.epoch + 1, st["nb"]
        gs = e * nb
        m = tr.metrics or {}
        row = {"imgsz": IMGSZ, "seed": SEED, "epoch": e, "global_step": gs,
               "frac_budget": round(gs / (nb * tr.epochs), 6),
               "learning_rate": tr.optimizer.param_groups[0]["lr"],
               "validated": bool(tr.args.val), "is_checkpoint": e in CKPT,
               "mosaic_on": tr.epoch < st["mos"],
               "val_mAP50": m.get("metrics/mAP50(B)"), "val_mAP50_95": m.get("metrics/mAP50-95(B)"),
               "precision": m.get("metrics/precision(B)"), "recall": m.get("metrics/recall(B)"),
               "val_box_loss": m.get("val/box_loss"), "val_cls_loss": m.get("val/cls_loss"),
               "val_dfl_loss": m.get("val/dfl_loss"),
               "wall_s": round(time.time() - st["t0"], 1)}
        rows.append(row)
        keys = sorted({k for r in rows for k in r})
        with open(rd / "curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        if e in CKPT:
            src = rd / "weights" / "last.pt"
            if src.exists():
                shutil.copy2(src, rd / "weights" / ("ckpt_ep%d_step%d.pt" % (e, CKPT[e])))

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(**a)
    hrs = (time.time() - st["t0"]) / 3600.0
    present = sorted(p.name for p in (rd / "weights").glob("ckpt_ep*_step*.pt"))
    json.dump({"imgsz": IMGSZ, "seed": SEED, "epochs": EPOCHS,
               "total_updates": st["nb"] * EPOCHS, "train_hours": hrs,
               "checkpoints": present, "data": data,
               "args_source": "configs/experiments.yaml [train], identical to runs_p4/r512"},
              open(rd / "p13_done.json", "w"), indent=1)
    print("P13DONE %.3f h, checkpoints %s" % (hrs, present))

if __name__ == "__main__":
    main()
