"""PHASE 16A -- LPWA-50 (localization-preserving width allocation), FROM SCRATCH. FROZEN by phase16/p16_prereg.json.
Training code is the Phase 15 runner; only the model, output root and pre-registration differ.

Identical to runs_p4/r512 in every training field except:
  model   widths/yolo11n_w{1p00,0p75,0p50,0p35}.yaml, random init (pretrained=False)
  output  runs_p15/<width>/seed<n>

Gates (abort before step 1):
  schedule  trainer's computed schedule == runs_p4/r512/seed0/schedule.json   (exit 3)
  params    built params == audited params in p15_prereg.json                 (exit 4)
  init      w100 first conv weight != yolo11n.pt first conv weight            (exit 5)

Frozen checkpoint grid: epoch 10 -> 6300, 19 -> 11970, 29 -> 18270, 39 -> 24570.
After training, every checkpoint gets an Ultralytics clean-VAL eval (descriptive only).
TEST IS NOT TOUCHED anywhere in this phase.
"""
import argparse, csv, json, shutil, time
from pathlib import Path
import yaml

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
OUT = REPO / "runs_p16"
REF = REPO / "runs_p4" / "r512" / "seed0"
PREREG = BASE / "phase16" / "p16_prereg.json"
CKPT = {10: 6300, 19: 11970, 29: 18270, 39: 24570}
EPOCHS, WARMUP_EPOCHS, CLOSE_MOSAIC, IMGSZ = 39, 1.480952, 5, 512
WIDTHS = ("lpwa50",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", required=True, choices=WIDTHS)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params

    pre = json.loads(PREREG.read_text())
    assert args.seed in pre["seeds"], "seed not pre-registered"
    arch = pre["architectures"][args.width]
    want = json.loads((REF / "schedule.json").read_text())

    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    a = dict(cfg["train"])
    data = str(REPO / "data" / "tomato_village" / "dataset.yaml")
    a.update(data=data, imgsz=IMGSZ, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
             close_mosaic=CLOSE_MOSAIC, patience=10 ** 6, seed=args.seed, deterministic=True,
             pretrained=False, val=True, plots=False, multi_scale=False, rect=False,
             project=str(OUT / args.width), name=f"seed{args.seed}", exist_ok=True)
    a.pop("cv", None)

    rd = Path(a["project"]) / a["name"]
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "p16_done.json").exists():
        print(f"SKIP {args.width} seed{args.seed}"); return

    rows, st = [], {"t0": time.time()}
    model = YOLO(arch["yaml"])

    def on_train_start(tr):
        st["nb"] = len(tr.train_loader)
        st["nw"] = tr._get_warmup_iterations(st["nb"])
        st["mos"] = tr.epochs - tr.args.close_mosaic
        got = {"steps_per_epoch": st["nb"], "epochs": tr.epochs,
               "total_updates": st["nb"] * tr.epochs, "warmup_updates": st["nw"],
               "mosaic_close_epoch": st["mos"]}
        bad = {k: (want[k], got[k]) for k in got if want[k] != got[k]}
        net = tr.model.module if hasattr(tr.model, "module") else tr.model
        n_par = get_num_params(net)
        first = next(m for m in net.modules() if isinstance(m, torch.nn.Conv2d)).weight.detach().cpu()
        same_as_pre = None
        if True:
            pre_sd = torch.load(BASE / "tomato-adcf-yolo" / "yolo11n.pt", map_location="cpu",
                                weights_only=False)["model"].float().state_dict()
            pre_w = next(v for k, v in pre_sd.items() if k.endswith("conv.weight"))
            same_as_pre = bool(pre_w.shape == first.shape and torch.allclose(pre_w, first.float()))
        json.dump({"width": args.width, "imgsz": IMGSZ, "seed": args.seed, **got,
                   "ckpt_epochs": CKPT, "baseline": {k: want[k] for k in got}, "mismatch": bad,
                   "params": n_par, "audited_params": arch["params"],
                   "first_conv_equals_pretrained": same_as_pre, "init": "random (pretrained=False)"},
                  open(rd / "schedule.json", "w"), indent=1)
        if bad:
            print("SCHEDULE_MISMATCH " + json.dumps(bad), flush=True); raise SystemExit(3)
        if n_par != arch["params"]:
            print(f"PARAM_MISMATCH {n_par} != {arch['params']}", flush=True); raise SystemExit(4)
        if same_as_pre:
            print("PRETRAINED_LEAK first conv equals yolo11n.pt", flush=True); raise SystemExit(5)
        print("gates pass: schedule " + json.dumps(got) + f" params {n_par} random-init", flush=True)

    def on_fit_epoch_end(tr):
        if tr.epoch + 1 > tr.epochs:          # Ultralytics fires this again from final_eval
            return
        e, nb = tr.epoch + 1, st["nb"]
        gs = e * nb
        m = tr.metrics or {}
        row = {"width": args.width, "imgsz": IMGSZ, "seed": args.seed, "epoch": e,
               "global_step": gs, "frac_budget": round(gs / (nb * tr.epochs), 6),
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

    out = {"width": args.width, "seed": args.seed, "imgsz": IMGSZ, "train_hours": round(hours, 4),
           "total_updates": st["nb"] * EPOCHS, "params": arch["params"], "checkpoints": {}}
    for e, step in CKPT.items():
        w = rd / "weights" / f"ckpt_ep{e}_step{step}.pt"
        if not w.exists():
            out["checkpoints"][str(step)] = {"error": "missing"}; continue
        r = YOLO(str(w)).val(data=data, split="val", imgsz=IMGSZ, batch=a["batch"],
                             device=0, workers=0, plots=False, verbose=False)
        out["checkpoints"][str(step)] = {"epoch": e, "val_mAP50": float(r.box.map50),
                                         "val_mAP50_95": float(r.box.map),
                                         "val_precision": float(r.box.mp),
                                         "val_recall": float(r.box.mr)}
    json.dump(out, open(rd / "p16_done.json", "w"), indent=1)
    print("P16_DONE " + json.dumps({k: v for k, v in out.items() if k != "checkpoints"}), flush=True)


if __name__ == "__main__":
    main()
