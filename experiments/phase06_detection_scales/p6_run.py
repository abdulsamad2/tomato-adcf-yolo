"""PHASE 6 -- detection-scale experiment. FROZEN; do not modify after any seed completes.

Phase 4 YOLO11n @512 protocol, identical in every field except the detection-scale topology.
Arm A (P3+P4+P5) is NOT retrained -- it is runs_p4/r512/seed{0..4}.

  data        data/tomato_village (leakage-free, 10080 train files)
  imgsz 512 | batch 16 | steps_per_epoch 630 | epochs 39 | total updates 24570
  patience 1e6 (early stopping disabled)
  warmup_epochs 1.480952 -> 933 updates ; close_mosaic 5 -> 3150 mosaic-free updates
  SGD lr0 0.01 lrf 0.01, cos_lr false ; multi_scale false ; rect false ; deterministic
  FROZEN CHECKPOINTS: ep10->6300, 19->11970, 29->18270, 39->24570 ; PRIMARY 18270
  TEST IS NEVER TOUCHED.

Initialization uses the audited remap of yolo11n.pt and is VERIFIED against the audited values
inside on_train_start, by tensor-value equality, before a single optimizer step is taken.
A material deviation aborts the run rather than continuing silently.

Windows: __main__ guard required (Ultralytics dataloader uses spawn).
"""
import argparse, csv, json, shutil, time
from pathlib import Path
import torch, yaml

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
SCALES = BASE / "scales"
OUT = REPO / "runs_p6"
CKPT = {10: 6300, 19: 11970, 29: 18270, 39: 24570}
EPOCHS, WARMUP_EPOCHS, CLOSE_MOSAIC = 39, 1.480952, 5

# audited remap (phase6_detection_scale_audit.md section 2)
SPEC = {
 "B": {"body": {}, "det_old": 23, "det_new": 20, "branches": {0: 0, 1: 1}},
 "C": {"body": {19: 14, 20: 15, 21: 16, 22: 17}, "det_old": 23, "det_new": 18,
       "branches": {1: 0, 2: 1}},
}
# audited expectations: transferred %, body-new params, detect-new params
EXPECT = {"B": {"pct": 98.70, "body_new": 0, "detect_new": 24208},
          "C": {"pct": 95.76, "body_new": 16384, "detect_new": 89360}}
PCT_TOL = 0.05


def build_remapped(arm):
    """Return (model, remapped_tensors, stats) with audited pretrained weights loaded."""
    from ultralytics import YOLO
    sp = SPEC[arm]
    y = json.loads((SCALES / "variant_yamls.json").read_text())[arm]
    m = YOLO(y)
    net = m.model.float()
    dn = sp["det_new"]

    ck = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
    src = ck["model"].float().state_dict()
    remapped = {}
    for k, v in src.items():
        p = k.split(".")
        if len(p) > 1 and p[0] == "model" and p[1].isdigit():
            old = int(p[1])
            if old == sp["det_old"]:
                if len(p) > 3 and p[2] in ("cv2", "cv3") and p[3].isdigit():
                    nb = sp["branches"].get(int(p[3]))
                    if nb is None:
                        continue
                    p = ["model", str(dn), p[2], str(nb)] + p[4:]
                else:
                    p = ["model", str(dn)] + p[2:]
            elif old in sp["body"]:
                p = ["model", str(sp["body"][old])] + p[2:]
        remapped[".".join(p)] = v

    tgt = net.state_dict()
    total = int(sum(v.numel() for v in tgt.values()))
    load, miss = {}, []
    for k, v in tgt.items():
        s = remapped.get(k)
        if s is not None and s.shape == v.shape:
            load[k] = s.clone()
        else:
            miss.append(k)
    net.load_state_dict(load, strict=False)

    body_miss = [k for k in miss if not k.startswith(f"model.{dn}.")]
    det_miss = [k for k in miss if k.startswith(f"model.{dn}.")]
    stats = {"arm": arm, "target_tensors": len(tgt), "target_params": total,
             "transferred_tensors": len(load),
             "transferred_params": int(sum(v.numel() for v in load.values())),
             "transferred_pct": round(100 * sum(v.numel() for v in load.values()) / total, 2),
             "body_new_tensors": len(body_miss),
             "body_new_params": int(sum(tgt[k].numel() for k in body_miss)),
             "body_new_layers": sorted({k.split(".")[1] for k in body_miss}),
             "detect_new_tensors": len(det_miss),
             "detect_new_params": int(sum(tgt[k].numel() for k in det_miss)),
             "remap_body": {str(k): v for k, v in sp["body"].items()},
             "remap_detect": f"{sp['det_old']}->{dn}",
             "remap_branches": {str(k): v for k, v in sp["branches"].items()}}
    return m, load, stats


def check(stats, arm):
    e = EXPECT[arm]
    bad = []
    if abs(stats["transferred_pct"] - e["pct"]) > PCT_TOL:
        bad.append(f"transferred_pct {stats['transferred_pct']} != audited {e['pct']}")
    if stats["body_new_params"] != e["body_new"]:
        bad.append(f"body_new_params {stats['body_new_params']} != audited {e['body_new']}")
    if stats["detect_new_params"] != e["detect_new"]:
        bad.append(f"detect_new_params {stats['detect_new_params']} != audited {e['detect_new']}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["B", "C"])
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    from adcf_yolo.eval.predictions import predict_split
    from adcf_yolo.eval.coco_eval import coco_size_eval

    rd = OUT / args.arm / f"seed{args.seed}"
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "p6_done.json").exists():
        print(f"SKIP {args.arm} seed{args.seed}"); return

    model, loaded, stats = build_remapped(args.arm)
    bad = check(stats, args.arm)
    json.dump({"pre_train": stats, "audited_expected": EXPECT[args.arm],
               "deviations": bad}, open(rd / "init_report.json", "w"), indent=1)
    print("INIT " + json.dumps(stats))
    if bad:
        print("INIT_MISMATCH_ABORT " + json.dumps(bad))
        raise SystemExit(2)

    # persist as an Ultralytics checkpoint so the trainer receives the remapped weights
    ckpt_path = rd / "init_remapped.pt"
    torch.save({"model": model.model, "ema": None, "updates": 0, "optimizer": None,
                "train_args": {"task": "detect", "imgsz": 512},
                "date": time.strftime("%Y-%m-%d"), "version": "p6"}, ckpt_path)
    m = YOLO(str(ckpt_path))

    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    a = dict(cfg["train"])
    data = str(REPO / "data" / "tomato_village" / "dataset.yaml")
    a.update(data=data, imgsz=512, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
             close_mosaic=CLOSE_MOSAIC, patience=10 ** 6, seed=args.seed, deterministic=True,
             val=True, plots=False, multi_scale=False, rect=False,
             # NOTE: pretrained must NOT be set False here. Ultralytics does
             #   weights = None if pretrained is False else self.model
             # so pretrained=False discards the remapped checkpoint and rebuilds from YAML.
             # Leaving the Phase 4 config value seeds the trainer from our remapped weights.
             project=str(OUT / args.arm), name=f"seed{args.seed}", exist_ok=True)
    a.pop("cv", None)

    yn = yaml.safe_load(Path(data).read_text(encoding="utf-8")).get("names") or {}
    NAMES = {int(k): v for k, v in yn.items()} if isinstance(yn, dict) else dict(enumerate(yn))

    rows, st = [], {"t0": time.time()}

    def on_train_start(tr):
        # VERIFY the weights the trainer actually holds, by value, before any optimizer step
        sd = tr.model.state_dict()
        ok = sum(1 for k, v in loaded.items()
                 if k in sd and sd[k].shape == v.shape and torch.equal(sd[k].detach().cpu().float(), v.float()))
        okp = int(sum(v.numel() for k, v in loaded.items()
                      if k in sd and sd[k].shape == v.shape
                      and torch.equal(sd[k].detach().cpu().float(), v.float())))
        ver = {"verified_tensors": ok, "expected_tensors": len(loaded),
               "verified_params": okp, "expected_params": stats["transferred_params"],
               "verified_pct": round(100 * okp / stats["target_params"], 2)}
        json.dump(ver, open(rd / "init_verified.json", "w"), indent=1)
        print("VERIFY " + json.dumps(ver), flush=True)
        if ok != len(loaded) or okp != stats["transferred_params"]:
            raise RuntimeError(f"INIT_VERIFY_FAILED {ver}")
        st["nb"] = len(tr.train_loader)
        st["nw"] = tr._get_warmup_iterations(st["nb"])
        st["mos"] = tr.epochs - tr.args.close_mosaic
        json.dump({"arm": args.arm, "seed": args.seed, "steps_per_epoch": st["nb"],
                   "epochs": tr.epochs, "total_updates": st["nb"] * tr.epochs,
                   "warmup_updates": st["nw"], "mosaic_close_epoch": st["mos"],
                   "mosaic_free_updates": tr.args.close_mosaic * st["nb"],
                   "ckpt_epochs": CKPT}, open(rd / "schedule.json", "w"), indent=1)

    def on_fit_epoch_end(tr):
        if tr.epoch + 1 > tr.epochs:          # Ultralytics fires this again from final_eval
            return
        e, nb = tr.epoch + 1, st["nb"]
        gs = e * nb
        m_ = tr.metrics or {}
        row = {"arm": args.arm, "seed": args.seed, "epoch": e, "global_step": gs,
               "frac_budget": round(gs / (nb * tr.epochs), 6),
               "learning_rate": tr.optimizer.param_groups[0]["lr"],
               "validated": bool(tr.args.val), "is_checkpoint": e in CKPT,
               "mosaic_on": tr.epoch < st["mos"],
               "val_mAP50": m_.get("metrics/mAP50(B)"), "val_mAP50_95": m_.get("metrics/mAP50-95(B)"),
               "precision": m_.get("metrics/precision(B)"), "recall": m_.get("metrics/recall(B)"),
               "val_box_loss": m_.get("val/box_loss"), "val_cls_loss": m_.get("val/cls_loss"),
               "val_dfl_loss": m_.get("val/dfl_loss"),
               "wall_s": round(time.time() - st["t0"], 1)}
        rows.append(row)
        keys = sorted({k for r in rows for k in r})
        with open(rd / "curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        if e in CKPT:
            src = rd / "weights" / "last.pt"
            if src.exists():
                shutil.copy2(src, rd / "weights" / f"ckpt_ep{e}_step{CKPT[e]}.pt")

    m.add_callback("on_train_start", on_train_start)
    m.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    t0 = time.time()
    m.train(**a)
    hours = (time.time() - t0) / 3600

    from ultralytics.utils.torch_utils import get_num_params, get_flops
    out = {"arm": args.arm, "seed": args.seed, "train_hours": round(hours, 4),
           "total_updates": st["nb"] * EPOCHS, "init": stats, "checkpoints": {}}
    for e, step in CKPT.items():
        w = rd / "weights" / f"ckpt_ep{e}_step{step}.pt"
        if not w.exists():
            out["checkpoints"][str(step)] = {"error": "missing"}; continue
        mm = YOLO(str(w))
        r = mm.val(data=data, split="val", imgsz=512, batch=a["batch"],
                   device=0, workers=0, plots=False, verbose=False)
        pc = {}
        try:
            for i, ci in enumerate(r.box.ap_class_index):
                pc[str(int(ci))] = {"ap50_95": float(r.box.maps[int(ci)]),
                                    "ap50": float(r.box.ap50[i])}
        except Exception:
            pass
        images = predict_split(w, data, split="val", imgsz=512, device="0")
        out["checkpoints"][str(step)] = {
            "epoch": e, "val_mAP50": float(r.box.map50), "val_mAP50_95": float(r.box.map),
            "val_precision": float(r.box.mp), "val_recall": float(r.box.mr),
            "per_class": pc, "coco": coco_size_eval(images, NAMES)}
        if e == EPOCHS:
            try:
                out["params_M"] = round(get_num_params(mm.model) / 1e6, 4)
                out["GFLOPs"] = round(get_flops(mm.model, 512), 3)
            except Exception:
                pass
    json.dump(out, open(rd / "p6_done.json", "w"), indent=1)
    print("P6_DONE " + json.dumps({k: v for k, v in out.items()
                                   if k not in ("checkpoints", "init")}))


if __name__ == "__main__":
    main()
