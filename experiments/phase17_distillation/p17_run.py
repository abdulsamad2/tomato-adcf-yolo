"""PHASE 17 -- single-configuration localization distillation into the Phase 15 0.75x student.
FROZEN by phase17/p17_prereg.json. Training code = the Phase 15 runner + ONE KD loss (p17_kd.py).

Identical to runs_p15/w075 in every field (yaml, random init, data, imgsz 512, SGD, batch 16, 24570 updates,
warmup 933, close_mosaic 5, seeds, augmentation, deterministic) EXCEPT the added training-time KD loss.

Gates before step 1 (abort with the code shown):
  3 schedule != runs_p4/r512/seed0/schedule.json        4 student params != 1,565,628 (Phase 15 audit)
  5 student first conv equals yolo11n.pt (pretrained leak)
  6 teacher checkpoint sha256 != pre-registered         7 any teacher param requires_grad or teacher in train mode
Checkpoints at 6300 / 11970 / 18270 / 24570 updates. TEST IS NOT TOUCHED.
"""
import argparse, csv, hashlib, json, shutil, sys, time
from pathlib import Path
import yaml

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
REF = REPO / "runs_p4" / "r512" / "seed0"
STUDENT_YAML = BASE / "widths" / "yolo11n_w0p75.yaml"
STUDENT_PARAMS = 1565628
CKPT = {10: 6300, 19: 11970, 29: 18270, 39: 24570}
EPOCHS, WARMUP_EPOCHS, CLOSE_MOSAIC, IMGSZ = 39, 1.480952, 5, 512
sys.path.insert(0, str(BASE))


class StopDry(Exception):
    pass


def sd_sha(net):
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train(seed, rd, kd=None, stop_after=None, record=None):
    """kd = None (ordinary Phase 15 trainer) or {teacher, teacher_sha256, lam_ld, lam_at, t_ld, probe}."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params
    import p17_kd

    want = json.loads((REF / "schedule.json").read_text())
    cfg = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    a = dict(cfg["train"])
    data = str(REPO / "data" / "tomato_village" / "dataset.yaml")
    rd = Path(rd)
    a.update(data=data, imgsz=IMGSZ, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, close_mosaic=CLOSE_MOSAIC,
             patience=10 ** 6, seed=seed, deterministic=True, pretrained=False, val=stop_after is None,
             plots=False, multi_scale=False, rect=False, project=str(rd.parent), name=rd.name, exist_ok=True)
    a.pop("cv", None)
    rd.mkdir(parents=True, exist_ok=True)
    rows, st = [], {"t0": time.time()}
    rec = record if record is not None else {}
    model = YOLO(str(STUDENT_YAML))

    def on_train_start(tr):
        st["nb"] = len(tr.train_loader); st["nw"] = tr._get_warmup_iterations(st["nb"])
        st["mos"] = tr.epochs - tr.args.close_mosaic
        got = {"steps_per_epoch": st["nb"], "epochs": tr.epochs, "total_updates": st["nb"] * tr.epochs,
               "warmup_updates": st["nw"], "mosaic_close_epoch": st["mos"]}
        bad = {k: (want[k], got[k]) for k in got if want[k] != got[k]}
        net = tr.model.module if hasattr(tr.model, "module") else tr.model
        n_par = get_num_params(net)
        first = next(m for m in net.modules() if isinstance(m, torch.nn.Conv2d)).weight.detach().cpu().float()
        pre_sd = torch.load(REPO / "yolo11n.pt", map_location="cpu", weights_only=False)["model"].float().state_dict()
        pre_w = next(v for k, v in pre_sd.items() if k.endswith("conv.weight"))
        same_as_pre = bool(pre_w.shape == first.shape and torch.allclose(pre_w, first))
        rec.update(schedule=got, mismatch=bad, params=n_par, init_sd_sha256=sd_sha(net),
                   first_conv_equals_pretrained=same_as_pre, first_conv_shape=list(first.shape))
        if kd is not None:
            th = hashlib.sha256(Path(kd["teacher"]).read_bytes()).hexdigest()
            crit, teacher = p17_kd.attach(tr, kd["teacher"], kd["lam_ld"], kd["lam_at"], kd["t_ld"], kd.get("probe", False))
            st["crit"] = crit
            rec.update(teacher=str(kd["teacher"]), teacher_sha256=th,
                       teacher_any_requires_grad=any(p.requires_grad for p in teacher.parameters()),
                       teacher_training_mode=bool(teacher.training),
                       teacher_params=get_num_params(teacher), kd=dict(kd))
        json.dump({"seed": seed, **got, "ckpt_epochs": CKPT, "baseline": {k: want[k] for k in got}, **rec},
                  open(rd / "schedule.json", "w"), indent=1, default=str)
        if bad: print("SCHEDULE_MISMATCH " + json.dumps(bad), flush=True); raise SystemExit(3)
        if n_par != STUDENT_PARAMS: print("PARAM_MISMATCH %d" % n_par, flush=True); raise SystemExit(4)
        if same_as_pre: print("PRETRAINED_LEAK", flush=True); raise SystemExit(5)
        if kd is not None:
            if kd.get("teacher_sha256") and rec["teacher_sha256"] != kd["teacher_sha256"]:
                print("TEACHER_HASH_MISMATCH", flush=True); raise SystemExit(6)
            if rec["teacher_any_requires_grad"] or rec["teacher_training_mode"]:
                print("TEACHER_NOT_FROZEN", flush=True); raise SystemExit(7)
        print("gates pass: " + json.dumps(got) + " params %d kd=%s" % (n_par, kd is not None), flush=True)

    def on_train_batch_end(tr):
        st["b"] = st.get("b", 0) + 1
        if record is not None:
            rec.setdefault("batch_loss", []).append(float(tr.loss.detach()))
            li = tr.loss_items.values() if isinstance(tr.loss_items, dict) else tr.loss_items
            rec.setdefault("batch_items", []).append([float(x) for x in li])
        if stop_after is not None and st["b"] >= stop_after:
            raise StopDry()

    def on_fit_epoch_end(tr):
        if tr.epoch + 1 > tr.epochs: return
        e, nb = tr.epoch + 1, st["nb"]; gs = e * nb; m = tr.metrics or {}
        row = {"seed": seed, "epoch": e, "global_step": gs, "learning_rate": tr.optimizer.param_groups[0]["lr"],
               "is_checkpoint": e in CKPT, "mosaic_on": tr.epoch < st["mos"],
               "val_mAP50": m.get("metrics/mAP50(B)"), "val_mAP50_95": m.get("metrics/mAP50-95(B)"),
               "wall_s": round(time.time() - st["t0"], 1)}
        try:
            row.update({k: float(v) for k, v in tr.label_loss_items(tr.tloss, prefix="train").items()})
        except Exception:
            pass
        c = st.get("crit")
        if c is not None and c.log["n"]:
            row.update(kd_ld_raw=c.log["ld"] / c.log["n"], kd_at_raw=c.log["at"] / c.log["n"],
                       det_loss=c.log["det"] / c.log["n"])
            c.log.update(n=0, ld=0.0, at=0.0, det=0.0); c.batch_log.clear()
        rows.append(row)
        keys = sorted({k for r in rows for k in r})
        with open(rd / "curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        if e in CKPT and (rd / "weights" / "last.pt").exists():
            shutil.copy2(rd / "weights" / "last.pt", rd / "weights" / ("ckpt_ep%d_step%d.pt" % (e, CKPT[e])))

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    t0 = time.time()
    try:
        model.train(**a)
    except StopDry:
        rec["stopped_after_batches"] = st["b"]
        c = st.get("crit")
        if c is not None:
            rec["kd_batch_log"] = c.batch_log; rec["kd_shapes"] = c.first_shapes
        return rec, None
    return rec, (time.time() - t0) / 3600, data, a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    from ultralytics import YOLO
    pre = json.loads((BASE / "phase17" / "p17_prereg.json").read_text())
    assert args.seed in pre["seeds"], "seed not pre-registered"
    rd = REPO / "runs_p17" / ("seed%d" % args.seed)
    if (rd / "p17_done.json").exists():
        print("SKIP seed%d" % args.seed); return
    K = pre["distillation"]
    kd = {"teacher": pre["teacher"]["checkpoint"], "teacher_sha256": pre["teacher"]["sha256"],
          "lam_ld": K["lam_ld"], "lam_at": K["lam_at"], "t_ld": K["T_ld"]}
    rec, hours, data, a = train(args.seed, rd, kd=kd)
    out = {"seed": args.seed, "train_hours": round(hours, 4), "total_updates": rec["schedule"]["total_updates"],
           "params": rec["params"], "teacher_sha256": rec["teacher_sha256"], "checkpoints": {}}
    for e, step in CKPT.items():
        w = rd / "weights" / ("ckpt_ep%d_step%d.pt" % (e, step))
        if not w.exists():
            out["checkpoints"][str(step)] = {"error": "missing"}; continue
        r = YOLO(str(w)).val(data=data, split="val", imgsz=IMGSZ, batch=a["batch"], device=0, workers=0,
                             plots=False, verbose=False)
        out["checkpoints"][str(step)] = {"epoch": e, "val_mAP50": float(r.box.map50), "val_mAP50_95": float(r.box.map)}
    json.dump(out, open(rd / "p17_done.json", "w"), indent=1)
    print("P17_DONE " + json.dumps({k: v for k, v in out.items() if k != "checkpoints"}), flush=True)


if __name__ == "__main__":
    main()
