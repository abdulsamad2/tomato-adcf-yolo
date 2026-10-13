"""Resolution audit: every run whose base model is yolo11n, plus every run at imgsz != 640."""
import csv, json, re
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
counts = {}
for d in (REPO / "data").iterdir():
    if d.is_dir() and (d / "images" / "train").is_dir():
        counts[d.name] = len(list((d / "images" / "train").iterdir()))

PROTO = ["data","imgsz","epochs","patience","optimizer","lr0","lrf","cos_lr","warmup_epochs",
         "momentum","weight_decay","batch","nbs","close_mosaic","mosaic","mixup","cutmix",
         "copy_paste","scale","translate","fliplr","flipud","degrees","shear","perspective",
         "hsv_h","hsv_s","hsv_v","erasing","auto_augment","multi_scale","rect","box","cls","dfl",
         "seed","deterministic","pretrained","model","fraction","resume","val"]

n_rows, res_rows = [], []
for root in ("runs","runs_cv","runs_timing","runs_bcd","runs_p2b","runs_p3"):
    rd = REPO / root
    if not rd.is_dir():
        continue
    for a in sorted(x for x in rd.iterdir() if x.is_dir()):
        for s in sorted(x for x in a.iterdir() if x.is_dir()):
            ay = s / "args.yaml"
            if not ay.exists():
                continue
            cfg = yaml.safe_load(ay.read_text(encoding="utf-8")) or {}
            base = str(cfg.get("model",""))
            imgsz = cfg.get("imgsz")
            is_n = ("yolo11n" in base.lower()) or ("yolo11n" in a.name.lower())
            if not (is_n or imgsz != 640):
                continue
            dpath = str(cfg.get("data",""))
            dname = Path(dpath).parent.name if dpath else ""
            ntr = counts.get(dname)
            spe = (ntr // int(cfg.get("batch",16))) if ntr else None
            rcsv = s / "results.csv"
            ep = (sum(1 for _ in rcsv.open(encoding="utf-8")) - 1) if rcsv.exists() else None
            r = {"root":root,"run":a.name,"seed_dir":s.name,"is_yolo11n":is_n,
                 "dataset_dir":dname,"train_images":ntr,"steps_per_epoch":spe,
                 "epochs_run":ep,"updates_run":(spe*ep) if (spe and ep) else None,
                 "done":any((s/f).exists() for f in
                            ("train_done.json","p2b_done.json","bcd_done.json","p3_done.json"))}
            for k in PROTO:
                r[k] = cfg.get(k)
            for tag, fn in (("val","results_val.json"),("test","results_test.json")):
                p = s / fn
                if p.exists():
                    j = json.loads(p.read_text())
                    r[f"{tag}_mAP50_95"] = j.get("mAP50_95"); r[f"{tag}_mAP50"] = j.get("mAP50")
                    for kk in ("params_M","GFLOPs"):
                        if j.get(kk) is not None: r[kk] = j[kk]
                    co = j.get("coco") or {}
                    r[f"{tag}_AP_small"] = co.get("AP_small")
            for fn in ("p3_done.json","p2b_done.json","train_done.json","bcd_done.json"):
                p = s / fn
                if p.exists():
                    j = json.loads(p.read_text())
                    r["train_hours"] = j.get("train_hours") or j.get("hours")
                    if "checkpoints" in j:
                        c18 = (j["checkpoints"] or {}).get("18270") or {}
                        c24 = (j["checkpoints"] or {}).get("24570") or {}
                        r["val_mAP50_95_18270"] = c18.get("val_mAP50_95")
                        r["val_mAP50_95_24570"] = c24.get("val_mAP50_95")
                    if j.get("params_M"): r["params_M"] = j["params_M"]
                    if j.get("GFLOPs_640"): r["GFLOPs"] = j["GFLOPs_640"]
                    break
            (n_rows if is_n else res_rows).append(r)

allr = n_rows + res_rows
keys = sorted({k for r in allr for k in r})
with open(REPO / "results" / "resolution_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(allr)
print(json.dumps({
 "yolo11n_runs": len(n_rows),
 "yolo11n_by_imgsz": {str(i): sum(1 for r in n_rows if r["imgsz"] == i)
                      for i in sorted({r["imgsz"] for r in n_rows})},
 "non640_runs_any_model": sorted({(r["run"], r["imgsz"]) for r in res_rows}),
 "dataset_train_counts": counts}, indent=1))
