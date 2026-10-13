"""Phase 9 feasibility: run registry + source-family exposure partition. NO evaluation, NO training.

The clean 268-photo TEST split is CLOSED. This script never reads clean-test labels or images for
metric purposes; it reads clean-test FAMILY IDS only, and only in order to EXCLUDE those families
from the candidate independent-evaluation pool. That exclusion is the point: without it the
independent arm would silently contain closed test material.
"""
import csv, json, re
from collections import Counter
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = Path(r"C:\Users\absamad\cs\phase9"); OUT.mkdir(exist_ok=True)
RAW = REPO / "data" / "raw" / "Tomato-Village" / "Variant-c(Object Detection)"
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
fam = lambda stem: norm(AUG.sub("", stem))

# ---------- A. run registry ----------
ROOTS = ["runs", "runs_cv", "runs_timing", "runs_bcd", "runs_p2b", "runs_p3", "runs_p4", "runs_p6"]
FIELDS = ["data", "model", "imgsz", "epochs", "patience", "batch", "nbs", "optimizer", "lr0", "lrf",
          "cos_lr", "warmup_epochs", "close_mosaic", "mosaic", "mixup", "cutmix", "copy_paste",
          "multi_scale", "rect", "pretrained", "deterministic", "seed", "fraction", "resume"]
rows = []
for root in ROOTS:
    rd = REPO / root
    if not rd.exists():
        continue
    for a in rd.rglob("args.yaml"):
        d = a.parent
        try:
            y = yaml.safe_load(a.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec = {"root": root, "run": d.parent.name, "seed_dir": d.name,
               "path": str(d.relative_to(REPO))}
        for f in FIELDS:
            rec[f] = y.get(f)
        ds = str(rec.get("data") or "")
        rec["dataset_dir"] = Path(ds).parent.name if ds else None
        # completion markers + metrics
        for mk in ("train_done.json", "p2b_done.json", "p3_done.json", "p4_done.json",
                   "p6_done.json", "bcd_done.json"):
            p = d / mk
            if p.exists():
                try:
                    j = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    j = {}
                rec["marker"] = mk
                rec["train_hours"] = j.get("train_hours")
                rec["epochs_run"] = j.get("epochs_run") or j.get("epochs")
                rec["total_updates"] = j.get("total_updates")
                ck = j.get("checkpoints") or {}
                rec["ckpt_steps"] = ",".join(sorted(ck, key=lambda x: int(x))) if ck else None
                break
        rv = d / "results_val.json"
        if rv.exists():
            try:
                rec["val_mAP50_95"] = json.loads(rv.read_text(encoding="utf-8")).get("mAP50_95")
            except Exception:
                pass
        rt = d / "results_test.json"
        rec["has_test_result"] = rt.exists()
        # weights available for re-evaluation without retraining
        w = d / "weights"
        wl = sorted(x.name for x in w.glob("*.pt")) if w.exists() else []
        rec["weights"] = ";".join(wl)
        rec["n_weight_files"] = len(wl)
        rec["has_last"] = "last.pt" in wl
        rec["has_best"] = "best.pt" in wl
        rec["n_frozen_ckpts"] = sum(1 for x in wl if x.startswith("ckpt_ep"))
        rec["reusable_for_eval"] = bool(wl)
        c = d / "results.csv"
        if c.exists():
            try:
                rec["csv_epochs"] = sum(1 for _ in c.open(encoding="utf-8")) - 1
            except Exception:
                pass
        rows.append(rec)

keys = sorted({k for r in rows for k in r})
with open(OUT / "run_registry.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

# ---------- B. family membership of every split we are allowed to reason about ----------
def fams_of_labels(d):
    out = {}
    if not d.exists():
        return out
    for p in sorted(d.iterdir()):
        if p.suffix != ".txt":
            continue
        out.setdefault(fam(p.stem), []).append(p.name)
    return out

CLEAN = REPO / "data" / "tomato_village" / "labels"
clean_train = fams_of_labels(CLEAN / "train")
clean_val = fams_of_labels(CLEAN / "val")
clean_test = fams_of_labels(CLEAN / "test")     # FAMILY IDS ONLY - for exclusion
off_train = fams_of_labels(RAW / "train" / "yolo")   # released YOLO-format labels
off_val = fams_of_labels(RAW / "val" / "yolo")

CT, CV, CTE = set(clean_train), set(clean_val), set(clean_test)
OT, OV = set(off_train), set(off_val)

# class distribution + object counts, read from label text only (no images, no model)
def stats_for(fams, label_dirs):
    cls = Counter(); boxes = 0; files = 0
    for d in label_dirs:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix != ".txt" or fam(p.stem) not in fams:
                continue
            files += 1
            for line in p.read_text().splitlines():
                q = line.split()
                if len(q) >= 5:
                    cls[int(float(q[0]))] += 1; boxes += 1   # released labels write ids as floats
    return {"files": files, "boxes": boxes, "classes": dict(sorted(cls.items()))}

OV_DIR = [RAW / "val" / "yolo"]
exposed = OV & CT                       # official-val families the clean-trained model SAW
indep_permitted = (OV & CV)             # official-val families in clean-VAL -> permitted
indep_closed = (OV & CTE)               # official-val families in clean-TEST -> MUST be excluded
indep_other = OV - CT - CV - CTE        # official-val families in no clean split

part = {
 "definitions": {
   "A_source_exposed": "official-val family that is also a clean-TRAIN family (model saw a relative)",
   "B_source_independent_permitted": "official-val family that is a clean-VAL family (never trained on, not test)",
   "B_excluded_closed_test": "official-val family that is a clean-TEST family - EXCLUDED to protect the closed test set",
   "unassigned": "official-val family present in no clean split"},
 "counts": {
   "official_val_families": len(OV), "official_train_families": len(OT),
   "clean_train_families": len(CT), "clean_val_families": len(CV), "clean_test_families": len(CTE),
   "A_exposed_families": len(exposed),
   "B_independent_permitted_families": len(indep_permitted),
   "B_excluded_closed_test_families": len(indep_closed),
   "unassigned_families": len(indep_other)},
 "A_exposed": stats_for(exposed, OV_DIR),
 "B_independent_permitted": stats_for(indep_permitted, OV_DIR),
 "B_excluded_closed_test": stats_for(indep_closed, OV_DIR),
 "unassigned": stats_for(indep_other, OV_DIR),
 "clean_val_whole": stats_for(CV, [CLEAN / "val"]),
}
# overlap sanity
part["sanity"] = {
  "clean_splits_disjoint": len(CT & CV) == 0 and len(CT & CTE) == 0 and len(CV & CTE) == 0,
  "A_and_B_disjoint": len(exposed & indep_permitted) == 0,
  "A_plus_B_plus_excluded_plus_unassigned_eq_officialval":
      len(exposed) + len(indep_permitted) + len(indep_closed) + len(indep_other) == len(OV),
  "official_val_families_also_in_official_train": len(OV & OT)}
json.dump(part, open(OUT / "exposure_partition.json", "w"), indent=1)

print(json.dumps({"registry_rows": len(rows),
                  "roots_seen": sorted({r["root"] for r in rows}),
                  "datasets_seen": sorted({str(r.get("dataset_dir")) for r in rows}),
                  "runs_with_weights": sum(1 for r in rows if r["reusable_for_eval"]),
                  "counts": part["counts"], "sanity": part["sanity"]}, indent=1))
print("A exposed:", part["A_exposed"]["files"], "files,", part["A_exposed"]["boxes"], "boxes")
print("B permitted:", part["B_independent_permitted"]["files"], "files,",
      part["B_independent_permitted"]["boxes"], "boxes")
print("B excluded (closed test):", part["B_excluded_closed_test"]["files"], "files")
print("REGDONE")
