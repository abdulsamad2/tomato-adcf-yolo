import csv, json
from pathlib import Path
ROOT = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
rows = []
for base, kind in ((ROOT/"runs", "seed"), (ROOT/"runs_cv", "fold")):
    if not base.exists():
        continue
    for done in sorted(base.glob("*/*/train_done.json")):
        d = done.parent
        rec = {"run": d.parent.name, "unit": d.name, "kind": kind, "root": base.name}
        td = json.loads(done.read_text())
        rec["train_hours"] = round(td.get("train_hours", float("nan")), 4)
        rec["seed"] = td.get("seed"); rec["fold"] = td.get("fold"); rec["data"] = td.get("data")
        spec = (td.get("run") or {}).get("adcf")
        rec["adcf_spec"] = json.dumps(spec) if spec else ""
        rec["train_override"] = json.dumps((td.get("run") or {}).get("train") or {})
        for split in ("val", "test"):
            p = d/f"results_{split}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            for k in ("mAP50", "mAP50_95", "precision", "recall", "params_M", "GFLOPs", "imgsz"):
                if k in m:
                    rec[f"{split}_{k}"] = m[k]
            coco = m.get("coco") or {}
            for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                      "AR_small", "AR_medium", "AR_large"):
                if k in coco:
                    rec[f"{split}_coco_{k}"] = coco[k]
            sp = m.get("speed") or {}
            for k, v in sp.items():
                rec[f"{split}_speed_{k}"] = v
            for cls, cm in (m.get("per_class") or {}).items():
                if "AP50" in cm:
                    rec[f"{split}_AP50_{cls.replace(' ', '_')}"] = cm["AP50"]
        rows.append(rec)
cols = sorted({k for r in rows for k in r},
              key=lambda k: (k not in ("run", "unit", "kind", "root", "seed", "fold"), k))
out = ROOT/"results"/"all_runs_record.csv"
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("rows", len(rows), "cols", len(cols), "bytes", out.stat().st_size)
