"""Hardlink the original images into the derived single-class view. No reparse points."""
import json, os, shutil
from pathlib import Path
REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
SRC = REPO / "data" / "tomato_village" / "images"
D1 = REPO / "data" / "tomato_village_1cls"
DST = D1 / "images"
made = {"hardlink": 0, "copied": 0, "existing": 0}
for sp in ("train", "val"):
    (DST / sp).mkdir(parents=True, exist_ok=True)
    for f in sorted((SRC / sp).iterdir()):
        if not f.is_file(): continue
        t = DST / sp / f.name
        if t.exists(): made["existing"] += 1; continue
        try:
            os.link(str(f), str(t)); made["hardlink"] += 1
        except Exception:
            shutil.copy2(f, t); made["copied"] += 1
rep = {"made": made, "images_is_reparse_point": os.path.islink(str(DST))}
for sp in ("train", "val"):
    ni = len([p for p in (DST / sp).iterdir() if p.is_file()])
    nl = len([p for p in (D1 / "labels" / sp).iterdir() if p.suffix == ".txt"])
    rep[sp] = {"images": ni, "labels": nl, "match": ni == nl}
s0 = sorted((SRC / "val").iterdir())[0]
d0 = DST / "val" / s0.name
rep["bytes_identical_sample"] = s0.stat().st_size == d0.stat().st_size
lab = (D1 / "labels" / "val" / (s0.stem + ".txt")).read_text().splitlines()
rep["derived_label_class_fields"] = sorted({l.split()[0] for l in lab if l.split()})
orig = (REPO / "data" / "tomato_village" / "labels" / "val" / (s0.stem + ".txt")).read_text().splitlines()
rep["original_label_class_fields"] = sorted({l.split()[0] for l in orig if l.split()})
json.dump(rep, open(D1 / "link_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print("LINKDONE")
