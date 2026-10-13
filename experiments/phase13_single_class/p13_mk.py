"""Phase 13: derived SINGLE-CLASS training view. Originals are never modified.

All six populated Tomato-Village classes map to one class, "labelled foliar region".
Images are shared with the original split via a directory junction (no copy, no admin).
Label files are NEW files with the class field rewritten to 0; geometry is byte-identical.
Source-family assignment is inherited unchanged because the file names are unchanged.

The derived view declares train + val only. Test is deliberately absent so that no code path
in this phase can reach it.
"""
import json, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
SRC = REPO / "data" / "tomato_village"
DST = REPO / "data" / "tomato_village_1cls"
SPLITS = ("train", "val")

DST.mkdir(parents=True, exist_ok=True)
# images: junction to the ORIGINAL image tree (read-only use)
imgs = DST / "images"
if not imgs.exists():
    subprocess.run(["cmd", "/c", "mklink", "/J", str(imgs), str(SRC / "images")],
                   check=True, capture_output=True)
report = {"splits": {}, "classes_seen": {}, "geometry_identical": True}
for sp in SPLITS:
    sd = SRC / "labels" / sp
    dd = DST / "labels" / sp
    dd.mkdir(parents=True, exist_ok=True)
    n_files = n_boxes = 0
    seen = {}
    for f in sorted(sd.iterdir()):
        if f.suffix != ".txt": continue
        out = []
        for line in f.read_text().splitlines():
            t = line.split()
            if not t: continue
            c = int(float(t[0])); seen[c] = seen.get(c, 0) + 1
            out.append(" ".join(["0"] + t[1:]))
            n_boxes += 1
        (dd / f.name).write_text(("\n".join(out) + "\n") if out else "")
        n_files += 1
    report["splits"][sp] = {"label_files": n_files, "boxes": n_boxes}
    report["classes_seen"][sp] = dict(sorted(seen.items()))

# verify: geometry preserved exactly, only the class field changed
bad = 0
for sp in SPLITS:
    for f in sorted((SRC / "labels" / sp).iterdir()):
        if f.suffix != ".txt": continue
        a = [l.split() for l in f.read_text().splitlines() if l.split()]
        b = [l.split() for l in (DST / "labels" / sp / f.name).read_text().splitlines() if l.split()]
        if len(a) != len(b) or any(x[1:] != y[1:] for x, y in zip(a, b)) or any(y[0] != "0" for y in b):
            bad += 1
report["files_with_altered_geometry"] = bad
report["geometry_identical"] = (bad == 0)

(DST / "dataset.yaml").write_text(
    "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: labelled foliar region\n" % DST,
    encoding="utf-8")
# originals must be untouched
report["original_dataset_yaml_unchanged"] = (SRC / "dataset.yaml").read_text().count("Early Blight") == 1
report["original_train_label_classes"] = sorted({int(float(l.split()[0]))
    for f in list((SRC / "labels" / "train").iterdir())[:400] if f.suffix == ".txt"
    for l in f.read_text().splitlines() if l.split()})
json.dump(report, open(DST / "derive_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
print("MKDONE" if bad == 0 else "MK_GEOMETRY_MISMATCH")
