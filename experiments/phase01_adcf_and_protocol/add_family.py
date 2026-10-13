"""Phase 1 of the cross-family plan: add seven non-YOLO11 detector architectures.

Screening entries only (1 seed via --seeds 0); the shortlist is retrained at 3 seeds.
yolov8-ghost has no pretrained checkpoint on the hub, so it is built from YAML and trained
from scratch -- recorded explicitly because it is not comparable to the pretrained arms.
"""
from pathlib import Path
import shutil

p = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\configs\experiments.yaml")
shutil.copy2(p, p.with_suffix(".yaml.bak_family"))
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

entries = {
 # transformer detector, NMS-free -- the biggest architectural jump from YOLO
 "rtdetr-l":       "  - {name: rtdetr-l,          base: rtdetr-l, tier: 4, tables: [family]}\n",
 # newer YOLO generations not yet tested
 "yolov9s":        "  - {name: yolov9s,           base: yolov9s, tier: 4, tables: [family]}\n",
 "yolov10s":       "  - {name: yolov10s,          base: yolov10s, tier: 4, tables: [family]}\n",
 "yolo12s":        "  - {name: yolo12s,           base: yolo12s, tier: 4, tables: [family]}\n",
 "yolov6s":        "  - {name: yolov6s,           base: yolov6s, tier: 4, tables: [family]}\n",
 "yolov5su":       "  - {name: yolov5su,          base: yolov5su, tier: 4, tables: [family]}\n",
 # Ghost convolutions: module-level compute reduction, no pretrained weights on the hub
 "yolov8-ghost":   "  - {name: yolov8-ghost,      base: yolov8s, cfg: yolov8-ghost.yaml, tier: 4, tables: [family], pretrained: false}\n",
}
added, present = [], []
anchor = max(i for i, l in enumerate(lines) if l.strip().startswith("- {name:"))
for nm, text in entries.items():
    if any(f"name: {nm}," in l for l in lines):
        present.append(nm)
    else:
        lines.insert(anchor + 1, text)
        added.append(nm)
p.write_text("".join(lines), encoding="utf-8")
print("added:", added)
print("already present:", present or "none")
