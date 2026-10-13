"""Restore the config and insert the cross-family entries safely.

Two defects in the first attempt:
  1. The anchor was the last line starting with "- {name:", but some entries span two lines
     (continuation "adcf: *final}"), so the insert split an entry and broke the YAML.
     Fix: anchor on the last COMPLETE single-line entry (starts "- {name:" and ends "}").
  2. tables: [family] referenced a table that does not exist; the defined set is
     ladder/main/control/resolution/fusion/placement/scale/data. Fix: define `family`.
"""
import re
from pathlib import Path
import shutil

p = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\configs\experiments.yaml")
bak = p.with_suffix(".yaml.bak_family")
shutil.copy2(bak, p)                      # undo the broken insert
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

# 1. define the `family` table next to `data`
ti = next(i for i, l in enumerate(lines) if re.match(r"\s+data:\s+\{reference:", l))
if not any("family:" in l and "reference:" in l for l in lines):
    indent = re.match(r"(\s*)", lines[ti]).group(1)
    lines.insert(ti + 1, f'{indent}family:     {{reference: yolo11s, title: "Cross-family comparison"}}\n')

# 2. insert after the last COMPLETE one-line run entry
COMPLETE = re.compile(r"^\s*- \{name: .*\}\s*$")
anchor = max(i for i, l in enumerate(lines) if COMPLETE.match(l))
entries = [
 "  - {name: rtdetr-l,          base: rtdetr-l, tier: 4, tables: [family]}\n",
 "  - {name: yolov9s,           base: yolov9s, tier: 4, tables: [family]}\n",
 "  - {name: yolov10s,          base: yolov10s, tier: 4, tables: [family]}\n",
 "  - {name: yolo12s,           base: yolo12s, tier: 4, tables: [family]}\n",
 "  - {name: yolov5su,          base: yolov5su, tier: 4, tables: [family]}\n",
]
added = []
for e in entries:
    nm = e.split("name: ")[1].split(",")[0]
    if not any(f"name: {nm}," in l for l in lines):
        lines.insert(anchor + 1, e)
        added.append(nm)
p.write_text("".join(lines), encoding="utf-8")

# 3. structural check: YAML must still parse and every entry must be a dict with a name
import yaml
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
runs = cfg["runs"]
assert all(isinstance(r, dict) and "name" in r for r in runs), "malformed run entry"
print(f"yaml OK: {len(runs)} run entries, tables={sorted(cfg['tables'])}")
print("added:", added)
print("anchor line was:", lines[anchor].strip()[:70])
