from pathlib import Path
import shutil
p = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\configs\experiments.yaml")
shutil.copy2(p, p.with_suffix(".yaml.bak_dup"))
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
entry = "  - {name: yolo11s-dup8,      base: yolo11s, tier: 3, tables: [data], data: data/tomato_village_dup8/dataset.yaml}\n"
if any("name: yolo11s-dup8," in l for l in lines):
    print("already present")
else:
    a = max(i for i, l in enumerate(lines) if "name: yolo11s-noaug," in l)
    lines.insert(a + 1, entry)
    p.write_text("".join(lines), encoding="utf-8")
    print("added yolo11s-dup8")
