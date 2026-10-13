from pathlib import Path
import shutil
p = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\configs\experiments.yaml")
shutil.copy2(p, p.with_suffix(".yaml.bak_eff"))
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
want = {
 "yolo11s-320":  "  - {name: yolo11s-320,       base: yolo11s, tier: 3, tables: [resolution], train: {imgsz: 320}}\n",
 "yolo11s-480":  "  - {name: yolo11s-480,       base: yolo11s, tier: 3, tables: [resolution], train: {imgsz: 480}}\n",
 "yolo11s-aug2": "  - {name: yolo11s-aug2,      base: yolo11s, tier: 3, tables: [data], data: data/tomato_village_aug2/dataset.yaml}\n",
 "yolo11s-aug4": "  - {name: yolo11s-aug4,      base: yolo11s, tier: 3, tables: [data], data: data/tomato_village_aug4/dataset.yaml}\n",
}
added = []
for nm, text in want.items():
    if any(f"name: {nm}," in l for l in lines):
        continue
    anchor = max(i for i, l in enumerate(lines) if "name: yolo11s-noaug," in l)
    lines.insert(anchor + 1, text)
    added.append(nm)
p.write_text("".join(lines), encoding="utf-8")
print("added:", added or "none")
