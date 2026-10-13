from pathlib import Path
import shutil
p = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\configs\experiments.yaml")
shutil.copy2(p, p.with_suffix(".yaml.bak_nano"))
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
want = {
 "yolov8n": "  - {name: yolov8n,       base: yolov8n, tier: 3, tables: [scale]}\n",
 "yolo26n": "  - {name: yolo26n,       base: yolo26n, tier: 3, tables: [scale]}\n",
}
added = []
for nm, text in want.items():
    if any(f"name: {nm}," in l for l in lines):
        continue
    anchor = max(i for i, l in enumerate(lines) if "name: yolo11m-adcf" in l)
    lines.insert(anchor + 1, text)
    added.append(nm)
p.write_text("".join(lines), encoding="utf-8")
print("added:", added or "none (already present)")
