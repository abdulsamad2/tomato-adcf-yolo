from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p9_registry.py")
s = p.read_text(encoding="utf-8")
a = 'off_train = fams_of_labels(RAW / "train" / "labels")\noff_val = fams_of_labels(RAW / "val" / "labels")'
b = 'off_train = fams_of_labels(RAW / "train" / "yolo")   # released YOLO-format labels\noff_val = fams_of_labels(RAW / "val" / "yolo")'
assert a in s, "anchor1"
s = s.replace(a, b, 1)
a2 = 'OV_DIR = [RAW / "val" / "labels"]'
b2 = 'OV_DIR = [RAW / "val" / "yolo"]'
assert a2 in s, "anchor2"
s = s.replace(a2, b2, 1)
p.write_text(s, encoding="utf-8")
print("patched")
