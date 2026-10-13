from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p12_infer.py")
s = p.read_text(encoding="utf-8")
a = '    armids = {k: set(arms[k]) for k in ("A", "B")}'
b = '    armids = {k: set(arms["arm_image_ids"][k]) for k in ("A", "B")}'
assert a in s, "anchor missing"
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched")
