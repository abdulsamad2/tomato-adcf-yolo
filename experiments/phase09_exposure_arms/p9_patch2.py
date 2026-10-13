from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p9_registry.py")
s = p.read_text(encoding="utf-8")
a = "                    cls[int(q[0])] += 1; boxes += 1"
b = "                    cls[int(float(q[0]))] += 1; boxes += 1   # released labels write ids as floats"
assert a in s, "anchor"
s = s.replace(a, b, 1)
p.write_text(s, encoding="utf-8")
print("patched")
