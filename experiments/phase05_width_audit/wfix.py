from pathlib import Path
p = Path(r"C:\Users\absamad\cs\width_audit.py")
s = p.read_text(encoding="utf-8")
a = '    cfg["scales"] = {"w": [N_DEPTH, round(w, 6), N_MAXC]}'
b = '    cfg["scales"] = {"n": [N_DEPTH, round(w, 6), N_MAXC]}  # key must match the letter Ultralytics parses from the filename'
assert a in s
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched")
