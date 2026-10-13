from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p8_sample.py")
s = p.read_text(encoding="utf-8")
assert "TARGET_PER_CELL = 7" in s
s = s.replace("TARGET_PER_CELL = 7", "TARGET_PER_CELL = 11")
p.write_text(s, encoding="utf-8")
print("patched to 11 per cell")
