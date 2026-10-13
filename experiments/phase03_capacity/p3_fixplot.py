from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p3_plot.py")
s = p.read_text(encoding="utf-8")
a = 'ax1.text(PRIMARY + 450, 16.2, "primary\\nendpoint\\n18,270", ha="left", va="bottom",\n         fontsize=7, color="0.3")'
b = 'ax1.text(PRIMARY - 400, 25.25, "primary endpoint 18,270", ha="right", va="top",\n         fontsize=7, color="0.3")'
assert a in s, "anchor missing"
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched")
