from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p12_plot.py")
s = p.read_text(encoding="utf-8")
a = 'axes[0].legend(loc="center right", frameon=False)'
b = 'axes[0].legend(loc="center left", frameon=False, bbox_to_anchor=(0.02, 0.30))'
assert a in s, "anchor missing"
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched")
