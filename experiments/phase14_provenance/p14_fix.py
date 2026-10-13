from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p14_prov.py")
s = p.read_text(encoding="utf-8")
a = 'RAW.parent.parent.parent / "results"'
b = 'Path(r"C:\\Users\\absamad\\cs\\phase14")'
assert s.count(a) == 2, s.count(a)
s = s.replace(a, b)
s = s.replace('import csv, json, re', 'import csv, json, os, re')
s = s.replace('fams = sorted(idx)', 'os.makedirs(r"C:\\Users\\absamad\\cs\\phase14", exist_ok=True)\nfams = sorted(idx)')
p.write_text(s, encoding="utf-8")
print("patched")
