from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p7_cand.py")
s = p.read_text(encoding="utf-8")
a = ('    if name == "R8":\n'
     '        # Detect.__init__(nc, reg_max, end2end, ch); parse_model appends ch last,\n'
     '        # so the YAML args list must carry all three leading positionals.\n'
     '        c["head"][-1][3] = ["nc", 8, False]\n')
b = ('    if name == "R8":\n'
     '        c["reg_max"] = 8   # parse_model reads reg_max as a top-level YAML key\n')
assert a in s, "anchor missing"
s = s.replace(a, b, 1)
p.write_text(s, encoding="utf-8")
print("patched")
