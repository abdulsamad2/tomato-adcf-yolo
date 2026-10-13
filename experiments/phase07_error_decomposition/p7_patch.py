from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p7_cand.py")
s = p.read_text(encoding="utf-8")

# 1. set reg_max through the Detect YAML args: Detect(nc, reg_max, end2end, ch)
a1 = '    if name == "S32HALF":'
b1 = ('    if name == "R8":\n'
      '        # Detect.__init__(nc, reg_max, end2end, ch); parse_model appends ch last,\n'
      '        # so the YAML args list must carry all three leading positionals.\n'
      '        c["head"][-1][3] = ["nc", 8, False]\n'
      '    if name == "S32HALF":')
assert a1 in s
s = s.replace(a1, b1, 1)

# 2. drop the class-attribute patching, which does not exist in this version
a2 = ('    from ultralytics.nn.modules import Detect\n'
      '    old = Detect.reg_max\n'
      '    if name == "R8":\n'
      '        Detect.reg_max = 8\n'
      '    try:\n'
      '        p = OUT / f"cand_{name}.yaml"\n'
      '        p.write_text(yaml.safe_dump(cfg_for(name), sort_keys=False, default_flow_style=None), encoding="utf-8")\n'
      '        m = YOLO(str(p))\n'
      '    finally:\n'
      '        Detect.reg_max = old\n'
      '    return m, str(p)')
b2 = ('    p = OUT / f"cand_{name}.yaml"\n'
      '    p.write_text(yaml.safe_dump(cfg_for(name), sort_keys=False, default_flow_style=None), encoding="utf-8")\n'
      '    m = YOLO(str(p))\n'
      '    return m, str(p)')
assert a2 in s, "build anchor missing"
s = s.replace(a2, b2, 1)
p.write_text(s, encoding="utf-8")
print("patched")
