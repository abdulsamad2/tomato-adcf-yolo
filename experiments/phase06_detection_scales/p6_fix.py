from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p6_run.py")
s = p.read_text(encoding="utf-8")
a = "             val=True, plots=False, multi_scale=False, rect=False, pretrained=False,"
b = ("             val=True, plots=False, multi_scale=False, rect=False,\n"
     "             # NOTE: pretrained must NOT be set False here. Ultralytics does\n"
     "             #   weights = None if pretrained is False else self.model\n"
     "             # so pretrained=False discards the remapped checkpoint and rebuilds from YAML.\n"
     "             # Leaving the Phase 4 config value seeds the trainer from our remapped weights.")
assert a in s, "anchor missing"
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched")
