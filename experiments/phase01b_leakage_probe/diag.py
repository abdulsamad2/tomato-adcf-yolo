import csv, re
from pathlib import Path
REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
CV = REPO/"data"/"tomato_village_cv"
AUG = re.compile(r"_aug\d+$")
gid = lambda p: AUG.sub("", Path(p).stem)
man = CV/"fold0"/"split_manifest.csv"
rows = list(csv.DictReader(man.open(encoding="utf-8")))
keys = {r["original_id"] for r in rows}
files = [f.name for f in (CV/"fold0"/"images"/"test").iterdir()]
ids = sorted({gid(f) for f in files})
unmapped = [i for i in ids if i not in keys]
print("manifest key samples:", sorted(keys)[:4])
print("test id samples     :", ids[:4])
print("unmapped count:", len(unmapped), "of", len(ids))
print("unmapped samples:", unmapped[:6])
print("file samples:", sorted(files)[:4])
# do unmapped ids match a manifest key after a looser transform?
import difflib
for u in unmapped[:3]:
    close = difflib.get_close_matches(u, keys, n=2, cutoff=0.6)
    print(f"  {u!r} -> closest manifest keys {close}")
