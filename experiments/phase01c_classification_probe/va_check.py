import re, json
from pathlib import Path
from collections import defaultdict, Counter

VA = Path(r"C:\Users\absamad\cs\audit\Tomato-Village\Variant-a(Multiclass Classification)")
CROP = re.compile(r"_\d+$")
rows = []
for sp in ("train", "val", "test"):
    for cd in (VA/sp).iterdir():
        if cd.is_dir():
            for f in cd.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append((sp, cd.name, f.stem, CROP.sub("", f.stem)))

print("images:", len(rows))
stems = defaultdict(set)
stem_cls = defaultdict(set)
for sp, cls, st, src in rows:
    stems[src].add(sp)
    stem_cls[src].add(cls)
print("distinct source photos:", len(stems))
print("source photos spanning >1 split:", sum(1 for v in stems.values() if len(v) > 1))
print("crops per photo:", dict(Counter(sum(1 for r in rows if r[3] == s) for s in stems).most_common(6)))

train_src = {src for sp, _, _, src in rows if sp == "train"}
for sp in ("val", "test"):
    n = sum(1 for s, _, _, src in rows if s == sp)
    leaked = sum(1 for s, _, _, src in rows if s == sp and src in train_src)
    print(f"{sp}: {leaked}/{n} images share a source photo with train = {100*leaked/n:.1f}%")

multi = sum(1 for v in stem_cls.values() if len(v) > 1)
print("source photos carrying >1 class label:", multi)
json.dump({"rows": rows}, open(r"C:\Users\absamad\cs\audit\va_rows.json", "w"))
