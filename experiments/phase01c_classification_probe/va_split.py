import csv, json, re, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

VA = Path(r"C:\Users\absamad\cs\audit\Tomato-Village\Variant-a(Multiclass Classification)")
OUT = Path(r"C:\Users\absamad\cs\audit\va_split")
OUT.mkdir(parents=True, exist_ok=True)
CROP = re.compile(r"_\d+$")

rows = []
for sp in ("train", "val", "test"):
    for cd in sorted((VA/sp).iterdir()):
        if cd.is_dir():
            for f in sorted(cd.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append({"path": str(f), "cls": cd.name, "stem": f.stem,
                                 "src": CROP.sub("", f.stem), "official": sp})

groups = defaultdict(list)
for r in rows:
    groups[r["src"]].append(r)

def dhash(p, size=8):
    im = Image.open(p).convert("L").resize((size+1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

srcs = sorted(groups)
bits = np.array([dhash(groups[s][0]["path"]) for s in srcs], dtype=np.uint8)
parent = list(range(len(srcs)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i in range(len(srcs)):
    d = (bits[i+1:] != bits[i]).sum(axis=1)
    for off in np.nonzero(d <= 6)[0]:
        a, b = find(i), find(i + 1 + int(off))
        if a != b:
            parent[b] = a
cluster = {s: find(i) for i, s in enumerate(srcs)}
merged = defaultdict(list)
for s in srcs:
    merged[cluster[s]].extend(groups[s])
print(f"images={len(rows)} source_photos={len(groups)} groups_after_merge={len(merged)}")

# stratified group split 70/15/15 by the group's (single) class
rng = random.Random(0)
by_cls = defaultdict(list)
for gid, items in merged.items():
    by_cls[items[0]["cls"]].append(gid)
assign = {}
for cls, gids in sorted(by_cls.items()):
    rng.shuffle(gids)
    sizes = {g: len(merged[g]) for g in gids}
    total = sum(sizes.values())
    want = {"train": 0.70*total, "val": 0.15*total, "test": 0.15*total}
    got = {"train": 0, "val": 0, "test": 0}
    for g in sorted(gids, key=lambda g: -sizes[g]):
        pick = max(("train", "val", "test"), key=lambda s: (want[s]-got[s]) / max(want[s], 1))
        assign[g] = pick
        got[pick] += sizes[g]

with (OUT/"manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["path", "cls", "stem", "src", "group", "official", "clean"])
    w.writeheader()
    for gid, items in merged.items():
        for r in items:
            w.writerow({**{k: r[k] for k in ("path", "cls", "stem", "src", "official")},
                        "group": gid, "clean": assign[gid]})

counts = defaultdict(lambda: defaultdict(int))
for gid, items in merged.items():
    for r in items:
        counts[assign[gid]][r["cls"]] += 1
summary = {s: {"total": sum(c.values()), **dict(sorted(c.items()))} for s, c in sorted(counts.items())}
print(json.dumps(summary, indent=1))

# verify no group spans splits, and quantify the official split for comparison
spans = defaultdict(set)
for gid, items in merged.items():
    for r in items:
        spans[gid].add(assign[gid])
print("clean groups spanning splits:", sum(1 for v in spans.values() if len(v) > 1))
tr = {r["src"] for r in rows if r["official"] == "train"}
for sp in ("val", "test"):
    sub = [r for r in rows if r["official"] == sp]
    print(f"official {sp}: {sum(1 for r in sub if r['src'] in tr)}/{len(sub)} share a source photo with official train")
json.dump(summary, open(OUT/"split_summary.json", "w"), indent=1)
