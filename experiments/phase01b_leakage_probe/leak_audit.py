"""Generic train/test leakage audit for image datasets.

For each dataset: group images by perceptual hash (dHash, Hamming <= 6) and by
augmentation-suffix-stripped filename, then report whether any group has members
in more than one split. A group spanning splits means the test set is contaminated.
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

AUG = re.compile(r"[_-](aug|rot|flip|bright|dark|noise|blur|crop|zoom|shear|mirror)\w*\d*$", re.I)
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_WORDS = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test", "eval": "test"}

def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

def split_of(p, root):
    for part in p.relative_to(root).parts:
        w = SPLIT_WORDS.get(part.lower())
        if w:
            return w
    return None

def audit(root, name, max_images=None):
    root = Path(root)
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in EXT]
    if max_images and len(imgs) > max_images:
        imgs = imgs[:max_images]
    rows = []
    for p in imgs:
        s = split_of(p, root)
        if s is None:
            continue
        try:
            h = dhash(p)
        except Exception:
            continue
        rows.append((p, s, h))
    if not rows:
        return {"dataset": name, "error": "no split-labelled images found", "images": len(imgs)}

    per_split = defaultdict(int)
    for _, s, _ in rows:
        per_split[s] += 1

    # group 1: filename stem with augmentation suffix stripped
    byname = defaultdict(set)
    for p, s, _ in rows:
        byname[AUG.sub("", p.stem).lower()].add(s)
    name_span = sum(1 for v in byname.values() if len(v) > 1)

    # group 2: near-duplicate clusters by dHash (bucketed exact match + Hamming <= 6)
    hashes = np.array([h for _, _, h in rows], dtype=np.uint64)
    splits = [s for _, s, _ in rows]
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    bits = np.unpackbits(hashes.view(np.uint8).reshape(len(rows), 8)[:, ::-1], axis=1)
    for i in range(len(rows)):
        d = (bits[i + 1:] != bits[i]).sum(axis=1)
        for off in np.nonzero(d <= 6)[0]:
            union(i, i + 1 + int(off))
    clusters = defaultdict(set)
    cluster_members = defaultdict(list)
    for i, s in enumerate(splits):
        clusters[find(i)].add(s)
        cluster_members[find(i)].append(i)
    hash_span = sum(1 for v in clusters.values() if len(v) > 1)

    # how much of each non-train split is contaminated by a train-side twin
    contaminated = defaultdict(int)
    for cid, members in cluster_members.items():
        ss = clusters[cid]
        if "train" in ss and len(ss) > 1:
            for i in members:
                if splits[i] != "train":
                    contaminated[splits[i]] += 1

    return {
        "dataset": name,
        "images_scanned": len(rows),
        "images_per_split": dict(per_split),
        "distinct_name_groups": len(byname),
        "name_groups_spanning_splits": name_span,
        "distinct_hash_clusters": len(clusters),
        "hash_clusters_spanning_splits": hash_span,
        "contaminated_eval_images": dict(contaminated),
        "contamination_pct": {k: round(100 * v / per_split[k], 2)
                              for k, v in contaminated.items() if per_split.get(k)},
    }

if __name__ == "__main__":
    targets = json.loads(Path(sys.argv[1]).read_text())
    out = []
    for t in targets:
        if Path(t["root"]).exists():
            r = audit(t["root"], t["name"], t.get("max_images"))
        else:
            r = {"dataset": t["name"], "error": "path not found"}
        out.append(r)
        print(json.dumps(r, indent=1), flush=True)
    Path(r"C:\Users\absamad\cs\audit\leakage_audit.json").write_text(json.dumps(out, indent=1))
