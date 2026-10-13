"""Leakage audit that reads images straight out of a git repo.

Needed because PlantDoc contains a filename with '?' in it, which Windows cannot
create, so `git checkout` fails. Blobs are streamed via `git cat-file --batch`
and hashed in memory; nothing is written to disk.
"""
import io, json, re, subprocess, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

GIT = r"C:\Users\absamad\cs\mingit\cmd\git.exe"
EXT = (".jpg", ".jpeg", ".png", ".bmp")
AUG = re.compile(r"[_-](aug|rot|flip|bright|dark|noise|blur|crop|zoom|shear|mirror)\w*\d*$", re.I)
SPLIT_WORDS = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}

def dhash_bytes(b, size=8):
    im = Image.open(io.BytesIO(b)).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

def list_paths(repo, ref="HEAD"):
    out = subprocess.run([GIT, "-C", repo, "ls-tree", "-r", "--name-only", ref],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [p for p in out.stdout.splitlines() if p.lower().endswith(EXT)]

def split_of(path):
    for part in path.split("/"):
        w = SPLIT_WORDS.get(part.lower())
        if w:
            return w
    return None

def audit_git(repo, name, ref="HEAD"):
    paths = list_paths(repo, ref)
    labelled = [(p, split_of(p)) for p in paths]
    labelled = [(p, s) for p, s in labelled if s]
    if not labelled:
        return {"dataset": name, "error": "no split-labelled images", "images_in_repo": len(paths)}

    proc = subprocess.Popen([GIT, "-C", repo, "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    rows = []
    for p, s in labelled:
        proc.stdin.write(f"{ref}:{p}\n".encode("utf-8"))
        proc.stdin.flush()
        header = proc.stdout.readline().decode("utf-8", "replace").split()
        if len(header) < 3:
            continue
        n = int(header[2])
        blob = proc.stdout.read(n)
        proc.stdout.read(1)
        try:
            rows.append((p, s, dhash_bytes(blob)))
        except Exception:
            pass
    proc.stdin.close()
    proc.wait()

    per_split = defaultdict(int)
    for _, s, _ in rows:
        per_split[s] += 1
    byname = defaultdict(set)
    for p, s, _ in rows:
        byname[AUG.sub("", Path(p).stem).lower()].add(s)

    bits = np.array([h for _, _, h in rows], dtype=np.uint8)
    splits = [s for _, s, _ in rows]
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(rows)):
        d = (bits[i+1:] != bits[i]).sum(axis=1)
        for off in np.nonzero(d <= 6)[0]:
            a, b = find(i), find(i + 1 + int(off))
            if a != b:
                parent[b] = a
    clusters, members = defaultdict(set), defaultdict(list)
    for i, s in enumerate(splits):
        clusters[find(i)].add(s); members[find(i)].append(i)
    contaminated = defaultdict(int)
    for cid, ms in members.items():
        if "train" in clusters[cid] and len(clusters[cid]) > 1:
            for i in ms:
                if splits[i] != "train":
                    contaminated[splits[i]] += 1
    return {
        "dataset": name,
        "images_scanned": len(rows),
        "images_per_split": dict(per_split),
        "distinct_name_groups": len(byname),
        "name_groups_spanning_splits": sum(1 for v in byname.values() if len(v) > 1),
        "distinct_hash_clusters": len(clusters),
        "hash_clusters_spanning_splits": sum(1 for v in clusters.values() if len(v) > 1),
        "contaminated_eval_images": dict(contaminated),
        "contamination_pct": {k: round(100*v/per_split[k], 2) for k, v in contaminated.items()},
    }

if __name__ == "__main__":
    res = audit_git(sys.argv[1], sys.argv[2])
    print(json.dumps(res, indent=1))
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
