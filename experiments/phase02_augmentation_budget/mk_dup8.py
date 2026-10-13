"""Arm B of the controlled copy ablation: 8 BYTE-IDENTICAL duplicates per training photo.

Same split, same file count and same steps/epoch as the shipped 8x dataset, but the offline
copies carry no augmentation diversity. A-vs-B therefore isolates diversity at matched budget;
the online pipeline (mosaic/flip/HSV/scale) is identical in both arms.

Verification: every _augK label must equal the original label byte-for-byte, and every _augK
image must be a hardlink to the same inode as the original.
"""
import os, re, shutil, hashlib
from pathlib import Path

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
SRC = REPO/"data"/"tomato_village"
OUT = REPO/"data"/"tomato_village_dup8"
AUG = re.compile(r"_aug(\d+)$")

for kind in ("images", "labels"):
    for split in ("train", "val", "test"):
        (OUT/kind/split).mkdir(parents=True, exist_ok=True)

# val/test: originals only, as in every other arm
for kind in ("images", "labels"):
    for split in ("val", "test"):
        for f in (SRC/kind/split).iterdir():
            d = OUT/kind/split/f.name
            if not d.exists():
                try: os.link(f, d)
                except OSError: shutil.copy2(f, d)

# train: original + 7 identical links named _aug1.._aug7
originals = [f for f in (SRC/"images"/"train").iterdir() if not AUG.search(f.stem)]
made = 0
for img in originals:
    lab = SRC/"labels"/"train"/(img.stem + ".txt")
    for k in range(8):
        stem = img.stem if k == 0 else f"{img.stem}_aug{k}"
        for src, kind, ext in ((img, "images", img.suffix), (lab, "labels", ".txt")):
            if not src.exists():
                continue
            d = OUT/kind/"train"/(stem + ext)
            if not d.exists():
                try: os.link(src, d)
                except OSError: shutil.copy2(src, d)
                made += 1

y = (SRC/"dataset.yaml").read_text(encoding="utf-8")
(OUT/"dataset.yaml").write_text(y.replace("tomato_village", "tomato_village_dup8"), encoding="utf-8")

n_tr = len(list((OUT/"images"/"train").iterdir()))
n_sh = len(list((SRC/"images"/"train").iterdir()))
print(f"originals={len(originals)} train_files={n_tr} (shipped 8x has {n_sh}) links_made={made}")

# verify identity: sample 40 photos, confirm each _augK is the same bytes as the original
bad = 0
for img in originals[:40]:
    h0 = hashlib.md5((OUT/"images"/"train"/img.name).read_bytes()).hexdigest()
    for k in range(1, 8):
        p = OUT/"images"/"train"/f"{img.stem}_aug{k}{img.suffix}"
        if not p.exists() or hashlib.md5(p.read_bytes()).hexdigest() != h0:
            bad += 1
print(f"identity_check: {bad} mismatches in 40 photos x 7 copies (expect 0)")
assert bad == 0 and n_tr == n_sh, (bad, n_tr, n_sh)
print("DUP8_OK")
