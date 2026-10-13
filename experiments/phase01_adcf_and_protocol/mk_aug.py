"""Derive 2x and 4x augmentation-copy datasets from the prepared 8x split.

Keeps the identical photo split (same manifest), so the only variable is how many
offline copies of each training photo the model sees. val/test are untouched
(originals only), and files are hardlinked rather than copied.
"""
import os, re, shutil, sys
from pathlib import Path

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
SRC = REPO/"data"/"tomato_village"
AUG = re.compile(r"_aug(\d+)$")

def build(keep, out_name):
    out = REPO/"data"/out_name
    if (out/"dataset.yaml").exists():
        print(f"{out_name}: exists, skipping"); return
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (out/kind/split).mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            for f in (SRC/kind/split).iterdir():
                m = AUG.search(f.stem)
                if split == "train" and m and int(m.group(1)) > keep:
                    dropped += 1 if kind == "images" else 0
                    continue
                dst = out/kind/split/f.name
                if not dst.exists():
                    try:
                        os.link(f, dst)
                    except OSError:
                        shutil.copy2(f, dst)
                kept += 1 if kind == "images" else 0
    y = (SRC/"dataset.yaml").read_text(encoding="utf-8")
    (out/"dataset.yaml").write_text(y.replace("tomato_village", out_name), encoding="utf-8")
    n_tr = len(list((out/"images"/"train").iterdir()))
    print(f"{out_name}: train_images={n_tr} (dropped {dropped} aug copies)")

# keep=1 -> original + _aug1 = 2 copies per photo; keep=3 -> original + _aug1..3 = 4 copies
build(1, "tomato_village_aug2")
build(3, "tomato_village_aug4")
