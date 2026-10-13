import csv, re
from pathlib import Path
REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RAW = REPO/"data"/"raw"/"Tomato-Village"/"Variant-c(Object Detection)"
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
gid = lambda p: norm(AUG.sub("", Path(p).stem))

for split in ("train", "val"):
    d = RAW/split
    print(f"--- official {split} ---")
    for sub in sorted(x for x in d.iterdir() if x.is_dir()):
        n = sum(1 for _ in sub.iterdir())
        print(f"   {sub.name}/ = {n} files")
    loose = [x.name for x in d.iterdir() if x.is_file()]
    if loose:
        print(f"   loose files: {len(loose)} e.g. {loose[:2]}")

def ids_of(split):
    d = RAW/split
    img = d/"images" if (d/"images").exists() else d
    return {gid(f.name) for f in img.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")}

off = {s: ids_of(s) for s in ("train", "val")}
man = list(csv.DictReader((REPO/"data"/"tomato_village"/"split_manifest.csv").open(encoding="utf-8")))
clean = {}
for r in man:
    clean.setdefault(r["split"], set()).add(norm(r["original_id"]))

print("\n--- photo counts ---")
print("official train photos:", len(off["train"]), "| official val photos:", len(off["val"]))
print("official train n val (leak within official split):", len(off["train"] & off["val"]))
for cs in ("train", "val", "test"):
    print(f"clean {cs}: {len(clean[cs])}")
print("\n--- contamination of the official val w.r.t. our clean-split model ---")
inter = off["val"] & clean["train"]
print(f"official-val photos that are in our clean TRAIN: {len(inter)} of {len(off['val'])} "
      f"({100*len(inter)/max(1,len(off['val'])):.1f}%)")
print(f"official-val photos in our clean val : {len(off['val'] & clean['val'])}")
print(f"official-val photos in our clean test: {len(off['val'] & clean['test'])}")
print("\n--- contamination of our clean TEST w.r.t. a model trained on official train ---")
inter2 = clean["test"] & off["train"]
print(f"clean-test photos inside official train: {len(inter2)} of {len(clean['test'])} "
      f"({100*len(inter2)/len(clean['test']):.1f}%)")
print(f"clean-test photos usable as held-out for an official-trained model: {len(clean['test'] - off['train'])}")
