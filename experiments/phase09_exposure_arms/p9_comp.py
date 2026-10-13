import json, re
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RAW = REPO / "data" / "raw" / "Tomato-Village" / "Variant-c(Object Detection)"
OUT = Path(r"C:\Users\absamad\cs\phase9")
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
fam = lambda s: norm(AUG.sub("", s))
CLEAN = REPO / "data" / "tomato_village" / "labels"
def fams(d): return {fam(p.stem) for p in d.iterdir() if p.suffix == ".txt"}
CT, CV, CTE = fams(CLEAN/"train"), fams(CLEAN/"val"), fams(CLEAN/"test")
out = {}
for arm, sel in (("A_exposed", CT), ("B_independent_permitted", CV), ("B_excluded_closed_test", CTE)):
    orig = augc = 0; perfam = Counter(); cls = Counter(); boxes = 0
    for p in sorted((RAW/"val"/"yolo").iterdir()):
        if p.suffix != ".txt": continue
        f = fam(p.stem)
        if f not in sel: continue
        perfam[f] += 1
        if "_aug" in p.stem: augc += 1
        else: orig += 1
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) >= 5: cls[int(float(q[0]))] += 1; boxes += 1
    out[arm] = {"files": orig+augc, "original_files": orig, "augmented_files": augc,
                "pct_augmented": round(100*augc/(orig+augc), 1) if (orig+augc) else None,
                "families": len(perfam), "files_per_family_mean": round(sum(perfam.values())/len(perfam), 2) if perfam else None,
                "files_per_family_dist": dict(sorted(Counter(perfam.values()).items())),
                "boxes": boxes, "boxes_per_family": round(boxes/len(perfam), 2) if perfam else None,
                "class_counts": dict(sorted(cls.items())),
                "class_pct": {k: round(100*v/boxes, 2) for k, v in sorted(cls.items())} if boxes else None}
json.dump(out, open(OUT/"arm_composition.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k}: {v['files']} files ({v['pct_augmented']}% augmented), {v['families']} families, "
          f"{v['boxes']} boxes, {v['boxes_per_family']}/family, files/family {v['files_per_family_dist']}")
    print(f"   class %: {v['class_pct']}")
print("COMPDONE")
