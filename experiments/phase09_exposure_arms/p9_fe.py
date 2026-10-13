import json, re
from pathlib import Path
REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = Path(r"C:\Users\absamad\cs\phase9")
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
fam = lambda s: norm(AUG.sub("", s))
def fams(d):
    return {fam(p.stem) for p in d.iterdir() if p.suffix == ".txt"} if d.exists() else set()

base = fams(REPO/"data"/"tomato_village"/"labels"/"train")
out = {"reference": "data/tomato_village/labels/train", "reference_families": len(base), "datasets": {}}
for name in ("tomato_village", "tomato_village_noaug", "tomato_village_dup8"):
    d = REPO/"data"/name/"labels"
    e = {}
    for sp in ("train", "val", "test"):
        f = fams(d/sp)
        n_files = len(list((d/sp).iterdir())) if (d/sp).exists() else 0
        e[sp] = {"families": len(f), "label_files": n_files,
                 "files_per_family": round(n_files/len(f), 2) if f else None}
        if sp == "train":
            e["train_identical_to_reference"] = (f == base)
            e["train_missing_vs_reference"] = len(base - f)
            e["train_extra_vs_reference"] = len(f - base)
    out["datasets"][name] = e

# CV folds: each fold has its own train families -> its own exposure partition
cvd = REPO/"data"/"tomato_village_cv"
out["cv_folds"] = {}
if cvd.exists():
    for fd in sorted(x for x in cvd.iterdir() if x.is_dir()):
        tr = fams(fd/"labels"/"train"); te = fams(fd/"labels"/"test") or fams(fd/"labels"/"val")
        out["cv_folds"][fd.name] = {"train_families": len(tr), "eval_families": len(te),
                                    "train_subset_of_full_1796": len(tr - (base | set())) if False else None,
                                    "overlap_train_eval": len(tr & te)}
json.dump(out, open(OUT/"dataset_family_equality.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("FEDONE")
