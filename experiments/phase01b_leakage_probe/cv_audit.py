import csv, json, re
from pathlib import Path

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
CV = REPO/"data"/"tomato_village_cv"
AUG = re.compile(r"_aug\d+$")
gid = lambda p: AUG.sub("", Path(p).stem)

cands = list(CV.glob("**/*manifest*.csv")) + [REPO/"data"/"tomato_village"/"split_manifest.csv"]
man = next(p for p in cands if p.exists())
rows = list(csv.DictReader(man.open(encoding="utf-8")))
cluster = {r["original_id"]: r["cluster"] for r in rows}
out = {"manifest_used": str(man), "manifest_rows": len(rows),
       "distinct_clusters": len(set(cluster.values())), "folds": {}}

test_sets, all_ids = {}, set()
for fold in sorted(d.name for d in CV.iterdir() if d.is_dir()):
    ids, nf, naug = {}, {}, {}
    for split in ("train", "val", "test"):
        d = CV/fold/"images"/split
        files = [f.name for f in d.iterdir() if f.is_file()] if d.exists() else []
        ids[split] = {gid(f) for f in files}
        nf[split] = len(files)
        naug[split] = sum(1 for f in files if AUG.search(Path(f).stem))
    cl = {s: {cluster.get(i, "UNMAPPED") for i in ids[s]} for s in ids}
    out["folds"][fold] = {
        "files": nf, "photos": {s: len(ids[s]) for s in ids}, "aug_files": naug,
        "photo_overlap_train_test": len(ids["train"] & ids["test"]),
        "photo_overlap_train_val": len(ids["train"] & ids["val"]),
        "photo_overlap_val_test": len(ids["val"] & ids["test"]),
        "cluster_overlap_train_test": len(cl["train"] & cl["test"]),
        "cluster_overlap_train_val": len(cl["train"] & cl["val"]),
        "unmapped_ids": sum(1 for s in ids for i in ids[s] if i not in cluster),
    }
    test_sets[fold] = ids["test"]
    all_ids |= ids["train"] | ids["val"] | ids["test"]

names = sorted(test_sets)
out["test_pairwise_overlap"] = {f"{a}|{b}": len(test_sets[a] & test_sets[b])
                                for i, a in enumerate(names) for b in names[i+1:]}
union = set().union(*test_sets.values())
out["union_of_test_sets"] = len(union)
out["all_photos_seen"] = len(all_ids)
out["photos_never_tested"] = len(all_ids - union)
(REPO/"results"/"cv_fold_audit.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
