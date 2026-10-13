import csv, json, re
from pathlib import Path
REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
CV = REPO/"data"/"tomato_village_cv"
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
gid = lambda p: norm(AUG.sub("", Path(p).stem))

man = CV/"fold0"/"split_manifest.csv"
rows = list(csv.DictReader(man.open(encoding="utf-8")))
cluster = {norm(r["original_id"]): norm(r["cluster"]) for r in rows}
out = {"manifest": str(man), "rows": len(rows), "distinct_clusters": len(set(cluster.values())), "folds": {}}
test_sets, all_ids = {}, set()
for fold in sorted(d.name for d in CV.iterdir() if d.is_dir()):
    ids, naug = {}, {}
    for split in ("train", "val", "test"):
        d = CV/fold/"images"/split
        fs = [f.name for f in d.iterdir() if f.is_file()] if d.exists() else []
        ids[split] = {gid(f) for f in fs}
        naug[split] = sum(1 for f in fs if AUG.search(Path(f).stem))
    unmapped = sorted({i for s in ids for i in ids[s] if i not in cluster})
    cl = {s: {cluster[i] for i in ids[s] if i in cluster} for s in ids}
    out["folds"][fold] = {
        "photos": {s: len(ids[s]) for s in ids}, "aug_files": naug,
        "photo_overlap_train_test": len(ids["train"] & ids["test"]),
        "photo_overlap_train_val": len(ids["train"] & ids["val"]),
        "cluster_overlap_train_test": len(cl["train"] & cl["test"]),
        "cluster_overlap_train_val": len(cl["train"] & cl["val"]),
        "cluster_overlap_val_test": len(cl["val"] & cl["test"]),
        "unmapped_ids": len(unmapped), "unmapped_samples": unmapped[:3],
    }
    test_sets[fold] = ids["test"]; all_ids |= ids["train"] | ids["val"] | ids["test"]
names = sorted(test_sets)
out["test_pairwise_overlap"] = {f"{a}|{b}": len(test_sets[a] & test_sets[b])
                                for i, a in enumerate(names) for b in names[i+1:]}
out["union_of_test_sets"] = len(set().union(*test_sets.values()))
out["all_photos_seen"] = len(all_ids)
out["verdict"] = ("GROUPED - folds respect photo groups and near-duplicate clusters"
                  if all(f["photo_overlap_train_test"] == 0 and f["cluster_overlap_train_test"] == 0
                         and f["aug_files"]["test"] == 0 for f in out["folds"].values())
                  else "CONTAMINATED - see per-fold overlaps")
(REPO/"results"/"cv_fold_audit.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1))
for f, v in out["folds"].items():
    print(f, "photos", v["photos"], "aug_in_test", v["aug_files"]["test"],
          "| photo_ovl_tr_te", v["photo_overlap_train_test"],
          "| clus_ovl_tr_te", v["cluster_overlap_train_test"],
          "| clus_ovl_tr_val", v["cluster_overlap_train_val"],
          "| unmapped", v["unmapped_ids"])
