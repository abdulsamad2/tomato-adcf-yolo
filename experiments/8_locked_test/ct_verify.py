"""Clean-test pre-run verification. Reads file NAMES, label text, the split manifest and file BYTES (for
sha256) only. No image is decoded, no model is loaded, no metric is computed.

Checks
  V1 268 test images and 268 test label files; stems == manifest test original_ids; no _augN copies
  V2 test photos / clusters disjoint from train and val (manifest original_id AND cluster)
  V3 on-disk train/val file families (stem without _augN) disjoint from test families
  V4 Phase 9 evaluation arms (A and B) contain no test family
  V5 prior-access audit: every earlier artefact that scored the clean test split, and confirmation that
     none of the 20 checkpoints in this plan (or their runs) ever wrote a test result
Writes clean_test/verify_report.json.
"""
import csv, hashlib, json, re
from pathlib import Path

B = Path(r"C:\Users\absamad\cs"); R = B / "tomato-adcf-yolo"; D = R / "data" / "tomato_village"; OUT = B / "clean_test"
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
fam = lambda s: norm(AUG.sub("", s))
rep, ok = {}, True

man = list(csv.DictReader(open(R / "splits" / "split_manifest.csv", encoding="utf-8")))
by = {sp: [r for r in man if r["split"] == sp] for sp in ("train", "val", "test")}
ids = {sp: {r["original_id"] for r in v} for sp, v in by.items()}
cl = {sp: {r["cluster"] for r in v} for sp, v in by.items()}
imgs = sorted(p for p in (D / "images" / "test").iterdir() if p.is_file())
labs = sorted(p for p in (D / "labels" / "test").iterdir() if p.suffix == ".txt")
stems = {p.stem for p in imgs}
v1 = {"n_images": len(imgs), "n_labels": len(labs), "n_manifest_test": len(by["test"]),
      "stems_equal_manifest": {norm(s) for s in stems} == {norm(s) for s in ids["test"]},
      "stems_equal_manifest_note": "compared after the Phase 9 name normalisation ('Jaipur_Pots (102)' == 'Jaipur_Pots_102')",
      "label_stems_equal_images": {p.stem for p in labs} == stems,
      "aug_copies_in_test": sum(1 for s in stems if AUG.search(s))}
v1["pass"] = v1["n_images"] == v1["n_labels"] == v1["n_manifest_test"] == 268 and v1["stems_equal_manifest"] \
    and v1["label_stems_equal_images"] and v1["aug_copies_in_test"] == 0
rep["V1_counts"] = v1
v2 = {"test_clusters": len(cl["test"]), "id_overlap_train": len(ids["test"] & ids["train"]),
      "id_overlap_val": len(ids["test"] & ids["val"]), "cluster_overlap_train": len(cl["test"] & cl["train"]),
      "cluster_overlap_val": len(cl["test"] & cl["val"])}
v2["pass"] = v2["id_overlap_train"] == v2["id_overlap_val"] == v2["cluster_overlap_train"] == v2["cluster_overlap_val"] == 0
rep["V2_manifest_disjoint"] = v2
disk = {sp: {fam(p.stem) for p in (D / "labels" / sp).iterdir() if p.suffix == ".txt"} for sp in ("train", "val")}
tf = {fam(s) for s in stems}
tcf = {fam(r["original_id"]) for r in by["test"] if r["original_id"] != r["cluster"]} | {fam(r["cluster"]) for r in by["test"]}
v3 = {"train_file_families": len(disk["train"]), "val_file_families": len(disk["val"]),
      "overlap_train": len((tf | tcf) & disk["train"]), "overlap_val": len((tf | tcf) & disk["val"])}
v3["pass"] = v3["overlap_train"] == v3["overlap_val"] == 0
rep["V3_disk_disjoint"] = v3
arms = json.loads((B / "phase9" / "arm_image_ids.json").read_text())
armfams = {fam(f) for f in arms["family_of_image"].values()}
rep["V4_phase9_arms"] = {"arm_families": len(armfams), "overlap_test": len((tf | tcf) & armfams)}
rep["V4_phase9_arms"]["pass"] = rep["V4_phase9_arms"]["overlap_test"] == 0

prior = {}
for p in R.glob("runs*/**/results_test.json"):
    k = p.parts[len(R.parts)] + "/" + p.parts[len(R.parts) + 1]
    prior[k] = prior.get(k, 0) + 1
plan_roots = ["runs_p17", "runs_p15/w075", "runs_p15/w100", "runs_p4/r512"]
touched = {r: sorted(str(p.relative_to(R)) for p in (R / r).rglob("*") if p.is_file() and "test" in p.name.lower())
           for r in plan_roots}
rep["V5_prior_access"] = {
  "statement": ("The clean test split HAS been scored before, for models outside this plan: the original adcf-run "
                "pipeline (results_test.json per run, results/tables_test.md, Sep 21 - Oct 3) and the Phase 2B single "
                "test evaluation (results/p2b_test_eval.json, Oct 5, 15 YOLO11s checkpoints). capacity_audit.py and "
                "res_audit.py later read those stored numbers. No model in this plan, and no protocol choice in "
                "Phases 4-17, was scored on or selected with the test split."),
  "earlier_runs_with_test_results": prior,
  "other_test_artefacts": [str(p.relative_to(R)) for p in [R / "results" / "tables_test.md", R / "results" / "p2b_test_eval.json",
                                                         R / "results" / "all_seeds_test.csv"] if p.exists()],
  "test_named_files_in_plan_runs": touched,
  "pass_plan_models_never_scored_on_test": all(len(v) == 0 for v in touched.values())}

h = hashlib.sha256()
for p in imgs + labs:
    h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
rep["test_set_sha256"] = h.hexdigest()
rep["n_boxes"] = sum(1 for p in labs for ln in p.read_text().splitlines() if len(ln.split()) >= 5)
rep["n_empty_label_images"] = sum(1 for p in labs if not any(len(l.split()) >= 5 for l in p.read_text().splitlines()))
rep["all_pass"] = all(rep[k]["pass"] for k in ("V1_counts", "V2_manifest_disjoint", "V3_disk_disjoint", "V4_phase9_arms")) \
    and rep["V5_prior_access"]["pass_plan_models_never_scored_on_test"]
json.dump(rep, open(OUT / "verify_report.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) or len(json.dumps(v)) < 400 else {kk: vv for kk, vv in v.items() if kk != "statement"})
                  for k, v in rep.items()}, indent=1))
print("VERIFYDONE all_pass=%s" % rep["all_pass"])
