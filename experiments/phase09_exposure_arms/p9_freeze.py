"""Phase 9 step 1: freeze + hash the two evaluation arms BEFORE any model is loaded.

Integrity gates (ABORT on any failure):
  G1 A and B disjoint by source family
  G2 neither arm contains a clean-TEST family
  G3 every B family is ABSENT from clean training
  G4 every A family is PRESENT in clean training
  G5 counts match the accepted Phase 9 audit exactly
  G6 augmentation composition matches the Phase 9 audit
  G7 released class ids are identical to the prepared-split class ids for shared families
      (guards against a class remap in tv-prepare that would silently misalign the metric)

No model is imported or loaded anywhere in this script.
"""
import hashlib, json, re, os, shutil, sys
from collections import Counter
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
RAW = REPO / "data" / "raw" / "Tomato-Village" / "Variant-c(Object Detection)" / "val"
CLEAN = REPO / "data" / "tomato_village"
OUT = Path(r"C:\Users\absamad\cs\phase9")
ARMS = OUT / "arms"
AUG = re.compile(r"_aug\d+$")
norm = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
fam = lambda s: norm(AUG.sub("", s))

EXPECT = {"A": {"families": 1047, "files": 2028, "boxes": 23225},
          "B": {"families": 213,  "files": 411,  "boxes": 4683}}
EXPECT_AUGPCT = {"A": 87.0, "B": 87.1}

def die(msg):
    print("ABORT:", msg); sys.exit(2)

def fams_of(d):
    return {fam(p.stem) for p in d.iterdir() if p.suffix == ".txt"}

CT = fams_of(CLEAN / "labels" / "train")
CV = fams_of(CLEAN / "labels" / "val")
CTE = fams_of(CLEAN / "labels" / "test")        # identifiers ONLY, for the exclusion guard
print(f"clean families: train={len(CT)} val={len(CV)} test={len(CTE)}")

# ---- assign every released val label file to an arm ----
rows = {"A": [], "B": []}
excluded_test = []
boxes = {"A": 0, "B": 0}
clsids = {"A": Counter(), "B": Counter()}
for p in sorted((RAW / "yolo").iterdir()):
    if p.suffix != ".txt":
        continue
    f = fam(p.stem)
    arm = "A" if f in CT else ("B" if f in CV else None)
    if f in CTE:
        excluded_test.append(p.name); continue
    if arm is None:
        continue
    n = 0; ids = []
    for line in p.read_text().splitlines():
        q = line.split()
        if len(q) >= 5:
            n += 1; ids.append(int(float(q[0])))
    rows[arm].append({"family": f, "label": p.name, "stem": p.stem,
                      "augmented": int("_aug" in p.stem), "boxes": n})
    boxes[arm] += n
    clsids[arm].update(ids)

# ---- gates ----
FA = {r["family"] for r in rows["A"]}
FB = {r["family"] for r in rows["B"]}
if FA & FB: die(f"G1 A/B share {len(FA & FB)} families")
if (FA | FB) & CTE: die(f"G2 arms contain {len((FA|FB)&CTE)} clean-TEST families")
if FB & CT: die(f"G3 {len(FB & CT)} B families are present in clean training")
if FA - CT: die(f"G4 {len(FA - CT)} A families are absent from clean training")
for a in ("A", "B"):
    got = {"families": len({r['family'] for r in rows[a]}), "files": len(rows[a]), "boxes": boxes[a]}
    if got != EXPECT[a]: die(f"G5 arm {a} counts {got} != audit {EXPECT[a]}")
    pct = round(100 * sum(r["augmented"] for r in rows[a]) / len(rows[a]), 1)
    if abs(pct - EXPECT_AUGPCT[a]) > 0.15: die(f"G6 arm {a} augmented {pct}% != {EXPECT_AUGPCT[a]}%")
    print(f"gate pass arm {a}: {got}, augmented {pct}%")

# ---- G7 class-id alignment: released vs prepared labels for the SAME families ----
prep_tr = {}
for p in (CLEAN / "labels" / "train").iterdir():
    if p.suffix == ".txt":
        prep_tr.setdefault(fam(p.stem), p)
shared = sorted(FA & set(prep_tr))[:400]
rel_ids, pre_ids = Counter(), Counter()
relmap = {}
for r in rows["A"]:
    relmap.setdefault(r["family"], r["label"])
for f in shared:
    for line in (RAW / "yolo" / relmap[f]).read_text().splitlines():
        q = line.split()
        if len(q) >= 5: rel_ids[int(float(q[0]))] += 1
    for line in prep_tr[f].read_text().splitlines():
        q = line.split()
        if len(q) >= 5: pre_ids[int(float(q[0]))] += 1
dy = yaml.safe_load((CLEAN / "dataset.yaml").read_text(encoding="utf-8"))
names = dy.get("names")
if set(rel_ids) != set(pre_ids):
    die(f"G7 class-id sets differ: released {sorted(rel_ids)} vs prepared {sorted(pre_ids)}")
print(f"gate pass G7: class ids identical {sorted(rel_ids)} over {len(shared)} shared families")
print(f"  released id histogram  {dict(sorted(rel_ids.items()))}")
print(f"  prepared id histogram  {dict(sorted(pre_ids.items()))}")
print(f"  dataset.yaml nc={dy.get('nc')} names={names}")

# ---- write immutable manifests + hashes ----
ARMS.mkdir(parents=True, exist_ok=True)
hashes = {}
for a in ("A", "B"):
    recs = sorted(rows[a], key=lambda r: (r["family"], r["label"]))
    lines = ["# Phase 9 evaluation arm " + a + " - IMMUTABLE",
             "# arm A = source-EXPOSED (official-val family in clean-TRAIN)",
             "# arm B = source-INDEPENDENT PERMITTED (official-val family in clean-VAL, never clean-TEST)",
             "# columns: family<TAB>label_file<TAB>augmented<TAB>boxes"]
    lines += [f"{r['family']}\t{r['label']}\t{r['augmented']}\t{r['boxes']}" for r in recs]
    txt = "\n".join(lines) + "\n"
    mp = ARMS / f"manifest_arm_{a}.tsv"
    mp.write_text(txt, encoding="utf-8", newline="\n")
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    fh = hashlib.sha256(("\n".join(r["label"] for r in recs) + "\n").encode()).hexdigest()
    gh = hashlib.sha256(("\n".join(sorted({r["family"] for r in recs})) + "\n").encode()).hexdigest()
    hashes[a] = {"manifest_sha256": h, "filelist_sha256": fh, "familylist_sha256": gh,
                 "files": len(recs), "families": len({r["family"] for r in recs}),
                 "boxes": sum(r["boxes"] for r in recs),
                 "augmented_files": sum(r["augmented"] for r in recs)}
    print(f"arm {a}: manifest sha256 {h}")

rec = {"frozen_before_model_load": True, "released_source": str(RAW),
       "clean_split": str(CLEAN), "arms": hashes,
       "excluded_clean_test_label_files": len(excluded_test),
       "gates_passed": ["G1","G2","G3","G4","G5","G6","G7"],
       "class_ids_present": sorted(rel_ids), "dataset_names": names, "nc": dy.get("nc"),
       "class_pct_A": {k: round(100*v/boxes['A'],2) for k,v in sorted(clsids['A'].items())},
       "class_pct_B": {k: round(100*v/boxes['B'],2) for k,v in sorted(clsids['B'].items())}}
json.dump(rec, open(OUT / "arm_freeze_record.json", "w"), indent=1)
print("excluded clean-test label files:", len(excluded_test))
print("FREEZEDONE")
