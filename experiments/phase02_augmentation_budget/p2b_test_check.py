import hashlib, json
from pathlib import Path
import yaml

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
CK = {"B": ("ckpt_ep230_step18170.pt", 18170), "C": ("ckpt_ep29_step18270.pt", 18270),
      "D": ("ckpt_ep29_step18270.pt", 18270)}
DATA = {"B": "data/tomato_village_noaug/dataset.yaml", "C": "data/tomato_village_dup8/dataset.yaml",
        "D": "data/tomato_village/dataset.yaml"}

out = {"checkpoints": {}, "test_sets": {}}
for arm, (fn, step) in CK.items():
    for s in range(5):
        p = REPO / "runs_p2b" / arm / f"seed{s}" / "weights" / fn
        out["checkpoints"][f"{arm}{s}"] = {"exists": p.exists(),
                                           "mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None}

for arm, rel in DATA.items():
    y = yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))
    root = (REPO / rel).parent
    tp = y.get("test") or y.get("val")
    d = (root / tp) if not Path(tp).is_absolute() else Path(tp)
    d = d if d.is_dir() else d.parent
    files = sorted(f.name for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out["test_sets"][arm] = {
        "declared_test": str(tp), "resolved": str(d), "n_images": len(files),
        "name_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest()[:16],
        "content_sha256": hashlib.sha256(b"".join(
            hashlib.md5((d / f).read_bytes()).digest() for f in files)).hexdigest()[:16],
        "nc": y.get("nc"),
    }
h = {v["content_sha256"] for v in out["test_sets"].values()}
n = {v["name_sha256"] for v in out["test_sets"].values()}
out["all_arms_identical_test_set"] = (len(h) == 1 and len(n) == 1)
print(json.dumps(out, indent=1))
