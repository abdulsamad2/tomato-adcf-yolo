"""Download the 20 locked-test checkpoints from the GitHub Release and verify their SHA-256.

    python tools/download_weights.py                 # all 20 -> weights/
    python tools/download_weights.py --model kd075   # only the distilled 0.75x seeds
    python tools/download_weights.py --restore       # also copy into the original run paths (runs_p15/, runs_p17/, runs_p4/)
                                                     # so experiments/8_locked_test/ct_eval.py and the stats scripts find them

Hashes come from artifacts/weights_manifest.json and are identical to artifacts/clean_test/prereg.json.
"""
import argparse, hashlib, json, shutil, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["kd075", "w075", "w100", "r512"], help="download only this model's 5 seeds")
    ap.add_argument("--out", type=Path, default=REPO / "weights")
    ap.add_argument("--restore", action="store_true", help="also place files at their original run paths")
    a = ap.parse_args()
    man = json.loads((REPO / "artifacts" / "weights_manifest.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for e in man["checkpoints"]:
        if a.model and e["model"] != a.model:
            continue
        dst = a.out / e["file"]
        if not (dst.exists() and sha256(dst) == e["sha256"]):
            url = man["url_template"].format(file=e["file"])
            print(f"downloading {e['file']} ({e['bytes'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(url, dst)
        ok = sha256(dst) == e["sha256"]
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {e['file']}  {e['description']}, seed {e['seed']}")
        if ok and a.restore:
            tgt = REPO / e["original_path"]
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, tgt)
    if bad:
        sys.exit(f"{bad} file(s) failed SHA-256 verification")
    print("all requested checkpoints verified")


if __name__ == "__main__":
    main()
