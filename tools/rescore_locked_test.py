"""Re-score the locked-test table from the released weights with the FROZEN evaluator.

Runs a copy of experiments/8_locked_test/ct_eval.py in which only its folder constants are redirected:
checkpoint paths point to weights/ (SHA-256 verified against the original prereg hashes) and outputs go to a
scratch folder. Scoring, settings, bootstrap and table code are unchanged.

    python tools/download_weights.py
    python tools/rescore_locked_test.py --out rescored/          # writes rescored/results.json + table

Needs the dataset rebuilt at data/tomato_village (tv-download + tv-prepare --manifest ...).
--split val runs the same pipeline on the validation split (smoke test that does not touch the test set).
"""
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=REPO / "weights")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    pre = json.loads((REPO / "artifacts" / "clean_test" / "prereg.json").read_text(encoding="utf-8-sig"))
    man = {(e["model"], str(e["seed"])): e for e in json.loads((REPO / "artifacts" / "weights_manifest.json").read_text())["checkpoints"]}
    for m, seeds in pre["checkpoints"].items():
        for s, c in seeds.items():
            e = man[(m, s)]
            assert e["sha256"] == c["sha256"], (m, s)
            c["path"] = str((a.weights / e["file"]).resolve())          # hash is re-checked by ct_eval itself
    (out / "prereg.json").write_text(json.dumps(pre, indent=1), encoding="utf-8")
    vr = json.loads((REPO / "artifacts" / "clean_test" / "verify_report.json").read_text(encoding="utf-8-sig"))
    (out / "verify_report.json").write_text(json.dumps(vr), encoding="utf-8")
    (out / "results").mkdir(exist_ok=True)

    src = (REPO / "experiments" / "8_locked_test" / "ct_eval.py").read_text(encoding="utf-8-sig")
    rep = [('B = Path(r"C:\\Users\\absamad\\cs"); R = B / "tomato-adcf-yolo"; OUT = B / "clean_test"',
            'B = Path(r"%s"); R = Path(r"%s"); OUT = Path(r"%s")' % (REPO.parent, REPO, out)),
           ('R / "results" / "clean_test_table', 'OUT / "results" / "clean_test_table')]
    if a.split == "val":
        rep += [('D / "images" / "test"', 'D / "images" / "val"'), ('D / "labels" / "test"', 'D / "labels" / "val"'),
                ('if r["split"] == "test"}', 'if r["split"] == "val"}'),
                ('assert vr["all_pass"] and vr["test_set_sha256"] == pre["test_set"]["sha256"], "test set changed or verification failed"',
                 'assert vr["all_pass"]')]
    for old, new in rep:
        assert old in src, "frozen evaluator changed unexpectedly: " + old[:60]
        src = src.replace(old, new)
    runner = out / "ct_eval_rescore.py"; runner.write_text(src, encoding="utf-8")
    sys.exit(subprocess.run([sys.executable, str(runner)], cwd=REPO).returncode)


if __name__ == "__main__":
    main()
