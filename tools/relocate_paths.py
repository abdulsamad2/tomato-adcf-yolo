"""Make a runnable copy of experiments/ for a new machine.

The phase scripts were run from C:\\Users\\absamad\\cs (the parent of this repository) and hard-code that root.
They are kept byte-identical here so their SHA-256 hashes still match the pre-registrations
(see experiments/MANIFEST.json). This tool writes a COPY with the root replaced; it never edits experiments/.

    python tools/relocate_paths.py --root /data/work --out /data/work
    # -> /data/work/<script>.py etc., expecting the repo at /data/work/tomato-adcf-yolo

Scripts write their outputs to sibling folders of the repository (phase15/, phase16/, ..., posthoc/), exactly as
in the original layout. PowerShell launchers (*.ps1) are copied too; on Linux run the python commands they contain.
"""
import argparse, shutil
from pathlib import Path

OLD = [r"C:\Users\absamad\cs", "C:/Users/absamad/cs", r"C:\\Users\\absamad\\cs"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="new parent directory that contains tomato-adcf-yolo/")
    ap.add_argument("--out", required=True, help="where to write the relocated scripts (usually the same as --root)")
    a = ap.parse_args()
    src = Path(__file__).resolve().parents[1] / "experiments"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    new = str(Path(a.root).resolve())
    n = 0
    for p in src.rglob("*"):
        if p.suffix not in (".py", ".ps1"): continue
        dst = out / p.name if p.parent.name not in ("clean_test", "posthoc") else out / p.parent.name / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = p.read_text(encoding="utf-8-sig")
        for o in OLD:
            text = text.replace(o, new)
        dst.write_text(text, encoding="utf-8"); n += 1
    print(f"wrote {n} relocated scripts to {out} (root -> {new})")


if __name__ == "__main__":
    main()
