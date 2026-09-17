"""Download only the object-detection variant (Variant-c) of Tomato-Village.

The full GitHub repo is ~3.3 GB; a sparse, blob-less clone fetches just Variant-c.
The exact commit is written to SOURCE.txt so the paper can cite the data version.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/mamta-joshi-gehlot/Tomato-Village.git"
SUBDIR = "Variant-c(Object Detection)"


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def download(dest: Path, commit: str | None = None) -> Path:
    dest = dest.resolve()
    if (dest / ".git").exists():
        print(f"[download] {dest} already exists, skipping clone")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] sparse-cloning {SUBDIR} into {dest} (several GB of images, be patient)")
        _git("clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(dest))
        _git("sparse-checkout", "set", SUBDIR, cwd=dest)
        _git("checkout", commit or "main", cwd=dest)
    head = _git("rev-parse", "HEAD", cwd=dest)
    (dest / "SOURCE.txt").write_text(f"repo: {REPO_URL}\nsubdir: {SUBDIR}\ncommit: {head}\n")
    print(f"[download] done, commit {head}")
    return dest / SUBDIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, default=Path("data/raw/Tomato-Village"))
    p.add_argument("--commit", help="pin a specific commit (default: latest main)")
    args = p.parse_args()
    download(args.dest, args.commit)


if __name__ == "__main__":
    main()
