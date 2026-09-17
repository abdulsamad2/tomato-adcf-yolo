"""Synthetic Tomato-Village Variant-c layout that reproduces the real dataset's quirks."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adcf_yolo.data.download import SUBDIR

N_CLASSES = 8


def _write_image(path: Path, rng: np.random.Generator, base: np.ndarray | None = None) -> np.ndarray:
    arr = base if base is not None else rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).resize((160, 120), Image.Resampling.NEAREST).save(path, quality=95)
    return arr


def make_raw_dataset(root: Path, n_originals: int = 48, n_aug: int = 3, seed: int = 0) -> Path:
    """Each original gets `n_aug` augmented copies; copies are scattered across the official
    train/val folders (as in the real data), labels use float class ids ("3.0")."""
    rng = np.random.default_rng(seed)
    variant = root / SUBDIR
    for split in ("train", "val"):
        (variant / split / "images").mkdir(parents=True)
        (variant / split / "yolo").mkdir(parents=True)
    (variant / "Varient-C Labels.txt").write_text(
        "0 : 'Early_blight'\n1 : 'Healthy'\n2 : 'Late_blight'\n3 : 'Leaf Miner'\n"
        "4 : 'Magnesium Deficiency'\n5 : 'Nitrogen Deficiency'\n6 : 'Pottassium Deficiency'\n7 : 'Spotted Wilt Virus'\n"
    )
    for i in range(n_originals):
        cls = i % N_CLASSES
        base = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
        stem = f"IMG2022{i:04d}" if i % 3 else f"Jaipur_Pots ({i})"
        for k in range(n_aug + 1):
            name = stem if k == 0 else f"{stem}_aug{k}"
            split = "val" if rng.random() < 0.2 else "train"
            arr = base if k == 0 else np.flip(base, axis=1).copy()
            _write_image(variant / split / "images" / f"{name}.jpg", rng, arr)
            lines = [f"{float(cls)} 0.5 0.5 0.2 0.3", f"{float(cls)} 0.2 0.2 0.1 0.1"]
            if i == 0 and k == 0:
                lines += ["9.0 0.5 0.5 0.1 0.1", "1.0 0.99 0.5 0.1 0.1", "garbage"]  # bad class, clipped, bad line
            (variant / split / "yolo" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    # A near-duplicate photo (same scene re-shot): must end up in the same split as IMG20220001.
    src = next(variant.glob("*/images/IMG20220001.jpg"))
    first = np.asarray(Image.open(src))
    dup = variant / "train" / "images" / "IMG20229999.jpg"
    Image.fromarray(np.clip(first.astype(int) + 3, 0, 255).astype(np.uint8)).save(dup, quality=95)
    (variant / "train" / "yolo" / "IMG20229999.txt").write_text("1.0 0.5 0.5 0.2 0.3\n")
    return root


@pytest.fixture
def raw_dataset(tmp_path):
    return make_raw_dataset(tmp_path / "raw")
