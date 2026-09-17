import csv
from collections import Counter

import pytest
import yaml

from adcf_yolo.data.prepare import (
    AUG_RE,
    clean_class_name,
    parse_label,
    prepare,
    prepare_cv,
    safe_name,
    stratified_group_split,
)
from adcf_yolo.data.stats import compute_stats


def test_clean_class_name():
    assert clean_class_name("'Pottassium Deficiency'") == "Potassium Deficiency"
    assert clean_class_name("'Early_blight'") == "Early Blight"


def test_parse_label(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("3.0 0.5 0.5 0.2 0.2\n3.0 0.5 0.5 0.2 0.2\n2.5 0.5 0.5 0.1 0.1\n1 0.99 0.5 0.1 0.1\n0 0.5 0.5 0 0.1\nbad\n")
    problems = Counter()
    boxes = parse_label(f, 8, problems)
    assert len(boxes) == 2
    assert all(isinstance(b[0], int) for b in boxes)
    clipped = [b for b in boxes if b[0] == 1][0]
    assert clipped[1] + clipped[3] / 2 == pytest.approx(1.0)
    assert problems == Counter(duplicate_box=1, bad_class=1, clipped_box=1, degenerate_box=1, bad_line=1)


def test_stratified_group_split_is_deterministic_and_balanced():
    strata = {f"g{i}": f"c{i % 4}" for i in range(400)}
    a = stratified_group_split(strata, (0.7, 0.15, 0.15), seed=1)
    assert a == stratified_group_split(strata, (0.7, 0.15, 0.15), seed=1)
    for c in range(4):
        counts = Counter(a[g] for g in strata if strata[g] == f"c{c}")
        assert counts == Counter(train=70, val=15, test=15)


def test_prepare_has_no_leakage(raw_dataset, tmp_path):
    out = tmp_path / "prepared"
    report = prepare(raw_dataset, out, seed=0)
    assert report["official_split_leaked_photos"] > 0  # the synthetic data leaks, like the real one

    rows = list(csv.DictReader(open(out / "split_manifest.csv")))
    split_of = {r["original_id"]: r["split"] for r in rows}
    assert set(split_of.values()) == {"train", "val", "test"}
    assert split_of["IMG20229999"] == split_of["IMG20220001"]  # near-duplicate merged

    split_of_written = {safe_name(g): s for g, s in split_of.items()}
    for split in ("train", "val", "test"):
        for img in (out / "images" / split).iterdir():
            assert split_of_written[AUG_RE.sub("", img.stem)] == split  # every copy sits with its original
            assert (out / "labels" / split / f"{img.stem}.txt").exists()
        if split != "train":
            assert not any("_aug" in p.name for p in (out / "images" / split).iterdir())
    assert any("_aug" in p.name for p in (out / "images" / "train").iterdir())

    cfg = yaml.safe_load((out / "dataset.yaml").read_text())
    assert cfg["names"][6] == "Potassium Deficiency"
    for label in out.glob("labels/*/*.txt"):
        for line in label.read_text().splitlines():
            assert line.split()[0].isdigit()  # "3.0" normalised to "3"

    stats = compute_stats(out / "dataset.yaml")
    assert stats["test"]["images"] == report["files_per_split"]["test"]

    # Reusing the manifest reproduces the split exactly.
    out2 = tmp_path / "prepared2"
    prepare(raw_dataset, out2, seed=123, manifest=out / "split_manifest.csv", train_aug=False)
    rows2 = {r["original_id"]: r["split"] for r in csv.DictReader(open(out2 / "split_manifest.csv"))}
    assert rows2 == split_of
    assert not any("_aug" in p.name for p in (out2 / "images" / "train").iterdir())

    with pytest.raises(FileExistsError):
        prepare(raw_dataset, out)


def test_prepare_cv_every_photo_tested_once(raw_dataset, tmp_path):
    out = tmp_path / "cv"
    reports = prepare_cv(raw_dataset, out, folds=3, val_ratio=0.15, seed=0)
    assert len(reports) == 3
    tested = Counter()
    for k in range(3):
        rows = list(csv.DictReader(open(out / f"fold{k}" / "split_manifest.csv")))
        split_of = {r["original_id"]: r["split"] for r in rows}
        cluster_splits = {}
        for r in rows:  # a near-duplicate cluster never straddles splits
            assert cluster_splits.setdefault(r["cluster"], r["split"]) == r["split"]
        tested.update(g for g, s in split_of.items() if s == "test")
        assert {"train", "val", "test"} == set(split_of.values())
        assert (out / f"fold{k}" / "dataset.yaml").exists()
    assert set(tested.values()) == {1}
    assert len(tested) == reports[0]["original_photos"]
