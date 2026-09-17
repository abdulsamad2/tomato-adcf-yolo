"""End-to-end: prepare -> train (1 epoch, CPU) -> test eval -> bench -> gates -> collect."""

import json
import subprocess
import sys

import pytest
import yaml


def cli(module, *args, cwd):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=cwd, check=True)


@pytest.mark.slow
def test_full_pipeline(raw_dataset, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    cli("adcf_yolo.data.prepare", "--raw", raw_dataset, "--out", "data/tv", cwd=work)
    cli("adcf_yolo.data.stats", "--data", "data/tv/dataset.yaml", "--out", "results/dataset", "--preview", 4, cwd=work)

    plan = {
        "data": "data/tv/dataset.yaml",
        "project": "runs",
        "seeds": [0],
        "pretrained": False,
        "train": {"epochs": 1, "imgsz": 64, "batch": 4, "workers": 0, "device": "cpu", "plots": False, "amp": False},
        "eval": {"split": "test", "imgsz": 64, "batch": 4, "conf": 0.001, "iou": 0.7},
        "tables": {"main": {"reference": "base", "title": "Main"}},
        "runs": [
            {"name": "base", "base": "yolo11n", "tier": 1, "tables": ["main"]},
            {"name": "adcf", "base": "yolo11n", "tier": 1, "tables": ["main"],
             "adcf": {"placement": "all", "fusion": "gated", "gate": "spatial"}},
        ],
    }
    (work / "plan.yaml").write_text(yaml.safe_dump(plan))
    cli("adcf_yolo.experiments", "--config", "plan.yaml", "--device", "cpu", cwd=work)

    res = json.loads((work / "runs/adcf/seed0/results_test.json").read_text())
    assert {"mAP50", "mAP50_95", "coco", "params_M", "GFLOPs", "per_class"} <= set(res)
    base = json.loads((work / "runs/base/seed0/results_test.json").read_text())
    assert res["params_M"] != base["params_M"]  # the ADCF model really is different

    # Second invocation must skip finished work.
    out = subprocess.run(
        [sys.executable, "-m", "adcf_yolo.experiments", "--config", "plan.yaml", "--dry-run"],
        cwd=work, check=True, capture_output=True, text=True,
    ).stdout
    assert "0 remaining" in out

    cli("adcf_yolo.eval.efficiency", "--runs", "runs", "--imgsz", 64, "--warmup", 1, "--iters", 3, "--device", "cpu", cwd=work)
    assert (work / "runs/adcf/speed.json").exists()
    cli("adcf_yolo.gates", "--weights", "runs/adcf/seed0/weights/best.pt", "--data", "data/tv/dataset.yaml",
        "--n", 2, "--imgsz", 64, "--device", "cpu", "--out", "results/gates", cwd=work)
    stats = json.loads((work / "results/gates/gate_stats.json").read_text())
    assert set(stats) == {"p4_td", "p3_out", "p4_out", "p5_out"}
    cli("adcf_yolo.collect", "--config", "plan.yaml", "--out", "results", cwd=work)
    md = (work / "results/tables_test.md").read_text()
    assert "| adcf | 1 |" in md
