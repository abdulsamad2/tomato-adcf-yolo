"""End-to-end on synthetic data, through the real CLIs:
prepare (+CV) -> train (1 epoch, CPU) -> val/test eval -> bench -> gates -> compare -> collect."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

P2_CFG = str(Path(__file__).resolve().parents[1] / "configs/models/yolo11n-p2.yaml")


def cli(module, *args, cwd, capture=False):
    r = subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=cwd, check=True, capture_output=capture, text=True)
    return r.stdout


@pytest.mark.slow
def test_full_pipeline(raw_dataset, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    cli("adcf_yolo.data.prepare", "--raw", raw_dataset, "--out", "data/tv", cwd=work)
    cli("adcf_yolo.data.prepare", "--raw", raw_dataset, "--out", "data/tv_cv", "--cv-folds", 2, cwd=work)
    cli("adcf_yolo.data.stats", "--data", "data/tv/dataset.yaml", "--out", "results/dataset", "--preview", 4, cwd=work)

    final = {"placement": "all", "mode": "residual", "fusion": "gated", "gate": "spatial"}
    plan = {
        "data": "data/tv/dataset.yaml",
        "project": "runs",
        "seeds": [0],
        "pretrained": False,
        "train": {"epochs": 1, "imgsz": 64, "batch": 4, "workers": 0, "device": "cpu", "plots": False, "amp": False},
        "eval": {"splits": ["val", "test"], "batch": 4, "conf": 0.001, "iou": 0.7},
        "tables": {
            "ladder": {"kind": "ladder", "reference": "base", "title": "Ladder", "order": ["base", "adcf", "p2-adcf"]},
            "main": {"reference": "base", "title": "Main"},
        },
        "cv": {"folds": 2, "data": "data/tv_cv/fold{fold}/dataset.yaml", "project": "runs_cv", "seeds": [0],
               "runs": ["base", "adcf"]},
        "runs": [
            {"name": "base", "base": "yolo11n", "tier": 1, "tables": ["ladder", "main"]},
            {"name": "adcf", "base": "yolo11n", "tier": 1, "tables": ["ladder", "main"], "adcf": final},
            {"name": "replace", "base": "yolo11n", "tier": 1, "tables": ["main"], "adcf": {**final, "mode": "replace"}},
            {"name": "p2-adcf", "base": "yolo11n", "cfg": P2_CFG, "tier": 1, "tables": ["ladder"], "adcf": final,
             "train": {"imgsz": 96}},
        ],
    }
    (work / "plan.yaml").write_text(yaml.safe_dump(plan))
    cli("adcf_yolo.experiments", "--config", "plan.yaml", "--device", "cpu", cwd=work)

    res = json.loads((work / "runs/adcf/seed0/results_test.json").read_text())
    assert {"mAP50", "mAP50_95", "coco", "params_M", "GFLOPs", "per_class"} <= set(res)
    assert (work / "runs/adcf/seed0/results_val.json").exists()
    base = json.loads((work / "runs/base/seed0/results_test.json").read_text())
    assert res["params_M"] > base["params_M"]  # residual ADCF adds parameters
    p2 = json.loads((work / "runs/p2-adcf/seed0/results_test.json").read_text())
    assert p2["imgsz"] == 96  # evaluated at its own training resolution

    assert "0 remaining" in cli("adcf_yolo.experiments", "--config", "plan.yaml", "--dry-run", cwd=work, capture=True)

    cli("adcf_yolo.experiments", "--config", "plan.yaml", "--cv", "--device", "cpu", cwd=work)
    assert (work / "runs_cv/adcf/fold1_seed0/results_test.json").exists()

    cli("adcf_yolo.eval.efficiency", "--runs", "runs", "--warmup", 1, "--iters", 3, "--device", "cpu", cwd=work)
    assert json.loads((work / "runs/p2-adcf/speed.json").read_text())["imgsz"] == 96

    cli("adcf_yolo.gates", "--weights", "runs/adcf/seed0/weights/best.pt", "--data", "data/tv/dataset.yaml",
        "--n", 2, "--imgsz", 64, "--device", "cpu", "--out", "results/gates", cwd=work)
    stats = json.loads((work / "results/gates/gate_stats.json").read_text())
    assert set(stats) == {"p4_td", "p3_out", "p4_out", "p5_out"}

    out = cli("adcf_yolo.compare", "--config", "plan.yaml", "--a", "base", "--b", "adcf", "--n-boot", 20,
              "--device", "cpu", cwd=work, capture=True)
    assert "95% CI" in out

    cli("adcf_yolo.collect", "--config", "plan.yaml", "--out", "results", cwd=work)
    md = (work / "results/tables_val.md").read_text()
    assert "| adcf | 1 |" in md and "Δ mAP50-95 (step)" in md
    cli("adcf_yolo.collect", "--config", "plan.yaml", "--out", "results", "--cv", cwd=work)
    assert "| adcf | 2 |" in (work / "results/tables_cv.md").read_text()
