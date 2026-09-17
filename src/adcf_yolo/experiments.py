"""Run the experiment plan: train -> evaluate on test -> save metrics. Resumable.

Output per run and seed:
    runs/<name>/seed<k>/weights/best.pt    selected on VAL fitness
    runs/<name>/seed<k>/train_done.json    marker: training finished
    runs/<name>/seed<k>/results_test.json  TEST metrics (the numbers for the paper)

Re-running the same command skips finished work and resumes interrupted training.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from adcf_yolo.build import make_trainer
from adcf_yolo.eval.coco_eval import coco_size_eval
from adcf_yolo.eval.efficiency import model_complexity


def load_plan(path: Path) -> dict[str, Any]:
    plan = yaml.safe_load(Path(path).read_text())
    names = [r["name"] for r in plan["runs"]]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate run names: {sorted(dupes)}")
    for r in plan["runs"]:
        unknown = set(r.get("tables", [])) - set(plan.get("tables", {}))
        if unknown:
            raise ValueError(f"run {r['name']} references unknown tables {sorted(unknown)}")
    return plan


def select_runs(plan: dict, tier: int | None, only: list[str] | None) -> list[dict]:
    runs = plan["runs"]
    if only:
        missing = set(only) - {r["name"] for r in runs}
        if missing:
            raise ValueError(f"unknown run names: {sorted(missing)}")
        return [r for r in runs if r["name"] in only]
    return [r for r in runs if tier is None or r["tier"] <= tier]


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def train_run(run: dict, seed: int, plan: dict, project: Path, device: str | None) -> Path:
    seed_dir = project / run["name"] / f"seed{seed}"
    done = seed_dir / "train_done.json"
    best = seed_dir / "weights" / "best.pt"
    last = seed_dir / "weights" / "last.pt"
    if done.exists() and best.exists():
        return best

    trainer = make_trainer(run.get("adcf"))
    t0 = time.time()
    if last.exists():
        print(f"[run] resuming {seed_dir}")
        YOLO(str(last)).train(resume=True, trainer=trainer)
    else:
        weights = f"{run['base']}.pt" if plan.get("pretrained", True) else f"{run['base']}.yaml"
        args = {**plan["train"], **run.get("train", {})}
        if device is not None:
            args["device"] = device
        YOLO(weights).train(
            trainer=trainer,
            data=str(run.get("data", plan["data"])),
            project=str(project / run["name"]),
            name=f"seed{seed}",
            exist_ok=True,
            seed=seed,
            **args,
        )
    done.write_text(
        json.dumps(
            {"run": run, "seed": seed, "train_hours": (time.time() - t0) / 3600, "env": environment()},
            indent=2,
        )
    )
    return best


def evaluate_run(run: dict, seed: int, plan: dict, best: Path, device: str | None) -> dict[str, Any]:
    ev = plan["eval"]
    data = str(run.get("data", plan["data"]))
    seed_dir = best.parents[1]
    metrics = YOLO(str(best)).val(
        data=data,
        split=ev["split"],
        imgsz=ev["imgsz"],
        batch=ev["batch"],
        conf=ev["conf"],
        iou=ev["iou"],
        device=device,
        project=str(seed_dir),
        name=f"eval_{ev['split']}",
        exist_ok=True,
        plots=True,
    )
    box, names = metrics.box, metrics.names
    per_class = {
        names[int(c)]: {"AP50": float(box.ap50[k]), "AP50_95": float(box.ap[k])}
        for k, c in enumerate(box.ap_class_index)
    }
    result = {
        "run": run["name"],
        "seed": seed,
        "split": ev["split"],
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": per_class,
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
        "coco": coco_size_eval(
            best, data, ev["split"], ev["imgsz"], ev["conf"], ev["iou"], batch=ev["batch"], device=device
        ),
        **model_complexity(best, ev["imgsz"]),
    }
    (seed_dir / f"results_{ev['split']}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    p.add_argument("--tier", type=int, help="run all runs with tier <= this (default: all)")
    p.add_argument("--only", nargs="+", help="run only these run names")
    p.add_argument("--seeds", type=int, nargs="+", help="override the seeds in the config")
    p.add_argument("--device", help="e.g. 0, cpu, mps (default: Ultralytics auto-select)")
    p.add_argument("--reeval", action="store_true", help="recompute test metrics for finished runs")
    p.add_argument("--dry-run", action="store_true", help="print what would run and exit")
    args = p.parse_args()

    plan = load_plan(args.config)
    project = Path(plan.get("project", "runs")).resolve()
    runs = select_runs(plan, args.tier, args.only)
    seeds = args.seeds or plan["seeds"]
    split = plan["eval"]["split"]

    jobs = [(r, s) for r in runs for s in seeds]
    todo = [(r, s) for r, s in jobs if args.reeval or not (project / r["name"] / f"seed{s}" / f"results_{split}.json").exists()]
    print(f"[run] {len(jobs)} jobs ({len(runs)} runs x {len(seeds)} seeds), {len(todo)} remaining")
    for r, s in todo:
        print(f"  - {r['name']} seed{s}  base={r['base']}  adcf={r.get('adcf')}")
    if args.dry_run:
        return

    for k, (run, seed) in enumerate(todo, 1):
        print(f"\n[run] ===== {k}/{len(todo)}: {run['name']} seed{seed} =====")
        best = train_run(run, seed, plan, project, args.device)
        res = evaluate_run(run, seed, plan, best, args.device)
        coco = res["coco"]
        print(
            f"[run] {run['name']} seed{seed}: mAP50={res['mAP50']:.4f} mAP50-95={res['mAP50_95']:.4f} "
            f"AP_small={coco['AP_small']} params={res['params_M']:.2f}M GFLOPs={res['GFLOPs']:.1f}"
        )


if __name__ == "__main__":
    main()
