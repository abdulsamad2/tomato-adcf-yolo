"""Run the experiment plan: train -> evaluate on val and test -> save metrics. Resumable.

Output per job (run x seed, or run x fold x seed with --cv):
    <project>/<name>/seed<k>/weights/best.pt      selected on VAL fitness
    <project>/<name>/seed<k>/train_done.json      marker: training finished (+ env, hours)
    <project>/<name>/seed<k>/results_val.json     for development decisions
    <project>/<name>/seed<k>/results_test.json    for the paper, once decisions are frozen

Re-running the same command skips finished work and resumes interrupted training.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from adcf_yolo.build import make_trainer
from adcf_yolo.eval.coco_eval import coco_size_eval
from adcf_yolo.eval.efficiency import model_complexity
from adcf_yolo.eval.predictions import predict_split


@dataclass
class Job:
    run: dict
    seed: int
    fold: int | None
    data: str
    dir: Path

    @property
    def label(self) -> str:
        return f"{self.run['name']} {self.dir.name}"


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
    for name in plan.get("cv", {}).get("runs", []):
        find_run(plan, name)
    return plan


def find_run(plan: dict, name: str) -> dict:
    for r in plan["runs"]:
        if r["name"] == name:
            return r
    raise ValueError(f"unknown run name: {name}")


def eval_settings(plan: dict, run: dict) -> dict[str, Any]:
    """Evaluate at the resolution the run was trained at."""
    train = {**plan["train"], **run.get("train", {})}
    return {"splits": ["val", "test"], "batch": 16, "conf": 0.001, "iou": 0.7, **plan.get("eval", {}), "imgsz": train["imgsz"]}


def build_jobs(plan: dict, runs: list[dict], seeds: list[int] | None, cv: bool) -> list[Job]:
    jobs = []
    if cv:
        spec = plan["cv"]
        project = Path(spec.get("project", "runs_cv")).resolve()
        for r in runs:
            for k in range(spec["folds"]):
                for s in seeds or spec.get("seeds", [0]):
                    jobs.append(Job(r, s, k, spec["data"].format(fold=k), project / r["name"] / f"fold{k}_seed{s}"))
    else:
        project = Path(plan.get("project", "runs")).resolve()
        for r in runs:
            for s in seeds or plan["seeds"]:
                jobs.append(Job(r, s, None, str(r.get("data", plan["data"])), project / r["name"] / f"seed{s}"))
    return jobs


def select_runs(plan: dict, tier: int | None, only: list[str] | None, cv: bool) -> list[dict]:
    if only:
        return [find_run(plan, n) for n in only]
    if cv:
        return [find_run(plan, n) for n in plan["cv"]["runs"]]
    return [r for r in plan["runs"] if tier is None or r["tier"] <= tier]


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _finished_checkpoint(path: Path) -> bool:
    """Ultralytics strips the optimizer from last.pt when training completes."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("optimizer") is None


def train_job(job: Job, plan: dict, device: str | None) -> Path:
    done = job.dir / "train_done.json"
    best, last = job.dir / "weights" / "best.pt", job.dir / "weights" / "last.pt"
    if done.exists() and best.exists():
        return best

    run = job.run
    t0 = time.time()
    if best.exists() and last.exists() and _finished_checkpoint(last):
        print(f"[run] {job.label}: training already complete, writing marker")
    elif last.exists():
        print(f"[run] resuming {job.dir}")
        YOLO(str(last)).train(resume=True, trainer=make_trainer(run.get("adcf")))
    else:
        args = {**plan["train"], **run.get("train", {})}
        if device is not None:
            args["device"] = device
        pretrained = plan.get("pretrained", True)
        if run.get("cfg"):  # custom architecture (e.g. P2): build from YAML, init from the base checkpoint
            model = YOLO(run["cfg"])
            args["pretrained"] = f"{run['base']}.pt" if pretrained else False
        else:
            model = YOLO(f"{run['base']}.pt" if pretrained else f"{run['base']}.yaml")
        model.train(
            trainer=make_trainer(run.get("adcf")),
            data=job.data,
            project=str(job.dir.parent),
            name=job.dir.name,
            exist_ok=True,
            seed=job.seed,
            **args,
        )
    done.write_text(
        json.dumps(
            {"run": run, "seed": job.seed, "fold": job.fold, "data": job.data,
             "train_hours": (time.time() - t0) / 3600, "env": environment()},
            indent=2,
        )
    )
    return best


def evaluate_job(job: Job, plan: dict, split: str, best: Path, device: str | None) -> dict[str, Any]:
    ev = eval_settings(plan, job.run)
    metrics = YOLO(str(best)).val(
        data=job.data, split=split, imgsz=ev["imgsz"], batch=ev["batch"], conf=ev["conf"], iou=ev["iou"],
        device=device, project=str(job.dir), name=f"eval_{split}", exist_ok=True, plots=True,
    )
    box, names = metrics.box, metrics.names
    images = predict_split(
        best, job.data, split, ev["imgsz"], ev["conf"], ev["iou"], batch=ev["batch"], device=device,
        cache=job.dir / f"preds_{split}.pkl",
    )
    result = {
        "run": job.run["name"],
        "seed": job.seed,
        "fold": job.fold,
        "split": split,
        "imgsz": ev["imgsz"],
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": {
            names[int(c)]: {"AP50": float(box.ap50[k]), "AP50_95": float(box.ap[k])}
            for k, c in enumerate(box.ap_class_index)
        },
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
        "coco": coco_size_eval(images, names),
        **model_complexity(best, ev["imgsz"]),
    }
    (job.dir / f"results_{split}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    p.add_argument("--tier", type=int, help="run all runs with tier <= this (default: all)")
    p.add_argument("--only", nargs="+", help="run only these run names")
    p.add_argument("--seeds", type=int, nargs="+", help="override the seeds in the config")
    p.add_argument("--cv", action="store_true", help="run the cross-validation jobs from the `cv:` section")
    p.add_argument("--device", help="e.g. 0, cpu, mps (default: Ultralytics auto-select)")
    p.add_argument("--reeval", action="store_true", help="recompute metrics for finished runs")
    p.add_argument("--dry-run", action="store_true", help="print what would run and exit")
    args = p.parse_args()

    plan = load_plan(args.config)
    runs = select_runs(plan, args.tier, args.only, args.cv)
    jobs = build_jobs(plan, runs, args.seeds, args.cv)

    def pending(job: Job) -> bool:
        splits = eval_settings(plan, job.run)["splits"]
        return args.reeval or not all((job.dir / f"results_{s}.json").exists() for s in splits)

    todo = [j for j in jobs if pending(j)]
    print(f"[run] {len(jobs)} jobs ({len(runs)} runs), {len(todo)} remaining")
    for j in todo:
        print(f"  - {j.label}  base={j.run['base']}  cfg={j.run.get('cfg')}  adcf={j.run.get('adcf')}")
    if args.dry_run:
        return

    for k, job in enumerate(todo, 1):
        print(f"\n[run] ===== {k}/{len(todo)}: {job.label} =====")
        best = train_job(job, plan, args.device)
        for split in eval_settings(plan, job.run)["splits"]:
            res = evaluate_job(job, plan, split, best, args.device)
            print(
                f"[run] {job.label} [{split}] mAP50={res['mAP50']:.4f} mAP50-95={res['mAP50_95']:.4f} "
                f"AP_small={res['coco']['AP_small']} params={res['params_M']:.2f}M GFLOPs={res['GFLOPs']:.1f}"
            )


if __name__ == "__main__":
    main()
