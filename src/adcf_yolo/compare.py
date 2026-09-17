"""Paired bootstrap: is model B really better than model A on this test set?

    adcf-compare --a yolo11s --b yolo11s-adcf-res

LEARN (see docs/notes/05-metrics.md): seeds capture *training* noise, but the
test set is also a sample. With only a few hundred test photos, a different
sample could flip a small gain. The bootstrap resamples test *images* with
replacement many times, recomputes mAP for both models on the SAME resample
(paired), and looks at the distribution of the difference:

  * 95% CI of Δ entirely above 0  -> the gain is robust to test-set sampling
  * CI crossing 0                 -> can't claim an improvement from this data

Each model's value on a resample is the mean over its seeds, so training noise
is averaged in as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adcf_yolo.eval.predictions import MapEvaluator, predict_split
from adcf_yolo.experiments import eval_settings, find_run, load_plan


def evaluators(plan: dict, name: str, split: str, device: str | None) -> list[MapEvaluator]:
    run = find_run(plan, name)
    project = Path(plan.get("project", "runs"))
    seed_dirs = sorted(d for d in (project / name).glob("seed*") if (d / "weights" / "best.pt").exists())
    if not seed_dirs:
        raise SystemExit(f"no trained seeds for {name} under {project / name}")
    ev = eval_settings(plan, run)
    out = []
    for d in seed_dirs:
        images = predict_split(
            d / "weights" / "best.pt", run.get("data", plan["data"]), split, ev["imgsz"], ev["conf"], ev["iou"],
            batch=ev["batch"], device=device, cache=d / f"preds_{split}.pkl",
        )
        out.append(MapEvaluator(images))
    return out


def paired_bootstrap(a: list[MapEvaluator], b: list[MapEvaluator], n_boot: int = 1000, seed: int = 0) -> dict:
    names = a[0].names
    if any(e.names != names for e in a + b):
        raise ValueError("models were evaluated on different image sets")
    n = len(names)

    def score(evs, idx):
        vals = np.array([e(idx) for e in evs])  # (seeds, 2)
        return vals.mean(0)

    point_a, point_b = score(a, None), score(b, None)
    rng = np.random.default_rng(seed)
    deltas = np.empty((n_boot, 2))
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = score(b, idx) - score(a, idx)
    result = {"images": n, "seeds_a": len(a), "seeds_b": len(b), "n_boot": n_boot}
    for j, metric in enumerate(("mAP50", "mAP50_95")):
        d = deltas[:, j]
        result[metric] = {
            "a": float(point_a[j]),
            "b": float(point_b[j]),
            "delta": float(point_b[j] - point_a[j]),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_delta_le_0": float((d <= 0).mean()),
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    p.add_argument("--a", required=True, help="reference run name")
    p.add_argument("--b", required=True, help="candidate run name")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device")
    p.add_argument("--out", type=Path, default=Path("results/compare"))
    args = p.parse_args()

    plan = load_plan(args.config)
    res = paired_bootstrap(
        evaluators(plan, args.a, args.split, args.device), evaluators(plan, args.b, args.split, args.device), args.n_boot
    )
    res.update({"a_run": args.a, "b_run": args.b, "split": args.split})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.b}__vs__{args.a}_{args.split}.json").write_text(json.dumps(res, indent=2))
    for metric in ("mAP50", "mAP50_95"):
        r = res[metric]
        print(
            f"{metric:9s} {args.a}={100 * r['a']:.2f}  {args.b}={100 * r['b']:.2f}  Δ={100 * r['delta']:+.2f}  "
            f"95% CI [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}]  P(Δ≤0)={r['p_delta_le_0']:.3f}"
        )


if __name__ == "__main__":
    main()
