"""Aggregate per-seed test results into paper tables (mean ± std, delta, p-value).

LEARN: with 3 seeds a t-test has little power. Treat p-values as supporting
evidence, not proof; a gain smaller than the seed std is not a real gain.
See docs/notes/05-metrics.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from adcf_yolo.experiments import load_plan

COLUMNS = [  # (key, header, scale)
    ("mAP50", "mAP50", 100),
    ("mAP50_95", "mAP50-95", 100),
    ("precision", "P", 100),
    ("recall", "R", 100),
    ("coco.AP_small", "AP_S", 100),
    ("coco.AP_medium", "AP_M", 100),
    ("coco.AP_large", "AP_L", 100),
]


def get(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


def load_results(project: Path, split: str) -> dict[str, list[dict]]:
    results = defaultdict(list)
    for f in sorted(project.glob(f"*/seed*/results_{split}.json")):
        r = json.loads(f.read_text())
        speed = f.parents[1] / "speed.json"
        r["speed"] = json.loads(speed.read_text()) if speed.exists() else {}
        results[r["run"]].append(r)
    return results


def fmt(values: list[float], scale: float) -> str:
    v = np.array(values, dtype=float) * scale
    return f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def table(title: str, run_names: list[str], results: dict, reference: str | None) -> str:
    headers = ["Model", "seeds"] + [h for _, h, _ in COLUMNS] + ["Params (M)", "GFLOPs", "Latency (ms)", "FPS", "Δ mAP50-95", "p"]
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    ref_vals = [r["mAP50_95"] for r in results.get(reference, [])]
    for name in run_names:
        rs = results.get(name)
        if not rs:
            lines.append(f"| {name} | 0 |" + " – |" * (len(headers) - 2))
            continue
        cells = [name, str(len(rs))]
        for key, _, scale in COLUMNS:
            vals = [get(r, key) for r in rs]
            cells.append(fmt(vals, scale) if all(v is not None for v in vals) else "–")
        cells += [f"{rs[0]['params_M']:.2f}", f"{rs[0]['GFLOPs']:.1f}"]
        sp = rs[0]["speed"]
        cells += [f"{sp['latency_ms_mean']:.2f}", f"{sp['fps']:.0f}"] if sp else ["–", "–"]
        vals = [r["mAP50_95"] for r in rs]
        if name != reference and ref_vals:
            cells.append(f"{100 * (np.mean(vals) - np.mean(ref_vals)):+.1f}")
            if len(vals) > 1 and len(ref_vals) > 1:
                cells.append(f"{stats.ttest_ind(vals, ref_vals, equal_var=False).pvalue:.3f}")  # Welch's t-test
            else:
                cells.append("–")
        else:
            cells += ["ref" if name == reference else "–", "–"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def per_class_table(run_names: list[str], results: dict) -> str:
    present = [n for n in run_names if n in results]
    if not present:
        return ""
    classes = list(results[present[0]][0]["per_class"])
    lines = ["### Per-class AP50-95 (test, mean over seeds)", "", "| Class | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for c in classes:
        cells = [f"{100 * np.mean([r['per_class'][c]['AP50_95'] for r in results[n] if c in r['per_class']]):.1f}" for n in present]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    plan = load_plan(args.config)
    project = Path(plan.get("project", "runs"))
    split = plan["eval"]["split"]
    results = load_results(project, split)
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / f"all_seeds_{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed"] + [h for _, h, _ in COLUMNS] + ["params_M", "GFLOPs"])
        for name, rs in results.items():
            for r in sorted(rs, key=lambda r: r["seed"]):
                writer.writerow([name, r["seed"]] + [get(r, k) for k, _, _ in COLUMNS] + [r["params_M"], r["GFLOPs"]])

    parts = [f"# Results ({split} split)\n", "Metrics ×100. mean ± sample std over seeds. p = Welch's t-test on mAP50-95 vs the reference.\n"]
    for key, spec in plan["tables"].items():
        names = [r["name"] for r in plan["runs"] if key in r.get("tables", [])]
        parts.append(table(spec["title"], names, results, spec.get("reference")))
    parts.append(per_class_table([r["name"] for r in plan["runs"] if "main" in r.get("tables", [])], results))
    md = "\n".join(parts)
    (args.out / f"tables_{split}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
