"""Aggregate results into paper tables.

    adcf-collect                 # val and test tables for the single-split runs
    adcf-collect --cv            # cross-validation table (mean ± std over folds, paired t-test)

LEARN: make every design decision from the *val* tables. Look at the test tables
only once the final model is frozen (docs/notes/07-critical-review.md). With 3 seeds
a t-test has little power; use adcf-compare (paired bootstrap) or --cv for the
headline claim.
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

COLUMNS = [  # (key, header)
    ("mAP50", "mAP50"),
    ("mAP50_95", "mAP50-95"),
    ("precision", "P"),
    ("recall", "R"),
    ("coco.AP_small", "AP_S"),
    ("coco.AP_medium", "AP_M"),
    ("coco.AP_large", "AP_L"),
]


def get(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


def load_results(project: Path, split: str) -> dict[str, list[dict]]:
    results = defaultdict(list)
    for f in sorted(project.glob(f"*/*/results_{split}.json")):
        r = json.loads(f.read_text())
        speed = f.parents[1] / "speed.json"
        r["speed"] = json.loads(speed.read_text()) if speed.exists() else {}
        results[r["run"]].append(r)
    return results


def mean_std(values: list[float]) -> str:
    v = 100 * np.array(values, dtype=float)
    return f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def _row_cells(name: str, rs: list[dict]) -> list[str]:
    cells = [name, str(len(rs))]
    for key, _ in COLUMNS:
        vals = [get(r, key) for r in rs]
        cells.append(mean_std(vals) if all(v is not None for v in vals) else "–")
    sp = rs[0]["speed"]
    cells += [f"{rs[0]['imgsz']}", f"{rs[0]['params_M']:.2f}", f"{rs[0]['GFLOPs']:.1f}"]
    cells += [f"{sp['latency_ms_mean']:.2f}", f"{sp['fps']:.0f}"] if sp else ["–", "–"]
    return cells


def table(spec: dict, run_names: list[str], results: dict) -> str:
    """kind=compare: Δ and Welch p vs `reference`. kind=ladder: Δ vs the previous row (cumulative story)."""
    ladder = spec.get("kind") == "ladder"
    headers = ["Model", "n"] + [h for _, h in COLUMNS] + ["imgsz", "Params (M)", "GFLOPs", "Latency (ms)", "FPS"]
    headers += ["Δ mAP50-95 (step)", "Δ (total)"] if ladder else ["Δ mAP50-95", "p (Welch)"]
    lines = [f"### {spec['title']}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    reference = spec.get("reference") or (run_names[0] if run_names else None)
    ref_vals = [r["mAP50_95"] for r in results.get(reference, [])]
    prev_vals = None
    for name in run_names:
        rs = results.get(name)
        if not rs:
            lines.append(f"| {name} | 0 |" + " – |" * (len(headers) - 2))
            prev_vals = None
            continue
        cells = _row_cells(name, rs)
        vals = [r["mAP50_95"] for r in rs]
        if ladder:
            cells.append(f"{100 * (np.mean(vals) - np.mean(prev_vals)):+.1f}" if prev_vals else "–")
            cells.append(f"{100 * (np.mean(vals) - np.mean(ref_vals)):+.1f}" if ref_vals and name != reference else "–")
            prev_vals = vals
        elif name == reference or not ref_vals:
            cells += ["ref" if name == reference else "–", "–"]
        else:
            cells.append(f"{100 * (np.mean(vals) - np.mean(ref_vals)):+.1f}")
            ok = len(vals) > 1 and len(ref_vals) > 1
            cells.append(f"{stats.ttest_ind(vals, ref_vals, equal_var=False).pvalue:.3f}" if ok else "–")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def per_class_table(run_names: list[str], results: dict) -> str:
    present = [n for n in run_names if n in results]
    if not present:
        return ""
    classes = list(results[present[0]][0]["per_class"])
    lines = ["### Per-class AP50-95 (mean over seeds)", "", "| Class | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for c in classes:
        cells = []
        for n in present:
            vals = [r["per_class"][c]["AP50_95"] for r in results[n] if c in r["per_class"]]
            cells.append(f"{100 * np.mean(vals):.1f}" if vals else "–")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cv_table(plan: dict, split: str = "test") -> str:
    spec = plan["cv"]
    results = load_results(Path(spec.get("project", "runs_cv")), split)
    names = spec["runs"]
    per_fold = {}  # name -> {fold: mean over seeds}
    for n in names:
        folds = defaultdict(list)
        for r in results.get(n, []):
            folds[r["fold"]].append(r)
        per_fold[n] = {k: {m: np.mean([get(r, m) or 0 for r in rs]) for m, _ in COLUMNS} for k, rs in folds.items()}
    ref = names[0]
    lines = [
        f"### {spec['folds']}-fold grouped cross-validation ({split} fold, mean ± std over folds)",
        "",
        "| Model | folds | " + " | ".join(h for _, h in COLUMNS) + " | Δ mAP50-95 | p (paired t) | folds improved |",
        "|---|---|" + "---|" * (len(COLUMNS) + 3),
    ]
    for n in names:
        folds = per_fold[n]
        if not folds:
            lines.append(f"| {n} | 0 |" + " – |" * (len(COLUMNS) + 3))
            continue
        cells = [n, str(len(folds))] + [mean_std([f[m] for f in folds.values()]) for m, _ in COLUMNS]
        common = sorted(set(folds) & set(per_fold[ref]))
        if n != ref and len(common) > 1:
            a = np.array([per_fold[ref][k]["mAP50_95"] for k in common])
            b = np.array([folds[k]["mAP50_95"] for k in common])
            cells += [f"{100 * (b - a).mean():+.1f}", f"{stats.ttest_rel(b, a).pvalue:.3f}", f"{int((b > a).sum())}/{len(common)}"]
        else:
            cells += ["ref" if n == ref else "–", "–", "–"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=("val", "test"))
    p.add_argument("--cv", action="store_true", help="write the cross-validation table instead")
    args = p.parse_args()

    plan = load_plan(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cv:
        md = "# Cross-validation results\n\n" + cv_table(plan)
        (args.out / "tables_cv.md").write_text(md)
        print(md)
        return

    project = Path(plan.get("project", "runs"))
    for split in args.splits:
        results = load_results(project, split)
        with open(args.out / f"all_seeds_{split}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "seed", "imgsz"] + [h for _, h in COLUMNS] + ["params_M", "GFLOPs"])
            for name, rs in results.items():
                for r in sorted(rs, key=lambda r: r["seed"]):
                    writer.writerow([name, r["seed"], r["imgsz"]] + [get(r, k) for k, _ in COLUMNS] + [r["params_M"], r["GFLOPs"]])

        parts = [
            f"# Results ({split} split)\n",
            "Metrics ×100, mean ± sample std over seeds. " + (
                "**VAL: use these for design decisions.**\n" if split == "val"
                else "**TEST: report only after the final model is frozen.**\n"
            ),
        ]
        for key, spec in plan["tables"].items():
            names = [r["name"] for r in plan["runs"] if key in r.get("tables", [])]
            if spec.get("order"):
                names = [n for n in spec["order"] if n in names] + [n for n in names if n not in spec["order"]]
            parts.append(table(spec, names, results))
        parts.append(per_class_table([r["name"] for r in plan["runs"] if "main" in r.get("tables", [])], results))
        md = "\n".join(parts)
        (args.out / f"tables_{split}.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
