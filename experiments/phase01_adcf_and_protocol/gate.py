import argparse, json, statistics as st, sys
from pathlib import Path

def mean_val(root, name):
    vals = [json.loads(p.read_text())["mAP50_95"] * 100
            for p in sorted(Path(root, name).glob("seed*/results_val.json"))]
    return (st.mean(vals), len(vals)) if vals else (None, 0)

ap = argparse.ArgumentParser(description="Decision gate: is a candidate worth more GPU?")
ap.add_argument("--root", default=r"C:\Users\absamad\cs\tomato-adcf-yolo\runs")
ap.add_argument("--a", required=True, help="reference run")
ap.add_argument("--b", nargs="+", required=True, help="candidate run(s)")
ap.add_argument("--min-delta", type=float, default=1.5)
ap.add_argument("--winner-file")
args = ap.parse_args()

a, na = mean_val(args.root, args.a)
if a is None:
    print(f"[gate] reference {args.a} has no results -> FAIL"); sys.exit(1)
print(f"[gate] reference {args.a} = {a:.2f} (n={na})")
best = bestname = None
for nm in args.b:
    v, n = mean_val(args.root, nm)
    print(f"[gate] candidate {nm} = {'n/a' if v is None else format(v, '.2f')} (n={n})")
    if v is not None and (best is None or v > best):
        best, bestname = v, nm
if best is None:
    print("[gate] no candidate produced results -> FAIL"); sys.exit(1)
d = best - a
print(f"[gate] best={bestname} {best:.2f}  delta={d:+.2f}  threshold={args.min_delta:+.2f}")
if d >= args.min_delta:
    print(f"[gate] PASS -> spend more GPU confirming {bestname}")
    if args.winner_file:
        Path(args.winner_file).write_text(bestname)
    sys.exit(0)
print("[gate] FAIL -> skip confirmation, move to the next stage")
sys.exit(1)
