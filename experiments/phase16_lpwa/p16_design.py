"""Phase 16A design (no training): solve the Localization-Preserving Width Allocation (LPWA-50).

Rule fixed BEFORE solving (revision 2, user-set budget box; stated in the Phase 16A pre-registration):
  * topology, depth, head design identical to YOLO11n / uniform 0.50x (runs_p15/w050)
  * stem layers 0-2 stay at the uniform 0.50x width
  * every P3/P4-resolution layer (backbone 3-6, neck 13, 16, 17, 19) is scaled by ONE shared factor a
  * every P5-resolution layer (backbone 7-10, neck 20 and 22) is scaled by ONE shared factor b,
    b in {0.50, 0.55, ..., 1.00} of the uniform 0.50x width
  * HARD budget: params <= 811,144 (uniform 0.50x). TARGET: GFLOPs@512 <= 2.0. ABSOLUTE cap: < 2.707 (0.75x)
  * choose the LARGEST a (0.01 grid) for which some b meets params and the 2.0 target; at that a take
    the LARGEST b that meets params. If no a > 1 meets the 2.0 target, fall back to the 2.707 cap
    with b = 0.50 and record that the target was infeasible.
  * the head is unchanged: the class branch is <2% of params and the box branch is already floored
    at 64 channels (reg_max*4) at every width (phase16/p16_budget_audit.json)
Channels are realised with Ultralytics' make_divisible(.., 8), i.e. the same rounding as Phase 15.
"""
import copy, json, math, sys, warnings
from pathlib import Path
import yaml
warnings.filterwarnings("ignore")
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

W = Path(r"C:\Users\absamad\cs\widths")
OUT = Path(r"C:\Users\absamad\cs\phase16"); OUT.mkdir(exist_ok=True)
BUDGET, GF_TARGET, GF_CAP, U = 811144, 2.0, 2.707, 0.125   # U = yolo11n width 0.25 x 0.50
BGRID = [round(0.50 + 0.05 * k, 2) for k in range(11)]
BASE = yaml.safe_load((W / "yolo11n_w0p50.yaml").read_text())
STEM, P34, P5 = {0, 1, 2}, {3, 4, 5, 6, 13, 16, 17, 19}, {7, 8, 9, 10, 20, 22}
md = lambda x: int(math.ceil(x / 8) * 8)

def build(a, b):
    cfg = copy.deepcopy(BASE)
    cfg["scales"] = {"n": [0.5, 1.0, 100000]}       # channels below are ABSOLUTE
    layers = cfg["backbone"] + cfg["head"]
    chans = {}
    for i, L in enumerate(layers):
        args = L[3]
        if L[2] in ("Conv", "C3k2", "SPPF", "C2PSA"):
            f = U if i in STEM else (U * a if i in P34 else U * b)
            args[0] = md(args[0] * f); chans[i] = args[0]
    return cfg, chans

_M = {}
def measure(a, b, tag="lpwa_probe"):
    if tag == "lpwa_probe" and (a, b) in _M:
        return _M[(a, b)]
    cfg, chans = build(a, b)
    p = OUT / ("yolo11n_%s.yaml" % tag)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    net = YOLO(str(p)).model
    r = (p, chans, get_num_params(net), round(get_flops(net, 512), 3))
    if tag == "lpwa_probe":
        _M[(a, b)] = r
    return r

# sanity: a = 1.0, b = 1.0 must reproduce w050 exactly
_, _, n0, g0 = measure(1.0, 1.0)
assert n0 == BUDGET, "absolute-channel rebuild of w050 gives %d params, expected %d" % (n0, BUDGET)
print("rebuild check: absolute-channel w050 = %d params, %.3f GFLOPs (matches)" % (n0, g0))

grid = []
def feasible(a, b, cap):
    _, _, n, g = measure(a, b); grid.append({"a": a, "b": b, "params": n, "GFLOPs": g})
    return n <= BUDGET and g <= cap

def max_a(b, cap):
    """largest a on the 0.01 grid in [1.00, 3.00] with feasible(a, b); params and FLOPs rise with a."""
    lo, hi = 100, 300
    if not feasible(lo / 100, b, cap): return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid / 100, b, cap): lo = mid
        else: hi = mid - 1
    return lo / 100

target_met = True
cand = {b: max_a(b, GF_TARGET) for b in BGRID}
cand = {b: a for b, a in cand.items() if a is not None and a > 1.0}
if not cand:
    target_met = False
    cand = {0.50: max_a(0.50, GF_CAP - 1e-9)}
a = max(cand.values())
b = max(bb for bb, aa in cand.items() if aa == a)
print("per-b max a under %s GFLOPs: %s" % (GF_TARGET if target_met else GF_CAP, cand))
p, chans, n, g = measure(a, b, "lpwa50")
assert n <= BUDGET and g < GF_CAP, (n, g)
(OUT / "yolo11n_lpwa_probe.yaml").unlink(missing_ok=True)
uni = {i: md(L[3][0] * U) for i, L in enumerate(BASE["backbone"] + BASE["head"])
       if L[2] in ("Conv", "C3k2", "SPPF", "C2PSA")}
import hashlib
rec = {"rule": __doc__, "a": a, "b": b, "params": n, "GFLOPs_512": g, "budget_params": BUDGET,
       "gflops_target": GF_TARGET, "gflops_target_met": target_met, "gflops_cap": GF_CAP,
       "w050_GFLOPs_512": 1.549, "yaml": str(p),
       "yaml_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
       "channels": {str(i): {"lpwa50": chans[i], "uniform_w050": uni[i],
                             "group": "stem" if i in STEM else ("P3/P4" if i in P34 else "P5")}
                    for i in sorted(chans)},
       "search_grid": grid}
json.dump(rec, open(OUT / "p16_design.json", "w"), indent=1)
print("LPWA-50: a=%.2f b=%.2f params=%d (budget %d) GFLOPs=%.3f (uniform 1.549, target %.1f met=%s, cap %.3f)"
      % (a, b, n, BUDGET, g, GF_TARGET, target_met, GF_CAP))
for i in sorted(chans):
    print("  layer %2d %-6s uniform %4d -> lpwa %4d" % (i, rec["channels"][str(i)]["group"], uni[i], chans[i]))
print("DESIGNDONE")
