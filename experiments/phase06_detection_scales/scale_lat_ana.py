import json
import numpy as np
from scipy import stats
from pathlib import Path
S = Path(r"C:\Users\absamad\cs\scales")
L = json.loads((S / "scale_latency.json").read_text())
A = json.loads((S / "scale_audit.json").read_text())["variants"]

per = {}
for r in L["runs"]:
    per.setdefault(r["variant"], []).append(r)
rows = {}
for v, rs in per.items():
    b1 = float(np.mean([x["batches"]["1"]["ms_median"] for x in rs]))
    b16 = float(np.mean([x["batches"]["16"]["ms_median"] for x in rs]))
    b1m = float(np.mean([x["batches"]["1"]["ms_mean"] for x in rs]))
    b16m = float(np.mean([x["batches"]["16"]["ms_mean"] for x in rs]))
    sd1 = float(np.mean([x["batches"]["1"]["ms_sd"] for x in rs]))
    sd16 = float(np.mean([x["batches"]["16"]["ms_sd"] for x in rs]))
    mem = float(np.mean([x["batches"]["16"]["peak_mem_MB"] for x in rs]))
    rows[v] = {"n_runs": len(rs), "b1_med": round(b1, 3), "b16_med": round(b16, 3),
               "b1_mean": round(b1m, 3), "b16_mean": round(b16m, 3),
               "b1_sd": round(sd1, 3), "b16_sd": round(sd16, 3),
               "b16_img_per_s": round(16000.0 / b16, 1), "peak_mem_MB": round(mem, 1),
               "conv2d": A[v]["n_conv2d"], "modules": A[v]["n_modules"],
               "GFLOPs": A[v]["GFLOPs_512"], "params_M": A[v]["params_M"]}
ref = rows["A"]
for v in rows:
    rows[v]["b16_ratio"] = round(rows[v]["b16_med"] / ref["b16_med"], 4)
    rows[v]["b1_ratio"] = round(rows[v]["b1_med"] / ref["b1_med"], 4)

order = ["A", "B", "C", "C_lean", "B_disc", "C_disc"]
lat = np.array([rows[v]["b16_med"] for v in order])
out = {"per_variant": rows, "gate_before": L["gate_before"], "gate_after": L["gate_after"],
       "reference_drift_pct": L["reference_drift_pct"], "predictors": {}}
for name in ("conv2d", "modules", "GFLOPs", "params_M"):
    x = np.array([rows[v][name] for v in order], dtype=float)
    pr, pp = stats.pearsonr(x, lat)
    sr, sp = stats.spearmanr(x, lat)
    out["predictors"][name] = {"pearson_r": round(float(pr), 4), "pearson_p": float(pp),
                               "spearman_rho": round(float(sr), 4), "spearman_p": float(sp)}
    print(f"{name:<9} vs batch-16 median latency: Pearson r={pr:+.3f} (p={pp:.4f})  "
          f"Spearman rho={sr:+.3f} (p={sp:.4f})")
print()
for v in order:
    r_ = rows[v]
    print(f"{v:<8} conv={r_['conv2d']:>3} GF={r_['GFLOPs']:.3f} par={r_['params_M']:.3f}M | "
          f"b1 {r_['b1_med']:.2f} ({r_['b1_ratio']:.3f}x) b16 {r_['b16_med']:.2f} "
          f"({r_['b16_ratio']:.3f}x) {r_['b16_img_per_s']:.0f} img/s mem {r_['peak_mem_MB']:.0f}MB")
json.dump(out, open(S / "scale_latency_summary.json", "w"), indent=1)
print("ANADONE")
