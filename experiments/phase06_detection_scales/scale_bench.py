"""Phase 6 latency re-measurement with an explicit idle gate and a repeated reference arm.

The first attempt was contaminated: the GPU was at 100% utilisation / 22.5 GB from another
session's workload. This run verifies idleness before and after, records GPU state alongside
every measurement, and measures the reference arm twice (first and last) so drift is visible.
"""
import json, statistics as st, subprocess, time, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

OUT = Path(r"C:\Users\absamad\cs\scales")
Y = json.loads((OUT / "variant_yamls.json").read_text())
RES, WARMUP, ITERS = 512, 200, 1000
SEQ = ["A", "B", "C", "C_lean", "B_disc", "C_disc", "A"]   # A repeated last


def gpu():
    q = ("utilization.gpu,memory.used,temperature.gpu,clocks.sm,"
         "clocks_throttle_reasons.active")
    o = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    p = [x.strip() for x in o.split(",")]
    return {"util_pct": int(p[0].split()[0]), "mem_MB": int(p[1].split()[0]),
            "temp_C": int(p[2]), "sm_MHz": int(p[3].split()[0]), "throttle": p[4]}


def main():
    from ultralytics import YOLO
    g0 = gpu()
    if g0["util_pct"] > 5 or g0["mem_MB"] > 1500:
        print("ABORT_NOT_IDLE " + json.dumps(g0)); return

    res = {"gate_before": g0, "imgsz": RES, "warmup": WARMUP, "iters": ITERS,
           "sequence": SEQ, "runs": []}
    for pos, name in enumerate(SEQ):
        net = YOLO(Y[name]).model.float().eval().cuda()
        rec = {"pos": pos, "variant": name, "gpu_before": gpu(), "batches": {}}
        for bs in (1, 16):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.rand(bs, 3, RES, RES, device="cuda")
            with torch.no_grad():
                for _ in range(WARMUP):
                    net(x)
                torch.cuda.synchronize()
                ts = []
                for _ in range(ITERS):
                    t0 = time.perf_counter(); net(x); torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
            ts_s = sorted(ts)
            rec["batches"][str(bs)] = {
                "ms_mean": round(st.mean(ts), 3), "ms_sd": round(st.stdev(ts), 3),
                "ms_median": round(st.median(ts), 3),
                "ms_p05": round(ts_s[int(0.05 * len(ts_s))], 3),
                "ms_p95": round(ts_s[int(0.95 * len(ts_s))], 3),
                "img_per_s": round(bs * 1000.0 / st.mean(ts), 1),
                "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
            del x
        rec["gpu_after"] = gpu()
        res["runs"].append(rec)
        b1, b16 = rec["batches"]["1"], rec["batches"]["16"]
        print(f"[{pos}] {name:<7} b1 {b1['ms_mean']:6.2f}±{b1['ms_sd']:<5.2f} "
              f"(med {b1['ms_median']:6.2f}) b16 {b16['ms_mean']:6.2f}±{b16['ms_sd']:<5.2f} "
              f"(med {b16['ms_median']:6.2f}) | gpu {rec['gpu_before']['util_pct']}% "
              f"{rec['gpu_before']['sm_MHz']}MHz {rec['gpu_before']['temp_C']}C", flush=True)
        del net

    res["gate_after"] = gpu()
    a1 = res["runs"][0]["batches"]["16"]["ms_median"]
    a2 = res["runs"][-1]["batches"]["16"]["ms_median"]
    res["reference_drift_pct"] = round(100 * (a2 - a1) / a1, 2)
    res["reference_repeat_ok"] = abs(res["reference_drift_pct"]) < 5.0
    print(f"reference A b16 median first={a1:.2f} last={a2:.2f} "
          f"drift={res['reference_drift_pct']:+.2f}% ok={res['reference_repeat_ok']}")
    json.dump(res, open(OUT / "scale_latency.json", "w"), indent=1)
    print("BENCHDONE")


if __name__ == "__main__":
    main()
