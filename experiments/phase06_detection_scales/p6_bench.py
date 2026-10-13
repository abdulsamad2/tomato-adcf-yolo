"""Phase 6 efficiency on the TRAINED models, idle-gated, reference arm measured twice."""
import json, statistics as st, subprocess, time, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = REPO / "results" / "p6_efficiency.json"
W = {"A": REPO / "runs_p4" / "r512" / "seed0" / "weights" / "ckpt_ep39_step24570.pt",
     "B": REPO / "runs_p6" / "B" / "seed0" / "weights" / "ckpt_ep39_step24570.pt",
     "C": REPO / "runs_p6" / "C" / "seed0" / "weights" / "ckpt_ep39_step24570.pt"}
SEQ = ["A", "B", "C", "A"]
RES, WARMUP, ITERS = 512, 200, 1000


def gpu():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    p = [x.strip() for x in o.split(",")]
    return {"util_pct": int(p[0].split()[0]), "mem_MB": int(p[1].split()[0]),
            "temp_C": int(p[2]), "sm_MHz": int(p[3].split()[0])}


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    if OUT.exists():
        print("REFUSING: exists"); return
    g0 = gpu()
    if g0["util_pct"] > 5 or g0["mem_MB"] > 1500:
        print("ABORT_NOT_IDLE " + json.dumps(g0)); return
    res = {"gate_before": g0, "imgsz": RES, "warmup": WARMUP, "iters": ITERS,
           "weights": {k: str(v) for k, v in W.items()}, "sequence": SEQ, "runs": [], "static": {}}
    for arm in ("A", "B", "C"):
        m = YOLO(str(W[arm]))
        net = m.model.float().eval()
        nconv = sum(1 for x in net.modules() if isinstance(x, torch.nn.Conv2d))
        res["static"][arm] = {"params_M": round(get_num_params(net) / 1e6, 4),
                              "GFLOPs_512": round(get_flops(net, RES), 3),
                              "conv2d": nconv, "n_modules": len(net.model),
                              "strides": [int(s) for s in net.model[-1].stride.tolist()]}
    for pos, arm in enumerate(SEQ):
        m = YOLO(str(W[arm]))
        net = m.model.float().eval().fuse().cuda()
        nconv_f = sum(1 for x in net.modules() if isinstance(x, torch.nn.Conv2d))
        rec = {"pos": pos, "arm": arm, "fused_conv2d": nconv_f, "gpu_before": gpu(), "batches": {}}
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
            s_ = sorted(ts)
            rec["batches"][str(bs)] = {"ms_mean": round(st.mean(ts), 3), "ms_sd": round(st.stdev(ts), 3),
                                       "ms_median": round(st.median(ts), 3),
                                       "ms_p05": round(s_[int(0.05 * len(s_))], 3),
                                       "ms_p95": round(s_[int(0.95 * len(s_))], 3),
                                       "img_per_s": round(bs * 1000.0 / st.mean(ts), 1),
                                       "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
            del x
        rec["gpu_after"] = gpu()
        res["runs"].append(rec)
        b1, b16 = rec["batches"]["1"], rec["batches"]["16"]
        print(f"[{pos}] {arm} fusedconv={nconv_f} b1 {b1['ms_median']:.2f} b16 {b16['ms_median']:.2f} "
              f"mem {b16['peak_mem_MB']:.0f}MB gpu {rec['gpu_before']['util_pct']}%", flush=True)
        del net
    res["gate_after"] = gpu()
    a = [r for r in res["runs"] if r["arm"] == "A"]
    f, l = a[0]["batches"]["16"]["ms_median"], a[-1]["batches"]["16"]["ms_median"]
    res["reference_drift_pct"] = round(100 * (l - f) / f, 2)
    res["reference_repeat_ok"] = abs(res["reference_drift_pct"]) < 5.0
    print(f"reference A b16 median first={f:.2f} last={l:.2f} drift={res['reference_drift_pct']:+.2f}% "
          f"ok={res['reference_repeat_ok']}")
    json.dump(res, open(OUT, "w"), indent=1)
    print("P6BENCHDONE")


if __name__ == "__main__":
    main()
