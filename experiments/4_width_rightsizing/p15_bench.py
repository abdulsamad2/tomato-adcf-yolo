"""Phase 15 resources on the TRAINED seed-0 24570-update checkpoints. Idle-gated; the w100
reference is measured first and last to bound drift. Desktop GPU/CPU numbers only: they
support NO edge-device speed claim.
"""
import io, json, statistics as stt, subprocess, time, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
OUT = REPO / "results" / "p15_efficiency.json"
WID = ["w100", "w075", "w050", "w035"]
W = {w: REPO / "runs_p15" / w / "seed0" / "weights" / "ckpt_ep39_step24570.pt" for w in WID}
SEQ = WID + ["w100"]
RES, WARMUP, ITERS, CPU_ITERS = 512, 200, 1000, 100


def gpu():
    o = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu,clocks.sm",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    p = [x.strip() for x in o.split(",")]
    return {"name": p[0], "util_pct": int(p[1].split()[0]), "mem_MB": int(p[2].split()[0]),
            "temp_C": int(p[3]), "sm_MHz": int(p[4].split()[0])}


def timeit(net, x, warm, iters, cuda):
    with torch.no_grad():
        for _ in range(warm): net(x)
        if cuda: torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); net(x)
            if cuda: torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    s_ = sorted(ts)
    return {"ms_mean": round(stt.mean(ts), 3), "ms_sd": round(stt.stdev(ts), 3),
            "ms_median": round(stt.median(ts), 3), "ms_p05": round(s_[int(0.05 * len(s_))], 3),
            "ms_p95": round(s_[int(0.95 * len(s_))], 3),
            "img_per_s": round(x.shape[0] * 1000.0 / stt.median(ts), 1)}


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    if OUT.exists():
        print("REFUSING: exists"); return
    g0 = gpu()
    if g0["util_pct"] > 5 or g0["mem_MB"] > 1500:
        print("ABORT_NOT_IDLE " + json.dumps(g0)); return
    res = {"gate_before": g0, "imgsz": RES, "warmup": WARMUP, "iters": ITERS, "cpu_iters": CPU_ITERS,
           "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
           "weights": {k: str(v) for k, v in W.items()}, "sequence": SEQ, "static": {}, "runs": [],
           "claim_scope": "desktop GPU / CPU only; no edge-device claim"}
    for w in WID:
        net = YOLO(str(W[w])).model.float().eval()
        buf = io.BytesIO(); torch.save({"model": net.half().state_dict()}, buf)
        net = net.float()
        res["static"][w] = {"params": int(get_num_params(net)), "params_M": round(get_num_params(net) / 1e6, 4),
                            "GFLOPs_512": round(get_flops(net, RES), 3),
                            "fp16_state_dict_MB": round(buf.tell() / 1e6, 3),
                            "checkpoint_file_MB": round(W[w].stat().st_size / 1e6, 3)}
    for pos, w in enumerate(SEQ):
        net = YOLO(str(W[w])).model.float().eval().fuse()
        rec = {"pos": pos, "width": w, "gpu_before": gpu(), "gpu": {}, "cpu": {}}
        netc = net.cuda()
        for bs in (1, 16):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.rand(bs, 3, RES, RES, device="cuda")
            r = timeit(netc, x, WARMUP, ITERS, True)
            r["peak_mem_MB"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            rec["gpu"][str(bs)] = r; del x
        netp = netc.cpu()
        rec["cpu"]["1"] = timeit(netp, torch.rand(1, 3, RES, RES), 20, CPU_ITERS, False)
        rec["gpu_after"] = gpu()
        res["runs"].append(rec)
        print(f"[{pos}] {w} gpu b1 {rec['gpu']['1']['ms_median']:.2f}ms  b16 {rec['gpu']['16']['ms_median']:.2f}ms "
              f"({rec['gpu']['16']['img_per_s']:.0f} img/s, {rec['gpu']['16']['peak_mem_MB']:.0f}MB)  "
              f"cpu b1 {rec['cpu']['1']['ms_median']:.1f}ms", flush=True)
        del net, netc, netp
    a = [r for r in res["runs"] if r["width"] == "w100"]
    f, l = a[0]["gpu"]["16"]["ms_median"], a[-1]["gpu"]["16"]["ms_median"]
    res["reference_drift_pct_b16"] = round(100 * (l - f) / f, 2)
    res["reference_repeat_ok"] = abs(res["reference_drift_pct_b16"]) < 5.0
    res["gate_after"] = gpu()
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"reference drift {res['reference_drift_pct_b16']:+.2f}% ok={res['reference_repeat_ok']}")
    print("P15BENCHDONE")


if __name__ == "__main__":
    main()
