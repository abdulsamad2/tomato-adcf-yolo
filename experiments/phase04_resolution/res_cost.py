"""Resolution cost curve for YOLO11n: GFLOPs, batch-1 and batch-16 latency, inference memory.
Idle GPU, fp32, 200 warmup + 1000 measured iterations, identical conditions at every resolution.
No training. Weights are the trained Phase 3 final checkpoint (8-class head).
"""
import json, statistics as st, time
from pathlib import Path
import torch

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
W = REPO / "runs_p3" / "yolo11n" / "seed0" / "weights" / "ckpt_ep39_step24570.pt"
RES = [320, 416, 512, 640, 960]
WARMUP, ITERS = 200, 1000


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    assert W.exists(), W
    y = YOLO(str(W))
    net = y.model.float().eval().cuda()
    out = {"params_M": round(get_num_params(net) / 1e6, 4), "weights": str(W),
           "gpu": torch.cuda.get_device_name(0), "warmup": WARMUP, "iters": ITERS,
           "precision": "fp32", "resolutions": {}}
    for sz in RES:
        assert sz % 32 == 0, sz
        rec = {"GFLOPs": round(get_flops(net, sz), 3), "batches": {}}
        for bs in (1, 16):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.rand(bs, 3, sz, sz, device="cuda")
            with torch.no_grad():
                for _ in range(WARMUP):
                    net(x)
                torch.cuda.synchronize()
                ts = []
                for _ in range(ITERS):
                    t0 = time.perf_counter(); net(x); torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
            rec["batches"][str(bs)] = {
                "ms_mean": round(st.mean(ts), 3), "ms_sd": round(st.stdev(ts), 3),
                "img_per_s": round(bs * 1000.0 / st.mean(ts), 1),
                "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
            del x
        out["resolutions"][str(sz)] = rec
        print(f"{sz}: {rec['GFLOPs']:.2f} GFLOPs  b1 {rec['batches']['1']['ms_mean']:.2f} ms  "
              f"b16 {rec['batches']['16']['ms_mean']:.2f} ms "
              f"({rec['batches']['16']['img_per_s']:.0f} img/s, "
              f"{rec['batches']['16']['peak_mem_MB']:.0f} MB)", flush=True)
    json.dump(out, open(REPO / "results" / "resolution_cost.json", "w"), indent=1)
    print("COSTDONE")


if __name__ == "__main__":
    main()
