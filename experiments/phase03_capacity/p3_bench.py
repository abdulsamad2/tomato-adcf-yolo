"""Phase 3 efficiency characterisation: params, GFLOPs@640, latency at batch 1 and 16.

Idle GPU, 200 warmup iterations then 1000 measured, identical conditions for every model.
Weights are the trained final checkpoints (24,570 updates) so the 8-class head is counted,
not the 80-class COCO head.
"""
import json, time
from pathlib import Path
import torch

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
CK = {
 "yolo11n": REPO / "runs_p3" / "yolo11n" / "seed0" / "weights" / "ckpt_ep39_step24570.pt",
 "yolo11s": REPO / "runs_p2b" / "D" / "seed0" / "weights" / "ckpt_ep39_step24570.pt",
 "yolo11m": REPO / "runs_p3" / "yolo11m" / "seed0" / "weights" / "ckpt_ep39_step24570.pt",
}
WARMUP, ITERS, IMGSZ = 200, 1000, 640


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    out = {}
    for name, w in CK.items():
        assert w.exists(), f"missing {w}"
        y = YOLO(str(w))
        net = y.model.float().eval().cuda()
        rec = {"weights": str(w), "params_M": round(get_num_params(net) / 1e6, 4),
               "GFLOPs_640": round(get_flops(net, IMGSZ), 3), "batches": {}}
        for bs in (1, 16):
            x = torch.rand(bs, 3, IMGSZ, IMGSZ, device="cuda")
            with torch.no_grad():
                for _ in range(WARMUP):
                    net(x)
                torch.cuda.synchronize()
                ts = []
                for _ in range(ITERS):
                    t0 = time.perf_counter()
                    net(x)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
            import statistics as st
            rec["batches"][str(bs)] = {
                "ms_mean": round(st.mean(ts), 3), "ms_sd": round(st.stdev(ts), 3),
                "ms_median": round(st.median(ts), 3),
                "ms_p95": round(sorted(ts)[int(0.95 * len(ts))], 3),
                "img_per_s": round(bs * 1000.0 / st.mean(ts), 1),
                "warmup": WARMUP, "iters": ITERS}
            print(f"{name} b{bs}: {st.mean(ts):.3f}+-{st.stdev(ts):.3f} ms  "
                  f"{bs*1000.0/st.mean(ts):.1f} img/s", flush=True)
        out[name] = rec
        del net, y
        torch.cuda.empty_cache()
    out["_meta"] = {"gpu": torch.cuda.get_device_name(0), "imgsz": IMGSZ,
                    "warmup": WARMUP, "iters": ITERS, "precision": "fp32",
                    "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(out, open(REPO / "results" / "p3_efficiency.json", "w"), indent=1)
    print("BENCHDONE")


if __name__ == "__main__":
    main()
