"""Grid granularity, depth invariance, and the resource curve for each width. No training."""
import json, time, warnings
from pathlib import Path
import torch, yaml
warnings.filterwarnings("ignore")

OUTDIR = Path(r"C:\Users\absamad\cs\widths")
A = json.loads((OUTDIR / "width_audit.json").read_text())
NOM = [64, 128, 256, 512, 1024]
N_WIDTH, N_MAXC = 0.25, 1024
RES, WARMUP, ITERS = 512, 200, 1000


def main():
    from ultralytics import YOLO
    from ultralytics.utils.ops import make_divisible
    from ultralytics.utils.torch_utils import get_flops

    # 1. granularity: which requested multipliers produce DISTINCT architectures
    sig2mult = {}
    for i in range(20, 105):
        k = i / 100.0
        sig = tuple(make_divisible(min(c, N_MAXC) * k * N_WIDTH, 8) for c in NOM)
        sig2mult.setdefault(sig, []).append(round(k, 2))
    bands = [{"channels": list(s), "multiplier_range": [min(v), max(v)], "n_multipliers": len(v)}
             for s, v in sorted(sig2mult.items(), key=lambda x: sum(x[0]))]

    # 2. depth invariance + head detail + latency
    out = {"granularity_bands": bands, "variants": {}}
    for lbl in ("1.00x", "0.75x", "0.50x", "0.35x"):
        p = A["candidates"][lbl]["yaml_path"]
        m = YOLO(p)
        net = m.model
        nmod = len(net.model)
        nconv = sum(1 for _ in net.modules() if isinstance(_, torch.nn.Conv2d))
        det = net.model[-1]
        depth_sig = [type(x).__name__ for x in net.model]
        net = net.float().eval().cuda()
        rec = {"n_modules": nmod, "n_conv2d": nconv, "module_sequence": depth_sig,
               "GFLOPs_512": round(get_flops(net, RES), 3), "batches": {}}
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
            import statistics as st
            rec["batches"][str(bs)] = {"ms_mean": round(st.mean(ts), 3), "ms_sd": round(st.stdev(ts), 3),
                                       "img_per_s": round(bs * 1000.0 / st.mean(ts), 1),
                                       "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
            del x
        out["variants"][lbl] = rec
        print(f"{lbl}: modules={nmod} conv2d={nconv} GFLOPs={rec['GFLOPs_512']} "
              f"b1={rec['batches']['1']['ms_mean']:.2f}ms b16={rec['batches']['16']['ms_mean']:.2f}ms "
              f"mem={rec['batches']['16']['peak_mem_MB']:.0f}MB", flush=True)

    seqs = {lbl: out["variants"][lbl]["module_sequence"] for lbl in out["variants"]}
    ref = seqs["1.00x"]
    out["depth_topology_identical_across_widths"] = all(v == ref for v in seqs.values())
    out["n_modules_identical"] = len({out["variants"][l]["n_modules"] for l in out["variants"]}) == 1
    json.dump(out, open(OUTDIR / "width_granularity.json", "w"), indent=1)
    print("depth/topology identical across all widths:", out["depth_topology_identical_across_widths"])
    print("distinct architectures in 0.20-1.04 multiplier sweep:", len(bands))
    for b in bands:
        print(f"   channels {b['channels']}  <- requested {b['multiplier_range'][0]:.2f}-{b['multiplier_range'][1]:.2f}")
    print("GRANDONE")


if __name__ == "__main__":
    main()
