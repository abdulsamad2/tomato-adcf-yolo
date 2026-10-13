"""Phase 6: instantiate every variant, measure topology/cost/transfer/latency. No training."""
import json, statistics as st, time, warnings
from pathlib import Path
import torch, yaml
warnings.filterwarnings("ignore")

OUT = Path(r"C:\Users\absamad\cs\scales")
Y = json.loads((OUT / "variant_yamls.json").read_text())
RES, WARMUP, ITERS = 512, 200, 1000
ORDER = ["A", "B", "C", "C_lean", "B_disc", "C_disc"]


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    from ultralytics.nn.modules import Detect

    ck = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
    src = {k: v for k, v in ck["model"].float().state_dict().items()}
    src_shapes = {k: tuple(v.shape) for k, v in src.items()}

    res = {"imgsz": RES, "nc": 8, "warmup": WARMUP, "iters": ITERS,
           "pretrained_tensors": len(src),
           "pretrained_params": int(sum(v.numel() for v in src.values())),
           "variants": {}}

    for name in ORDER:
        m = YOLO(Y[name])
        net = m.model
        nmod = len(net.model)
        nconv = sum(1 for x in net.modules() if isinstance(x, torch.nn.Conv2d))
        nbn = sum(1 for x in net.modules() if isinstance(x, torch.nn.BatchNorm2d))
        det = net.model[-1]
        seq = [f"{i}:{type(x).__name__}" for i, x in enumerate(net.model)]

        # strides from a real forward pass
        net.eval()
        with torch.no_grad():
            strides = [int(s) for s in det.stride.tolist()]
        head = {"nl": int(det.nl), "reg_max": int(det.reg_max), "strides": strides,
                "in_channels": [int(b[0].conv.in_channels) for b in det.cv2],
                "cv2_mid": [int(b[0].conv.out_channels) for b in det.cv2],
                "cv3_out": [int(b[-1].out_channels) for b in det.cv3],
                "detect_from": list(det.f) if hasattr(det, "f") else None}

        # pretrained transfer, naive key match
        tgt = net.float().state_dict()
        tgt_params = int(sum(v.numel() for v in tgt.values()))
        match_t = match_p = 0
        newly = []
        for k, v in tgt.items():
            if k in src_shapes and src_shapes[k] == tuple(v.shape):
                match_t += 1; match_p += v.numel()
            else:
                newly.append(k)
        body_new = [k for k in newly if not k.startswith(f"model.{nmod-1}.")]
        head_new = [k for k in newly if k.startswith(f"model.{nmod-1}.")]
        body_new_p = int(sum(tgt[k].numel() for k in body_new))
        head_new_p = int(sum(tgt[k].numel() for k in head_new))

        rec = {"yaml": Y[name], "n_modules": nmod, "n_conv2d": nconv, "n_batchnorm": nbn,
               "module_sequence": seq, "params": int(get_num_params(net)),
               "params_M": round(get_num_params(net) / 1e6, 4),
               "GFLOPs_512": round(get_flops(net, RES), 3), "detect": head,
               "transfer": {"target_tensors": len(tgt), "target_params": tgt_params,
                            "matched_tensors": match_t, "matched_params": int(match_p),
                            "pct_params": round(100 * match_p / tgt_params, 2),
                            "newly_init_tensors": len(newly),
                            "newly_init_body_tensors": len(body_new),
                            "newly_init_body_params": body_new_p,
                            "newly_init_body_pct": round(100 * body_new_p / tgt_params, 3),
                            "newly_init_detect_tensors": len(head_new),
                            "newly_init_detect_params": head_new_p,
                            "newly_init_detect_pct": round(100 * head_new_p / tgt_params, 3),
                            "newly_init_body_names": body_new[:40]}}

        net = net.cuda().eval()
        rec["batches"] = {}
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
            rec["batches"][str(bs)] = {"ms_mean": round(st.mean(ts), 3),
                                       "ms_sd": round(st.stdev(ts), 3),
                                       "img_per_s": round(bs * 1000.0 / st.mean(ts), 1),
                                       "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
            del x
        res["variants"][name] = rec
        print(f"{name}: mods={nmod} conv={nconv} params={rec['params_M']}M "
              f"GF={rec['GFLOPs_512']} strides={strides} headin={head['in_channels']} "
              f"xfer={rec['transfer']['pct_params']}% newbody={rec['transfer']['newly_init_body_pct']}% "
              f"b1={rec['batches']['1']['ms_mean']:.2f} b16={rec['batches']['16']['ms_mean']:.2f} "
              f"mem={rec['batches']['16']['peak_mem_MB']:.0f}MB", flush=True)

    a = res["variants"]["A"]
    for n in ORDER:
        v = res["variants"][n]
        v["params_ratio"] = round(v["params"] / a["params"], 4)
        v["GFLOPs_ratio"] = round(v["GFLOPs_512"] / a["GFLOPs_512"], 4)
        v["b16_latency_ratio"] = round(v["batches"]["16"]["ms_mean"] / a["batches"]["16"]["ms_mean"], 4)
        v["b1_latency_ratio"] = round(v["batches"]["1"]["ms_mean"] / a["batches"]["1"]["ms_mean"], 4)
    json.dump(res, open(OUT / "scale_audit.json", "w"), indent=1)
    print("SCALEDONE")


if __name__ == "__main__":
    main()
