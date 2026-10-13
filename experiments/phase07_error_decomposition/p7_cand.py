"""Phase 7 part 4: build every plausible candidate, measure statics + initialization, then
benchmark the feasible ones fused on a verified-idle GPU. No training.

Candidates follow the resource decomposition, not the reverse:
  REF      reference YOLO11n-512, nc=8, reg_max=16
  R8       reg_max 16 -> 8  (box branch is 88% of the Detect head and sized entirely by reg_max)
  NC6      nc 8 -> 6        (only 6 populated classes)
  P5SLIM   layer 22 C3k2 1024->512  (single largest layer: 14.62% of parameters)
  S32HALF  layers 7,8,9,10,20,22 halved (the stride-32 block: 61% of parameters, 20% of MACs)
All keep P3+P4+P5 and standard convolutions only.
"""
import copy, json, statistics as st, subprocess, time, warnings
from pathlib import Path
import torch, yaml, ultralytics
warnings.filterwarnings("ignore")

OUT = Path(r"C:\Users\absamad\cs\phase7")
ULR = Path(ultralytics.__file__).parent
BASE = yaml.safe_load((ULR / "cfg" / "models" / "11" / "yolo11.yaml").read_text(encoding="utf-8"))
RES, WARMUP, ITERS = 512, 200, 1000


def cfg_for(name):
    c = copy.deepcopy(BASE)
    c["nc"] = 6 if name == "NC6" else 8
    c["scales"] = {"n": [0.50, 0.25, 1024]}
    if name == "P5SLIM":
        c["head"][11][3] = [512, True]                     # layer 22
    if name == "R8":
        c["reg_max"] = 8   # parse_model reads reg_max as a top-level YAML key
    if name == "S32HALF":
        for i, a in ((7, [512, 3, 2]), (8, [512, True]), (9, [512, 5]), (10, [512])):
            c["backbone"][i][3] = a
        c["head"][9][3] = [256, 3, 2]                      # layer 20
        c["head"][11][3] = [512, True]                     # layer 22
    return c


def build(name):
    from ultralytics import YOLO
    p = OUT / f"cand_{name}.yaml"
    p.write_text(yaml.safe_dump(cfg_for(name), sort_keys=False, default_flow_style=None), encoding="utf-8")
    m = YOLO(str(p))
    return m, str(p)


def transfer(net, src_shapes, src_vals):
    """Honest split: exact-key shape match only. Channel slicing is NOT counted as pretrained."""
    tgt = net.state_dict()
    tot = int(sum(v.numel() for v in tgt.values()))
    dn = len(net.model) - 1
    ok_p = body_p = neck_p = det_p = 0
    body_l, neck_l, det_t = set(), set(), 0
    for k, v in tgt.items():
        hit = k in src_shapes and src_shapes[k] == tuple(v.shape)
        idx = int(k.split(".")[1]) if k.split(".")[0] == "model" and k.split(".")[1].isdigit() else -1
        if hit:
            ok_p += v.numel()
        elif idx == dn:
            det_p += v.numel(); det_t += 1
        elif 0 <= idx <= 10:
            body_p += v.numel(); body_l.add(idx)
        else:
            neck_p += v.numel(); neck_l.add(idx)
    return {"target_params": tot, "pretrained_unchanged_params": int(ok_p),
            "pretrained_pct": round(100 * ok_p / tot, 2),
            "sliced_params": 0, "random_backbone_params": int(body_p),
            "random_backbone_pct": round(100 * body_p / tot, 3),
            "random_backbone_layers": sorted(body_l),
            "random_neck_params": int(neck_p), "random_neck_pct": round(100 * neck_p / tot, 3),
            "random_neck_layers": sorted(neck_l),
            "random_detect_params": int(det_p), "random_detect_pct": round(100 * det_p / tot, 3),
            "random_detect_tensors": det_t}


def gpu():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    p = [x.strip() for x in o.split(",")]
    return {"util_pct": int(p[0].split()[0]), "mem_MB": int(p[1].split()[0]), "temp_C": int(p[2])}


def main():
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    ck = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
    sd = ck["model"].float().state_dict()
    src_shapes = {k: tuple(v.shape) for k, v in sd.items()}

    names = ["REF", "R8", "NC6", "P5SLIM", "S32HALF"]
    res = {"imgsz": RES, "candidates": {}}
    nets = {}
    for n in names:
        m, p = build(n)
        net = m.model.float().eval()
        det = net.model[-1]
        rec = {"yaml": p, "n_modules": len(net.model),
               "conv2d": sum(1 for x in net.modules() if isinstance(x, torch.nn.Conv2d)),
               "params": int(get_num_params(net)), "params_M": round(get_num_params(net) / 1e6, 4),
               "GFLOPs_512": round(get_flops(net, RES), 3),
               "nc": int(det.nc), "reg_max": int(det.reg_max),
               "strides": [int(s) for s in det.stride.tolist()],
               "detect_in_ch": [int(b[0].conv.in_channels) for b in det.cv2],
               "cv2_mid": [int(b[0].conv.out_channels) for b in det.cv2],
               "cv2_out": [int(b[-1].out_channels) for b in det.cv2],
               "cv3_mid": [int(b[1][-1].conv.out_channels) for b in det.cv3],
               "out_ch_per_location": 4 * int(det.reg_max) + int(det.nc),
               "detect_params": int(sum(x.numel() for x in det.parameters())),
               "box_branch_params": int(sum(x.numel() for b in det.cv2 for x in b.parameters())),
               "cls_branch_params": int(sum(x.numel() for b in det.cv3 for x in b.parameters())),
               "transfer": transfer(net, src_shapes, sd)}
        res["candidates"][n] = rec
        nets[n] = net
    ref = res["candidates"]["REF"]
    for n in names:
        r = res["candidates"][n]
        r["params_ratio"] = round(r["params"] / ref["params"], 4)
        r["GFLOPs_ratio"] = round(r["GFLOPs_512"] / ref["GFLOPs_512"], 4)
        print(f"{n:<8} mods={r['n_modules']} conv={r['conv2d']} par={r['params_M']:.4f}M "
              f"({r['params_ratio']:.3f}x) GF={r['GFLOPs_512']:.3f} ({r['GFLOPs_ratio']:.3f}x) "
              f"nc={r['nc']} rm={r['reg_max']} c2={r['cv2_mid']} out/loc={r['out_ch_per_location']} "
              f"| xfer={r['transfer']['pretrained_pct']}% randBB={r['transfer']['random_backbone_pct']}% "
              f"randNeck={r['transfer']['random_neck_pct']}% randDet={r['transfer']['random_detect_pct']}%",
              flush=True)

    g0 = gpu()
    res["gate_before"] = g0
    if g0["util_pct"] > 5 or g0["mem_MB"] > 1500:
        print("ABORT_NOT_IDLE " + json.dumps(g0))
        json.dump(res, open(OUT / "candidates.json", "w"), indent=1)
        return
    seq = ["REF"] + [n for n in names if n != "REF"] + ["REF"]
    res["latency"] = []
    for pos, n in enumerate(seq):
        net = build(n)[0].model.float().eval().fuse().cuda()
        rec = {"pos": pos, "cand": n, "gpu_before": gpu(), "batches": {}}
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
            rec["batches"][str(bs)] = {"ms_mean": round(st.mean(ts), 3), "ms_sd": round(st.stdev(ts), 3),
                                       "ms_median": round(st.median(ts), 3),
                                       "img_per_s": round(bs * 1000.0 / st.mean(ts), 1),
                                       "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
            del x
        rec["gpu_after"] = gpu()
        res["latency"].append(rec)
        print(f"[{pos}] {n:<8} b1 {rec['batches']['1']['ms_median']:.2f} "
              f"b16 {rec['batches']['16']['ms_median']:.2f} "
              f"mem {rec['batches']['16']['peak_mem_MB']:.0f}MB gpu {rec['gpu_before']['util_pct']}%", flush=True)
        del net
    res["gate_after"] = gpu()
    rr = [x for x in res["latency"] if x["cand"] == "REF"]
    f, l = rr[0]["batches"]["16"]["ms_median"], rr[-1]["batches"]["16"]["ms_median"]
    res["reference_drift_pct"] = round(100 * (l - f) / f, 2)
    res["reference_repeat_ok"] = abs(res["reference_drift_pct"]) < 5.0
    print(f"reference b16 median first={f:.2f} last={l:.2f} drift={res['reference_drift_pct']:+.2f}% "
          f"ok={res['reference_repeat_ok']}")
    json.dump(res, open(OUT / "candidates.json", "w"), indent=1)
    print("CANDDONE")


if __name__ == "__main__":
    main()
