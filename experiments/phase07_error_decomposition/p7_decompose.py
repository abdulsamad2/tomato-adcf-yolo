"""Phase 7 part 1: layer-level resource decomposition of YOLO11n-512. No training.

MACs are counted directly from measured input/output tensor shapes via forward hooks, then
validated against Ultralytics' own get_flops. Roles are assigned from the YAML graph, not guessed.
"""
import json, warnings
from pathlib import Path
import torch, yaml
warnings.filterwarnings("ignore")

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
SCALES = Path(r"C:\Users\absamad\cs\scales")
OUT = Path(r"C:\Users\absamad\cs\phase7"); OUT.mkdir(exist_ok=True)
RES = 512
W_A = REPO / "runs_p4" / "r512" / "seed0" / "weights" / "ckpt_ep39_step24570.pt"
W_B = REPO / "runs_p6" / "B" / "seed0" / "weights" / "ckpt_ep39_step24570.pt"


def macs_of(mod, inp, out):
    if isinstance(mod, torch.nn.Conv2d):
        oh, ow = out.shape[-2:]
        return mod.in_channels // mod.groups * mod.out_channels * mod.kernel_size[0] * mod.kernel_size[1] * oh * ow
    if isinstance(mod, torch.nn.Linear):
        return mod.in_features * mod.out_features
    return 0


def decompose(net, tag):
    recs, hooks = {}, []
    leaf = [(n, m) for n, m in net.named_modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d))]
    top_shape = {}

    def mk(name, mod):
        def h(m, i, o):
            o0 = o[0] if isinstance(o, (list, tuple)) else o
            recs.setdefault(name, {"type": type(m).__name__,
                                   "params": sum(p.numel() for p in m.parameters(recurse=False)),
                                   "macs": 0, "in_shape": None, "out_shape": None, "n_calls": 0})
            r = recs[name]
            i0 = i[0] if isinstance(i, (list, tuple)) else i
            r["in_shape"] = list(i0.shape) if hasattr(i0, "shape") else None
            r["out_shape"] = list(o0.shape) if hasattr(o0, "shape") else None
            r["macs"] += int(macs_of(m, i, o0))
            r["n_calls"] += 1
        return h

    for n, m in leaf:
        hooks.append(m.register_forward_hook(mk(n, m)))

    def mktop(idx, mod):
        def h(m, i, o):
            o0 = o[0] if isinstance(o, (list, tuple)) else o
            if hasattr(o0, "shape"):
                top_shape[idx] = list(o0.shape)
        return h

    for idx, m in enumerate(net.model):
        hooks.append(m.register_forward_hook(mktop(idx, m)))

    net.eval()
    with torch.no_grad():
        net(torch.rand(1, 3, RES, RES))
    for h in hooks:
        h.remove()

    layers = {}
    for name, r in recs.items():
        parts = name.split(".")
        idx = int(parts[1]) if parts[0] == "model" and parts[1].isdigit() else -1
        e = layers.setdefault(idx, {"params": 0, "macs": 0, "conv2d": 0, "bn": 0, "leaves": []})
        e["params"] += r["params"]; e["macs"] += r["macs"]
        if r["type"] == "Conv2d":
            e["conv2d"] += 1
        if r["type"] == "BatchNorm2d":
            e["bn"] += 1
        e["leaves"].append({"name": name, **r})
    return layers, top_shape


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params, get_flops

    out = {"imgsz": RES, "models": {}}
    ck = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
    src = {k: tuple(v.shape) for k, v in ck["model"].float().state_dict().items()}

    for tag, wp in (("A", W_A), ("B", W_B)):
        net = YOLO(str(wp)).model.float()
        ycfg = net.yaml
        layers, shapes = decompose(net, tag)
        tot_p = int(get_num_params(net))
        tot_m = sum(v["macs"] for v in layers.values())
        gflops_ul = get_flops(net, RES)

        # graph: from-fields and consumers
        rows_cfg = [list(x) for x in (ycfg["backbone"] + ycfg["head"])]
        froms, consumers = {}, {}
        for i, r in enumerate(rows_cfg):
            f = r[0]
            f = f if isinstance(f, list) else [f]
            f = [(i - 1 if x == -1 else x) for x in f]
            froms[i] = f
            for x in f:
                consumers.setdefault(x, []).append(i)

        sd = net.state_dict()
        tbl = []
        for idx in sorted(k for k in layers if k >= 0):
            e = layers[idx]
            mod = net.model[idx]
            osh = shapes.get(idx)
            pre = sum(v.numel() for k, v in sd.items()
                      if k.startswith(f"model.{idx}.") and k in src and src[k] == tuple(v.shape))
            tbl.append({
                "idx": idx, "type": type(mod).__name__,
                "from": froms.get(idx), "consumers": consumers.get(idx, []),
                "out_shape": osh,
                "in_ch": (e["leaves"][0]["in_shape"][1] if e["leaves"] and e["leaves"][0]["in_shape"] else None),
                "out_ch": (osh[1] if osh and len(osh) > 2 else None),
                "stride_at_output": (RES // osh[2] if osh and len(osh) > 2 and osh[2] else None),
                "params": e["params"], "params_pct": round(100 * e["params"] / tot_p, 2),
                "macs": e["macs"], "gmacs": round(e["macs"] / 1e9, 4),
                "macs_pct": round(100 * e["macs"] / tot_m, 2),
                "conv2d": e["conv2d"], "bn": e["bn"],
                "pretrained_params": int(pre),
                "pretrained_pct_of_layer": (round(100 * pre / e["params"], 1) if e["params"] else None)})
        out["models"][tag] = {"total_params": tot_p, "total_gmacs": round(tot_m / 1e9, 4),
                              "ultralytics_gflops": gflops_ul,
                              "counting_agreement_pct": round(100 * (tot_m / 1e9) / gflops_ul, 1),
                              "n_modules": len(net.model), "layers": tbl,
                              "detect_strides": [int(s) for s in net.model[-1].stride.tolist()]}
        print(f"{tag}: params {tot_p:,}  my GMACs {tot_m/1e9:.4f}  ultralytics {gflops_ul:.3f}  "
              f"agreement {100*(tot_m/1e9)/gflops_ul:.1f}%  modules {len(net.model)}", flush=True)

    json.dump(out, open(OUT / "resource_decomposition.json", "w"), indent=1)
    print("DECDONE")


if __name__ == "__main__":
    main()
