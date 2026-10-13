"""Can variant C's lost neck pretraining be recovered by index remapping? CPU only.

C deletes reference head layers 14-18, so every later layer is renumbered and its state_dict
keys stop matching even where the tensor shapes are unchanged. This measures how much comes
back under the explicit correspondence old 19,20,21,22 -> new 14,15,16,17.
"""
import json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

OUT = Path(r"C:\Users\absamad\cs\scales")
Y = json.loads((OUT / "variant_yamls.json").read_text())
REMAP = {"C": {19: 14, 20: 15, 21: 16, 22: 17},
         "C_lean": {19: 13, 20: 14, 21: 15, 22: 16},
         "B": {}, "A": {}}


def main():
    from ultralytics import YOLO
    ck = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
    src = ck["model"].float().state_dict()
    res = {}
    for name, mp in REMAP.items():
        net = YOLO(Y[name]).model.float()
        tgt = net.state_dict()
        tot = sum(v.numel() for v in tgt.values())
        det_idx = len(net.model) - 1

        def transferred(use_remap):
            keys = {}
            for k, v in src.items():
                parts = k.split(".")
                if use_remap and len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
                    old = int(parts[1])
                    if old in mp:
                        parts = ["model", str(mp[old])] + parts[2:]
                keys[".".join(parts)] = v
            t = p = 0
            miss = []
            for k, v in tgt.items():
                s = keys.get(k)
                if s is not None and s.shape == v.shape:
                    t += 1; p += v.numel()
                else:
                    miss.append(k)
            return t, p, miss

        t0, p0, _ = transferred(False)
        t1, p1, miss1 = transferred(True)
        body_miss = [k for k in miss1 if not k.startswith(f"model.{det_idx}.")]
        body_miss_p = sum(tgt[k].numel() for k in body_miss)
        byl = {}
        for k in body_miss:
            i = k.split(".")[1]
            byl[i] = byl.get(i, 0) + tgt[k].numel()
        res[name] = {
            "naive_tensors": t0, "naive_params": int(p0), "naive_pct": round(100 * p0 / tot, 2),
            "remapped_tensors": t1, "remapped_params": int(p1), "remapped_pct": round(100 * p1 / tot, 2),
            "remap_applied": {str(k): v for k, v in mp.items()},
            "body_newly_init_params_after_remap": int(body_miss_p),
            "body_newly_init_pct_after_remap": round(100 * body_miss_p / tot, 3),
            "body_newly_init_by_layer": {k: int(v) for k, v in sorted(byl.items(), key=lambda x: -x[1])},
            "target_params": int(tot)}
        print(f"{name}: naive {res[name]['naive_pct']}% -> remapped {res[name]['remapped_pct']}% "
              f"| body still new {res[name]['body_newly_init_pct_after_remap']}% "
              f"({res[name]['body_newly_init_params_after_remap']:,} params) "
              f"layers {list(byl.keys())}")
    json.dump(res, open(OUT / "scale_remap.json", "w"), indent=1)
    print("REMAPDONE")


if __name__ == "__main__":
    main()
