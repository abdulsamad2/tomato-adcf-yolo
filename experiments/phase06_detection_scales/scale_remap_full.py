"""Full initialization remap: body layer indices AND Detect branch indices. CPU only.

B keeps reference detection branches (P3,P4) = A branches 0,1.
C keeps (P4,P5) = A branches 1,2.
The final class conv never matches (80 COCO classes vs 8) in ANY variant, including the
reference - that is the harmless newly-initialised detection output.
"""
import json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

OUT = Path(r"C:\Users\absamad\cs\scales")
Y = json.loads((OUT / "variant_yamls.json").read_text())
SPEC = {
 "A":      {"body": {}, "det_old": 23, "det_new": 23, "branches": {0: 0, 1: 1, 2: 2}},
 "B":      {"body": {}, "det_old": 23, "det_new": 20, "branches": {0: 0, 1: 1}},
 "C":      {"body": {19: 14, 20: 15, 21: 16, 22: 17}, "det_old": 23, "det_new": 18,
            "branches": {1: 0, 2: 1}},
 "C_lean": {"body": {19: 13, 20: 14, 21: 15, 22: 16}, "det_old": 23, "det_new": 17,
            "branches": {1: 0, 2: 1}},
}


def main():
    from ultralytics import YOLO
    ck = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
    src = ck["model"].float().state_dict()
    res = {}
    for name, sp in SPEC.items():
        net = YOLO(Y[name]).model.float()
        tgt = net.state_dict()
        tot = sum(v.numel() for v in tgt.values())
        dn = sp["det_new"]

        remapped = {}
        for k, v in src.items():
            p = k.split(".")
            if len(p) > 1 and p[0] == "model" and p[1].isdigit():
                old = int(p[1])
                if old == sp["det_old"]:
                    if len(p) > 3 and p[2] in ("cv2", "cv3") and p[3].isdigit():
                        ob = int(p[3])
                        nb = {v2: k2 for k2, v2 in sp["branches"].items()}
                        inv = {o: n for n, o in {nn: oo for oo, nn in sp["branches"].items()}.items()}
                        tgt_b = sp["branches"].get(ob)
                        if tgt_b is None:
                            continue
                        p = ["model", str(dn), p[2], str(tgt_b)] + p[4:]
                    else:
                        p = ["model", str(dn)] + p[2:]
                elif old in sp["body"]:
                    p = ["model", str(sp["body"][old])] + p[2:]
            remapped[".".join(p)] = v

        t = pr = 0
        miss = []
        for k, v in tgt.items():
            s = remapped.get(k)
            if s is not None and s.shape == v.shape:
                t += 1; pr += v.numel()
            else:
                miss.append(k)
        body_miss = [k for k in miss if not k.startswith(f"model.{dn}.")]
        det_miss = [k for k in miss if k.startswith(f"model.{dn}.")]
        bmp = sum(tgt[k].numel() for k in body_miss)
        dmp = sum(tgt[k].numel() for k in det_miss)
        res[name] = {"target_params": int(tot), "transferred_tensors": t,
                     "transferred_params": int(pr), "transferred_pct": round(100 * pr / tot, 2),
                     "body_new_params": int(bmp), "body_new_pct": round(100 * bmp / tot, 3),
                     "body_new_layers": sorted({k.split(".")[1] for k in body_miss}),
                     "detect_new_params": int(dmp), "detect_new_pct": round(100 * dmp / tot, 3),
                     "detect_new_tensors": len(det_miss)}
        print(f"{name}: full-remap transfer {res[name]['transferred_pct']}% | "
              f"body new {res[name]['body_new_pct']}% ({bmp:,}) layers {res[name]['body_new_layers']} | "
              f"detect new {res[name]['detect_new_pct']}% ({dmp:,})")
    json.dump(res, open(OUT / "scale_remap_full.json", "w"), indent=1)
    print("FULLDONE")


if __name__ == "__main__":
    main()
