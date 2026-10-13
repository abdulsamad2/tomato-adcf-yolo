"""Phase 7 part 2: Detect-head internals, the floor analysis, and the exact A-vs-B difference."""
import json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
OUT = Path(r"C:\Users\absamad\cs\phase7")
RES = 512
W = {"A": REPO / "runs_p4" / "r512" / "seed0" / "weights" / "ckpt_ep39_step24570.pt",
     "B": REPO / "runs_p6" / "B" / "seed0" / "weights" / "ckpt_ep39_step24570.pt"}


def macs(m, out_hw):
    if isinstance(m, torch.nn.Conv2d):
        oh, ow = out_hw
        return m.in_channels // m.groups * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * oh * ow
    return 0


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params
    res = {}
    for tag, wp in W.items():
        net = YOLO(str(wp)).model.float()
        det = net.model[-1]
        tot = int(get_num_params(net))
        grids = [RES // int(s) for s in det.stride.tolist()]
        e = {"nc": int(det.nc), "reg_max": int(det.reg_max), "nl": int(det.nl),
             "strides": [int(s) for s in det.stride.tolist()], "grids": grids,
             "in_channels": [int(b[0].conv.in_channels) for b in det.cv2],
             "cv2_mid": [int(b[0].conv.out_channels) for b in det.cv2],
             "cv2_out": [int(b[-1].out_channels) for b in det.cv2],
             "cv3_mid": [int(b[1][-1].conv.out_channels) for b in det.cv3],
             "cv3_out": [int(b[-1].out_channels) for b in det.cv3],
             "total_model_params": tot, "branches": {}}
        sums = {"cv2": [0, 0], "cv3": [0, 0], "dfl": [0, 0]}
        for i, g in enumerate(grids):
            br = {}
            for which, seq in (("cv2", det.cv2[i]), ("cv3", det.cv3[i])):
                p = sum(x.numel() for x in seq.parameters())
                mm = 0
                for sub in seq.modules():
                    mm += macs(sub, (g, g))
                br[which] = {"params": int(p), "gmacs": round(mm / 1e9, 5)}
                sums[which][0] += int(p); sums[which][1] += mm
            e["branches"][f"P{i+3 if tag=='A' else ''}_{det.stride[i]:.0f}"] = br
        e["dfl_params"] = int(sum(x.numel() for x in det.dfl.parameters()))
        e["dfl_learnable"] = int(sum(x.numel() for x in det.dfl.parameters() if x.requires_grad))
        e["totals"] = {"cv2_box_params": sums["cv2"][0], "cv2_box_gmacs": round(sums["cv2"][1] / 1e9, 5),
                       "cv3_cls_params": sums["cv3"][0], "cv3_cls_gmacs": round(sums["cv3"][1] / 1e9, 5),
                       "detect_params": int(sum(x.numel() for x in det.parameters())),
                       "detect_pct_of_model": round(100 * sum(x.numel() for x in det.parameters()) / tot, 2)}
        # floor arithmetic, straight from the Detect constructor
        ch = e["in_channels"]
        e["floors"] = {
          "c2_formula": "max(16, ch//4, reg_max*4)",
          "c2_terms": [{"ch": c, "ch_div4": c // 4, "regmax4": 4 * e["reg_max"],
                        "chosen": max(16, c // 4, 4 * e["reg_max"]),
                        "binding": ("reg_max*4" if 4 * e["reg_max"] >= max(16, c // 4) else
                                    ("ch//4" if c // 4 >= 16 else "16"))} for c in ch],
          "c3_formula": "max(ch[0], min(nc, 100))",
          "c3_terms": {"ch0": ch[0], "nc": e["nc"], "chosen": max(ch[0], min(e["nc"], 100)),
                       "binding": "ch[0]" if ch[0] >= min(e["nc"], 100) else "nc"},
          "output_channels_per_location": 4 * e["reg_max"] + e["nc"]}
        res[tag] = e
        t = e["totals"]
        print(f"{tag}: Detect {t['detect_params']:,} params ({t['detect_pct_of_model']}% of model) | "
              f"box {t['cv2_box_params']:,}p/{t['cv2_box_gmacs']:.4f}G | "
              f"cls {t['cv3_cls_params']:,}p/{t['cv3_cls_gmacs']:.4f}G | "
              f"dfl learnable {e['dfl_learnable']} | c2={e['cv2_mid']} c3={e['cv3_mid']}", flush=True)
        print(f"   c2 binding: {[x['binding'] for x in e['floors']['c2_terms']]}  "
              f"c3 binding: {e['floors']['c3_terms']['binding']}  "
              f"out/location: {e['floors']['output_channels_per_location']}", flush=True)

    # exact A - B layer diff
    dec = json.loads((OUT / "resource_decomposition.json").read_text())
    la = {r["idx"]: r for r in dec["models"]["A"]["layers"]}
    lb = {r["idx"]: r for r in dec["models"]["B"]["layers"]}
    diff = {"removed_layers": [], "detect_change": {}, "unchanged_layers": []}
    for i, r in la.items():
        if i == 23:
            continue
        if i in lb:
            diff["unchanged_layers"].append({"idx": i, "params": r["params"], "gmacs": r["gmacs"],
                                             "same_params": r["params"] == lb[i]["params"]})
        else:
            diff["removed_layers"].append({"idx": i, "type": r["type"], "out_shape": r["out_shape"],
                                           "stride": r["stride_at_output"], "params": r["params"],
                                           "gmacs": r["gmacs"], "conv2d": r["conv2d"]})
    dA, dB = la[23], lb[20]
    diff["detect_change"] = {"A_params": dA["params"], "B_params": dB["params"],
                             "A_gmacs": dA["gmacs"], "B_gmacs": dB["gmacs"],
                             "A_conv2d": dA["conv2d"], "B_conv2d": dB["conv2d"],
                             "delta_params": dA["params"] - dB["params"],
                             "delta_gmacs": round(dA["gmacs"] - dB["gmacs"], 5)}
    rp = sum(x["params"] for x in diff["removed_layers"])
    rm = sum(x["gmacs"] for x in diff["removed_layers"])
    tp, tm = dec["models"]["A"]["total_params"], dec["models"]["A"]["total_gmacs"]
    diff["summary"] = {
      "neck_layers_removed_params": rp, "neck_layers_removed_gmacs": round(rm, 5),
      "detect_branch_removed_params": diff["detect_change"]["delta_params"],
      "detect_branch_removed_gmacs": diff["detect_change"]["delta_gmacs"],
      "total_param_saving": rp + diff["detect_change"]["delta_params"],
      "total_gmac_saving": round(rm + diff["detect_change"]["delta_gmacs"], 5),
      "param_saving_pct": round(100 * (rp + diff["detect_change"]["delta_params"]) / tp, 2),
      "gmac_saving_pct": round(100 * (rm + diff["detect_change"]["delta_gmacs"]) / tm, 2),
      "share_of_param_saving_from_neck": round(100 * rp / (rp + diff["detect_change"]["delta_params"]), 1),
      "share_of_gmac_saving_from_neck": round(100 * rm / (rm + diff["detect_change"]["delta_gmacs"]), 1)}
    res["A_minus_B"] = diff
    print("\nA-B: removed neck layers " + ", ".join(f"{x['idx']}({x['type']},{x['params']:,}p)"
                                                   for x in diff["removed_layers"]))
    print("     " + json.dumps(diff["summary"], indent=1))
    json.dump(res, open(OUT / "detect_and_diff.json", "w"), indent=1)
    print("DETDONE")


if __name__ == "__main__":
    main()
