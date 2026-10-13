"""Phase 16 pre-design audit (no training): where do the parameters and FLOPs of each width live?

Splits every audited width into backbone stages, neck nodes and the Detect head (box branch cv2,
class branch cv3, DFL) per pyramid level, so an asymmetric allocation can be designed against
real numbers rather than assumptions.
"""
import json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

W = Path(r"C:\Users\absamad\cs\widths")
OUT = Path(r"C:\Users\absamad\cs\phase16"); OUT.mkdir(exist_ok=True)
FILES = {"w100": "yolo11n_w1p00.yaml", "w075": "yolo11n_w0p75.yaml",
         "w050": "yolo11n_w0p50.yaml", "w035": "yolo11n_w0p35.yaml"}
# YOLO11 layer index -> role
ROLE = {0: "stem", 1: "stem", 2: "bb_P2", 3: "bb_P3", 4: "bb_P3", 5: "bb_P4", 6: "bb_P4",
        7: "bb_P5", 8: "bb_P5", 9: "bb_P5_SPPF", 10: "bb_P5_C2PSA",
        11: "neck", 12: "neck", 13: "neck_td_P4", 14: "neck", 15: "neck", 16: "neck_out_P3",
        17: "neck_bu_P3to4", 18: "neck", 19: "neck_out_P4", 20: "neck_bu_P4to5", 21: "neck",
        22: "neck_out_P5"}
n = lambda m: sum(p.numel() for p in m.parameters())

res = {}
for k, f in FILES.items():
    net = YOLO(str(W / f)).model
    tot = n(net); det = net.model[-1]
    rows = {}
    for i, m in enumerate(net.model[:-1]):
        rows[ROLE[i]] = rows.get(ROLE[i], 0) + n(m)
    for lvl, (b, c) in enumerate(zip(det.cv2, det.cv3)):
        rows["head_box_P%d" % (lvl + 3)] = n(b)
        rows["head_cls_P%d" % (lvl + 3)] = n(c)
    rows["head_dfl"] = n(det.dfl)
    grp = {"backbone_P5_total": sum(v for r, v in rows.items() if r.startswith("bb_P5")),
           "backbone_P3P4": rows["bb_P3"] + rows["bb_P4"],
           "neck_total": sum(v for r, v in rows.items() if r.startswith("neck")),
           "head_box_total": sum(v for r, v in rows.items() if r.startswith("head_box")),
           "head_cls_total": sum(v for r, v in rows.items() if r.startswith("head_cls"))}
    res[k] = {"params": tot, "GFLOPs_512": round(get_flops(net, 512), 3),
              "head_box_mid_channels": [int(b[0].conv.out_channels) for b in det.cv2],
              "head_cls_mid_channels": [int(c[0][0].conv.out_channels) if isinstance(c[0], torch.nn.Sequential)
                                        else int(c[0].conv.out_channels) for c in det.cv3],
              "head_in_channels": [int(b[0].conv.in_channels) for b in det.cv2],
              "rows": rows, "groups": grp,
              "group_pct": {g: round(100 * v / tot, 1) for g, v in grp.items()}}
    print("%s  %.3fM  GFLOPs %.2f | box mid %s cls mid %s in %s" % (k, tot / 1e6, res[k]["GFLOPs_512"],
          res[k]["head_box_mid_channels"], res[k]["head_cls_mid_channels"], res[k]["head_in_channels"]))
    print("     " + "  ".join("%s %.1f%%" % (g, p) for g, p in res[k]["group_pct"].items()))
json.dump(res, open(OUT / "p16_budget_audit.json", "w"), indent=1)
print("BUDGETDONE")
