"""Phase 5 architecture audit: what YOLO11n width scaling actually does.

No training. Instantiates each candidate as the real 8-class detector and measures channels,
params, GFLOPs at 512, and pretrained-weight transfer.
"""
import copy, json, math, warnings
from pathlib import Path
import torch, yaml
warnings.filterwarnings("ignore")

REPO = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo")
UL = Path(torch.__file__).parent  # placeholder, replaced below
import ultralytics
ULR = Path(ultralytics.__file__).parent
BASE = ULR / "cfg" / "models" / "11" / "yolo11.yaml"
OUTDIR = Path(r"C:\Users\absamad\cs\widths"); OUTDIR.mkdir(exist_ok=True)
NC = 8
IMGSZ = 512
N_DEPTH, N_WIDTH, N_MAXC = 0.50, 0.25, 1024
CAND = {"1.00x": 1.00, "0.75x": 0.75, "0.50x": 0.50, "0.35x": 0.35}

from ultralytics import YOLO
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import get_num_params, get_flops
from ultralytics.nn.modules import Conv, Detect

base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
nominal = sorted({a[0] for sec in ("backbone", "head") for _, _, m, a in
                  [(x[0], x[1], x[2], x[3]) for x in base[sec]]
                  if isinstance(a, list) and a and isinstance(a[0], int)})

out = {"base_yaml": str(BASE), "nc": NC, "imgsz": IMGSZ,
       "yolo11n_scale": {"depth": N_DEPTH, "width": N_WIDTH, "max_channels": N_MAXC},
       "make_divisible": "math.ceil(x/8)*8", "nominal_yaml_channels": nominal,
       "candidates": {}}

# --- what the rounding rule produces, before building anything ---
for lbl, k in CAND.items():
    w = N_WIDTH * k
    rows = {}
    for ch in nominal:
        n_ch = make_divisible(min(ch, N_MAXC) * N_WIDTH, 8)
        v_ch = make_divisible(min(ch, N_MAXC) * k * N_WIDTH, 8)
        rows[str(ch)] = {"yolo11n": n_ch, "variant": v_ch,
                         "effective_ratio": round(v_ch / n_ch, 4),
                         "clipped_by_max_channels": bool(ch > N_MAXC)}
    out["candidates"][lbl] = {"requested_multiplier": k, "yaml_width": round(w, 6),
                              "channel_map": rows}

ref_sd = None
for lbl, k in CAND.items():
    w = N_WIDTH * k
    cfg = copy.deepcopy(base)
    cfg["nc"] = NC
    cfg["scales"] = {"n": [N_DEPTH, round(w, 6), N_MAXC]}  # key must match the letter Ultralytics parses from the filename
    p = OUTDIR / f"yolo11n_w{lbl.replace('.','p').replace('x','')}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    m = YOLO(str(p))
    net = m.model
    params = get_num_params(net)
    gf = get_flops(net, IMGSZ)

    # realised channels per module
    chans = []
    for i, mod in enumerate(net.model):
        o = getattr(mod, "c2", None) or getattr(mod, "cv2", None)
        oc = None
        if isinstance(mod, Conv):
            oc = mod.conv.out_channels
        elif isinstance(mod, Detect):
            oc = None
        else:
            for sub in mod.modules():
                if isinstance(sub, torch.nn.Conv2d):
                    oc = sub.out_channels
        chans.append({"idx": i, "type": type(mod).__name__, "out_channels": oc})

    det = net.model[-1]
    head = {"nc": int(det.nc), "reg_max": int(det.reg_max), "no": int(det.no),
            "in_channels": [int(x) for x in det.ch] if hasattr(det, "ch") else None,
            "cv2_mid": [int(b[0].conv.out_channels) for b in det.cv2],
            "cv3_out": [int(b[-1].out_channels) for b in det.cv3]}
    try:
        head["in_channels"] = [int(b[0].conv.in_channels) for b in det.cv2]
    except Exception:
        pass

    e = out["candidates"][lbl]
    e.update({"yaml_path": str(p), "params": int(params),
              "params_M": round(params / 1e6, 4), "GFLOPs_512": round(gf, 3),
              "n_layers": len(net.model), "detect_head": head,
              "stage_channels": [c for c in chans if c["out_channels"]]})

    sd = net.state_dict()
    if lbl == "1.00x":
        ref_sd = {kk: vv.shape for kk, vv in sd.items()}
    out["candidates"][lbl]["_sd_shapes"] = {kk: list(vv.shape) for kk, vv in sd.items()}

ref = out["candidates"]["1.00x"]
for lbl in CAND:
    e = out["candidates"][lbl]
    e["params_ratio_vs_1.00x"] = round(e["params"] / ref["params"], 4)
    e["GFLOPs_ratio_vs_1.00x"] = round(e["GFLOPs_512"] / ref["GFLOPs_512"], 4)

json.dump(out, open(OUTDIR / "width_audit.json", "w"), indent=1)
for lbl in CAND:
    e = out["candidates"][lbl]
    print(f"{lbl}: yaml_width={e['yaml_width']} params={e['params_M']}M "
          f"({e['params_ratio_vs_1.00x']:.3f}x) GFLOPs@512={e['GFLOPs_512']} "
          f"({e['GFLOPs_ratio_vs_1.00x']:.3f}x) head_in={e['detect_head']['in_channels']}")
    print("    channel map " + "  ".join(
        f"{c}->{v['variant']}(n={v['yolo11n']},r={v['effective_ratio']})"
        for c, v in e["channel_map"].items()))
print("AUDITDONE")
