"""Phase 5 initialization audit: how much of yolo11n.pt actually transfers into each width."""
import json, warnings
from pathlib import Path
import torch, yaml
warnings.filterwarnings("ignore")

OUTDIR = Path(r"C:\Users\absamad\cs\widths")
A = json.loads((OUTDIR / "width_audit.json").read_text())
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

ck = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
src = ck["model"].float().state_dict()
src_shapes = {k: tuple(v.shape) for k, v in src.items()}
src_params = sum(v.numel() for v in src.values())

res = {"pretrained": "yolo11n.pt (COCO, nc=80)",
       "pretrained_tensors": len(src), "pretrained_params": int(src_params), "variants": {}}

for lbl in ("1.00x", "0.75x", "0.50x", "0.35x"):
    p = Path(A["candidates"][lbl]["yaml_path"])
    m = YOLO(str(p))                 # fresh 8-class model at this width
    tgt = m.model.float().state_dict()
    tgt_params = sum(v.numel() for v in tgt.values())

    match_t = match_p = 0
    name_only = 0
    for k, v in tgt.items():
        if k in src_shapes:
            name_only += 1
            if src_shapes[k] == tuple(v.shape):
                match_t += 1
                match_p += v.numel()
    # what Ultralytics' own .load() would actually copy
    try:
        m2 = YOLO(str(p))
        before = {k: v.clone() for k, v in m2.model.state_dict().items()}
        m2.load("yolo11n.pt")
        after = m2.model.state_dict()
        changed = sum(1 for k in after if k in before and before[k].shape == after[k].shape
                      and not torch.equal(before[k], after[k]))
        changed_p = sum(after[k].numel() for k in after if k in before
                        and before[k].shape == after[k].shape and not torch.equal(before[k], after[k]))
    except Exception as e:
        changed = changed_p = -1

    res["variants"][lbl] = {
        "target_tensors": len(tgt), "target_params": int(tgt_params),
        "names_present_in_pretrained": name_only,
        "shape_matching_tensors": match_t,
        "shape_matching_params": int(match_p),
        "pct_tensors_transferable": round(100 * match_t / len(tgt), 2),
        "pct_params_transferable": round(100 * match_p / tgt_params, 2),
        "ultralytics_load_changed_tensors": changed,
        "ultralytics_load_changed_params": int(changed_p),
        "pct_params_actually_loaded": round(100 * changed_p / tgt_params, 2) if changed_p >= 0 else None,
    }
    print(f"{lbl}: {res['variants'][lbl]['pct_params_transferable']}% of params shape-match "
          f"({match_t}/{len(tgt)} tensors); .load() copied "
          f"{res['variants'][lbl]['pct_params_actually_loaded']}%")

json.dump(res, open(OUTDIR / "width_transfer.json", "w"), indent=1)
print("TRANSFERDONE")
