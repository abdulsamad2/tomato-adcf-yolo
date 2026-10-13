import json
import numpy as np
from pathlib import Path
P = Path(r"C:\Users\absamad\cs\phase9")
f = P / "match_cache" / "runs_p4__r512__pNone__seed0.npz"
d = np.load(f, allow_pickle=True)
print("keys:", list(d.keys()))
for k in d.keys():
    a = d[k]
    print("  %-22s shape=%-16s dtype=%s  sample=%s" % (k, a.shape, a.dtype, str(a.ravel()[:4])[:90]))
g = json.loads((P / "coco_gt_union.json").read_text())
print("\ncoco_gt: images=%d anns=%d cats=%s" % (len(g["images"]), len(g["annotations"]),
      [(c["id"], c["name"]) for c in g["categories"]]))
print("ann sample:", {k: v for k, v in g["annotations"][0].items() if k != "segmentation"})
print("img sample:", g["images"][0])
