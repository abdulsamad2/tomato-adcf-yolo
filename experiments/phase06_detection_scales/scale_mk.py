"""Phase 6 architecture audit: build the detection-scale variants. No training.

Reference head indices (yolo11.yaml):
 11 Upsample | 12 Concat[-1,6] | 13 C3k2 512 | 14 Upsample | 15 Concat[-1,4] | 16 C3k2 256 -> P3/8
 17 Conv 256 s2 | 18 Concat[-1,13] | 19 C3k2 512 -> P4/16 | 20 Conv 512 s2 | 21 Concat[-1,10]
 22 C3k2 1024 -> P5/32 | 23 Detect[16,19,22]

B  (P3+P4): delete 20,21,22 -> Detect[16,19].  Indices 0-19 UNCHANGED (tail removal).
C  (P4+P5): delete 14,15,16,17,18 -> old-19 now fed by 13. Indices after 13 shift by 5.
C_lean    : C, also dropping old-19 (13 becomes the P4 output). One fewer neck block.
B_disc/C_disc: disconnect-only controls - identical topology to A, only the Detect input list
               changes. Used to separate "real removal" from "ignoring an output".
"""
import copy, json
from pathlib import Path
import yaml, ultralytics

ULR = Path(ultralytics.__file__).parent
BASE = yaml.safe_load((ULR / "cfg" / "models" / "11" / "yolo11.yaml").read_text(encoding="utf-8"))
OUT = Path(r"C:\Users\absamad\cs\scales"); OUT.mkdir(exist_ok=True)
NC = 8

UP = [-1, 1, "nn.Upsample", [None, 2, "nearest"]]
def C3k2(ch, shortcut): return [-1, 2, "C3k2", [ch, shortcut]]
def CONV(ch): return [-1, 1, "Conv", [ch, 3, 2]]
def CAT(i):  return [[-1, i], 1, "Concat", [1]]

HEADS = {
 "A": ([UP, CAT(6), C3k2(512, False), UP, CAT(4), C3k2(256, False), CONV(256), CAT(13),
        C3k2(512, False), CONV(512), CAT(10), C3k2(1024, True)], [16, 19, 22]),
 "B": ([UP, CAT(6), C3k2(512, False), UP, CAT(4), C3k2(256, False), CONV(256), CAT(13),
        C3k2(512, False)], [16, 19]),
 "C": ([UP, CAT(6), C3k2(512, False), C3k2(512, False), CONV(512), CAT(10),
        C3k2(1024, True)], [14, 17]),
 "C_lean": ([UP, CAT(6), C3k2(512, False), CONV(512), CAT(10), C3k2(1024, True)], [13, 16]),
 "B_disc": ([UP, CAT(6), C3k2(512, False), UP, CAT(4), C3k2(256, False), CONV(256), CAT(13),
             C3k2(512, False), CONV(512), CAT(10), C3k2(1024, True)], [16, 19]),
 "C_disc": ([UP, CAT(6), C3k2(512, False), UP, CAT(4), C3k2(256, False), CONV(256), CAT(13),
             C3k2(512, False), CONV(512), CAT(10), C3k2(1024, True)], [19, 22]),
}

written = {}
for name, (head, det) in HEADS.items():
    cfg = copy.deepcopy(BASE)
    cfg["nc"] = NC
    cfg["scales"] = {"n": [0.50, 0.25, 1024]}
    cfg["head"] = [list(h) for h in head] + [[det, 1, "Detect", ["nc"]]]
    p = OUT / f"yolo11n_{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None), encoding="utf-8")
    written[name] = str(p)
    n_backbone = len(cfg["backbone"])
    print(f"{name}: head modules {len(head)} (+Detect), detect inputs {det}, "
          f"total module rows {n_backbone + len(head) + 1}")
json.dump(written, open(OUT / "variant_yamls.json", "w"), indent=1)
print("WRITTEN")
