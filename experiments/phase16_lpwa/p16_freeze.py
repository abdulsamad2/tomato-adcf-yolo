"""Phase 16A step 0: freeze the LPWA-50 architecture test BEFORE any Phase 16 training.

Records exact layer widths, params, GFLOPs, per-layer tensor shapes and Detect-head dimensions of
the solved candidate (phase16/p16_design.json), plus every rule. ABORTS if a Phase 16 checkpoint exists.
"""
import hashlib, json, sys, time, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
OUT = BASE / "phase16"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

if (OUT / "p16_prereg.json").exists():
    sys.exit("ABORT: already frozen")
if (REPO / "runs_p16").exists() and any((REPO / "runs_p16").rglob("ckpt_ep*_step*.pt")):
    sys.exit("ABORT: Phase 16 checkpoints already exist")

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops
D = json.loads((OUT / "p16_design.json").read_text())
assert sha(D["yaml"]) == D["yaml_sha256"], "design yaml changed since the solve"
net = YOLO(D["yaml"]).model.eval()
n, g = get_num_params(net), round(get_flops(net, 512), 3)
assert n == D["params"] and abs(g - D["GFLOPs_512"]) < 1e-3, (n, g)

shapes, ys, x = [], [], torch.zeros(1, 3, 512, 512)
with torch.no_grad():
    for i, m in enumerate(net.model[:-1]):
        inp = ys[m.f] if isinstance(m.f, int) and m.f != -1 else (x if m.f == -1 else [ys[j] if j != -1 else x for j in m.f])
        x = m(inp); ys.append(x)
        shapes.append({"idx": i, "type": type(m).__name__, "from": m.f, "out": list(x.shape)})
det = net.model[-1]
head = {"in_channels": [int(b[0].conv.in_channels) for b in det.cv2],
        "box_branch_mid": [int(b[0].conv.out_channels) for b in det.cv2],
        "box_branch_out": [int(b[-1].out_channels) for b in det.cv2],
        "cls_branch_out": [int(c[-1].out_channels) for c in det.cv3],
        "reg_max": int(det.reg_max), "nc": int(det.nc), "strides": [8, 16, 32]}
p15 = json.loads((BASE / "phase15" / "p15_prereg.json").read_text())

R = {
 "phase": "16A", "title": "Localization-Preserving Width Allocation (LPWA-50), one prospectively frozen candidate",
 "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
 "motivation": ("Phase 15: uniform narrowing to 0.50x costs -1.80 AP on arm B; class correctness given a "
                "found box stays ~97% while localization recall@0.50 falls 7.0 points. Budget audit: the class "
                "head is <2% of params and the box branch is already floored at 64 ch, so the reallocatable "
                "budget is P5 (32.8%) -> P3/P4 (9.2%). Hypothesis: at equal params, moving width from the P5 "
                "path to the P3/P4 paths recovers localization and AP."),
 "one_candidate_rule": "Exactly one candidate. No variants, no ratio changes, no tuning after any result.",
 "candidate": {"name": "lpwa50", "yaml": D["yaml"], "yaml_sha256": D["yaml_sha256"], "params": n,
               "GFLOPs_512": g, "a_P3P4": D["a"], "b_P5": D["b"], "layer_channels": D["channels"],
               "tensor_shapes_512": shapes, "detect_head": head, "design_rule": D["rule"],
               "design_record_sha256": sha(OUT / "p16_design.json")},
 "comparators": {
   "U50": {"runs": "runs_p15/w050/seed0-4", "params": 811144, "GFLOPs_512": 1.549,
           "scores": "phase15/eval_rows.json (already computed, unchanged)"},
   "w100": {"runs": "runs_p15/w100/seed0-4", "params": 2591400, "GFLOPs_512": 4.15}},
 "architectures": {"lpwa50": {"label": "LPWA-50", "yaml": D["yaml"], "yaml_sha256": D["yaml_sha256"], "params": n}},
 "training": dict(p15["training"], note="identical to Phase 15 (runner p16_run.py derived from p15_run.py)"),
 "initialisation": "random (pretrained=False), as Phase 15",
 "seeds": [0, 1, 2, 3, 4],
 "seed_rule": "All five seeds run directly; no screening subset. Paired with U50/w100 seeds of the same index.",
 "checkpoints": p15["checkpoints"], "primary_checkpoint": 24570, "secondary_checkpoint": 18270,
 "evaluation": p15["evaluation"],
 "success_criteria": {
   "rule": "LPWA-50 SUCCEEDS iff all three gates pass",
   "G1_primary_AP": ("mean over seeds of Delta_i = AP_B(lpwa50, i) - AP_B(U50, i) at 24570 is >= +0.75 AP. "
                     "The paired 95% t CI (df=4) is reported; statistical significance is NOT required."),
   "G2_mechanism": ("mean over seeds of Delta locR50_B (localization recall @IoU .50, arm B, Phase 12 rules) > 0. "
                    "Mechanistic corroboration; no separate significance test."),
   "G3_resources": "params <= 811,144 AND GFLOPs@512 < 2.707 (target <= 2.0, met at design: %.3f)" % g},
 "secondary": {
   "noninferiority_vs_w100": "Phase 15 rule: one-sided paired t, margin -1.5 AP, arm B @24570 (single comparison, no Holm)",
   "tost_vs_w100": "+/-1.5 AP",
   "superiority_vs_U50": "one-sided paired t p reported (not a gate)",
   "at_18270": "all of the above repeated at 18270",
   "exposure": "Delta on arm A and interaction I = Delta_B - Delta_A vs U50 and vs w100; family bootstrap 1000",
   "decomposition": "class-agnostic AP, AP50/75, locR50/75, conditional class correctness, FP/family",
   "convergence": "Phase 15 rule: still improving iff mean gain 18270->24570 > 0.5 AP and one-sided p < 0.05",
   "resources": "params, GFLOPs, fp16 size, peak VRAM, GPU b1/b16 latency, CPU b1 latency vs U50 and w100"},
 "decision": {"success": "proceed to Phase 17: localization-focused distillation into LPWA-50 (separately frozen)",
              "failure": "STOP architecture work; no second allocation, no module search"},
 "integrity_rules": p15["integrity_rules"] + ["No change to the candidate after any seed is trained."],
 "script_sha256": {}}
for s in ["p16_budget.py", "p16_design.py", "p16_freeze.py", "p16_run.py", "p16_eval.py"]:
    R["script_sha256"][s] = sha(BASE / s) if (BASE / s).exists() else "NOT YET WRITTEN"
json.dump(R, open(OUT / "p16_prereg.json", "w"), indent=1)
md = ["# Phase 16A pre-registration: LPWA-50", "", "Frozen " + R["frozen_at"] + ". Machine-readable: `p16_prereg.json`.",
      "", "## Motivation", R["motivation"], "", "## Candidate (one, frozen)",
      "params %d, GFLOPs@512 %.3f, P3/P4 x%.2f, P5 x%.2f vs uniform 0.50x; head unchanged." % (n, g, D["a"], D["b"]),
      "", "| layer | group | uniform 0.50x | LPWA-50 |", "|---|---|---|---|"]
md += ["| %s | %s | %d | %d |" % (k, v["group"], v["uniform_w050"], v["lpwa50"]) for k, v in D["channels"].items()]
md += ["", "Detect head: " + json.dumps(head), "", "## Training", R["initialisation"] + "; " + R["training"]["note"],
       "Seeds 0-4. " + R["seed_rule"], "", "## Success (all three)"]
md += ["- **%s** %s" % (k, v) for k, v in R["success_criteria"].items()]
md += ["", "## Secondary"] + ["- %s: %s" % (k, v) for k, v in R["secondary"].items()]
md += ["", "## Decision"] + ["- %s: %s" % (k, v) for k, v in R["decision"].items()]
md += ["", "## Integrity"] + ["- " + x for x in R["integrity_rules"]] + [""]
(OUT / "p16_prereg.md").write_text("\n".join(md), encoding="utf-8")
print("candidate %d params %.3f GFLOPs; P3 out %s P4 out %s P5 out %s; head %s"
      % (n, g, shapes[4]["out"], shapes[6]["out"], shapes[10]["out"], head))
print("FREEZEDONE")
