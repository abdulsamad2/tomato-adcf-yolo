"""Phase 15 step 0: freeze the width right-sizing protocol BEFORE any Phase 15 training.

Re-builds the four audited width architectures (no training, no weights loaded), checks them
against widths/width_audit.json, and writes an immutable pre-registration record:
  phase15/p15_prereg.json  (machine-readable rules + sha256 of every width yaml and script)
  phase15/p15_prereg.md    (human-readable copy of the same rules)

ABORTS if a width yaml no longer reproduces the audited params / GFLOPs, or if any Phase 15
run directory already contains a checkpoint (the rules must predate every result).
"""
import hashlib, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\absamad\cs")
REPO = BASE / "tomato-adcf-yolo"
W = BASE / "widths"
OUT = BASE / "phase15"; OUT.mkdir(exist_ok=True)
RUNS = REPO / "runs_p15"
WIDTHS = {"w100": ("1.00x", "yolo11n_w1p00.yaml"), "w075": ("0.75x", "yolo11n_w0p75.yaml"),
          "w050": ("0.50x", "yolo11n_w0p50.yaml"), "w035": ("0.35x", "yolo11n_w0p35.yaml")}
SCRIPTS = ["p15_freeze.py", "p15_run.py", "p15_eval.py", "p15_stats.py", "p15_bench.py"]

def die(m):
    print("ABORT:", m); sys.exit(2)

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

if (OUT / "p15_prereg.json").exists():
    die("pre-registration already frozen; refusing to overwrite " + str(OUT / "p15_prereg.json"))
if RUNS.exists() and any(RUNS.rglob("ckpt_ep*_step*.pt")):
    die("Phase 15 checkpoints already exist; rules can no longer be frozen prospectively")

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops
audit = json.loads((W / "width_audit.json").read_text())["candidates"]
arch = {}
for key, (lbl, fn) in WIDTHS.items():
    net = YOLO(str(W / fn)).model
    p, g = get_num_params(net), round(get_flops(net, 512), 3)
    a = audit[lbl]
    if p != a["params"] or abs(g - a["GFLOPs_512"]) > 1e-3:
        die(f"{lbl}: rebuilt params {p} / GFLOPs {g} != audit {a['params']} / {a['GFLOPs_512']}")
    arch[key] = {"label": lbl, "yaml": str(W / fn), "yaml_sha256": sha(W / fn),
                 "params": p, "params_M": round(p / 1e6, 4), "GFLOPs_512": g,
                 "params_ratio_vs_w100": a["params_ratio_vs_1.00x"],
                 "GFLOPs_ratio_vs_w100": a["GFLOPs_ratio_vs_1.00x"]}
    print(f"{key} {lbl}: {p} params, {g} GFLOPs@512 - matches audit")

R = {
 "phase": 15, "title": "Leakage-aware width right-sizing (uniform channel width, from scratch)",
 "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
 "question": ("How much detector width is required when model selection uses source-INDEPENDENT "
              "rather than source-exposed performance?"),
 "scope_note": ("Separately authorised deployment/right-sizing study. The Phase 12-14 mechanism "
                "search stays closed. No new module of any kind in this phase."),
 "architectures": arch,
 "architecture_rule": "Depth and topology identical; only the YOLO11n width multiplier changes.",
 "initialisation": ("Random initialisation for ALL four widths, including 1.00x. No pretrained "
                    "tensor is loaded (yaml-built model, pretrained=False). Rationale: pretrained "
                    "transfer is 98.21% / 7.94% / 15.22% / 22.93% of params for 1.00x / 0.75x / "
                    "0.50x / 0.35x and would confound width with initialisation quality."),
 "seeds": [0, 1, 2, 3, 4],
 "n_runs": 20,
 "seed_rule": "Same five seeds for every width. No seed added post hoc to any width.",
 "run_order": "seed-major: (seed0: w100,w075,w050,w035), (seed1: ...), ...",
 "training": {"source": "configs/experiments.yaml [train], identical to runs_p4/r512 except init",
              "data": "data/tomato_village (leakage-free source-family split, 10080 train files)",
              "imgsz": 512, "batch": 16, "nbs": 64, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
              "cos_lr": False, "epochs": 39, "steps_per_epoch": 630, "total_updates": 24570,
              "warmup_updates": 933, "close_mosaic_epochs": 5, "patience": "1e6 (disabled)",
              "multi_scale": False, "rect": False, "deterministic": True,
              "augmentation": "unchanged Ultralytics defaults (as runs_p4/r512)"},
 "schedule_gate": ("Before step 1 of every run the trainer's computed schedule (steps/epoch, epochs, "
                   "total updates, warmup updates, mosaic-close epoch) must equal "
                   "runs_p4/r512/seed0/schedule.json, else the run aborts (exit 3)."),
 "init_gate": ("Before training, every run checks (a) built params == audited params and (b) for "
               "w100, the first conv weight differs from yolo11n.pt (no silent pretrained load)."),
 "checkpoints": {"epoch10": 6300, "epoch19": 11970, "epoch29": 18270, "epoch39": 24570},
 "primary_checkpoint": 24570,
 "secondary_checkpoint": 18270,
 "test_access": "NONE. The clean test split is not read by any Phase 15 script.",
 "evaluation": {
   "arms": ("Phase 9 frozen arms over the official-val union (2439 images): arm B = "
            "source-INDEPENDENT (213 families, 411 files), arm A = source-EXPOSED (1047 families, "
            "2028 files). Manifests/hashes: phase9/arm_freeze_record.json."),
   "inference": "imgsz 512, conf 0.001, NMS IoU 0.7, max_det 300, top-100 per image",
   "scorer": "pycocotools COCO AP@[.50:.95], maxDets 100, 8 categories (as Phase 9/12)",
   "decomposition": ("Phase 12 matching rules: conf 0.25, greedy one-to-one class-agnostic match at "
                     "IoU 0.50; strict 0.75"),
   "also": "clean-val (268 originals) Ultralytics val at every checkpoint, descriptive only"},
 "primary_endpoint": "six-class AP50-95 on arm B (source-independent) at 24570 updates",
 "primary_test": {
   "type": "non-inferiority, paired across the 5 seeds",
   "contrast": "Delta(w) = AP_B(w) - AP_B(w100), per seed",
   "margin_AP_points": 1.5,
   "hypotheses": "H0(w): mean Delta(w) <= -1.5  vs  H1(w): mean Delta(w) > -1.5, w in {w075,w050,w035}",
   "test": "one-sided paired t, df=4: t = (mean Delta + 1.5) / SE",
   "multiplicity": ("Holm step-down across the three reduced-width comparisons, familywise "
                    "alpha 0.05 (one-sided)."),
   "rule": ("w is NON-INFERIOR iff its Holm-adjusted one-sided p < 0.05 AND its "
            "multiplicity-adjusted one-sided lower confidence bound > -1.5, where the bound for "
            "the hypothesis at Holm rank k (k=1 smallest p) uses level alpha/(3-k+1); a "
            "Bonferroni alpha/3 simultaneous bound is also reported as the conservative check. "
            "Testing stops at the first non-rejected rank (standard Holm)."),
   "interpretation_guard": ("Failure to establish non-inferiority is NOT evidence of inferiority "
                            "and is reported as 'non-inferiority not established'."),
   "margin_rationale": ("Median paired 5-seed SD of arm-B AP differences across the 45 existing "
                        "Phase 9 model pairs is 0.84 AP; at +/-1.0 equivalence would only be "
                        "attainable for |mean diff| < 0.20. Margin -1.5 specified by the user "
                        "before any Phase 15 run."),
   "not_evidence": "A non-significant difference is never reported as equivalence."},
 "secondary_tests": {
   "tost_equivalence": "two-sided TOST at +/-1.5 AP, paired t df=4, per width vs w100, Holm over 3 widths",
   "trajectories": "all four checkpoints + per-epoch clean-val curves, descriptive",
   "superiority": ("one-sided paired t that w > w100 on arm B, Holm over 3 widths "
                   "(Outcome D screen)"),
   "family_bootstrap": ("1000 resamples of source families (arms A and B independently, all files "
                        "of a family together); paired seed-mean Delta and interaction per width"),
   "reported_at": [18270, 24570]},
 "secondary_endpoints": ["six-class AP50, AP75, APs, AR100 (A and B)",
                         "class-agnostic AP50-95, AP50, AP75 (A and B)",
                         "localization recall @IoU .50 and .75 (A and B)",
                         "conditional class correctness (A and B)",
                         "false positives per family (A and B)",
                         "per-class AP (A and B)"],
 "exposure_interaction": {"Delta_ind": "metric(w, B) - metric(w100, B)",
                          "Delta_exp": "metric(w, A) - metric(w100, A)",
                          "I": "Delta_ind - Delta_exp, per seed; paired t + family bootstrap"},
 "convergence_diagnostic": {
   "statistic": "g(w) = AP_B(w, 24570) - AP_B(w, 18270), per seed",
   "flag": ("w is 'still improving' iff mean g(w) > 0.5 AP AND one-sided paired t p < 0.05"),
   "consequence": ("A flagged width is reported as FIXED-BUDGET performance, never as an "
                   "asymptotic capacity limit."),
   "extension_rule": ("No schedule extension inside Phase 15. Any extension is a new, separately "
                      "frozen phase that must apply the same longer schedule to ALL four widths. "
                      "No rescue training of any single width.")},
 "resource_endpoints": ["trainable params", "GFLOPs @512", "serialized fp16 weight size",
                        "peak inference VRAM (batch 1 and 16)", "GPU batch-1 latency",
                        "GPU batch-16 throughput", "CPU batch-1 latency"],
 "resource_note": ("GPU speed-up is claimed only if measured. Untrained timing 8.90 ms (0.35x) vs "
                   "9.13 ms (1.00x) at batch 16 is treated as no meaningful latency gain."),
 "decision_rule": ("Among reduced widths that pass corrected non-inferiority, select the SMALLEST "
                   "by parameter count. If none pass, report the observed accuracy-size Pareto "
                   "frontier without claiming preserved accuracy."),
 "integrity_rules": ["The clean test split is not inspected during development.",
                     "The margin and all rules are not changed after any result is seen.",
                     "Best-validation checkpoints are never the primary outcome; the primary is the "
                     "fixed 24570-update checkpoint.",
                     "No edge-device speed claim is made from desktop GPU timings."],
 "decision_rules": {
   "A": "w035 non-inferior -> select 0.35x; STOP architecture experimentation.",
   "B": ("w050 non-inferior, w035 not -> frontier lies between 0.35x and 0.50x; authorise design "
         "of ONE matched-budget localization-preserving asymmetric-width candidate (separate "
         "frozen phase)."),
   "C": ("w050 not non-inferior -> report the Pareto frontier; architecture work only with a "
         "mechanism-based budget-redistribution hypothesis; no module invention."),
   "D": ("any narrower width superior to w100 on arm B (Holm one-sided p < 0.05) -> report as a "
         "capacity-generalization result and test whether arm A gives the opposite sign. "
         "D can co-occur with A/B/C.")},
 "kd": "No knowledge distillation in Phase 15.",
 "script_sha256": {}}
for s in SCRIPTS:
    p = BASE / s
    R["script_sha256"][s] = sha(p) if p.exists() else "NOT YET WRITTEN AT FREEZE"

json.dump(R, open(OUT / "p15_prereg.json", "w"), indent=1)

md = ["# Phase 15 pre-registration: leakage-aware width right-sizing", "",
      f"Frozen {R['frozen_at']}, before any Phase 15 training. Machine-readable copy: "
      "`p15_prereg.json`.", "", "## Question", R["question"], "", R["scope_note"], "",
      "## Architectures (from scratch, identical depth/topology)", "",
      "| key | width | params | GFLOPs@512 | yaml sha256 |", "|---|---|---|---|---|"]
for k, a in arch.items():
    md.append(f"| {k} | {a['label']} | {a['params_M']} M | {a['GFLOPs_512']} | `{a['yaml_sha256'][:16]}` |")
md += ["", "## Training", R["initialisation"], "",
       "; ".join(f"{k}={v}" for k, v in R["training"].items()), "",
       "Seeds " + str(R["seeds"]) + ". " + R["seed_rule"], "", R["schedule_gate"], "", R["init_gate"],
       "", "Checkpoints " + str(R["checkpoints"]) + ". **Primary 24570**, secondary 18270. "
       + R["test_access"], "", "## Primary test"]
md += ["- " + k + ": " + str(v) for k, v in R["primary_test"].items()]
md += ["", "## Secondary"] + ["- " + k + ": " + str(v) for k, v in R["secondary_tests"].items()]
md += ["- endpoints: " + "; ".join(R["secondary_endpoints"]),
       "- exposure interaction: " + json.dumps(R["exposure_interaction"]), "",
       "## Convergence"] + ["- " + k + ": " + v for k, v in R["convergence_diagnostic"].items()]
md += ["", "## Resources", "- " + "; ".join(R["resource_endpoints"]), "- " + R["resource_note"],
       "", "## Decision rule", R["decision_rule"], "", "## Integrity rules"]
md += ["- " + x for x in R["integrity_rules"]]
md += ["", "## Interpretation of outcomes"] + ["- **" + k + "** " + v for k, v in R["decision_rules"].items()]
md += ["", R["kd"], ""]
(OUT / "p15_prereg.md").write_text("\n".join(md), encoding="utf-8")
print("FREEZEDONE", OUT / "p15_prereg.json")
