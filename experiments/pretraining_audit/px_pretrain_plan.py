"""Exploratory pretraining-inflation analysis: write the plan BEFORE any r512 checkpoint is scored.

Not pre-registered in Phase 15/16; written here before scoring so the contrasts cannot be chosen
after seeing results. No training. Clean test not touched.
"""
import hashlib, json, time
from pathlib import Path
B = Path(r"C:\Users\absamad\cs"); OUT = B / "pretrain_audit"; OUT.mkdir(exist_ok=True)
p = OUT / "plan.json"
assert not p.exists(), "plan already written"
assert not (OUT / "eval_rows.json").exists(), "scoring already started"
plan = {
 "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "EXPLORATORY (not pre-registered)",
 "question": "Does source-exposed evaluation also exaggerate the value of COCO pretraining?",
 "pretrained": "runs_p4/r512 seeds 0-4 (yolo11n.pt COCO init, 98.21% of params transferred)",
 "scratch": "runs_p15/w100 seeds 0-4 (identical yolo11n architecture, random init)",
 "identical": "architecture, data, imgsz 512, optimizer, 24570 updates, warmup 933, mosaic close, checkpoint grid, seeds",
 "only_difference": "initialisation",
 "scorer": "p15_eval.py code path (Phase 9 arms, union inference, top-100, pycocotools) applied to r512 checkpoints",
 "contrasts": {"gain_B": "AP_B(pretrained) - AP_B(scratch), per seed",
               "gain_A": "AP_A(pretrained) - AP_A(scratch), per seed",
               "I": "gain_A - gain_B (positive = exposure exaggerates the value of pretraining)",
               "ratio": "gain_A / gain_B"},
 "steps_reported": [24570, 18270, 11970, 6300], "headline_step": 24570,
 "also": "class-agnostic AP, locR50/75, cond class correctness; family bootstrap 1000 for gain_A, gain_B, I",
 "inference": "paired t over seeds (df 4) and family bootstrap reported separately; exploratory, no decision rule"}
json.dump(plan, open(p, "w"), indent=1)
print("PLAN sha256", hashlib.sha256(p.read_bytes()).hexdigest())
