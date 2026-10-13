"""Phase 16A: hash the analysis code before any Phase 15 metric is computed or viewed."""
import hashlib, json, time
from pathlib import Path

B = Path(r"C:\Users\absamad\cs"); R = B / "tomato-adcf-yolo"
FS = ["p16_budget.py", "p16_design.py", "p16_freeze.py", "p16_run.py", "p16_queue.ps1", "p16_eval.py", "p16_stats.py", "p16_bench.py"]
p = B / "phase16" / "p16_analysis_freeze.json"
assert not p.exists(), "analysis code already frozen"
rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
       "note": "analysis code frozen while LPWA-50 seed 0 is training; no Phase 15 metric computed or viewed",
       "completed_runs_at_freeze": len(list((R / "runs_p16").rglob("p15_done.json"))),
       "prereg_sha256": hashlib.sha256((B / "phase16" / "p16_prereg.json").read_bytes()).hexdigest(),
       "sha256": {f: hashlib.sha256((B / f).read_bytes()).hexdigest() for f in FS},
       "evaluator_control": ("p16_eval.py is the Phase 15 scorer with paths changed; p16_stats.py passed a planted +1.0 AP synthetic self-test. "
                             "Phase 15 control (0.0021 AP) carries over")}
json.dump(rec, open(p, "w"), indent=1)
print(json.dumps(rec, indent=1))
