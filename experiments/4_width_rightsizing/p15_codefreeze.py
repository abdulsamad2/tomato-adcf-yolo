"""Phase 15: hash the analysis code before any Phase 15 metric is computed or viewed."""
import hashlib, json, time
from pathlib import Path

B = Path(r"C:\Users\absamad\cs"); R = B / "tomato-adcf-yolo"
FS = ["p15_run.py", "p15_queue.ps1", "p15_eval.py", "p15_stats.py", "p15_bench.py", "p15_post.ps1"]
p = B / "phase15" / "p15_analysis_freeze.json"
assert not p.exists(), "analysis code already frozen"
rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
       "note": "analysis code frozen while run 1 of 20 is training; no Phase 15 metric computed or viewed",
       "revision": ("v2 supersedes p15_analysis_freeze_v1.json (03:09): p15_stats.py gained the "
                    "P15_TEST_ROOT synthetic self-test hook (paths only, unset in the real run) and "
                    "passed a planted-effect self-test. No rule, margin, endpoint or test changed."),
       "completed_runs_at_freeze": len(list((R / "runs_p15").rglob("p15_done.json"))),
       "prereg_sha256": hashlib.sha256((B / "phase15" / "p15_prereg.json").read_bytes()).hexdigest(),
       "sha256": {f: hashlib.sha256((B / f).read_bytes()).hexdigest() for f in FS},
       "evaluator_control": ("CONTROL r512 seed0 reproduced Phase 9 six-class AP within 0.0021 AP and "
                             "the Phase 12 decomposition exactly")}
json.dump(rec, open(p, "w"), indent=1)
print(json.dumps(rec, indent=1))
