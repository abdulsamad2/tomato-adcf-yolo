"""Pre-run addendum to the clean-test pre-registration (written before results exist)."""
import hashlib, json, time
from pathlib import Path
O = Path(r"C:\Users\absamad\cs\clean_test")
assert not (O / "results.json").exists() and not (O / "RUN_LOCK").exists(), "clean test already started"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "applies_to_prereg_sha256": sha(O / "prereg.json"),
     "approved_by_user": True,
     "wording_rule": "Describe the test split as 'held out from all Phase 4-17 models and decisions', never as 'never opened'.",
     "interpretation_rule": ("Whatever the clean-test ranking is, no model changes or new training phases follow from it. "
                             "Every outcome is reported as observed (e.g. pretrained 1.00x best; distilled 0.75x above or "
                             "below from-scratch 1.00x; results below expectation). The clean test is confirmation, not development.")}
json.dump(a, open(O / "prereg_addendum.json", "w"), indent=1); print(json.dumps(a, indent=1))
