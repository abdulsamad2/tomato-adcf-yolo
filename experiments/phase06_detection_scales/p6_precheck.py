import json, sys
sys.path.insert(0, r"C:\Users\absamad\cs")
import importlib.util
spec = importlib.util.spec_from_file_location("p6run", r"C:\Users\absamad\cs\p6_run.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for arm in ("B", "C"):
    m, loaded, stats = mod.build_remapped(arm)
    bad = mod.check(stats, arm)
    print(f"{arm}: transferred {stats['transferred_tensors']} tensors / "
          f"{stats['transferred_params']:,} params = {stats['transferred_pct']}% | "
          f"body new {stats['body_new_params']:,} (layers {stats['body_new_layers']}) | "
          f"detect new {stats['detect_new_params']:,} | GATE={'PASS' if not bad else 'FAIL ' + str(bad)}")
print("PRECHECK_DONE")
