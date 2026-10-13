from pathlib import Path
p = Path(r"C:\Users\absamad\cs\bcd_run.py")
s = p.read_text(encoding="utf-8")

old = """    def on_train_batch_end(tr):
        # per-iteration LR for the warmup window only (verification of warmup alignment)
        ni = tr.epoch * state["nb"] + getattr(tr, "_bi", 0)
        if len(state["iters"]) < 2200:
            state["iters"].append({"global_step": ni + 1,
                                   "lr": tr.optimizer.param_groups[0]["lr"]})"""
new = """    def on_train_batch_end(tr):
        # per-iteration LR for the warmup window only (verification of warmup alignment).
        # The trainer exposes no batch index, so count iterations here.
        state["ni"] = state.get("ni", 0) + 1
        if state["ni"] <= 2200:
            state["iters"].append({"global_step": state["ni"],
                                   "lr": tr.optimizer.param_groups[0]["lr"],
                                   "momentum": tr.optimizer.param_groups[0].get("momentum")})"""
assert old in s, "anchor not found"
s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("patched; 'ni' occurrences:", s.count('state["ni"]'))
import ast; ast.parse(s); print("syntax OK")
