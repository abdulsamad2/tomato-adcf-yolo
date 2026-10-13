import inspect, re
from ultralytics.engine import trainer as T
from ultralytics.cfg import DEFAULT_CFG_DICT

src = inspect.getsource(T.BaseTrainer)

print("=== does a val_period / validation-interval arg exist? ===")
print({k: v for k, v in DEFAULT_CFG_DICT.items()
       if any(s in k for s in ("val", "period", "patience", "close_mosaic", "warmup", "lrf", "cos_lr"))})

print("\n=== where validate() is called in the epoch loop ===")
for i, line in enumerate(src.splitlines()):
    if re.search(r"self\.validate\(\)|on_train_epoch_end|on_fit_epoch_end|final_epoch|stop\s*=|self\.scheduler\.step|warmup|self\.args\.val\b", line):
        print(f"{i:5d}: {line.rstrip()[:118]}")

print("\n=== LR scheduler construction (_setup_scheduler) ===")
print(inspect.getsource(T.BaseTrainer._setup_scheduler))

print("\n=== warmup / nw computation ===")
m = re.search(r"\n( *)nw = .*?\n", src)
print(m.group(0) if m else "nw not found")
for line in src.splitlines():
    if "nw" in line and ("round" in line or "max" in line or "warmup" in line):
        print("   ", line.strip()[:118])
