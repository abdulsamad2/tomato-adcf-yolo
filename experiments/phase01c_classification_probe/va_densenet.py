import csv, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image

MAN = Path(r"C:\Users\absamad\cs\audit\va_split\manifest.csv")
OUT = Path(r"C:\Users\absamad\cs\audit\densenet")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0"
EPOCHS, BATCH, PATIENCE = 30, 32, 8

rows = list(csv.DictReader(MAN.open(encoding="utf-8")))
CLASSES = sorted({r["cls"] for r in rows})
CIDX = {c: i for i, c in enumerate(CLASSES)}

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
TF_EVAL = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        r = self.items[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), CIDX[r["cls"]]

def metrics(y, p):
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    acc = cm.trace() / cm.sum()
    f1 = []
    for i in range(n):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"accuracy": float(acc), "macro_f1": float(np.mean(f1)),
            "per_class_f1": {c: round(float(v), 4) for c, v in zip(CLASSES, f1)}}

@torch.no_grad()
def evaluate(model, items):
    if not items:
        return None
    model.eval()
    dl = DataLoader(DS(items, TF_EVAL), batch_size=64, num_workers=2, pin_memory=True)
    ys, ps = [], []
    for x, y in dl:
        out = model(x.to(DEV, non_blocking=True))
        ps.extend(out.argmax(1).cpu().tolist()); ys.extend(y.tolist())
    return metrics(ys, ps)

def run(split_col, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tr = [r for r in rows if r[split_col] == "train"]
    va = [r for r in rows if r[split_col] == "val"]
    te = [r for r in rows if r[split_col] == "test"]
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(DS(tr, TF_TRAIN), batch_size=BATCH, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl:
            opt.zero_grad(set_to_none=True)
            crit(model(x.to(DEV, non_blocking=True)), y.to(DEV, non_blocking=True)).backward()
            opt.step()
        sched.step()
        m = evaluate(model, va)
        if m["macro_f1"] > best:
            best, best_ep, bad = m["macro_f1"], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        print(f"  [{split_col} seed{seed}] ep{ep} val_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}", flush=True)
    model.load_state_dict(best_state)
    res = {"split": split_col, "seed": seed, "best_epoch": best_ep,
           "epochs_run": ep + 1, "train_minutes": round((time.time()-t0)/60, 2),
           "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "val": evaluate(model, va), "test": evaluate(model, te)}
    if split_col == "clean":
        # controlled leakage probe: official test split, partitioned by training exposure
        seen_src = {r["src"] for r in tr}
        off = [r for r in rows if r["official"] == "test"]
        res["official_test_seen"] = evaluate(model, [r for r in off if r["src"] in seen_src])
        res["official_test_unseen"] = evaluate(model, [r for r in off if r["src"] not in seen_src])
        res["n_seen"] = sum(1 for r in off if r["src"] in seen_src)
        res["n_unseen"] = sum(1 for r in off if r["src"] not in seen_src)
    return res

def main():
    out = []
    for split_col in ("official", "clean"):
        for seed in (0, 1, 2):
            r = run(split_col, seed)
            out.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "per_class_f1"}, default=str)[:400], flush=True)
            (OUT/"densenet_results.json").write_text(json.dumps(out, indent=1))
    print("DENSENETDONE")

if __name__ == "__main__":
    main()
