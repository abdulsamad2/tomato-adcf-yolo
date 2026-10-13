import csv
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mt
from pathlib import Path

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
rows = list(csv.DictReader((RES/"p2b_learning_curves.csv").open(encoding="utf-8")))
COL = {"B": "#0072B2", "C": "#009E73", "D": "#D55E00"}
LBL = {"B": "B: 1x source photos", "C": "C: 8x byte-identical copies", "D": "D: 8x shipped augmented"}
CK = {"B": [6320, 12008, 18170, 24569], "C": [6300, 11970, 18270, 24570]}
CK["D"] = CK["C"]

def ser(arm, seed, field, onlyval):
    g = [r for r in rows if r["arm"] == arm and int(r["seed"]) == seed
         and r.get(field) not in (None, "", "None")
         and (r["validated"] in ("True","TRUE") or not onlyval)]
    g.sort(key=lambda r: int(r["global_step"]))
    return np.array([int(r["global_step"]) for r in g]), np.array([float(r[field]) for r in g])

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "savefig.dpi": 300})
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
panels = [("val_mAP50_95", "validation mAP50-95", True),
          ("val_mAP50", "validation mAP50", True),
          ("val_cls_loss", "validation classification loss", True),
          ("val_box_loss", "validation box loss", True)]
for ax, (field, ylab, onlyval) in zip(axes.ravel(), panels):
    for arm in "BCD":
        ys = []
        for s in range(5):
            x, y = ser(arm, s, field, onlyval)
            if len(x) == 0: continue
            ax.plot(x, y, "-", color=COL[arm], lw=0.55, alpha=0.32)
            ys.append((x, y))
        if ys:
            n = min(len(y) for _, y in ys)
            xm, ym = ys[0][0][:n], np.mean([y[:n] for _, y in ys], axis=0)
            ax.plot(xm, ym, "-", color=COL[arm], lw=1.8,
                    label=LBL[arm] if field == "val_mAP50_95" else None)
    for cx in CK["C"]:
        ax.axvline(cx, color="0.55", ls=(0, (2, 3)), lw=0.7, zorder=0)
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.01)
axes[0,0].legend(frameon=False, loc="lower right")
axes[0,0].set_title("thin lines = all 5 seeds; vertical lines = the 4 frozen checkpoints",
                    loc="left", fontsize=7, color="0.45")
axes[1,0].set_title("lower = better", loc="left", fontsize=7, color="0.45")
for ax in axes[1]:
    ax.set_xlabel("optimizer updates")
fig.suptitle("Phase 2B: offline augmentation diversity separates from both controls\n"
             "from ~12k updates onward (arm x budget interaction p = 0.025)",
             fontsize=9.5, y=1.035)
cap = ("Five paired seeds. Budget 24,569 updates (B) / 24,570 (C, D). Checkpoints frozen before training.\n"
       "Paired D-C: +0.77 (p=0.25) at ~6k, then +2.88 (p=0.007), +2.94 (p=0.008), +2.02 (p=0.022).\n"
       "Paired C-B: -0.58, -0.49, -0.44, -0.71 -- never significant, never equivalent within 1.0 at 5 seeds.\n"
       "Residual schedule differences: budget 1 update; checkpoints 20 / 38 / 100; mosaic tail 69. "
       "LR staircase 2.54% of lr0 for C and D, 0.32% for B; C and D share an identical grid.")
fig.text(0.5, -0.045, cap, fontsize=6.3, color="0.35", ha="center", va="top")
fig.tight_layout()
out = r"C:\Users\absamad\cs\audit\p2b_learning_curves"
fig.savefig(out + ".png", bbox_inches="tight", facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
r = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mt.Text) if t.get_text().strip() and t.get_visible()]
print("overlaps:", [(a.get_text()[:14], b.get_text()[:14]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:] if ba.overlaps(bb)][:6])