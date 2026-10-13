import csv
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mt
from pathlib import Path

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
rows = list(csv.DictReader((RES/"phase_2_learning_curves.csv").open(encoding="utf-8")))
COL = {"B": "#0072B2", "C": "#009E73", "D": "#D55E00"}
LBL = {"B": "B: 1x source photos", "C": "C: 8x byte-identical copies", "D": "D: 8x shipped augmented"}
BUDGET, MOSOFF = 49770, 43470

def ser(arm, seed, field, only_val):
    g = [r for r in rows if r["arm"] == arm and int(r["seed"]) == seed
         and r.get(field) not in (None, "", "None")
         and (r["validated"] == "True" or not only_val)]
    g.sort(key=lambda r: int(r["global_step"]))
    return np.array([int(r["global_step"]) for r in g]), np.array([float(r[field]) for r in g])

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "savefig.dpi": 300})
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
panels = [("val_mAP50_95", "validation mAP50-95", True),
          ("val_mAP50", "validation mAP50", True),
          ("train/box_loss", "training box loss", False),
          ("val_loss_box", "validation box loss", True)]
for ax, (field, ylab, onlyval) in zip(axes.ravel(), panels):
    for arm in "BCD":
        ys = []
        for s in (0, 1, 2):
            x, y = ser(arm, s, field, onlyval)
            if len(x) == 0: continue
            ax.plot(x, y, "-", color=COL[arm], lw=0.5, alpha=0.3)
            ys.append((x, y))
        if ys:
            n = min(len(y) for _, y in ys)
            xm, ym = ys[0][0][:n], np.mean([y[:n] for _, y in ys], axis=0)
            ax.plot(xm, ym, "-", color=COL[arm], lw=1.7,
                    label=LBL[arm] if field == "val_mAP50_95" else None)
            if field == "val_mAP50_95":
                i = int(np.argmax(ym))
                ax.plot(xm[i], ym[i], "o", color=COL[arm], ms=5.5, mfc="white", mew=1.5, zorder=6)
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.01)
    ax.axvline(MOSOFF, color="0.75", ls=":", lw=0.8)
axes[0,0].legend(frameon=False, loc="lower right")
axes[0,0].set_title("higher = better; open circle = peak of mean curve", loc="left", fontsize=7, color="0.45")
axes[0,1].set_title("dotted line = mosaic switched off", loc="left", fontsize=7, color="0.45")
for ax in axes[1]:
    ax.set_xlabel("optimizer updates")
fig.suptitle("Shipped offline augmentation raises the validation ceiling by 3.1 points;\n"
             "byte-identical duplication is statistically equivalent to no duplication",
             fontsize=9.5, y=1.035)
cap = ("Identical 49,770-update budget in all arms. Thin lines = individual seeds (n=3, paired); "
       "open circle = peak of the mean curve.\n"
       "At peak: D-B = +3.07 points (paired p=0.0069); C-B = -0.05, equivalent within 1.0 mAP50-95 "
       "(TOST p=0.019).\n"
       "Every arm declines 2.4-4.0 points from its peak by the fixed final checkpoint, so that "
       "registered endpoint sits in an overfit regime.")
fig.text(0.5, -0.04, cap, fontsize=6.4, color="0.35", ha="center", va="top")
fig.tight_layout()
out = r"C:\Users\absamad\cs\audit\phase2_learning_curves"
fig.savefig(out + ".png", bbox_inches="tight", facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
r = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mt.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text()[:14], b.get_text()[:14]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:] if ba.overlaps(bb)]
print("overlaps:", ov[:8])