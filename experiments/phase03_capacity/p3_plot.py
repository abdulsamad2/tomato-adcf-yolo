import csv, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
OUT = Path(r"C:\Users\absamad\cs\audit"); OUT.mkdir(exist_ok=True)
cur = list(csv.DictReader((RES / "p3_learning_curves.csv").open(encoding="utf-8")))
S = json.load(open(RES / "p3_stats.json"))
E = json.load(open(RES / "p3_efficiency.json"))

COL = {"yolo11n": "#0072B2", "yolo11s": "#009E73", "yolo11m": "#D55E00"}
LBL = {"yolo11n": "YOLO11n  2.59 M", "yolo11s": "YOLO11s  9.43 M", "yolo11m": "YOLO11m  20.06 M"}
M = ["yolo11n", "yolo11s", "yolo11m"]
CK = [6300, 11970, 18270, 24570]
PRIMARY = 18270

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.3, 3.3))

for m in M:
    for s in range(5):
        g = sorted([c for c in cur if c["model"] == m and int(c["seed"]) == s
                    and c.get("val_mAP50_95") not in ("", None)],
                   key=lambda c: int(c["global_step"]))
        ax1.plot([int(c["global_step"]) for c in g], [float(c["val_mAP50_95"]) * 100 for c in g],
                 color=COL[m], lw=0.5, alpha=0.26, zorder=1)
    steps = sorted({int(c["global_step"]) for c in cur if c["model"] == m})
    mean = [np.mean([float(c["val_mAP50_95"]) * 100 for c in cur
                     if c["model"] == m and int(c["global_step"]) == st]) for st in steps]
    ax1.plot(steps, mean, color=COL[m], lw=1.9, label=LBL[m], zorder=3)
    ck = [S["descriptives"][m][str(k)]["val_mAP50_95"]["mean"] for k in CK]
    ax1.plot(CK, ck, "o", color=COL[m], ms=4.6, mec="white", mew=0.7, zorder=4)

ax1.axvline(PRIMARY, color="0.35", ls="--", lw=0.9, zorder=2)
ax1.text(PRIMARY - 400, 25.25, "primary endpoint 18,270", ha="right", va="top",
         fontsize=7, color="0.3")
for k in CK:
    if k != PRIMARY:
        ax1.axvline(k, color="0.82", ls=":", lw=0.7, zorder=0)
ax1.set_xlabel("optimizer updates  (identical 24,570-update budget, all models)")
ax1.set_ylabel("validation mAP50-95 ($\\times$100)")
ax1.set_title("(a) Capacity helps early, then reverses", loc="left", fontweight="bold")
ax1.legend(loc="lower right", frameon=False)
ax1.set_xlim(0, 25400); ax1.set_ylim(15.2, 25.4)
an = S["rm_anova"]["model:step"]
ax1.text(0.035, 0.955, f"model $\\times$ budget interaction\nF({an['Num DF']:.0f},{an['Den DF']:.0f}) = "
         f"{an['F Value']:.1f},  $p$ < 0.0001", transform=ax1.transAxes, va="top", fontsize=7.3,
         color="0.2")

x = [E[m]["batches"]["16"]["ms_mean"] for m in M]
y = [S["descriptives"][m][str(PRIMARY)]["val_mAP50_95"]["mean"] for m in M]
yerr = [S["descriptives"][m][str(PRIMARY)]["val_mAP50_95"]["sd"] for m in M]
ax2.plot(x, y, "-", color="0.72", lw=1.0, zorder=1)
for i, m in enumerate(M):
    ax2.errorbar(x[i], y[i], yerr=yerr[i], fmt="o", color=COL[m], ms=8.5, capsize=3.2,
                 elinewidth=1.1, mec="white", mew=1.0, zorder=3)
    ax2.annotate(f"{m[-1].upper()}   {E[m]['GFLOPs_640']:.1f} GFLOPs",
                 (x[i], y[i]), textcoords="offset points",
                 xytext=(0, -44), ha="center", fontsize=7.3, color="0.25")
ax2.set_xlabel("batch-16 latency on an idle RTX 5000 Ada (ms)")
ax2.set_ylabel("validation mAP50-95 at 18,270 ($\\times$100)")
ax2.set_title("(b) 5.7$\\times$ the latency buys nothing", loc="left", fontweight="bold")
ax2.set_xlim(0, 80); ax2.set_ylim(20.2, 24.3)
ax2.text(0.97, 0.06, "mean $\\pm$ SD over 5 paired seeds\nfp32, 200 warmup + 1000 iters",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=7, color="0.4")

fig.tight_layout()
fig.savefig(OUT / "p3_capacity.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "p3_capacity.pdf", bbox_inches="tight", facecolor="white")
rend = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(rend)) for t in fig.findobj(matplotlib.text.Text)
      if t.get_text().strip() and t.get_visible()]
print("overlaps", [(a.get_text()[:16], b.get_text()[:16]) for i, (a, ba) in enumerate(tx)
                   for b, bb in tx[i+1:] if ba.overlaps(bb)])
print("saved")
