import csv, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
OUT = Path(r"C:\Users\absamad\cs\audit"); OUT.mkdir(exist_ok=True)
cur = list(csv.DictReader((RES / "p4_learning_curves.csv").open(encoding="utf-8")))
S = json.load(open(RES / "p4_stats.json"))
COST = json.load(open(RES / "resolution_cost.json"))

A = [320, 416, 512, 640]
COL = {320: "#CC79A7", 416: "#0072B2", 512: "#009E73", 640: "#D55E00"}
PRIMARY, CK = 18270, [6300, 11970, 18270, 24570]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))

for a in A:
    for s in range(5):
        g = sorted([c for c in cur if int(c["imgsz"]) == a and int(c["seed"]) == s
                    and c.get("val_mAP50_95") not in ("", None)],
                   key=lambda c: int(c["global_step"]))
        ax1.plot([int(c["global_step"]) for c in g], [float(c["val_mAP50_95"]) * 100 for c in g],
                 color=COL[a], lw=0.5, alpha=0.24, zorder=1)
    steps = sorted({int(c["global_step"]) for c in cur if int(c["imgsz"]) == a})
    mean = [np.mean([float(c["val_mAP50_95"]) * 100 for c in cur
                     if int(c["imgsz"]) == a and int(c["global_step"]) == st]) for st in steps]
    lab = f"{a} px" + ("  (reused)" if a == 640 else "")
    ax1.plot(steps, mean, color=COL[a], lw=1.9, label=lab, zorder=3)
    ax1.plot(CK, [S["descriptives"][str(a)][str(k)]["val_mAP50_95"]["mean"] for k in CK],
             "o", color=COL[a], ms=4.4, mec="white", mew=0.7, zorder=4)

ax1.axvline(PRIMARY, color="0.35", ls="--", lw=0.9, zorder=2)
ax1.text(PRIMARY - 400, 24.6, "primary endpoint 18,270", ha="right", va="top", fontsize=7, color="0.3")
for k in CK:
    if k != PRIMARY:
        ax1.axvline(k, color="0.84", ls=":", lw=0.7, zorder=0)
ax1.set_xlabel("optimizer updates  (24,570 in every arm)")
ax1.set_ylabel("validation mAP50-95 ($\\times$100)")
ax1.set_title("(a) 512 px matches 640; 320 px does not", loc="left", fontweight="bold")
ax1.legend(loc="lower right", frameon=False)
ax1.set_xlim(0, 25400); ax1.set_ylim(9.5, 24.9)

prim = {c["contrast"]: c for c in S["primary"]}
ys = [320, 416, 512]
yy = np.arange(len(ys))[::-1]
ax2.axvspan(-1.0, 1.0, color="#009E73", alpha=0.11, zorder=0)
ax2.axvline(0, color="0.45", lw=0.9, zorder=1)
for i, a in zip(yy, ys):
    c = prim[f"{a}-640"]
    ax2.errorbar(c["mean"], i, xerr=[[c["mean"] - c["ci_lo"]], [c["ci_hi"] - c["mean"]]],
                 fmt="o", color=COL[a], ms=8, capsize=4, elinewidth=1.5, mec="white", mew=1.0,
                 zorder=3)
    ax2.plot(c["diffs"], [i + 0.19] * 5, "|", color=COL[a], ms=6, alpha=0.65, zorder=2)
    ax2.text(4.45, i, f"TOST $p$ = {c['tost_p']:.3f}\nHolm {c['tost_p_holm']:.3f}",
             va="center", ha="right", fontsize=7, color="0.28")
ax2.set_yticks(yy); ax2.set_yticklabels([f"{a} px $-$ 640 px" for a in ys])
ax2.set_xlabel("paired difference at 18,270")
ax2.set_title("(b) Only 512 px reaches the $\\pm$1.0 margin", loc="left", fontweight="bold")
ax2.set_xlim(-4.4, 4.6); ax2.set_ylim(-0.55, 2.55)
ax2.text(-1.0, -0.44, "$\\pm$1.0 practical-equivalence margin", ha="left", va="bottom",
         fontsize=7, color="#00684d")

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "p4_resolution.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "p4_resolution.pdf", bbox_inches="tight", facecolor="white")
rend = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(rend)) for t in fig.findobj(matplotlib.text.Text)
      if t.get_text().strip() and t.get_visible()]
print("overlaps", [(a.get_text()[:18], b.get_text()[:18]) for i, (a, ba) in enumerate(tx)
                   for b, bb in tx[i+1:] if ba.overlaps(bb)])
print("saved")
