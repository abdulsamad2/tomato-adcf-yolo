import csv, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path(r"C:\Users\absamad\cs\tomato-adcf-yolo\results")
OUT = Path(r"C:\Users\absamad\cs\audit"); OUT.mkdir(exist_ok=True)
cur = list(csv.DictReader((RES / "p6_learning_curves.csv").open(encoding="utf-8")))
S = json.load(open(RES / "p6_stats.json"))

A = ["A", "B", "C"]
COL = {"A": "#D55E00", "B": "#0072B2", "C": "#009E73"}
LBL = {"A": "A  P3+P4+P5  (reference, reused)", "B": "B  P3+P4  (P5 branch removed)",
       "C": "C  P4+P5  (P3 branch removed)"}
PRIMARY, CK = 18270, [6300, 11970, 18270, 24570]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))

for a in A:
    xs = None
    allv = []
    for s in range(5):
        g = sorted([(int(c["global_step"]), float(c["val_mAP50_95"]) * 100) for c in cur
                    if c["arm"] == a and int(c["seed"]) == s and c.get("val_mAP50_95")])
        if not g:
            continue
        x, y = zip(*g)
        ax1.plot(x, y, color=COL[a], lw=0.5, alpha=0.3, zorder=1)
        xs = x; allv.append(y)
    if allv:
        n = min(len(v) for v in allv)
        m = np.mean([v[:n] for v in allv], axis=0)
        ax1.plot(xs[:n], m, color=COL[a], lw=1.9, label=LBL[a], zorder=3)
for ckp in CK:
    ax1.axvline(ckp, color="0.78", ls=":", lw=0.7, zorder=0)
ax1.axvline(PRIMARY, color="0.35", ls="--", lw=1.0, zorder=2)
ax1.text(PRIMARY - 500, 14.2, "primary endpoint 18,270", ha="right", va="bottom",
         fontsize=7, color="0.3", rotation=90)
ax1.set_xlabel("optimizer updates  (24,570 in every arm)")
ax1.set_ylabel("validation mAP50-95 (\u00d7100)")
ax1.set_title("Detection-scale removal costs accuracy at every budget")
ax1.legend(loc="lower right", frameon=False)
ax1.set_xlim(0, 25200)

P = {c["contrast"]: c for c in S["primary"]}
ks = ["B-A", "C-A"]
ypos = [1, 0]
ax2.axvspan(-1.0, 1.0, color="0.90", zorder=0)
ax2.axvline(0, color="0.45", lw=0.9, zorder=1)
for k, yp in zip(ks, ypos):
    c = P[k]
    col = COL[k.split("-")[0]]
    ax2.plot(c["diffs"], [yp + 0.17] * 5, "o", color=col, ms=4, alpha=0.55, zorder=3)
    ax2.errorbar(c["mean"], yp, xerr=[[c["mean"] - c["ci_lo"]], [c["ci_hi"] - c["mean"]]],
                 fmt="s", color=col, ms=8, capsize=4, lw=1.7, zorder=4)
    ax2.text(c["mean"], yp - 0.30, f"{c['mean']:+.2f}  [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]",
             ha="center", va="top", fontsize=7.5, color=col)
    ax2.text(-3.25, yp + 0.17,
             f"Holm $p$ = {c['p_holm']:.4f}\nTOST $p$ = {c['tost_p']:.2f}",
             ha="left", va="center", fontsize=7, color="0.28")
ax2.set_yticks(ypos)
ax2.set_yticklabels(["B \u2212 A\nP5 removed", "C \u2212 A\nP3 removed"])
ax2.set_ylim(-0.65, 1.65)
ax2.set_xlim(-3.4, 1.6)
ax2.set_xlabel("paired difference at 18,270")
ax2.set_title("Neither removal is equivalent within \u00b11.0")
ax2.text(0.0, 1.55, "\u00b11.0 practical-equivalence margin", ha="center", va="top",
         fontsize=7, color="0.42")

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT / "p6_detection_scale.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "p6_detection_scale.pdf", bbox_inches="tight", facecolor="white")
rend = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(rend)) for t in fig.findobj(matplotlib.text.Text)
      if t.get_text().strip() and t.get_visible()]
print("overlaps", [(a.get_text()[:20], b.get_text()[:20]) for i, (a, ba) in enumerate(tx)
                   for b, bb in tx[i+1:] if ba.overlaps(bb)])
print("saved")
