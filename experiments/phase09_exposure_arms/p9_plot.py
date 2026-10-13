import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"C:\Users\absamad\cs\phase9")
AUD = Path(r"C:\Users\absamad\cs\audit"); AUD.mkdir(exist_ok=True)
R = json.loads((OUT / "p9_results.json").read_text())
E = json.loads((OUT / "eval_rows.json").read_text())
BT = {b["comparison"]: b for b in R["bootstrap"]}

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300})
FIX, EARLY = "#0072B2", "#D55E00"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.9),
                               gridspec_kw={"width_ratios": [1.0, 1.25]})

# ---- panel a ----
xa = np.array([r["A"]["AP"]*100 for r in E]); yb = np.array([r["B"]["AP"]*100 for r in E])
isfix = np.array([r["patience"] is None for r in E])
ax1.scatter(xa[isfix], yb[isfix], s=26, facecolor=FIX, edgecolor="white", linewidth=0.5,
            label="fixed budget, frozen checkpoint (n=50)", zorder=3)
ax1.scatter(xa[~isfix], yb[~isfix], s=26, marker="s", facecolor=EARLY, edgecolor="white",
            linewidth=0.5, label="early-stopped, last.pt (n=20)", zorder=3)
ax1.axhspan(yb.min(), yb.max(), color="0.90", zorder=0)
ax1.annotate("", xy=(21.0, yb.min()), xytext=(21.0, yb.max()),
             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))
ax1.text(19.6, (yb.min()+yb.max())/2, "independent span %.1f pts" % (yb.max()-yb.min()),
         fontsize=7, color="0.25", va="center", ha="center", rotation=90)
ax1.annotate("", xy=(xa.min(), 21.6), xytext=(xa.max(), 21.6),
             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))
ax1.text((xa.min()+xa.max())/2, 21.15, "exposed span %.1f pts" % (xa.max()-xa.min()),
         fontsize=7, color="0.25", ha="center", va="top")
ax1.set_xlim(18, 78); ax1.set_ylim(10, 22.2)
ax1.set_xlabel("AP on source-EXPOSED families (arm A, n=1047)")
ax1.set_ylabel("AP on source-INDEPENDENT families\n(arm B, n=213)")
ax1.set_title("a  Every checkpoint inflated; apparent\n    differences almost all exposure", loc="left")
ax1.legend(loc="lower right", frameon=False)

# ---- panel b ----
CONF = [c for c in R["comparisons"] if c["tier"] == "PRIMARY"]
SEC = [c for c in R["comparisons"] if c["tier"] == "SECONDARY"]
items = CONF + SEC
lab = [c["comparison"] for c in items]
y = np.arange(len(items))[::-1]
for yy, c in zip(y, items):
    i = c["interaction"]; bt = BT[c["comparison"]]["interaction"]
    col = "#000000" if c["tier"] == "PRIMARY" else "0.35"
    ax2.plot([bt["ci_lo"], bt["ci_hi"]], [yy+0.17]*2, "-", color="#009E73", lw=1.4,
             solid_capstyle="butt", zorder=2)
    ax2.plot([i["ci_lo"], i["ci_hi"]], [yy-0.08]*2, "-", color=col, lw=2.6,
             solid_capstyle="butt", zorder=2)
    ax2.plot([i["mean"]], [yy-0.08], "o", color=col, ms=5, zorder=4,
             markeredgecolor="white", markeredgewidth=0.6)
    if "p_holm" in i:
        ax2.text(18.4, yy-0.08, "%.4f" % i["p_holm"], fontsize=6.5, va="center", ha="right",
                 color="black")
ax2.axvline(0, color="0.2", lw=0.9, zorder=1)
ax2.axhline(len(SEC)-0.5, color="0.8", lw=0.7, ls=":", zorder=1)
ax2.set_yticks(y); ax2.set_yticklabels(lab, fontsize=7.5)
for t, c in zip(ax2.get_yticklabels(), items):
    if c["tier"] == "PRIMARY":
        t.set_fontweight("bold")
ax2.set_xlim(-14.5, 18.6); ax2.set_ylim(-0.6, len(items)-0.1)
ax2.set_xlabel(r"interaction  $I=\Delta_{independent}-\Delta_{exposed}$  (AP points)")
ax2.set_title("b  Exposure alters the apparent effect of capacity and\n    resolution, but not of augmentation", loc="left")
ax2.text(18.4, len(items)-0.42, "$p_{Holm}$", fontsize=6.5, ha="right", va="center", color="black")
from matplotlib.lines import Line2D
ax2.legend(handles=[Line2D([0],[0], color="black", lw=2.6, label="95% CI across training seeds"),
                    Line2D([0],[0], color="#009E73", lw=1.4, label="95% CI across source families (bootstrap)")],
           loc="upper center", frameon=False, bbox_to_anchor=(0.46, -0.145), ncol=2)

fig.tight_layout()
fig.savefig(AUD / "p9_leakage_interaction.png", bbox_inches="tight", facecolor="white")
fig.savefig(AUD / "p9_leakage_interaction.pdf", bbox_inches="tight", facecolor="white")
rend = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(rend)) for t in fig.findobj(matplotlib.text.Text)
      if t.get_text().strip() and t.get_visible()]
print("overlaps", [(a.get_text()[:22], b.get_text()[:22]) for k, (a, ba) in enumerate(tx)
                   for b, bb in tx[k+1:] if ba.overlaps(bb)][:8])
print("saved")
