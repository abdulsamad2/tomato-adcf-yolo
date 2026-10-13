import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGET, LR0, LRF, WARM = 49770, 0.01, 0.01, 1890
B_C, CD_C, GREY = "#0072B2", "#D55E00", "#7a8085"

def lr_trace(steps, epochs):
    u = np.arange(1, BUDGET + 1)
    e = (u - 1) // steps
    base = LR0 * ((1 - e / epochs) * (1 - LRF) + LRF)
    lr = base.copy()
    w = u <= WARM
    lr[w] = np.interp(u[w], [1, WARM], [0.0, 1.0]) * base[w]
    return u, lr, e

uB, lrB, eB = lr_trace(79, 630)
uC, lrC, eC = lr_trace(630, 79)
obsB = [(79,0.000413),(632,0.003302),(1264,0.006525),(1896,0.009633)]
obsC = [(630,0.003328),(1260,0.006578),(1890,0.009744),(2520,0.009624)]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "figure.dpi": 110, "savefig.dpi": 300})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True,
                               gridspec_kw={"height_ratios": [2.4, 1]})
ax1.plot(uB, lrB, "-", color=B_C, lw=1.3, label="B: 1x source photos, 630 epochs x 79 steps")
ax1.plot(uC, lrC, "-", color=CD_C, lw=1.3, alpha=0.85, label="C and D: 8x copies, 79 epochs x 630 steps")
ax1.axvline(WARM, color=GREY, ls="--", lw=0.8)
ax1.annotate("warmup ends at 1890 updates\n(observed in all three arms)", xy=(WARM, 0.00974),
             xytext=(6000, 0.0086), fontsize=7, color="0.25",
             arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
for s, g in obsB: ax1.plot(s, g, "o", color=B_C, ms=4.5, mfc="white", mew=1.2, zorder=5)
for s, g in obsC: ax1.plot(s, g, "s", color=CD_C, ms=4.5, mfc="white", mew=1.2, zorder=5)
ax1.plot([], [], "o", color="0.4", ms=4.5, mfc="white", mew=1.2,
         label="measured optimizer LR (smoke tests; agrees within one epoch step)")
ax1.set_ylabel("learning rate")
ax1.set_title("Learning-rate trajectories coincide in update space to within one epoch step")
ax1.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.995, 0.62))
ax1.margins(x=0.01)
ax1.spines[["top", "right"]].set_visible(False)

mB, mCD = 550 * 79, 69 * 630
ax2.broken_barh([(0, mB)], (0.55, 0.3), facecolors=B_C, alpha=0.8)
ax2.broken_barh([(0, mCD)], (0.15, 0.3), facecolors=CD_C, alpha=0.8)
ax2.text(700, 0.70, "mosaic on", va="center", fontsize=6.5, color="white")
ax2.text(700, 0.30, "mosaic on", va="center", fontsize=6.5, color="white")
ax2.text(mB + 900, 0.70, "off: final 6320 updates", va="center", fontsize=6.5, color=B_C)
ax2.text(mCD + 900, 0.30, "off: final 6300 updates", va="center", fontsize=6.5, color=CD_C)
ax2.set_yticks([0.70, 0.30]); ax2.set_yticklabels(["B", "C, D"])
ax2.set_ylim(0, 1); ax2.set_xlim(0, BUDGET)
ax2.set_xlabel("optimizer updates (identical budget: 49,770 in every arm)")
fig.text(0.012, -0.045, "Curve: analytic reproduction of the trainer's epoch-quantised LambdaLR, verified against logged optimizer state.\n"
         "Agreement is within 0.05% of lr$_0$ through warmup; afterwards within one epoch step (0.16% of lr$_0$ for B, 1.25% for C and D),\n"
         "which is the sampling resolution, not a model error. Inter-arm divergence is therefore resolved only to one epoch step.",
         fontsize=6.2, color="0.35", ha="left", va="top")
ax2.set_title("Mosaic augmentation status", loc="left")
ax2.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = r"C:\Users\absamad\cs\audit\bcd_schedule_alignment"
fig.savefig(out + ".png", bbox_inches="tight", facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")

r = fig.canvas.get_renderer()
import matplotlib.text as mt
tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mt.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text()[:20], b.get_text()[:20]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:] if ba.overlaps(bb)]
print("overlaps:", ov)

# alignment table at checkpoints
rows = []
for f in (0.02, 0.04, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    u = int(f * BUDGET)
    rows.append({"update": u, "frac": f, "lr_B": round(float(lrB[u-1]), 6),
                 "lr_CD": round(float(lrC[u-1]), 6),
                 "rel_diff_pct": round(100*abs(float(lrB[u-1])-float(lrC[u-1]))/LR0, 3),
                 "mosaic_B": bool(u <= mB), "mosaic_CD": bool(u <= mCD)})
json.dump(rows, open(r"C:\Users\absamad\cs\audit\bcd_align_table.json", "w"), indent=1)
print(json.dumps(rows, indent=1))
