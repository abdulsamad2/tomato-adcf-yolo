import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

P12 = Path(r"C:\Users\absamad\cs\phase12"); OUT = Path(r"C:\Users\absamad\cs\audit"); OUT.mkdir(exist_ok=True)
R = json.loads((P12 / "p12_results.json").read_text())
CK, CAP = R["checkpoints"], R["capacity"]
MODELS = [("runs_p3__yolo11n", "11n"), ("runs_p2b__D", "11s"), ("runs_p3__yolo11m", "11m")]
EXP, IND, NEU = "#D55E00", "#0072B2", "#7a8085"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300})

def series(comp, arm):
    m, s = [], []
    for base, _ in MODELS:
        v = [CK["%s__seed%d" % (base, k)]["%s_%s" % (comp, arm)] for k in range(5)]
        m.append(np.mean(v)); s.append(np.std(v, ddof=1))
    return np.array(m), np.array(s)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
x = np.arange(3)
for ax, comp, ttl in ((axes[0], "loc_recall", "Localization recall\n(class identity ignored)"),
                      (axes[1], "cond_class_correct", "Class correctness GIVEN\nlocalization")):
    for arm, col, lab in (("A", EXP, "arm A  source-exposed"), ("B", IND, "arm B  source-independent")):
        m, s = series(comp, arm)
        ax.errorbar(x, m, yerr=s, fmt="o-", color=col, ms=6, lw=1.6, capsize=3, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in MODELS])
    ax.set_xlabel("model capacity"); ax.set_ylabel("%"); ax.set_title(ttl)
    ax.set_xlim(-0.3, 2.3)
axes[0].legend(loc="center left", frameon=False, bbox_to_anchor=(0.02, 0.30))
axes[1].set_ylim(94, 101)
axes[1].axhline(100, color="0.75", lw=0.8, ls=":")
axes[1].text(2.25, 100.15, "ceiling", fontsize=6.5, color="0.5", ha="right")

ax3 = axes[2]
COMP = [("ca_AP50", "class-agnostic AP50", 100.0), ("loc_recall", "localization recall", 1.0),
        ("full_correct_iou75", "full correct @IoU.75", 1.0), ("cond_class_correct", "cond. class correctness", 1.0)]
h = 0.35
for j, (lbl, mk, col) in enumerate((("s-n", "o", "#6a3d9a"), ("m-s", "s", "#33a02c"))):
    ys, xs, es = [], [], []
    for i, (comp, _, sc) in enumerate(COMP):
        c = CAP[lbl][comp]
        ys.append(i + (j - 0.5) * h); xs.append(c["I_mean"] * sc)
        es.append(abs(c["ci95"][1] - c["ci95"][0]) / 2 * sc)
    ax3.errorbar(xs, ys, xerr=es, fmt=mk, color=col, ms=6, lw=0, elinewidth=1.4,
                 capsize=3, label="capacity step %s" % lbl.replace("-", " \u2212 "))
ax3.axvline(0, color="0.5", lw=0.9, ls=":")
ax3.set_yticks(range(len(COMP))); ax3.set_yticklabels([n for _, n, _ in COMP])
ax3.invert_yaxis()
ax3.set_xlabel(r"interaction  $I=\Delta_B-\Delta_A$  (points)")
ax3.set_title("Which component carries the\ncapacity $\\times$ exposure interaction?")
ax3.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "p12_mechanism.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "p12_mechanism.pdf", bbox_inches="tight", facecolor="white")
rend = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(rend)) for t in fig.findobj(matplotlib.text.Text)
      if t.get_text().strip() and t.get_visible()]
print("overlaps", [(a.get_text()[:16], b.get_text()[:16]) for i, (a, ba) in enumerate(tx)
                   for b, bb in tx[i+1:] if ba.overlaps(bb)])
print("saved")
