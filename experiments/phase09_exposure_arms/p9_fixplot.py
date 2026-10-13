from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p9_plot.py")
s = p.read_text(encoding="utf-8")
A0 = 'ax1.axhspan(yb.min(), yb.max(), color="0.88", zorder=0)\nax1.annotate("", xy=(24, yb.min()), xytext=(24, yb.max()),\n             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))\nax1.text(26.5, (yb.min()+yb.max())/2, "independent\\nspan %.1f pts" % (yb.max()-yb.min()),\n         fontsize=7, color="0.25", va="center", ha="left")\nax1.annotate("", xy=(xa.min(), 10.6), xytext=(xa.max(), 10.6),\n             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))\nax1.text((xa.min()+xa.max())/2, 11.2, "exposed span %.1f pts" % (xa.max()-xa.min()),\n         fontsize=7, color="0.25", ha="center", va="bottom")\nax1.set_xlim(18, 78); ax1.set_ylim(10, 21.5)'
B0 = 'ax1.axhspan(yb.min(), yb.max(), color="0.90", zorder=0)\nax1.annotate("", xy=(21.0, yb.min()), xytext=(21.0, yb.max()),\n             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))\nax1.text(19.6, (yb.min()+yb.max())/2, "independent span %.1f pts" % (yb.max()-yb.min()),\n         fontsize=7, color="0.25", va="center", ha="center", rotation=90)\nax1.annotate("", xy=(xa.min(), 21.6), xytext=(xa.max(), 21.6),\n             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))\nax1.text((xa.min()+xa.max())/2, 21.15, "exposed span %.1f pts" % (xa.max()-xa.min()),\n         fontsize=7, color="0.25", ha="center", va="top")\nax1.set_xlim(18, 78); ax1.set_ylim(10, 22.2)'
assert A0 in s, "anchor 0"
s = s.replace(A0, B0, 1)
A1 = '        ax2.text(14.6, yy-0.08, "%.4f" % i["p_holm"], fontsize=6.5, va="center", ha="right",\n                 color="black")'
B1 = '        ax2.text(18.4, yy-0.08, "%.4f" % i["p_holm"], fontsize=6.5, va="center", ha="right",\n                 color="black")'
assert A1 in s, "anchor 1"
s = s.replace(A1, B1, 1)
A2 = 'ax2.set_xlim(-14, 15.4); ax2.set_ylim(-0.8, len(items)-0.2)'
B2 = 'ax2.set_xlim(-14.5, 18.6); ax2.set_ylim(-0.6, len(items)-0.1)'
assert A2 in s, "anchor 2"
s = s.replace(A2, B2, 1)
A3 = 'ax2.text(14.6, len(items)-0.45, "$p_{Holm}$", fontsize=6.5, ha="right", color="black")'
B3 = 'ax2.text(18.4, len(items)-0.42, "$p_{Holm}$", fontsize=6.5, ha="right", va="center", color="black")'
assert A3 in s, "anchor 3"
s = s.replace(A3, B3, 1)
A4 = '           loc="lower left", frameon=False, bbox_to_anchor=(0.0, -0.02))\nax2.text(-13.2, len(items)-1.45, "confirmatory\\n(bold, Holm-adjusted)", fontsize=6.5, color="0.3", va="top")'
B4 = '           loc="upper center", frameon=False, bbox_to_anchor=(0.46, -0.145), ncol=2)'
assert A4 in s, "anchor 4"
s = s.replace(A4, B4, 1)
A5 = 'fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6),'
B5 = 'fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.9),'
assert A5 in s, "anchor 5"
s = s.replace(A5, B5, 1)
p.write_text(s, encoding="utf-8")
print("patched", 6, "anchors")
