from pathlib import Path
p = Path(r"C:\Users\absamad\cs\bcd_plot.py")
s = p.read_text(encoding="utf-8")
s = s.replace('ax1.legend(frameon=False, loc="lower left")',
              'ax1.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.995, 0.62))')
s = s.replace('ax2.text(900, 0.70, "mosaic on", va="center", fontsize=7, color="white")',
              'ax2.text(700, 0.70, "mosaic on", va="center", fontsize=6.5, color="white")')
s = s.replace('ax2.text(900, 0.30, "mosaic on", va="center", fontsize=7, color="white")',
              'ax2.text(700, 0.30, "mosaic on", va="center", fontsize=6.5, color="white")')
p.write_text(s, encoding="utf-8")
print("patched")
