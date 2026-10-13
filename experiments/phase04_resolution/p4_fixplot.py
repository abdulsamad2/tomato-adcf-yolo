from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p4_plot.py")
s = p.read_text(encoding="utf-8")
reps = [
  ('ax2.text(2.55, i, f"TOST $p$ = {c[\'tost_p\']:.3f}\\nHolm {c[\'tost_p_holm\']:.3f}",\n             va="center", ha="right", fontsize=7, color="0.28")',
   'ax2.text(4.45, i, f"TOST $p$ = {c[\'tost_p\']:.3f}\\nHolm {c[\'tost_p_holm\']:.3f}",\n             va="center", ha="right", fontsize=7, color="0.28")'),
  ('ax2.set_xlim(-4.3, 2.7)', 'ax2.set_xlim(-4.4, 4.6)'),
  ('ax1.set_xlabel("optimizer updates  (identical 24,570-update budget, every resolution)")',
   'ax1.set_xlabel("optimizer updates  (24,570 in every arm)")'),
  ('ax2.set_xlabel("paired difference in validation mAP50-95 at 18,270")',
   'ax2.set_xlabel("paired difference at 18,270")'),
  ('fig.tight_layout()', 'fig.tight_layout(w_pad=2.0)'),
]
for a, b in reps:
    assert a in s, a[:40]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched")
