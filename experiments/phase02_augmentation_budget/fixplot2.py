from pathlib import Path
p = Path(r"C:\Users\absamad\cs\bcd_plot.py")
s = p.read_text(encoding="utf-8")
s = s.replace('label="measured optimizer LR (smoke tests, max error 0.06%)")',
              'label="measured optimizer LR (smoke tests; agrees within one epoch step)")')
s = s.replace('ax1.set_title("Learning-rate trajectories coincide in update space (max divergence 1.25% of lr$_0$)")',
              'ax1.set_title("Learning-rate trajectories coincide in update space to within one epoch step")')
s = s.replace('ax2.set_xlabel("optimizer updates (identical budget: 49,770 in every arm)")',
              'ax2.set_xlabel("optimizer updates (identical budget: 49,770 in every arm)")\n'
              'fig.text(0.012, -0.045, "Curve: analytic reproduction of the trainer\'s epoch-quantised LambdaLR, verified against logged optimizer state.\\n"\n'
              '         "Agreement is within 0.05% of lr$_0$ through warmup; afterwards within one epoch step (0.16% of lr$_0$ for B, 1.25% for C and D),\\n"\n'
              '         "which is the sampling resolution, not a model error. Inter-arm divergence is therefore resolved only to one epoch step.",\n'
              '         fontsize=6.2, color="0.35", ha="left", va="top")')
p.write_text(s, encoding="utf-8")
print("patched")
