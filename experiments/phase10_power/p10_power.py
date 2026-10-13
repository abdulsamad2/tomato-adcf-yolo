import numpy as np
from scipy import stats as st
def mde(n, sd, power=0.80, alpha=0.05):
    return (st.t.ppf(1-alpha/2, n-1) + st.t.ppf(power, n-1)) * sd / np.sqrt(n)
print("=== paired interaction MDE vs seeds (80% power, alpha 0.05 two-sided) ===")
print("%6s" % "seeds" + "".join("%12s" % ("sd=%.1f" % s) for s in (0.5,1.0,1.5,2.0)))
for n in (3,4,5):
    print("%6d" % n + "".join("%12.2f" % mde(n,s) for s in (0.5,1.0,1.5,2.0)))
print()
for n in (3,4,5):
    print("  n=%d : detectable |I|/sd = %.2f" % (n, mde(n,1.0)))
