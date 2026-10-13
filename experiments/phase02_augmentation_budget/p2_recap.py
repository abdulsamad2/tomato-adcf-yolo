import json
import numpy as np
from scipy import stats
D = {
 "yolo11s_val":  [0.239065,0.245372,0.247344,0.246565,0.254869],
 "p2_val":       [0.230036,0.234878,0.233918],
 "yolo11s_aps":  [0.074822,0.062126,0.112753,0.088761,0.049369],
 "p2_aps":       [0.040805,0.040765,0.077901],
 "yolo11s_test": [0.164011,0.212346,0.203556,0.207824,0.183815],
 "p2_test":      [0.176097,0.170842,0.195919],
}
out = {}
for lbl, a_k, b_k in (("val_mAP50_95","p2_val","yolo11s_val"),
                      ("val_AP_small","p2_aps","yolo11s_aps"),
                      ("test_mAP50_95","p2_test","yolo11s_test")):
    a = np.array(D[a_k])*100; b = np.array(D[b_k])*100
    t, p = stats.ttest_ind(a, b, equal_var=False)
    df = (a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))**2 / (
         (a.var(ddof=1)/len(a))**2/(len(a)-1) + (b.var(ddof=1)/len(b))**2/(len(b)-1))
    out[lbl] = {"p2_mean": round(float(a.mean()),3), "p2_sd": round(float(a.std(ddof=1)),3), "p2_n": len(a),
                "ref_mean": round(float(b.mean()),3), "ref_sd": round(float(b.std(ddof=1)),3), "ref_n": len(b),
                "delta": round(float(a.mean()-b.mean()),3), "welch_t": round(float(t),3),
                "welch_df": round(float(df),2), "p": float(p)}
    print(f'{lbl}: P2 {a.mean():.2f}±{a.std(ddof=1):.2f} (n={len(a)}) vs ref {b.mean():.2f}±{b.std(ddof=1):.2f} '
          f'(n={len(b)}) -> {a.mean()-b.mean():+.2f}, Welch t={t:.2f} df={df:.1f} p={p:.3f}')
json.dump(out, open(r"C:\Users\absamad\cs\scales\p2_recap.json","w"), indent=1)
print("P2DONE")
