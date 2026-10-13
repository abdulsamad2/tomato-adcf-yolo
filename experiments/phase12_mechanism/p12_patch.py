from pathlib import Path
p = Path(r"C:\Users\absamad\cs\p12_analyze.py")
s = p.read_text(encoding="utf-8")

# 1. apply the Phase 9 top-100-per-image rule, then re-verify six-class AP as a gated control
anchor = '    for f in sorted((P12 / "dets").glob("*.npz")):'
assert anchor in s
ins = [
 '    REF = json.loads((P12 / "p9_reference.json").read_text())',
 '    TOL = 0.05  # AP points; generous for GPU/batch nondeterminism, tight enough to catch a settings mismatch',
 '    ctrl, trunc_stats = [], []',
 '',
 '    def top100(iid, xyxy, cf, cl):',
 '        """Phase 9 rule: keep only the 100 highest-scoring detections per image."""',
 '        order = np.lexsort((-cf, iid))',
 '        iid2, cf2 = iid[order], cf[order]',
 '        keep = np.zeros(len(order), dtype=bool)',
 '        start = 0',
 '        bounds = np.flatnonzero(np.diff(iid2)) + 1',
 '        for end in list(bounds) + [len(iid2)]:',
 '            keep[start:min(start + 100, end)] = True',
 '            start = end',
 '        sel = order[keep]',
 '        return iid[sel], xyxy[sel], cf[sel], cl[sel]',
 '',
]
s = s.replace(anchor, chr(10).join(ins) + chr(10) + anchor)

a2 = '        iid, xyxy, cf, cl = d["image_id"], d["xyxy"], d["conf"], d["cls"]'
b2 = ('        iid, xyxy, cf, cl = d["image_id"], d["xyxy"], d["conf"], d["cls"]' + chr(10) +
      '        n_raw = len(iid)' + chr(10) +
      '        iid, xyxy, cf, cl = top100(iid, xyxy, cf, cl)' + chr(10) +
      '        trunc_stats.append((tag, n_raw, int(len(iid))))')
assert a2 in s; s = s.replace(a2, b2)

# 2. six-class control from the SAME truncated detections
a3 = '        # ---- GT-centric decomposition'
ins3 = [
 '        dt6 = [{"image_id": int(i), "category_id": int(c),',
 '                "bbox": [float(a), float(b), float(c2 - a), float(e - b)], "score": float(sc)}',
 '               for i, (a, b, c2, e), sc, c in zip(iid, xyxy, cf, cl)]',
 '        cocoDt6 = cocoGt.loadRes(dt6)',
 '        for arm in ("A", "B"):',
 '            E = COCOeval(cocoGt, cocoDt6, "bbox"); E.params.maxDets = [1, 10, 100]',
 '            E.params.imgIds = imgs_of[arm]',
 '            E.evaluate(); E.accumulate(); E.summarize()',
 '            rec["sc_AP_" + arm] = float(E.stats[0]) * 100',
 '            rec["sc_AP50_" + arm] = float(E.stats[1]) * 100',
 '            k = "%s|%s" % (tag, arm)',
 '            ctrl.append((k, REF[k]["AP"], rec["sc_AP_" + arm],',
 '                         abs(REF[k]["AP"] - rec["sc_AP_" + arm]), REF[k]["n_dets"]))',
 '',
]
s = s.replace(a3, chr(10).join(ins3) + chr(10) + a3)

# 3. gate before any capacity analysis
a4 = '    # ---- capacity x exposure'
ins4 = [
 '    md = max(c[3] for c in ctrl)',
 '    out["reproduction_control"] = {"pairs": len(ctrl), "max_abs_diff_AP_points": md,',
 '        "mean_abs_diff_AP_points": float(np.mean([c[3] for c in ctrl])), "tolerance": TOL,',
 '        "passed": bool(md <= TOL),',
 '        "worst": sorted(ctrl, key=lambda z: -z[3])[:5],',
 '        "truncation": {"rule": "top 100 detections per image (Phase 9)",',
 '            "n_raw_total": int(sum(t[1] for t in trunc_stats)),',
 '            "n_kept_total": int(sum(t[2] for t in trunc_stats))}}',
 '    print("REPRO max|diff| = %.4f AP pts (tol %.2f) over %d pairs -> %s"',
 '          % (md, TOL, len(ctrl), "PASS" if md <= TOL else "FAIL"), flush=True)',
 '    if md > TOL:',
 '        json.dump(out, open(P12 / "p12_results.json", "w"), indent=1)',
 '        print("STOP_REPRODUCTION_FAILED"); return',
 '',
]
s = s.replace(a4, chr(10).join(ins4) + chr(10) + a4)
p.write_text(s, encoding="utf-8")
print("patched, lines =", len(s.splitlines()))
