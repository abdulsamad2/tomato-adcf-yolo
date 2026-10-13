'''Phase 9 step 4: interaction statistics + source-family bootstrap. No model, no inference.

COCO AP over an arbitrary MULTISET of images is computed from cached per-detection match state.
Because subsetting never changes the relative score order of detections, each checkpoint/category is
sorted ONCE and a resample becomes a per-detection weight vector; AP is then vector multiplies plus
cumulative sums. The routine is VALIDATED against pycocotools' own per-arm AP before any statistic
is reported.
'''
import json, itertools, time
from pathlib import Path
import numpy as np
from scipy import stats as st

OUT = Path(r'C:\Users\absamad\cs\phase9')
CACHE = OUT / 'match_cache'
RECT = np.linspace(0, 1, 101)
NBOOT = 1000
RNG = np.random.default_rng(20261007)

ROWS = json.loads((OUT / 'eval_rows.json').read_text())
AI = json.loads((OUT / 'arm_image_ids.json').read_text())
IDS = {a: np.array(sorted(v), dtype=np.int64) for a, v in AI['arm_image_ids'].items()}
FAMOF = {int(k): v for k, v in AI['family_of_image'].items()}

ALLIMG = np.array(sorted(FAMOF), dtype=np.int64)
POS = {int(i): k for k, i in enumerate(ALLIMG)}
NIMG = ALLIMG.size
ARMMASK = {a: np.zeros(NIMG, dtype=bool) for a in ('A', 'B')}
for a in ('A', 'B'):
    for i in IDS[a]:
        ARMMASK[a][POS[int(i)]] = True
FAMPOS = {}
for a in ('A', 'B'):
    d = {}
    for i in IDS[a]:
        d.setdefault(FAMOF[int(i)], []).append(POS[int(i)])
    FAMPOS[a] = [np.array(v, dtype=np.int64) for k, v in sorted(d.items())]
NFAM = {a: len(FAMPOS[a]) for a in ('A', 'B')}
print('images', NIMG, '| families A', NFAM['A'], 'B', NFAM['B'], flush=True)

_CK = {}
def prep(tag):
    '''Per category: detections pre-sorted by descending score, with their image slot and
    per-image GT counts. Built once per checkpoint.'''
    if tag in _CK:
        return _CK[tag]
    z = np.load(CACHE / (tag + '.npz'), allow_pickle=False)
    meta, scores, tp, ig = z['meta'], z['scores'], z['tp'], z['ig']
    out = {}
    for c in z['catIds']:
        m = meta[meta[:, 0] == int(c)]
        if m.size == 0:
            continue
        npig = np.zeros(NIMG, dtype=np.int64)
        idx_l, img_l = [], []
        for cat_, iid, off, k, ng in m:
            p = POS.get(int(iid))
            if p is None:
                continue
            npig[p] += int(ng)
            if k > 0:
                idx_l.append(np.arange(off, off + k, dtype=np.int64))
                img_l.append(np.full(int(k), p, dtype=np.int64))
        if idx_l:
            idx = np.concatenate(idx_l); img = np.concatenate(img_l)
            o = np.argsort(-scores[idx], kind='mergesort')
            idx, img = idx[o], img[o]
            out[int(c)] = {'img': img, 'tp': tp[:, idx].astype(np.float64),
                           'notig': (ig[:, idx] == 0), 'npig': npig}
        else:
            out[int(c)] = {'img': np.zeros(0, np.int64), 'tp': np.zeros((10, 0)),
                           'notig': np.zeros((10, 0), bool), 'npig': npig}
    _CK[tag] = out
    return out

def ap_weighted(tag, w):
    '''COCO AP@[.50:.95] where w[j] = how many times image slot j is included (0,1,2,...).'''
    P_ = prep(tag)
    aps = []
    for c, d in P_.items():
        npig = float((d['npig'] * w).sum())
        if npig <= 0:
            continue
        if d['img'].size == 0:
            aps.append(0.0); continue
        ww = w[d['img']].astype(np.float64)
        if not ww.any():
            aps.append(0.0); continue
        per_t = np.empty(10)
        for t in range(10):
            keep = d['notig'][t] & (ww > 0)
            if not keep.any():
                per_t[t] = 0.0; continue
            wk = ww[keep]; tk = d['tp'][t][keep]
            ctp = np.cumsum(tk * wk)
            cfp = np.cumsum((1.0 - tk) * wk)
            rc = ctp / npig
            pr = ctp / (ctp + cfp + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            ins = np.searchsorted(rc, RECT, side='left')
            q = np.zeros(RECT.size)
            ok = ins < pr.size
            q[ok] = pr[ins[ok]]
            per_t[t] = q.mean()
        aps.append(float(per_t.mean()))
    return float(np.mean(aps)) if aps else 0.0

W1 = {a: ARMMASK[a].astype(np.int64) for a in ('A', 'B')}

# ---------- validate the routine against pycocotools ----------
val = []
for r in ROWS[:8]:
    for a in ('A', 'B'):
        mine = ap_weighted(r['tag'], W1[a])
        val.append({'tag': r['tag'], 'arm': a, 'mine': round(mine, 6),
                    'pycocotools': round(r[a]['AP'], 6), 'abs_diff': abs(mine - r[a]['AP'])})
mx = max(v['abs_diff'] for v in val)
print('scorer validation vs pycocotools: max |diff| = %.3e over %d checks' % (mx, len(val)), flush=True)
for v in val[:4]:
    print('   %-44s %s mine=%.6f coco=%.6f' % (v['tag'], v['arm'], v['mine'], v['pycocotools']))
assert mx < 5e-4, 'subset scorer does not reproduce pycocotools: ' + str(mx)
json.dump([{k: (float(x) if k == 'abs_diff' else x) for k, x in v.items()} for v in val],
          open(OUT / 'scorer_validation.json', 'w'), indent=1)
print('VALIDATED', flush=True)

# ---------- frozen comparison plan, identical to the accepted Phase 9 matrix ----------
P, S, X = 'PRIMARY', 'SECONDARY', 'EXPLORATORY'
PLAN = [
 (P, 'YOLO11n vs YOLO11s',          ('runs_p3/yolo11n', None),  ('runs_p2b/D', None),   [0,1,2,3,4]),
 (P, 'YOLO11m vs YOLO11s',          ('runs_p3/yolo11m', None),  ('runs_p2b/D', None),   [0,1,2,3,4]),
 (P, 'scale B (P3+P4) vs A',        ('runs_p6/B', None),        ('runs_p4/r512', None), [0,1,2,3,4]),
 (P, 'scale C (P4+P5) vs A',        ('runs_p6/C', None),        ('runs_p4/r512', None), [0,1,2,3,4]),
 (S, '320 vs 640',                  ('runs_p4/r320', None),     ('runs_p3/yolo11n', None), [0,1,2,3,4]),
 (S, '416 vs 640',                  ('runs_p4/r416', None),     ('runs_p3/yolo11n', None), [0,1,2,3,4]),
 (S, '512 vs 640',                  ('runs_p4/r512', None),     ('runs_p3/yolo11n', None), [0,1,2,3,4]),
 (S, 'no-aug originals vs shipped', ('runs_p2b/B', None),       ('runs_p2b/D', None),   [0,1,2,3,4]),
 (S, 'duplicate-8x vs shipped',     ('runs_p2b/C', None),       ('runs_p2b/D', None),   [0,1,2,3,4]),
 (S, 'ADCF residual vs baseline',   ('runs/yolo11s-adcf', 50),  ('runs/yolo11s', 50),   [0,1,2]),
 (S, 'ADCF replace vs baseline',    ('runs/yolo11s-adcf-replace', 50), ('runs/yolo11s', 50), [0,1,2]),
]
EXPL = [(X, 'CBAM', ('runs/yolo11s-cbam', 25), ('runs/yolo11s', 25)),
        (X, '960 px', ('runs/yolo11s-960', 25), ('runs/yolo11s', 25)),
        (X, 'P2 head', ('runs/yolo11s-p2', 25), ('runs/yolo11s', 25))]

BY = {(r['run'], r['patience'], r['seed']): r for r in ROWS}
def tg(spec, seed):
    return BY[(spec[0], spec[1], seed)]['tag']
def Mv(spec, seed, arm, key='AP'):
    return BY[(spec[0], spec[1], seed)][arm][key] * 100.0

def paired(d):
    d = np.asarray(d, float); n = d.size
    sd = float(d.std(ddof=1)) if n > 1 else float('nan')
    tt = st.ttest_rel(d, np.zeros(n)) if n > 1 else None
    ci = st.t.interval(0.95, n-1, loc=d.mean(), scale=st.sem(d)) if (n > 1 and sd > 0) else (float('nan'),)*2
    try:
        wp = float(st.wilcoxon(d).pvalue) if n > 1 else float('nan')
    except Exception:
        wp = float('nan')
    return {'per_seed': [round(float(x), 4) for x in d], 'mean': float(d.mean()), 'sd': sd,
            'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]),
            't': float(tt.statistic) if tt else float('nan'),
            'p': float(tt.pvalue) if tt else float('nan'),
            'dz': float(d.mean()/sd) if (sd and sd > 0) else float('nan'),
            'wilcoxon_p': wp, 'n_positive': int((d > 0).sum()), 'n': n,
            'sign_consistent': bool((d > 0).all() or (d < 0).all())}

RES = {'scorer': 'pycocotools-validated COCO AP@[.50:.95], maxDets=100, 8 categories',
       'comparisons': [], 'exploratory': [], 'inflation': []}

for r in ROWS:
    RES['inflation'].append({
        'run': r['run'], 'patience': r['patience'], 'seed': r['seed'], 'imgsz': r['imgsz'],
        'AP_A': round(r['A']['AP']*100, 4), 'AP_B': round(r['B']['AP']*100, 4),
        'inflation_A_minus_B': round((r['A']['AP']-r['B']['AP'])*100, 4),
        'ratio_A_over_B': round(r['A']['AP']/r['B']['AP'], 4) if r['B']['AP'] > 0 else None,
        'AP50_A': round(r['A']['AP50']*100, 4), 'AP50_B': round(r['B']['AP50']*100, 4),
        'AP75_A': round(r['A']['AP75']*100, 4), 'AP75_B': round(r['B']['AP75']*100, 4),
        'AR100_A': round(r['A']['AR100']*100, 4), 'AR100_B': round(r['B']['AR100']*100, 4),
        'APs_A': round(r['A']['APs']*100, 4), 'APs_B': round(r['B']['APs']*100, 4)})

for tier, name, iv, bl, seeds in PLAN:
    ps_rows, dA, dB, I = [], [], [], []
    for s in seeds:
        aiv, abl = Mv(iv, s, 'A'), Mv(bl, s, 'A')
        biv, bbl = Mv(iv, s, 'B'), Mv(bl, s, 'B')
        dA.append(aiv-abl); dB.append(biv-bbl); I.append((biv-bbl)-(aiv-abl))
        ps_rows.append({'seed': s, 'A_iv': round(aiv,4), 'A_bl': round(abl,4),
                        'B_iv': round(biv,4), 'B_bl': round(bbl,4),
                        'delta_A': round(aiv-abl,4), 'delta_B': round(biv-bbl,4),
                        'I': round((biv-bbl)-(aiv-abl),4)})
    RES['comparisons'].append({'tier': tier, 'comparison': name, 'intervention': iv[0],
                               'baseline': bl[0], 'seeds': seeds, 'per_seed': ps_rows,
                               'delta_A': paired(dA), 'delta_B': paired(dB),
                               'interaction': paired(I)})

prim = [c for c in RES['comparisons'] if c['tier'] == P]
pv = np.array([c['interaction']['p'] for c in prim]); m = pv.size
adj = np.empty(m); rm = 0.0
for rank, i in enumerate(np.argsort(pv)):
    rm = max(rm, (m-rank)*pv[i]); adj[i] = min(1.0, rm)
for c, a in zip(prim, adj):
    c['interaction']['p_holm'] = float(a)
RES['holm'] = {'family': 'the four confirmatory interaction hypotheses', 'method': 'Holm',
               'raw_p': pv.tolist(), 'adj_p': adj.tolist()}

for tier, name, iv, bl in EXPL:
    si = sorted({k[2] for k in BY if (k[0], k[1]) == iv})
    sb = sorted({k[2] for k in BY if (k[0], k[1]) == bl})
    RES['exploratory'].append({
        'tier': tier, 'comparison': name, 'intervention': iv[0], 'baseline': bl[0],
        'intervention_seeds': si, 'baseline_seeds': sb, 'paired_seeds': sorted(set(si)&set(sb)),
        'interaction': 'NOT COMPUTABLE - zero seed overlap between intervention and baseline',
        'inflation_intervention': [{'seed': s, 'AP_A': round(Mv(iv,s,'A'),4), 'AP_B': round(Mv(iv,s,'B'),4),
                                    'A_minus_B': round(Mv(iv,s,'A')-Mv(iv,s,'B'),4)} for s in si],
        'inflation_baseline': [{'seed': s, 'AP_A': round(Mv(bl,s,'A'),4), 'AP_B': round(Mv(bl,s,'B'),4),
                                'A_minus_B': round(Mv(bl,s,'A')-Mv(bl,s,'B'),4)} for s in sb]})

print('across-seed statistics:', flush=True)
for c in RES['comparisons']:
    i = c['interaction']
    print('  %-11s %-32s dA=%+8.3f dB=%+7.3f  I=%+8.3f+-%6.3f p=%.4f%s' %
          (c['tier'], c['comparison'], c['delta_A']['mean'], c['delta_B']['mean'],
           i['mean'], i['sd'], i['p'],
           '  holm=%.4f' % i['p_holm'] if 'p_holm' in i else ''), flush=True)

# ---------- source-family bootstrap: the family is the resampling unit ----------
# All files of a sampled family move together. Individual augmented files are NEVER resampled as
# independent observations. One resample draws families for arm A and arm B independently, then
# every needed checkpoint is scored under that single resample so the paired structure is preserved.
TAGS = sorted({tg(iv, s) for _, _, iv, bl, seeds in PLAN for s in seeds} |
              {tg(bl, s) for _, _, iv, bl, seeds in PLAN for s in seeds})
print('bootstrap over %d unique checkpoints, %d resamples' % (len(TAGS), NBOOT), flush=True)
t0 = time.time()
for t in TAGS:
    prep(t)
print('prep done in %.0fs' % (time.time()-t0), flush=True)

def wvec(arm, fidx):
    w = np.zeros(NIMG, dtype=np.int64)
    for k in fidx:
        np.add.at(w, FAMPOS[arm][k], 1)
    return w

BA = {t: np.empty(NBOOT) for t in TAGS}
BB = {t: np.empty(NBOOT) for t in TAGS}
t0 = time.time()
for b in range(NBOOT):
    wa = wvec('A', RNG.integers(0, NFAM['A'], NFAM['A']))
    wb = wvec('B', RNG.integers(0, NFAM['B'], NFAM['B']))
    for t in TAGS:
        BA[t][b] = ap_weighted(t, wa) * 100.0
        BB[t][b] = ap_weighted(t, wb) * 100.0
    if (b+1) % 50 == 0:
        el = time.time()-t0
        print('  boot %d/%d  %.0fs elapsed, ~%.0fs left' %
              (b+1, NBOOT, el, el/(b+1)*(NBOOT-b-1)), flush=True)
print('bootstrap resampling done in %.0fs' % (time.time()-t0), flush=True)

def summ(x):
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)),
            'ci_lo': float(np.percentile(x, 2.5)), 'ci_hi': float(np.percentile(x, 97.5)),
            'p_two_sided_sign': float(2*min((x > 0).mean(), (x < 0).mean()))}

BOOT = []
for tier, name, iv, bl, seeds in PLAN:
    ti = [tg(iv, s) for s in seeds]; tb = [tg(bl, s) for s in seeds]
    bdA = np.mean([BA[a] - BA[z] for a, z in zip(ti, tb)], axis=0)
    bdB = np.mean([BB[a] - BB[z] for a, z in zip(ti, tb)], axis=0)
    BOOT.append({'tier': tier, 'comparison': name, 'n_boot': NBOOT,
                 'resampling_unit': 'source-photo family; all files of a sampled family kept together',
                 'families_resampled': {'A': NFAM['A'], 'B': NFAM['B']},
                 'delta_A': summ(bdA), 'delta_B': summ(bdB), 'interaction': summ(bdB - bdA)})
    print('  boot %-32s I=%+8.3f ci[%+8.3f,%+8.3f]' %
          (name, BOOT[-1]['interaction']['mean'], BOOT[-1]['interaction']['ci_lo'],
           BOOT[-1]['interaction']['ci_hi']), flush=True)
RES['bootstrap'] = BOOT
RES['uncertainty_note'] = ('Two uncertainty sources are reported SEPARATELY and never pooled into a '
  'single n: (1) across-seed uncertainty - paired t over n=5 (or n=3) training seeds, which reflects '
  'training stochasticity; (2) finite-evaluation-family uncertainty - bootstrap over the 1047 exposed '
  'and 213 independent source families, which reflects the finite evaluation set. The family bootstrap '
  'holds the trained checkpoints fixed and therefore does NOT contain seed variance.')
RES['absolute_inflation_boot'] = {
    t: {'A': summ(BA[t]), 'B': summ(BB[t]), 'A_minus_B': summ(BA[t] - BB[t])} for t in TAGS}

# ---------- ranking stability over the frozen fixed-budget comparable family ----------
RANKSET = [(('runs_p3/yolo11n', None), 'YOLO11n 640'), (('runs_p2b/D', None), 'YOLO11s 640 shipped-aug'),
           (('runs_p3/yolo11m', None), 'YOLO11m 640'), (('runs_p4/r320', None), 'YOLO11n 320'),
           (('runs_p4/r416', None), 'YOLO11n 416'), (('runs_p4/r512', None), 'YOLO11n 512'),
           (('runs_p6/B', None), 'scale B 512'), (('runs_p6/C', None), 'scale C 512'),
           (('runs_p2b/B', None), 'YOLO11s no-aug'), (('runs_p2b/C', None), 'YOLO11s dup8')]
SD = [0, 1, 2, 3, 4]
rk = []
for spec, label in RANKSET:
    rk.append({'model': label, 'run': spec[0],
               'AP_A': round(float(np.mean([Mv(spec, s, 'A') for s in SD])), 4),
               'AP_B': round(float(np.mean([Mv(spec, s, 'B') for s in SD])), 4)})
ra = st.rankdata([-x['AP_A'] for x in rk]); rb = st.rankdata([-x['AP_B'] for x in rk])
for x, i, j in zip(rk, ra, rb):
    x['rank_A'] = int(i); x['rank_B'] = int(j); x['rank_shift'] = int(j-i)
sp = st.spearmanr([x['AP_A'] for x in rk], [x['AP_B'] for x in rk])
kt = st.kendalltau([x['AP_A'] for x in rk], [x['AP_B'] for x in rk])
rev = []
for i, j in itertools.combinations(range(len(rk)), 2):
    sa = np.sign(rk[i]['AP_A']-rk[j]['AP_A']); sb = np.sign(rk[i]['AP_B']-rk[j]['AP_B'])
    if sa and sb and sa != sb:
        rev.append({'pair': [rk[i]['model'], rk[j]['model']],
                    'A_gap': round(rk[i]['AP_A']-rk[j]['AP_A'], 3),
                    'B_gap': round(rk[i]['AP_B']-rk[j]['AP_B'], 3)})
# bootstrap the rank correlation at family level
rhos = np.empty(NBOOT)
for b in range(NBOOT):
    va = [np.mean([BA[tg(spec, s)][b] for s in SD]) for spec, _ in RANKSET]
    vb = [np.mean([BB[tg(spec, s)][b] for s in SD]) for spec, _ in RANKSET]
    rhos[b] = st.spearmanr(va, vb).statistic
RES['ranking'] = {
  'comparable_family': ('fixed 24,570-update budget, frozen checkpoint grid read at 18,270 '
                        '(18,170 for the no-aug arm), 5 paired seeds, no early stopping, no '
                        'validation-based checkpoint selection, identical optimizer and schedule shape'),
  'n_models': len(rk), 'models': rk, 'spearman_rho': float(sp.statistic),
  'spearman_p': float(sp.pvalue), 'kendall_tau': float(kt.statistic), 'kendall_p': float(kt.pvalue),
  'spearman_boot_ci': [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
  'n_pairs': len(rk)*(len(rk)-1)//2, 'n_reversals': len(rev), 'reversals': rev}
print('ranking: spearman rho=%.3f (boot ci %.3f..%.3f), reversals %d/%d' %
      (sp.statistic, np.percentile(rhos, 2.5), np.percentile(rhos, 97.5), len(rev),
       len(rk)*(len(rk)-1)//2), flush=True)

json.dump(RES, open(OUT / 'p9_results.json', 'w'), indent=1)
print('STATSDONE', flush=True)
