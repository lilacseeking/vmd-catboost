"""非零月条件覆盖率改进实验 (v2)。

对比三种上界策略:
  v1: upper = P × q_high                          (基线, 非零月覆盖0.68)
  v2-A: upper = max(q_high, P×q_high)? NO
        → 条件化: P>0.5 时 upper=q_high; else upper=P×q_high
  v2-B: 分位上调到0.9 + 条件化

指标: 边际覆盖率 / 零值月覆盖率 / 非零月覆盖率 / 区间宽度
"""
import sys, io, warnings, json, os
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb

TRAIN, TEST = 62, 12
CALIB = 12
FILES = ['inputs/data.xlsx', 'inputs/data_jibei_20.xlsx', 'inputs/data_highfreq.xlsx',
         'inputs/data_sgcc.xlsx', 'inputs/data_jibei.xlsx']


def load_material(file, sheet):
    df = pd.read_excel(file, sheet_name=sheet)
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if 'demand' in cl or '需求' in str(c): rename[c] = 'demand'
        elif 'date' in cl or '日期' in str(c): rename[c] = 'date'
    df = df.rename(columns=rename)
    df['date'] = pd.to_datetime(df['date'])
    return df['demand'].astype(float).values


def build_self(y_tr, y_te):
    y_all = np.concatenate([y_tr, y_te])
    def extr(N, offset):
        F = np.zeros((N, 8))
        for i in range(N):
            t = offset + i
            F[i,0]=y_all[t-1] if t>=1 else 0.0
            F[i,1]=y_all[t-2] if t>=2 else 0.0
            F[i,2]=y_all[t-3] if t>=3 else 0.0
            F[i,3]=y_all[t-6] if t>=6 else 0.0
            F[i,4]=y_all[t-12] if t>=12 else 0.0
            F[i,5]=np.mean(y_all[max(0,t-2):t+1]) if t>=1 else y_all[0]
            F[i,6]=np.mean(y_all[max(0,t-5):t+1]) if t>=1 else y_all[0]
            last=-1
            for j in range(t-1,-1,-1):
                if y_all[j]>0: last=j; break
            F[i,7]=float(t-last) if last>=0 else 99.0
        return F
    return extr(TRAIN, 0), extr(TEST, TRAIN)


def zi_conformal_v2(y_all, strategy, alpha=0.2):
    """零膨胀感知共形, 3种上界策略。
    strategy: 'v1' | 'A' | 'B'
    """
    y_tr, y_te = y_all[:TRAIN], y_all[TRAIN:]
    L_tr, L_te = build_self(y_tr, y_te)
    proper_n = TRAIN - CALIB
    # Stage1: 分类P
    y_bin = (y_tr > 0).astype(int)
    if y_bin.sum() < 5 or (len(y_bin)-y_bin.sum()) < 5:
        return None
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                             num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    clf.fit(L_tr[:proper_n], y_bin[:proper_n])
    P = np.clip(clf.predict_proba(L_te)[:, 1], 0.05, 0.95)
    # Stage2: 规模区间
    nz = y_tr > 0
    L_nz, y_nz = L_tr[nz], np.log1p(y_tr[nz])
    nz_idx = np.where(nz)[0]
    calib_mask = nz_idx >= proper_n
    if calib_mask.sum() < 3 or (~calib_mask).sum() < 5:
        return None
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                              num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    model.fit(L_nz[~calib_mask], y_nz[~calib_mask])
    calib_pred = model.predict(L_nz[calib_mask])
    scores = np.abs(y_nz[calib_mask] - calib_pred)
    if len(scores) < 3: return None
    # 分位数: v1/A 用 (1-alpha); B 用 0.9
    q_level = 0.9 if strategy == 'B' else (1 - alpha)
    q = np.quantile(scores, q_level)
    pred_log = model.predict(L_te)
    q_high = np.maximum(np.expm1(pred_log + q), 0)

    # 上界策略
    if strategy == 'v1':
        upper = P * q_high
    elif strategy == 'A':
        # 条件化: P>0.5 用纯规模, 否则 P×规模
        upper = np.where(P > 0.5, q_high, P * q_high)
    elif strategy == 'B':
        upper = np.where(P > 0.5, q_high, P * q_high)
    else:
        raise ValueError(strategy)
    lower = np.zeros_like(upper)

    nz_te = y_te > 0
    cov_all = np.mean((y_te >= lower) & (y_te <= upper))
    cov_zero = np.mean((y_te[~nz_te] >= lower[~nz_te]) & (y_te[~nz_te] <= upper[~nz_te])) if (~nz_te).sum() > 0 else np.nan
    cov_pos = np.mean((y_te[nz_te] >= lower[nz_te]) & (y_te[nz_te] <= upper[nz_te])) if nz_te.sum() > 0 else np.nan
    width = np.mean(upper - lower)
    # 相对宽度(仅非零预测月)
    nz_pred = upper > 0
    rel_width = np.mean((upper[nz_pred] - lower[nz_pred]) / upper[nz_pred]) if nz_pred.sum() > 0 else np.inf
    return {'cov_all': float(cov_all), 'cov_zero': float(cov_zero), 'cov_pos': float(cov_pos),
            'width': float(width), 'rel_width': float(rel_width), 'nz_te': int(nz_te.sum())}


def main_run():
    stats = {k: [] for k in ['v1', 'A', 'B']}
    rows = []
    seen = set()
    for f in FILES:
        if not os.path.exists(f): continue
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            if sheet in seen: continue
            seen.add(sheet)
            try:
                y = load_material(f, sheet)
                if len(y) < TRAIN + TEST: continue
                y_all = y[-81:][:TRAIN+TEST]
                if len(y_all) < TRAIN + TEST:
                    y_all = np.pad(y_all, (0, TRAIN+TEST-len(y_all)))
                nz_tr = int((y_all[:TRAIN] > 0).sum())
                rec = {'material': sheet, 'NZ': nz_tr}
                ok = True
                for strat in ['v1', 'A', 'B']:
                    r = zi_conformal_v2(y_all, strat)
                    if r is None:
                        ok = False; break
                    rec[f'{strat}_cov_all'] = r['cov_all']
                    rec[f'{strat}_cov_pos'] = r['cov_pos']
                    rec[f'{strat}_cov_zero'] = r['cov_zero']
                    rec[f'{strat}_width'] = r['width']
                    stats[strat].append(r)
                if ok:
                    rows.append(rec)
                    print(f"  {sheet[:26]:<28} NZ={nz_tr:>3} | "
                          f"v1 pos={rec['v1_cov_pos']:.2f} all={rec['v1_cov_all']:.2f} | "
                          f"A  pos={rec['A_cov_pos']:.2f} all={rec['A_cov_all']:.2f} | "
                          f"B  pos={rec['B_cov_pos']:.2f} all={rec['B_cov_all']:.2f}")
            except Exception as e:
                print(f"  [ERR] {sheet[:30]}: {str(e)[:50]}")

    print('\n' + '=' * 84)
    print(f'有效物资: {len(rows)}  目标覆盖率: 0.8')
    for strat, label in [('v1', 'v1 基线 P×q_high'), ('A', 'A 条件化上界'), ('B', 'B 分位0.9+条件化')]:
        s = stats[strat]
        if not s: continue
        ca = [x['cov_all'] for x in s]; cp = [x['cov_pos'] for x in s]; cz = [x['cov_zero'] for x in s]
        w = [x['width'] for x in s]
        print(f'\n{label}:')
        print(f'  边际覆盖率: mean={np.mean(ca):.3f} median={np.median(ca):.3f}')
        print(f'  零值月覆盖率: mean={np.mean(cz):.3f} median={np.median(cz):.3f}')
        print(f'  非零月覆盖率: mean={np.mean(cp):.3f} median={np.median(cp):.3f}  (>=0.7: {sum(1 for x in cp if x>=0.7)}/{len(cp)}, >=0.8: {sum(1 for x in cp if x>=0.8)}/{len(cp)})')
        print(f'  平均宽度: {np.mean(w):.2f}')
    # 配对比较 A vs v1
    print('\n=== 改进效果 (非零月覆盖率) ===')
    for strat in ['A', 'B']:
        pair = [(r[f'{strat}_cov_pos'], r['v1_cov_pos']) for r in rows]
        imp = sum(1 for a, b in pair if a > b)
        mean_d = np.mean([a - b for a, b in pair])
        print(f'  {strat} vs v1: 提升物资={imp}/{len(pair)}, 平均Δ非零覆盖={mean_d:+.3f}')

    with open('experiments/zi_conformal_v2.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print('\n已保存: experiments/zi_conformal_v2.json')


if __name__ == '__main__':
    main_run()
