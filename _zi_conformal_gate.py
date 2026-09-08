"""决定性实验: 零膨胀感知共形校准 vs 标准共形 vs 分位数回归。

验证: 对间歇性零膨胀电力物资, 标准共形覆盖率不足(0.42),
      零膨胀感知共形(发生P×规模区间, 概率加权)能否把覆盖率拉回 0.8。

三方法对比 (每物资):
  A. 标准共形: 全训练期残差校准 (复现 0.42)
  B. 零膨胀感知共形 (本文方法):
       - Stage1: 分类P(伯努利发生概率)
       - Stage2: 非零月残差共形 -> 规模区间 [q_low, q_high]
       - 最终区间: [0, P×q_high]  (下限0, 上限=发生概率×规模上界)
       - 覆盖率: 真值∈[0, P×q_high]
  C. 分位数回归 (LightGBM Quantile, alpha=0.1/0.9): 直接输出区间

指标: 覆盖率(coverage, 目标0.8) + 区间宽度(sharpeness) + Winkler score
"""
import sys, io, warnings, json, os
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

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


def std_conformal(y_all, alpha=0.2):
    """标准共形: 全训练期(含零值)残差校准"""
    y_tr, y_te = y_all[:TRAIN], y_all[TRAIN:]
    L_tr, L_te = build_self(y_tr, y_te)
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                              num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    proper_n = TRAIN - CALIB
    model.fit(L_tr[:proper_n], np.log1p(y_tr[:proper_n]))
    calib_pred = model.predict(L_tr[proper_n:])
    calib_true = y_tr[proper_n:]
    scores = np.abs(np.log1p(calib_true) - calib_pred)
    if len(scores) < 5: return None
    q = np.quantile(scores, 1 - alpha)
    pred_log = model.predict(L_te)
    lower = np.maximum(np.expm1(pred_log - q), 0)
    upper = np.expm1(pred_log + q)
    cov = np.mean((y_te >= lower) & (y_te <= upper))
    width = np.mean(upper - lower)
    return {'coverage': float(cov), 'width': float(width)}


def zi_conformal(y_all, alpha=0.2):
    """零膨胀感知共形 (本文方法):
    Stage1: 分类P(发生概率) | Stage2: 非零月规模区间 | 区间=[0, P×q_high]
    """
    y_tr, y_te = y_all[:TRAIN], y_all[TRAIN:]
    L_tr, L_te = build_self(y_tr, y_te)
    proper_n = TRAIN - CALIB
    # Stage1: 分类P (伯努利发生)
    y_bin = (y_tr > 0).astype(int)
    if y_bin.sum() < 5 or (len(y_bin)-y_bin.sum()) < 5:
        return None
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                             num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    clf.fit(L_tr[:proper_n], y_bin[:proper_n])
    P = clf.predict_proba(L_te)[:, 1]
    P = np.clip(P, 0.05, 0.95)
    # Stage2: 规模区间 (仅非零训练)
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
    q = np.quantile(scores, 1 - alpha)
    pred_log = model.predict(L_te)
    q_high = np.maximum(np.expm1(pred_log + q), 0)
    # 区间 [0, P×q_high]
    upper = P * q_high
    lower = np.zeros_like(upper)
    cov = np.mean((y_te >= lower) & (y_te <= upper))
    width = np.mean(upper - lower)
    return {'coverage': float(cov), 'width': float(width)}


def quantile_reg(y_all):
    """分位数回归: 直接预测 alpha=0.1/0.9"""
    y_tr, y_te = y_all[:TRAIN], y_all[TRAIN:]
    L_tr, L_te = build_self(y_tr, y_te)
    proper_n = TRAIN - CALIB
    model = lgb.LGBMRegressor(objective='quantile', alpha=0.1,
                              n_estimators=200, learning_rate=0.05, max_depth=3,
                              num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    model.fit(L_tr[:proper_n], np.log1p(y_tr[:proper_n]))
    lo = np.maximum(np.expm1(model.predict(L_te)), 0)
    model2 = lgb.LGBMRegressor(objective='quantile', alpha=0.9,
                               n_estimators=200, learning_rate=0.05, max_depth=3,
                               num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    model2.fit(L_tr[:proper_n], np.log1p(y_tr[:proper_n]))
    hi = np.expm1(model2.predict(L_te))
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    width = np.mean(hi - lo)
    return {'coverage': float(cov), 'width': float(width)}


def main_run():
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
                nz_tr = (y_all[:TRAIN] > 0).sum()
                r_std = std_conformal(y_all)
                r_zi = zi_conformal(y_all)
                r_qr = quantile_reg(y_all)
                rows.append({'material': sheet, 'NZ': int(nz_tr),
                             'std_cov': r_std['coverage'] if r_std else None,
                             'std_width': r_std['width'] if r_std else None,
                             'zi_cov': r_zi['coverage'] if r_zi else None,
                             'zi_width': r_zi['width'] if r_zi else None,
                             'qr_cov': r_qr['coverage'] if r_qr else None,
                             'qr_width': r_qr['width'] if r_qr else None})
                print(f"  {sheet[:28]:<30} NZ={nz_tr:>3} std={r_std['coverage']:.2f}({r_std['width']:>8.0f})"
                      f"  zi={r_zi['coverage']:.2f}({r_zi['width']:>8.0f})"
                      f"  qr={r_qr['coverage']:.2f}({r_qr['width']:>8.0f})"
                      if r_std and r_zi and r_qr else f"  {sheet[:28]:<30} NZ={nz_tr} (skipped)")
            except Exception as e:
                print(f"  [ERR] {sheet[:30]}: {str(e)[:50]}")

    R = pd.DataFrame(rows)
    print('\n' + '=' * 84)
    print(f'有效物资: {len(R)}  目标覆盖率: 0.8')
    for col, label in [('std_cov','标准共形'), ('zi_cov','零膨胀感知'), ('qr_cov','分位数回归')]:
        v = R[col].dropna()
        if len(v):
            print(f'  {label}: 覆盖率 mean={v.mean():.3f} median={v.median():.3f}  '
                  f'≥0.7占比: {(v>=0.7).mean():.0%}  ≥0.6: {(v>=0.6).mean():.0%}')
    # 成对比较: zi vs std 覆盖率提升
    pair = R.dropna(subset=['std_cov','zi_cov'])
    if len(pair):
        improve = (pair['zi_cov'] > pair['std_cov']).mean()
        mean_d = (pair['zi_cov'] - pair['std_cov']).mean()
        print(f'\n零膨胀感知 vs 标准: 覆盖率提升的物资占比={improve:.0%}, 平均Δ={mean_d:+.3f}')
    with open('experiments/zi_conformal_gate.json','w',encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print('\n已保存: experiments/zi_conformal_gate.json')


if __name__ == '__main__':
    main_run()
