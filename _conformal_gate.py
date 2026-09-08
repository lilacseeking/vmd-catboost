"""生死线实验：验证"共形区间宽度是否与需求特征相关"。

对全物资(48)跑 Split Conformal Prediction (Vovk 2005):
  - 用训练期非零需求训练 LightGBM 点预测器
  - calibration 集计算绝对残差 -> 分位数 -> 区间半径
  - 对测试集输出 P10-P90 区间, 计算相对宽度 (区间宽度/预测值)
验证: 区间相对宽度 与 需求特征(CV/需求密度/脉冲率/非零量) 的 Spearman 相关。

若相关显著 -> "区间宽度可预测" 成立 -> 方案2立住
"""
import sys, io, warnings, json, os
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

TRAIN, TEST = 62, 12
CALIB = 12
MONTHS = [f"{y}{m:02d}" for y in range(2019, 2027) for m in range(1, 13)]

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
    """自回归特征 (lag1/2/3/6/12 + roll3/6 + gap), 与 conformal_prediction.py 一致"""
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


def run_conformal(y_all, alpha=0.2):
    """Split Conformal: 返回 (点预测, 区间下界, 区间上界, 覆盖率, 平均相对宽度)"""
    y_tr, y_te = y_all[:TRAIN], y_all[TRAIN:]
    nz_tr = y_tr > 0
    if nz_tr.sum() < 8:
        return None
    L_tr, L_te = build_self(y_tr, y_te)
    X_tr = L_tr; X_te = L_te
    # 点预测器: 仅非零训练
    # split by TIME: proper_train = 前50月非零, calibration = 最后12月非零 (时间错位, 无泄漏)
    proper_n = TRAIN - CALIB
    y_pos = np.log1p(y_tr[nz_tr])
    X_pos = X_tr[nz_tr]
    # 非零月份的时间索引
    nz_idx = np.where(nz_tr)[0]
    # calibration = 最后12个月内的非零样本
    calib_mask = nz_idx >= proper_n
    proper_mask = ~calib_mask
    X_proper, X_calib = X_pos[proper_mask], X_pos[calib_mask]
    y_proper, y_calib = y_pos[proper_mask], y_pos[calib_mask]
    if len(y_proper) < 5 or len(y_calib) < 3:
        # 非零样本太少, 校准失败
        return None
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                              num_leaves=15, min_child_samples=3, reg_alpha=0.5, reg_lambda=0.5,
                              random_state=42, verbose=-1)
    try:
        model.fit(X_proper, y_proper)
    except Exception:
        return None
    # calibration: 非一致性分数 = |真实 - 预测| on non-zero calib months
    y_calib_pred = model.predict(X_calib)
    scores = np.abs(y_calib - y_calib_pred)
    if len(scores) < 3:
        return None
    q = np.quantile(scores, 1 - alpha)
    # 测试集预测 (全月, 包括零值)
    pred_all_log = model.predict(X_te)
    pred_all = np.maximum(np.expm1(pred_all_log), 0)
    lower = np.maximum(np.expm1(pred_all_log - q), 0)
    upper = np.expm1(pred_all_log + q)
    # 覆盖率: 真实值是否在 [lower, upper]
    cover = np.mean((y_te >= lower) & (y_te <= upper))
    # 平均相对宽度 (仅非零预测月, 避免除零)
    nz_pred = pred_all > 0
    if nz_pred.sum() > 0:
        rel_width = np.mean((upper[nz_pred] - lower[nz_pred]) / pred_all[nz_pred])
    else:
        rel_width = np.inf
    return {'pred': pred_all, 'lower': lower, 'upper': upper,
            'coverage': cover, 'rel_width': rel_width}


def material_features(y_tr):
    """需求特征"""
    nz = y_tr > 0
    nz_vals = y_tr[nz]
    if len(nz_vals) > 1 and np.mean(nz_vals) > 0:
        cv = float(np.std(nz_vals) / np.mean(nz_vals))
    else:
        cv = 0.0
    density = float(nz.sum() / len(y_tr))
    # 脉冲率: 相邻非零间隔的变异
    idx = np.where(nz)[0]
    gaps = np.diff(idx) if len(idx) > 1 else np.array([0])
    gap_cv = float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 0 and np.mean(gaps) > 0 else 0.0
    return {'NZ': int(nz.sum()), 'density': density, 'CV': cv, 'gap_CV': gap_cv,
            'nz_mean': float(np.mean(nz_vals)) if len(nz_vals) else 0.0}


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
                y_all = y[-81:][:TRAIN+TEST]  # 对齐前81月中的前74
                if len(y_all) < TRAIN + TEST:
                    y_all = np.pad(y_all, (0, TRAIN+TEST-len(y_all)))
                feat = material_features(y_all[:TRAIN])
                res = run_conformal(y_all)
                if res is None:
                    print(f"  {sheet[:30]:<32} 非零样本不足, 跳过")
                    continue
                rows.append({'material': sheet, **feat,
                             'coverage': res['coverage'], 'rel_width': res['rel_width']})
                print(f"  {sheet[:30]:<32} NZ={feat['NZ']:>3} CV={feat['CV']:.2f} "
                      f"coverage={res['coverage']:.2f} rel_width={res['rel_width']:.2f}")
            except Exception as e:
                print(f"  [ERR] {sheet[:30]}: {str(e)[:50]}")

    if not rows:
        print('\n无有效物资, 方案2生死线无法验证')
        return
    R = pd.DataFrame(rows)
    print('\n' + '=' * 80)
    print(f'有效物资: {len(R)}')
    print(f'平均覆盖率: {R["coverage"].mean():.3f} (目标 {1-0.2:.1f})')
    print(f'平均相对宽度: {R["rel_width"].mean():.3f}')
    # 相关性: 相对宽度 vs 各特征
    print('\nSpearman(相对区间宽度, 需求特征):')
    for col in ['density', 'CV', 'gap_CV', 'NZ']:
        valid = R.dropna(subset=['rel_width', col])
        valid = valid[np.isfinite(valid['rel_width'])]
        if len(valid) >= 8:
            rho, p = spearmanr(valid[col], valid['rel_width'])
            print(f'  rel_width × {col:<8}: rho={rho:+.3f}  p={p:.4f}  (n={len(valid)})')
    # 分组对比: 高频 vs 低频 相对宽度
    hi = R[R['NZ'] >= 30]; lo = R[R['NZ'] < 30]
    if len(hi) and len(lo):
        print(f'\n高频(NZ≥30, n={len(hi)}): rel_width mean={hi["rel_width"].mean():.3f} median={hi["rel_width"].median():.3f}')
        print(f'中低频(NZ<30, n={len(lo)}): rel_width mean={lo["rel_width"].mean():.3f} median={lo["rel_width"].median():.3f}')

    with open('experiments/conformal_gate.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print('\n已保存: experiments/conformal_gate.json')


if __name__ == '__main__':
    main_run()
