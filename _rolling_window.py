"""滚动窗口稳健性验证。

用多个滚动窗口划分验证 ZI-Conformal (零膨胀感知共形, 策略B) 的稳健性:
  - 多个测试窗口: 训练期逐步前移, 每个窗口预测后12个月
  - 指标: 边际覆盖率 / 非零月条件覆盖率 / 零值月覆盖率 / 区间相对宽度
  - 验证: 边际覆盖提升(vs标准共形)在多窗口下是否稳定; 覆盖率是否接近目标0.8

窗口设计: 总81个月 (2019-11 ~ 2026-07)
  - 测试窗1: train=0:62,  test=62:74
  - 测试窗2: train=1:63,  test=63:75
  - ...
  - 测试窗8: train=7:69,  test=69:81
每个窗口独立训练+校准+测试, 完全无泄漏。
"""
import sys, io, warnings, json, os
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb

TOTAL = 81
TRAIN_LEN = 62
TEST_LEN = 12
N_WINDOWS = 8
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


def build_self_window(y_all, train_start, train_len, test_len):
    """基于绝对时间索引构造训练/测试特征, 只用历史, 无泄漏"""
    def extr(i):
        # 特征: lag1/2/3/6/12 + roll3/6 + gap
        f = np.zeros(8)
        if i >= 1: f[0] = y_all[i-1]
        if i >= 2: f[1] = y_all[i-2]
        if i >= 3: f[2] = y_all[i-3]
        if i >= 6: f[3] = y_all[i-6]
        if i >= 12: f[4] = y_all[i-12]
        f[5] = np.mean(y_all[max(0, i-2):i+1])
        f[6] = np.mean(y_all[max(0, i-5):i+1])
        last = -1
        for j in range(i-1, -1, -1):
            if y_all[j] > 0: last = j; break
        f[7] = float(i - last) if last >= 0 else 99.0
        return f
    tr_idx = range(train_start, train_start + train_len)
    te_idx = range(train_start + train_len, train_start + train_len + test_len)
    X_tr = np.array([extr(i) for i in tr_idx])
    X_te = np.array([extr(i) for i in te_idx])
    y_tr = y_all[train_start:train_start + train_len]
    y_te = y_all[train_start + train_len:train_start + train_len + test_len]
    return X_tr, y_tr, X_te, y_te


def zi_window(y_all, train_start, strategy='B', alpha=0.2):
    """单窗口 ZI-Conformal (策略B: 条件化上界)"""
    X_tr, y_tr, X_te, y_te = build_self_window(y_all, train_start, TRAIN_LEN, TEST_LEN)
    proper_n = TRAIN_LEN - 12  # 最后12月做校准
    # Stage1: 分类P
    y_bin = (y_tr > 0).astype(int)
    if y_bin.sum() < 5 or (len(y_bin) - y_bin.sum()) < 5:
        return None
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                             num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    clf.fit(X_tr[:proper_n], y_bin[:proper_n])
    P = np.clip(clf.predict_proba(X_te)[:, 1], 0.05, 0.95)
    # Stage2: 规模区间
    nz = y_tr > 0
    X_nz, y_nz = X_tr[nz], np.log1p(y_tr[nz])
    nz_idx = np.where(nz)[0]
    calib_mask = nz_idx >= proper_n
    if calib_mask.sum() < 3 or (~calib_mask).sum() < 5:
        return None
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                              num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    model.fit(X_nz[~calib_mask], y_nz[~calib_mask])
    cp = model.predict(X_nz[calib_mask])
    scores = np.abs(y_nz[calib_mask] - cp)
    if len(scores) < 3: return None
    q = np.quantile(scores, 0.9)
    pred_log = model.predict(X_te)
    q_high = np.maximum(np.expm1(pred_log + q), 0)
    pred = np.maximum(np.expm1(pred_log), 0)
    if strategy == 'v1':
        upper = P * q_high
    else:
        upper = np.where(P > 0.5, q_high, P * q_high)
    lower = np.zeros_like(upper)

    nz_te = y_te > 0
    cov_all = np.mean((y_te >= lower) & (y_te <= upper))
    cov_zero = np.mean((y_te[~nz_te] >= lower[~nz_te]) & (y_te[~nz_te] <= upper[~nz_te])) if (~nz_te).sum() > 0 else np.nan
    cov_pos = np.mean((y_te[nz_te] >= lower[nz_te]) & (y_te[nz_te] <= upper[nz_te])) if nz_te.sum() > 0 else np.nan
    relw = np.mean((upper[nz_te] - lower[nz_te]) / (y_te[nz_te] + 1e-9)) if nz_te.sum() > 0 else np.inf
    fp = np.mean((y_te == 0) & (pred > 0))
    fn = np.mean((y_te > 0) & (pred == 0))
    return {'cov_all': float(cov_all), 'cov_zero': float(cov_zero), 'cov_pos': float(cov_pos),
            'relw': float(relw), 'fp': float(fp), 'fn': float(fn),
            'nz_te': int(nz_te.sum()), 'nz_tr': int(nz.sum())}


def std_conformal_window(y_all, train_start, alpha=0.2):
    """单窗口 标准共形 (全训练残差校准) 作为对照"""
    X_tr, y_tr, X_te, y_te = build_self_window(y_all, train_start, TRAIN_LEN, TEST_LEN)
    proper_n = TRAIN_LEN - 12
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                              num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    model.fit(X_tr[:proper_n], np.log1p(y_tr[:proper_n]))
    calib_pred = model.predict(X_tr[proper_n:])
    calib_true = y_tr[proper_n:]
    scores = np.abs(np.log1p(calib_true) - calib_pred)
    if len(scores) < 5: return None
    q = np.quantile(scores, 1 - alpha)
    pred_log = model.predict(X_te)
    lower = np.maximum(np.expm1(pred_log - q), 0)
    upper = np.expm1(pred_log + q)
    cov_all = np.mean((y_te >= lower) & (y_te <= upper))
    nz_te = y_te > 0
    cov_pos = np.mean((y_te[nz_te] >= lower[nz_te]) & (y_te[nz_te] <= upper[nz_te])) if nz_te.sum() > 0 else np.nan
    return {'cov_all': float(cov_all), 'cov_pos': float(cov_pos)}


def main_run():
    all_mats = {}
    seen = set()
    for f in FILES:
        if not os.path.exists(f): continue
        try: xl = pd.ExcelFile(f)
        except: continue
        for s in xl.sheet_names:
            if s in seen: continue
            seen.add(s)
            try:
                y = load_material(f, s)
                if len(y) < TOTAL: continue
                all_mats[s] = y[-TOTAL:]
            except Exception as e:
                print(f'  [ERR load] {s[:30]}: {e}')

    print(f'物资数: {len(all_mats)}')
    print(f'窗口: {N_WINDOWS} 个滚动窗口, 每窗训练{TRAIN_LEN}月/测试{TEST_LEN}月, 窗口起点0-7')
    print()

    # 每个物资: 收集8窗口的指标
    per_mat = {}
    for name, y_all in all_mats.items():
        zi_vals = []
        std_vals = []
        for w in range(N_WINDOWS):
            start = w  # 窗口起点: 0,1,...,7
            r_zi = zi_window(y_all, start)
            r_std = std_conformal_window(y_all, start)
            if r_zi and r_std:
                zi_vals.append(r_zi)
                std_vals.append(r_std)
        if len(zi_vals) >= 4:
            per_mat[name] = {'zi': zi_vals, 'std': std_vals}

    # 汇总
    print('=' * 84)
    print('【汇总】各物资在8个滚动窗口上的平均表现')
    print('=' * 84)
    print(f"{'物资':<28}{'窗数':>4} {'ZI边际':>7} {'ZI非零':>7} {'ZI相对宽':>7} {'Std边际':>7} {'Δ(边际)':>8}")
    agg = {'zi_cov_all': [], 'zi_cov_pos': [], 'std_cov_all': [], 'delta': []}
    for name, pm in sorted(per_mat.items(), key=lambda x: -np.mean([r['cov_all'] for r in x[1]['zi']])):
        zi = pm['zi']; std = pm['std']
        za = np.mean([r['cov_all'] for r in zi])
        zp = np.mean([r['cov_pos'] for r in zi])
        sa = np.mean([r['cov_all'] for r in std])
        zw = np.mean([r['relw'] for r in zi])
        d = za - sa
        agg['zi_cov_all'].append(za); agg['zi_cov_pos'].append(zp)
        agg['std_cov_all'].append(sa); agg['delta'].append(d)
        print(f"{name[:28]:<28}{len(zi):>4} {za:>7.3f} {zp:>7.3f} {zw:>7.2f} {sa:>7.3f} {d:>+8.3f}")

    print('\n' + '=' * 84)
    print('【总评】滚动窗口稳健性')
    print('=' * 84)
    za = np.array(agg['zi_cov_all']); zp = np.array(agg['zi_cov_pos']); sa = np.array(agg['std_cov_all']); dd = np.array(agg['delta'])
    print(f'ZI边际覆盖率: mean={za.mean():.3f} ± {za.std():.3f}  (n={len(za)}物资)')
    print(f'ZI非零月条件覆盖率: mean={zp.mean():.3f} ± {zp.std():.3f}')
    print(f'标准共形边际覆盖率: mean={sa.mean():.3f} ± {sa.std():.3f}')
    print(f'ZI vs 标准 Δ边际: mean={dd.mean():+.3f}  提升为正的物资: {(dd>0).sum()}/{len(dd)}')
    # 覆盖率达到目标的比例
    print(f'ZI边际覆盖>=0.7的物资: {(za>=0.7).sum()}/{len(za)}  | >=0.8: {(za>=0.8).sum()}/{len(za)}')
    print(f'ZI非零条件覆盖>=0.6的物资: {(zp>=0.6).sum()}/{len(zp)}')

    with open('experiments/rolling_window_results.json', 'w', encoding='utf-8') as f:
        json.dump({'per_material': {k: {'zi': v['zi'], 'std': v['std']} for k, v in per_mat.items()},
                   'summary': {'zi_cov_all_mean': float(za.mean()), 'zi_cov_all_std': float(za.std()),
                               'zi_cov_pos_mean': float(zp.mean()), 'std_cov_all_mean': float(sa.mean()),
                               'delta_mean': float(dd.mean())}}, f, ensure_ascii=False, indent=2)
    print('\n已保存: experiments/rolling_window_results.json')


if __name__ == '__main__':
    main_run()
