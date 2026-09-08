"""分位数回归 滚动窗口实验 (补跑, 与标准共形/ZI一致: 39物资 x 8窗口)。

对每个滚动窗口跑 LightGBM 分位数回归 (alpha=0.1/0.9),
记录 边际/非零/零值月覆盖。
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
    def extr(i):
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


def quantile_window(y_all, train_start, alpha_low=0.1, alpha_high=0.9):
    """分位数回归: alpha=0.1/0.9 输出区间 [lo, hi], 计算条件覆盖"""
    X_tr, y_tr, X_te, y_te = build_self_window(y_all, train_start, TRAIN_LEN, TEST_LEN)
    proper_n = TRAIN_LEN - 12
    # 训练用 log1p
    y_log = np.log1p(y_tr)
    try:
        model_lo = lgb.LGBMRegressor(objective='quantile', alpha=alpha_low,
                                     n_estimators=200, learning_rate=0.05, max_depth=3,
                                     num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
        model_lo.fit(X_tr[:proper_n], y_log[:proper_n])
        model_hi = lgb.LGBMRegressor(objective='quantile', alpha=alpha_high,
                                     n_estimators=200, learning_rate=0.05, max_depth=3,
                                     num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
        model_hi.fit(X_tr[:proper_n], y_log[:proper_n])
        lo = np.maximum(np.expm1(model_lo.predict(X_te)), 0)
        hi = np.maximum(np.expm1(model_hi.predict(X_te)), 0)
    except Exception:
        return None

    nz_te = y_te > 0
    cov_all = np.mean((y_te >= lo) & (y_te <= hi))
    cov_zero = np.mean((y_te[~nz_te] >= lo[~nz_te]) & (y_te[~nz_te] <= hi[~nz_te])) if (~nz_te).sum() > 0 else np.nan
    cov_pos = np.mean((y_te[nz_te] >= lo[nz_te]) & (y_te[nz_te] <= hi[nz_te])) if nz_te.sum() > 0 else np.nan
    return {'cov_all': float(cov_all), 'cov_zero': float(cov_zero), 'cov_pos': float(cov_pos),
            'nz_te': int(nz_te.sum()), 'nz_tr': int((y_tr > 0).sum())}


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
            except Exception:
                pass

    print(f'物资数: {len(all_mats)}')
    results = {}
    for name, y_all in all_mats.items():
        wins = []
        for w in range(N_WINDOWS):
            r = quantile_window(y_all, w)
            if r: wins.append(r)
        if len(wins) >= 4:
            results[name] = wins

    # 汇总
    ca = [np.mean([x['cov_all'] for x in v]) for v in results.values()]
    cp = [np.mean([x['cov_pos'] for x in v if not np.isnan(x['cov_pos'])]) for v in results.values()]
    cz = [np.mean([x['cov_zero'] for x in v if not np.isnan(x['cov_zero'])]) for v in results.values()]
    print(f'有效物资: {len(results)}')
    print(f'分位数回归 边际覆盖: {np.mean(ca):.4f} ± {np.std(ca):.4f}')
    print(f'分位数回归 非零覆盖: {np.nanmean(cp):.4f} ± {np.nanstd(cp):.4f}')
    print(f'分位数回归 零值覆盖: {np.nanmean(cz):.4f} ± {np.nanstd(cz):.4f}')

    with open('experiments/qr_rolling_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存: experiments/qr_rolling_results.json')


if __name__ == '__main__':
    main_run()
