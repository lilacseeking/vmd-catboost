"""方案B对比：把所有项目类型批次因子纳入候选池，让 Spearman 自动选择。

方案A (现状): _INTERNAL_FACTORS = [project_count, transformer_bids, monthly_bid_count, uhv_bids]
方案B (完整): 加 has_batch, digital_bids (data.xlsx/sgcc/jibei 有), 全部纳入 Spearman 候选

对比 ElasticNet-2S 的 R² / sMAPE。滚动窗口验证开关可随时关。
"""
import sys, io, warnings, json, os
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import main

# 强制灰色(与滚动窗口一致)
_orig_build_grey = main.build_grey_features
def build_grey_forced(demand_raw, train_len):
    if not main.USE_GREY_FEATURES:
        return np.zeros((len(demand_raw), 0)), np.zeros((len(demand_raw), 0))
    n = len(demand_raw); tl = train_len; d = np.maximum(demand_raw, 0)
    gm_fitted = np.zeros(n); gm_res = np.zeros(n); grey_a = np.zeros(n)
    for t in range(tl):
        start = max(0, t - 11); seg = d[start:t + 1]
        if len(seg) >= 4 and seg.sum() > 0:
            a, b = main._gm11_fit(seg); fitted = main._gm11_fitted(seg, a, b)
            gm_fitted[t] = fitted[-1]; gm_res[t] = d[t] - fitted[-1]; grey_a[t] = a
        else:
            gm_fitted[t] = np.mean(seg) if len(seg) > 0 else 0
    train_end_seg = d[max(0, tl - 12):tl]; test_len = n - tl
    if len(train_end_seg) >= 4 and train_end_seg.sum() > 0:
        a_hat, b_hat = main._gm11_fit(train_end_seg)
        preds = main._gm11_predict_next(train_end_seg[-1], a_hat, b_hat, test_len)
        gm_fitted[tl:] = preds; grey_a[tl:] = a_hat
    else:
        gm_fitted[tl:] = np.mean(train_end_seg) if len(train_end_seg) > 0 else 0
    best_r = 0.7; y_pos = np.maximum(d[:tl], 1e-6)
    if tl >= 5:
        best_smooth = float('inf')
        for r in [0.3, 0.5, 0.7, 0.9, 1.0]:
            try:
                fs = main._fractional_ago(y_pos[:min(20, tl)], r)
                sm = np.std(np.diff(fs))
                if sm < best_smooth: best_smooth = sm; best_r = r
            except: continue
    fago = np.zeros(n)
    try:
        fago_tr = main._fractional_ago(y_pos, best_r)
        scale = np.mean(y_pos[y_pos > 0]) / max(np.mean(fago_tr[fago_tr > 0]), 1e-6) if (y_pos > 0).any() else 1
        fago[:tl] = fago_tr * scale; fago[tl:] = fago[tl - 1]
    except: pass
    grey_tr = np.column_stack([gm_fitted[:tl], gm_res[:tl], grey_a[:tl], fago[:tl]])
    grey_te = np.column_stack([gm_fitted[tl:], gm_res[tl:], grey_a[tl:], fago[tl:]])
    return grey_tr, grey_te

main.build_grey_features = build_grey_forced
main.USE_GREY_FEATURES = True
main.USE_PAPER_FEATURES = False
main.USE_EVENT_FEATURES = True
main.USE_ENHANCED_LAG_FEATURES = True

# 可随时关掉的开关
USE_ALL_PROJECT_TYPES = True  # False = 方案A(只输变电), True = 方案B(全项目类型)

BASE_FACTORS = ['project_count', 'transformer_bids', 'monthly_bid_count', 'uhv_bids']
EXTRA_BATCH = ['has_batch', 'digital_bids']  # 其他项目类型批次

FILES = ['inputs/data.xlsx', 'inputs/data_sgcc.xlsx', 'inputs/data_jibei.xlsx', 'inputs/data_jibei_20.xlsx']


def load_material(file, sheet):
    df = pd.read_excel(file, sheet_name=sheet)
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if 'demand' in cl or '需求' in str(c): rename[c] = 'demand'
        elif 'date' in cl or '日期' in str(c): rename[c] = 'date'
        elif 'project' in cl or '项目' in str(c): rename[c] = 'project_count'
        elif 'transformer' in cl: rename[c] = 'transformer_bids'
        elif 'monthly' in cl or '公告' in str(c): rename[c] = 'monthly_bid_count'
        elif 'uhv' in cl or '特高压' in str(c): rename[c] = 'uhv_bids'
        elif 'has_batch' in cl or '批次' in str(c): rename[c] = 'has_batch'
        elif 'digital' in cl or '数字化' in str(c): rename[c] = 'digital_bids'
    df = df.rename(columns=rename)
    for col in ['project_count', 'transformer_bids', 'monthly_bid_count', 'uhv_bids', 'has_batch', 'digital_bids']:
        if col not in df.columns: df[col] = 0.0
    df['date'] = pd.to_datetime(df['date'])
    return df


def run_with_factors(df, material, factor_list):
    """用指定因子候选池跑 ElasticNet-2S"""
    main._INTERNAL_FACTORS = list(factor_list)
    main.FACTOR_NAMES = list(factor_list)
    main._top_factors_cache = {}
    main._mag_cache = {}
    main._grey_cache = {}
    X_tr, y_tr, X_te, y_te, _ = main.preprocess_data(df, material)
    yp, _, _, _ = main.run_elasticnet_2s(X_tr, y_tr, X_te, y_te, material)
    return float(r2_score(y_te, yp))


def smape(y, p):
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100)


def main_run():
    factor_A = BASE_FACTORS
    factor_B = BASE_FACTORS + EXTRA_BATCH

    rows = []
    seen = set()
    for f in FILES:
        if not os.path.exists(f): continue
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            if sheet in seen: continue
            seen.add(sheet)
            try:
                df = load_material(f, sheet)
                if 'demand' not in df.columns or len(df) < 20: continue
                # 检查该文件是否有 extra batch 列
                has_extra = any(df[col].abs().sum() > 0 for col in EXTRA_BATCH)
                if not has_extra:
                    continue  # 无 extra 列的文件跳过(如 jibei_20)
                nz = int((df['demand'].astype(float).values[:len(df) - main.N_TEST] > 0).sum())
                r_A = run_with_factors(df, sheet, factor_A)
                r_B = run_with_factors(df, sheet, factor_B)
                rows.append({'material': sheet, 'NZ': nz, 'R2_A': r_A, 'R2_B': r_B,
                             'delta': r_B - r_A})
                print(f"  {sheet[:26]:<28} NZ={nz:>3}  A={r_A:+.4f}  B={r_B:+.4f}  Δ={r_B-r_A:+.4f}")
            except Exception as e:
                print(f"  [ERR] {sheet[:30]}: {str(e)[:50]}")

    if not rows:
        print('无有效数据'); return
    R = pd.DataFrame(rows)
    print('\n' + '=' * 70)
    print('方案A(4因子) vs 方案B(全项目类型因子池)')
    print('=' * 70)
    print(f'有效物资: {len(R)}')
    print(f'方案A平均R²: {R["R2_A"].mean():.4f}')
    print(f'方案B平均R²: {R["R2_B"].mean():.4f}')
    print(f'平均ΔR²: {R["delta"].mean():+.4f}')
    print(f'B优于A的物资: {(R["delta"]>0).sum()}/{len(R)}')
    print(f'B劣于A的物资: {(R["delta"]<0).sum()}/{len(R)}')
    print(f'ΔR²>0.05: {(R["delta"]>0.05).sum()}  | ΔR²<-0.05: {(R["delta"]<-0.05).sum()}')
    # 最差/最好
    print('\nB明显优于A (Δ>0.05):')
    for _, r in R[R['delta']>0.05].sort_values('delta', ascending=False).iterrows():
        print(f"  {r['material'][:26]:<28} Δ={r['delta']:+.4f} (A={r['R2_A']:.3f}→B={r['R2_B']:.3f})")
    print('B明显劣于A (Δ<-0.05):')
    for _, r in R[R['delta']<-0.05].sort_values('delta').iterrows():
        print(f"  {r['material'][:26]:<28} Δ={r['delta']:+.4f} (A={r['R2_A']:.3f}→B={r['R2_B']:.3f})")

    with open('experiments/planB_factors.json', 'w', encoding='utf-8') as f:
        json.dump({'USE_ALL_PROJECT_TYPES': USE_ALL_PROJECT_TYPES, 'rows': rows}, f, ensure_ascii=False, indent=2)
    print('\n已保存: experiments/planB_factors.json')


if __name__ == '__main__':
    main_run()
