"""全物资灰色特征收益扫描。

对所有数据文件的所有物资，强制启用灰色先验特征（monkey-patch 掉 NZ>=15 门槛），
跑 ElasticNet-2S 无灰色 vs 有灰色 对比，找出灰色收益 ΔR² 最大的 Top5。

用法: python _grey_all_materials.py
"""
import sys, io, warnings, json, os
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import main

# ---- 强制所有物资计算灰色特征（去掉 NZ>=15 早退） ----
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
main.USE_PAPER_FEATURES = False      # 保证 _use_grey 恒 True
main.USE_ENHANCED_LAG_FEATURES = True
main.USE_EVENT_FEATURES = True

# ---- 数据文件 ----
FILES = ['inputs/data.xlsx', 'inputs/data_jibei_20.xlsx', 'inputs/data_highfreq.xlsx',
         'inputs/data_sgcc.xlsx', 'inputs/data_jibei.xlsx']


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
    df = df.rename(columns=rename)
    for col in ['project_count', 'transformer_bids', 'monthly_bid_count', 'uhv_bids']:
        if col not in df.columns:
            df[col] = 0.0
    df['date'] = pd.to_datetime(df['date'])
    return df


def smape(y, p):
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100)


def run_ab(df, material):
    nz_tr = int((df['demand'].astype(float).values[:len(df) - main.N_TEST] > 0).sum())
    main.USE_GREY_FEATURES = False
    X_off, y_tr, X_te, y_te, _ = main.preprocess_data(df, material)
    yp_off, _, _, _ = main.run_elasticnet_2s(X_off, y_tr, X_te, y_te, material)
    r_off = float(r2_score(y_te, yp_off))

    main.USE_GREY_FEATURES = True
    X_on, y_tr, X_te, y_te, _ = main.preprocess_data(df, material)
    yp_on, _, _, _ = main.run_elasticnet_2s(X_on, y_tr, X_te, y_te, material)
    r_on = float(r2_score(y_te, yp_on))

    return {
        'material': material, 'NZ': int(nz_tr),
        'R2_off': round(r_off, 4), 'R2_on': round(r_on, 4),
        'delta_R2': round(r_on - r_off, 4),
        'sMAPE_off': round(smape(y_te, yp_off), 1), 'sMAPE_on': round(smape(y_te, yp_on), 1),
        'y_test': [float(x) for x in y_te],
        'pred_off': [float(x) for x in yp_off], 'pred_on': [float(x) for x in yp_on],
    }


def main_run():
    results = {}
    seen_materials = set()
    for f in FILES:
        if not os.path.exists(f):
            print(f'SKIP missing {f}')
            continue
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            if sheet in seen_materials:
                continue
            try:
                df = load_material(f, sheet)
                if 'demand' not in df.columns or len(df) < 20:
                    continue
                seen_materials.add(sheet)
                r = run_ab(df, sheet)
                results[sheet] = r
                print(f"  {sheet[:34]:<36} NZ={r['NZ']:>3}  off={r['R2_off']:+.4f}  on={r['R2_on']:+.4f}  ΔR²={r['delta_R2']:+.4f}")
            except Exception as e:
                print(f"  [ERR] {sheet[:34]:<36} {str(e)[:80]}")

    # ---- 汇总 + Top5 ----
    rows = sorted(results.values(), key=lambda x: -x['delta_R2'])
    print('\n' + '=' * 96)
    print('灰色收益 ΔR² 排序（全部物资）')
    print('=' * 96)
    print(f"{'物资':<34}{'NZ':>4} {'off R²':>8} {'on R²':>8} {'ΔR²':>8}  {'sMAPE off/on'}")
    for r in rows:
        print(f"{r['material'][:34]:<34}{r['NZ']:>4} {r['R2_off']:>8.4f} {r['R2_on']:>8.4f} {r['delta_R2']:>+8.4f}  {r['sMAPE_off']:>6.1f}/{r['sMAPE_on']:<6.1f}")

    top5 = rows[:5]
    print('\n' + '=' * 96)
    print('灰色收益最大的 5 种物资')
    print('=' * 96)
    for i, r in enumerate(top5, 1):
        tag = '✓✓' if r['delta_R2'] > 0.10 else ('✓' if r['delta_R2'] > 0.02 else '—')
        print(f"  {i}. {r['material'][:34]:<34} NZ={r['NZ']:>3}  off={r['R2_off']:+.4f} → on={r['R2_on']:+.4f}  ΔR²={r['delta_R2']:+.4f}  {tag}")

    # 类内正增益率
    gpos = sum(1 for r in rows if r['delta_R2'] > 0)
    print(f'\n灰色正增益物资: {gpos}/{len(rows)} = {100*gpos/len(rows):.0f}%  | 平均ΔR² = {np.mean([r["delta_R2"] for r in rows]):+.4f}')

    with open('experiments/grey_all_materials.json', 'w', encoding='utf-8') as f:
        json.dump({'results': rows, 'top5': top5}, f, ensure_ascii=False, indent=2)
    print('\n已保存: experiments/grey_all_materials.json')


if __name__ == '__main__':
    main_run()
