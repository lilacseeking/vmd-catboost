"""批次条件化增益 × 可预测性 关系实验。

对全物资跑两种特征配置的 ElasticNet-2S:
  A. 无批次因子: FACTOR_NAMES=['project_count']           (批次事件完全移除)
  B. 含批次因子: FACTOR_NAMES=['project_count','transformer_bids','monthly_bid_count','uhv_bids']
批次增益 gain = R²_B - R²_A

再分析: 批次增益 与 该物资可预测性(R²_B) 的关系。
若 高可预测物资增益低、低可预测物资增益高 → 批次条件化应分层使用, 候选A有数据支撑。
"""
import sys, io, warnings, json, os
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import main

# ---- 强制灰色(与之前可预测性扫描一致) ----
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

FILES = ['inputs/data.xlsx', 'inputs/data_jibei_20.xlsx', 'inputs/data_highfreq.xlsx',
         'inputs/data_sgcc.xlsx', 'inputs/data_jibei.xlsx']

# 批次相关因子
BATCH_FACTORS = ['transformer_bids', 'monthly_bid_count', 'uhv_bids']
NON_BATCH_FACTORS = ['project_count']


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
        if col not in df.columns: df[col] = 0.0
    df['date'] = pd.to_datetime(df['date'])
    return df


def run_with_factors(df, material, factor_list):
    """用指定因子候选池跑 ElasticNet-2S, 返回 (R², y_test, y_pred)"""
    main._INTERNAL_FACTORS = factor_list
    main.FACTOR_NAMES = list(factor_list)
    main._top_factors_cache = {}   # 清因子选择缓存
    main._mag_cache = {}
    main._grey_cache = {}
    X_tr, y_tr, X_te, y_te, _ = main.preprocess_data(df, material)
    yp, _, _, _ = main.run_elasticnet_2s(X_tr, y_tr, X_te, y_te, material)
    return float(r2_score(y_te, yp)), np.asarray(y_te), np.asarray(yp)


def main_run():
    rows = []
    seen = set()
    for f in FILES:
        if not os.path.exists(f): continue
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            if sheet in seen: continue
            try:
                df = load_material(f, sheet)
                if 'demand' not in df.columns or len(df) < 20: continue
                seen.add(sheet)
                # 无批次
                r_nb, yt, _ = run_with_factors(df, sheet, NON_BATCH_FACTORS)
                # 含批次
                r_bf, yt2, _ = run_with_factors(df, sheet, NON_BATCH_FACTORS + BATCH_FACTORS)
                gain_abs = r_bf - r_nb
                gain_rel = (r_bf - r_nb) / (abs(r_nb) + 0.05)  # 归一化增益
                rows.append({'material': sheet, 'R2_nobatch': r_nb, 'R2_batch': r_bf,
                             'gain_abs': gain_abs, 'gain_rel': gain_rel,
                             'predictability': r_bf})
                print(f"  {sheet[:32]:<34} R²无批次={r_nb:+.4f}  R²有批次={r_bf:+.4f}  gain={gain_abs:+.4f}")
            except Exception as e:
                print(f"  [ERR] {sheet[:30]}: {str(e)[:60]}")

    if not rows:
        print('无数据'); return
    R = pd.DataFrame(rows)
    # 相关性
    valid = R.dropna(subset=['predictability', 'gain_abs'])
    rho_abs, p_abs = spearmanr(valid['predictability'], valid['gain_abs'])
    rho_rel, p_rel = spearmanr(valid['predictability'], valid['gain_rel'])
    print('\n' + '=' * 80)
    print(f'Spearman(可预测性R², 批次增益gain_abs) = {rho_abs:+.3f}  (p={p_abs:.3g}, n={len(valid)})')
    print(f'Spearman(可预测性R², 批次增益gain_rel) = {rho_rel:+.3f}  (p={p_rel:.3g}, n={len(valid)})')
    # 分组: 高/低可预测
    hi = R[R['predictability'] >= 0.5]
    lo = R[R['predictability'] < 0.5]
    print(f'\n高可预测(R²≥0.5, n={len(hi)}): 批次增益 mean={hi["gain_abs"].mean():+.3f}, median={hi["gain_abs"].median():+.3f}')
    print(f'低可预测(R²<0.5, n={len(lo)}): 批次增益 mean={lo["gain_abs"].mean():+.3f}, median={lo["gain_abs"].median():+.3f}')
    print(f'批次正增益物资: {(R["gain_abs"]>0).sum()}/{len(R)} = {100*(R["gain_abs"]>0).mean():.0f}%')

    # 画散点
    for f2 in fm.findSystemFonts():
        try:
            if any(n in f2.lower() for n in ['simhei', 'msyh', 'yahei', 'simsun']):
                fm.fontManager.addfont(f2)
        except: pass
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.axvline(0.5, color='gray', ls='--', lw=1, label='可预测性阈值0.5')
    ax.scatter(valid['predictability'], valid['gain_abs'], s=45, alpha=0.75, edgecolor='k', linewidth=0.5)
    for _, r in valid.iterrows():
        ax.annotate(r['material'][:6], (r['predictability'], r['gain_abs']), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('可预测性 (含批次R²)')
    ax.set_ylabel('批次条件化增益 ΔR² (有批次 - 无批次)')
    ax.set_title(f'批次增益 × 可预测性  (Spearman ρ={rho_abs:+.3f})')
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    out = 'outputs/figures/batch_gain_vs_predictability.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\n散点图已保存: {out}')

    with open('experiments/batch_gain_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(R.to_dict('records'), f, ensure_ascii=False, indent=2)
    print('数据已保存: experiments/batch_gain_analysis.json')


if __name__ == '__main__':
    main_run()
