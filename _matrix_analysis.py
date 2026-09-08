"""双可计划性矩阵实证：全物资 (X=数量可计划性, Y=发生可计划性) 散点 + 四象限占座。

X轴(数量可计划性) 候选:
  - R2_nonzero : 非零月上的R² (排除零值月, 只衡量"量"能否预测)
  - CV_size    : 非零规模的变异系数之倒数(1/CV), 越低越可计划
Y轴(发生可计划性) 候选:
  - det        : 1 - |2P-1|  (P∈[0,1], 0=不确定,1=完全确定)  取Stage1概率
  - density    : 需求密度 NZ/81 (非零月占比, 越高越"经常发生")
  - occ_r2     : Stage1分类AUC 或 零值命中率

输出: 散点图 + 四象限占座 + 两轴Spearman相关 + 与R²_total的对应
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
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr
import main

# 强制所有物资启用灰色(与全物资扫描一致, 保持可比)
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
        if col not in df.columns: df[col] = 0.0
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_material_metrics(df, material):
    """计算某物资的所有可计划性指标 + 两阶段P"""
    tl = len(df) - main.N_TEST
    demand = df['demand'].astype(float).values
    y_train, y_test = demand[:tl], demand[tl:]

    # ---- 特征 ----
    X_tr, y_tr, X_te, y_te, _ = main.preprocess_data(df, material)
    nz_tr = y_tr > 0

    # ---- Stage-1 分类概率 P (复刻 run_elasticnet_2s 的两阶段) ----
    y_bin = nz_tr.astype(int)
    from catboost import CatBoostClassifier
    hp = main.get_hp('stage1_cls', material)
    cls = CatBoostClassifier(**hp, loss_function='Logloss',
                             early_stopping_rounds=30, random_seed=main.RANDOM_SEED, verbose=0)
    nv = max(6, len(y_tr) // 4)
    cls.fit(X_tr[:-nv], y_bin[:-nv], eval_set=(X_tr[-nv:], y_bin[-nv:]))
    P = np.clip(cls.predict_proba(X_te)[:, 1], 0, 1)
    y_bin_te = (y_te > 0).astype(int)

    # ---- 指标 ----
    # 用回归器预测非零规模: 重新训练Stage2
    from sklearn.linear_model import ElasticNetCV
    X_nz = X_tr[nz_tr]; y_nz = y_tr[nz_tr]
    if len(y_nz) >= 8:
        reg = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                           cv=min(5, len(y_nz)), max_iter=5000, random_state=42)
        reg.fit(X_nz, y_nz)
        qty = np.maximum(reg.predict(X_te), 0)
    else:
        qty = np.full(len(y_te), np.mean(y_nz) if len(y_nz) else 0)
    yp = P * qty
    r2_total = float(r2_score(y_te, yp))
    # 非零规模可计划性: 只看测试期非零月的量预测 R²
    nz_te = y_te > 0
    r2_size = float(r2_score(y_te[nz_te], qty[nz_te])) if nz_te.sum() >= 2 else np.nan
    # 规模CV
    nz_all = demand[demand > 0]
    cv_size = float(np.std(nz_all) / np.mean(nz_all)) if len(nz_all) > 1 and np.mean(nz_all) > 0 else np.nan
    # Y轴: 发生可计划性
    det_mean = float(np.mean(1 - np.abs(2 * P - 1)))   # 0=不确定, 1=完全确定(取P极端)
    density = float(nz_tr.sum() / len(y_tr))     # 训练集需求密度
    # 发生分类AUC
    occ_auc = float(roc_auc_score(y_bin_te, P)) if (len(set(y_bin_te)) == 2) else np.nan
    # 零值命中率
    occ_acc = float(np.mean((P > 0.5).astype(int) == y_bin_te))
    # 发生确定性: 训练集里是否总是同一批月发生(月度周期性)
    cal_m = df['date'].dt.month.values
    month_prob = {m: np.mean(demand[:tl][cal_m[:tl] == m] > 0) for m in range(1, 13)}
    month_entropy = -np.sum([p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9) for p in month_prob.values()]) / 12

    return {
        'material': material, 'NZ': int(nz_tr.sum()),
        'r2_total': r2_total, 'r2_size': r2_size, 'cv_size': cv_size,
        'det_mean': det_mean, 'density': density,
        'occ_auc': occ_auc, 'occ_acc': occ_acc,
        'month_entropy': month_entropy,
        'P': [float(x) for x in P], 'y_test': [float(x) for x in y_te],
        'qty': [float(x) for x in qty], 'yp': [float(x) for x in yp],
    }


def main_run():
    results = {}
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
                m = compute_material_metrics(df, sheet)
                results[sheet] = m
            except Exception as e:
                print(f'  [ERR] {sheet[:30]}: {str(e)[:60]}')

    rows = list(results.values())
    # ---- 相关性 ----
    for a, b, la, lb in [('r2_total', 'det_mean', 'R²_total', '发生确定性1-|2P-1|'),
                         ('r2_total', 'density', 'R²_total', '需求密度NZ/81'),
                         ('r2_total', 'occ_auc', 'R²_total', '发生AUC'),
                         ('det_mean', 'density', '发生确定性', '需求密度')]:
        vals = [(r[a], r[b]) for r in rows if not np.isnan(r[a]) and not np.isnan(r[b])]
        rho, p = spearmanr([v[0] for v in vals], [v[1] for v in vals])
        print(f'  Spearman({la}, {lb}) = {rho:+.3f}  (p={p:.3g}, n={len(vals)})')

    # ---- 四象限 (X=R²_total, Y=det, 阈值 0.5/0.5) ----
    xthr, ythr = 0.5, 0.5
    quads = {'Q1框架协议(高量准&高确定)': [], 'Q2滚动备货(低量准&高确定)': [],
             'Q3安全库存(低量准&低确定)': [], 'Q4 VMI(高量准&低确定)': []}
    for r in rows:
        x, y = r['r2_total'], r['det_mean']
        if x >= xthr and y >= ythr: quads['Q1框架协议(高量准&高确定)'].append(r['material'])
        elif x < xthr and y >= ythr: quads['Q2滚动备货(低量准&高确定)'].append(r['material'])
        elif x < xthr and y < ythr: quads['Q3安全库存(低量准&低确定)'].append(r['material'])
        else: quads['Q4 VMI(高量准&低确定)'].append(r['material'])

    print('\n' + '=' * 100)
    print(f'四象限占座  (阈值 X=R²_total≥{xthr}, Y=发生确定性≥{ythr})  共{len(rows)}物资')
    print('=' * 100)
    for q, mats in quads.items():
        print(f'\n{q}: {len(mats)}个')
        for m in sorted(mats, key=lambda mm: -results[mm]['r2_total']):
            r = results[m]
            print(f'    {m[:32]:<34} R²={r["r2_total"]:+.3f} det={r["det_mean"]:.3f} NZ={r["NZ"]}')

    # ---- 画散点 ----
    for fm_f in fm.findSystemFonts():
        try:
            if any(n in fm_f.lower() for n in ['simhei', 'msyh', 'yahei', 'simsun']):
                fm.fontManager.addfont(fm_f)
        except: pass
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    panels = [
        (axes[0], 'r2_total', 'det_mean', 'R²(全月, 数量可计划性)', '发生确定性 1-|2P-1|', 'X=R²_total, Y=发生确定性'),
        (axes[1], 'r2_total', 'density', 'R²(全月)', '需求密度 NZ/81', 'X=R²_total, Y=需求密度'),
    ]
    for ax, xkey, ykey, xlab, ylab, ttl in panels:
        xs = [r[xkey] for r in rows if not np.isnan(r[xkey])]
        ys = [r[ykey] for r in rows if not np.isnan(r[ykey])]
        ax.axvline(xthr, color='gray', ls='--', lw=1)
        ax.axhline(ythr if ykey == 'det_mean' else 0.4, color='gray', ls='--', lw=1)
        ax.scatter(xs, ys, s=40, alpha=0.7, edgecolor='k', linewidth=0.5)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.grid(alpha=0.3)
        # 标注每点物资名(截断)
        for r in rows:
            ax.annotate(r['material'][:6], (r[xkey], r[ykey]), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
    plt.tight_layout()
    out = 'outputs/figures/predictability_matrix.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\n散点图已保存: {out}')

    with open('experiments/predictability_matrix.json', 'w', encoding='utf-8') as f:
        json.dump({'results': rows, 'quads': quads, 'thresholds': {'x': xthr, 'y': ythr}},
                  f, ensure_ascii=False, indent=2)
    print('数据已保存: experiments/predictability_matrix.json')


if __name__ == '__main__':
    main_run()
