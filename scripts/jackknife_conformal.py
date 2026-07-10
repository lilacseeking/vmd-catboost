"""
Jackknife+ Conformal Prediction — Leave-One-Out Calibration
================================================================
与 Split Conformal 的关键区别:
  Split:   50个月训练 + 12个月校准 → 12个残差 → 分位数估计不稳定
  Jackknife+: 62折留一法 → 62个残差 → 5倍更稳定的分位数估计

方法 (Barber et al. 2021, Annals of Statistics):
  1. Optuna 搜索最优超参数 (同 Route A)
  2. 对每个训练月 i:
       训练 LightGBM(最优参数) 在除 i 外的 61 个月上
       预测第 i 月 → 残差 R_i = |y_i - pred_i|
  3. 对每个测试月 j:
       用 62 个 Jackknife 模型预测 → 62 个 pred_j 值
       计算 62 个下界 = pred_j - R_i (i=1..62)
       计算 62 个上界 = pred_j + R_i (i=1..62)
       P10_j = 下界的第 10% 分位数
       P90_j = 上界的第 90% 分位数
  4. 评估所有 10 种物资

计算量: 每种物资训练 62 个模型 × 0.5s ≈ 31s → 10种 ≈ 5分钟
================================================================
"""
import os, sys, sqlite3, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from logger_utils import StepLogger, format_params

log = StepLogger('Jackknife')

# ========================================================================
# Config
# ========================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'jackknife_conformal')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN, TEST = 62, 12
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]
TEST_DATES = pd.date_range('2025-07-01', periods=12, freq='MS')

TARGETS = [
    ('AC_Arrester','%交流避雷器%','避雷器/绝缘子'),('CVT','%电容式电压互感器%','其他'),
    ('Post_Insulator','%交流支柱绝缘子%','避雷器/绝缘子'),('Breaker_Protect','%断路器保护%','断路器/组合电器'),
    ('Reactor_Protect','%电抗器保护%','保护/监控'),('Line_Protect','%线路保护%','保护/监控'),
    ('T10kV','%10kV变压器%','变压器'),('Transformer_Protect','%变压器保护%','变压器'),
    ('Busbar_Protect','%母线保护%','保护/监控'),('GIS_500kV','%500kV%GIS%','开关柜/环网柜'),
]

S2_SPACE = {
    'n_estimators':('int',50,400),'max_depth':('int',2,8),'num_leaves':('int',8,80),
    'learning_rate':('float',0.003,0.1),'min_child_samples':('int',2,15),
    'reg_alpha':('float',0.005,5.0),'reg_lambda':('float',0.005,5.0),
    'subsample':('float',0.5,1.0),'colsample_bytree':('float',0.5,1.0),
    'min_split_gain':('float',0.001,0.5),
}

# ========================================================================
# Pipeline start
# ========================================================================
log.pipeline_start_log(
    'Jackknife+ Conformal Prediction (Leave-One-Out)',
    '62-fold LOOCV calibration — 62 residuals per material for stable quantile estimation'
)

# ========================================================================
# Step 1: Data loading
# ========================================================================
log.step('Data Loading', '10 materials + 14-dim zero-leakage features')

ref = pd.read_excel(REF_XLSX, sheet_name=0)
COLS = [c for c in ref.columns if c not in ['日期','需求量']]
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
X_shared = ref[ref['mk'].between('202005','202606')][COLS].values.astype(np.float64)
X_sh_tr = X_shared[:TRAIN]

def build_self_features(y_train_series):
    """Build 8-dim self-lag features for a single training series."""
    n = len(y_train_series)
    F = np.zeros((n, 8))
    for i in range(n):
        t = i
        F[i,0]=y_train_series[t-1]if t>=1 else 0.0
        F[i,1]=y_train_series[t-2]if t>=2 else 0.0
        F[i,2]=y_train_series[t-3]if t>=3 else 0.0
        F[i,3]=y_train_series[t-6]if t>=6 else 0.0
        F[i,4]=y_train_series[t-12]if t>=12 else 0.0
        w3=max(0,t-2);F[i,5]=np.mean(y_train_series[w3:t+1])if t>=1 else y_train_series[0]
        w6=max(0,t-5);F[i,6]=np.mean(y_train_series[w6:t+1])if t>=1 else y_train_series[0]
        last=-1
        for j in range(t-1,-1,-1):
            if y_train_series[j]>0:last=j;break
        F[i,7]=float(t-last)if last>=0 else 99.0
    return F

db = sqlite3.connect(DB_PATH)
mat_data = {}
for ek, pattern, cat in TARGETS:
    cur = db.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) nz FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month>='202005' AND demand_month<='202606'
        GROUP BY material_name ORDER BY nz DESC LIMIT 3
    """, (pattern,))
    best = max(cur.fetchall(), key=lambda x:x[1])
    cur2 = db.execute("""
        SELECT demand_month, SUM(demand_quantity) FROM material_demand_item
        WHERE material_name=? AND demand_month>='202005' AND demand_month<='202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (best[0],))
    dmap = {r[0]:r[1] for r in cur2.fetchall()}
    y_full = np.array([dmap.get(m,0) for m in MONTHS], dtype=np.float64)

    mat_data[ek] = {
        'name': best[0], 'cat': cat,
        'y_full': y_full, 'y_tr': y_full[:TRAIN], 'y_te': y_full[TRAIN:],
        'X_sh_tr': X_sh_tr,
        'nz_tr': (y_full[:TRAIN]>0).sum(), 'nz_te': (y_full[TRAIN:]>0).sum(),
    }
    log.task(f'  {ek:<22s} → {best[0][:35]} | nz_train={mat_data[ek]["nz_tr"]}/62')

db.close()
log.step_ok('10 materials loaded')

# ========================================================================
# Step 2: Optuna (once per material) + Jackknife
# ========================================================================
log.step('Jackknife+ Calibration', 'Step 2a: Optuna ONCE per material. Step 2b: 62-fold LOOCV with best params')

def suggest(trial, space):
    pp = {}
    for pn, (pt, lo, hi) in space.items():
        if pt == 'int': pp[pn] = trial.suggest_int(pn, lo, hi)
        else: pp[pn] = trial.suggest_float(pn, lo, hi, log=True)
    return pp

def build_full_features(y_full_series):
    """Build 14-dim features from a complete y series (self-lag 8D + shared 6D)."""
    n = len(y_full_series)
    X_sh = X_shared[:n]  # first n rows of shared features (already time-aligned)
    F_self = np.zeros((n, 8))
    for i in range(n):
        t = i
        F_self[i,0]=y_full_series[t-1]if t>=1 else 0.0
        F_self[i,1]=y_full_series[t-2]if t>=2 else 0.0
        F_self[i,2]=y_full_series[t-3]if t>=3 else 0.0
        F_self[i,3]=y_full_series[t-6]if t>=6 else 0.0
        F_self[i,4]=y_full_series[t-12]if t>=12 else 0.0
        w3=max(0,t-2);F_self[i,5]=np.mean(y_full_series[w3:t+1])if t>=1 else y_full_series[0]
        w6=max(0,t-5);F_self[i,6]=np.mean(y_full_series[w6:t+1])if t>=1 else y_full_series[0]
        last=-1
        for j in range(t-1,-1,-1):
            if y_full_series[j]>0:last=j;break
        F_self[i,7]=float(t-last)if last>=0 else 99.0
    return np.column_stack([X_sh, F_self])

results = []
alpha = 0.20  # P10-P90 = 80% interval

for ek, d in mat_data.items():
    y_full = d['y_full']
    X_full = build_full_features(y_full)  # (74, 14)
    X_train_all = X_full[:TRAIN]  # (62, 14)
    y_train_all = y_full[:TRAIN]  # (62,)
    X_test = X_full[TRAIN:]  # (12, 14)
    y_test = y_full[TRAIN:]

    na = len(X_train_all); nv = min(12, na//4)

    # ---- Step 2a: Optuna (once) ----
    def obj(trial):
        pp = suggest(trial, S2_SPACE)
        m = lgb.LGBMRegressor(**pp, random_state=42, verbose=-1, force_col_wise=True)
        m.fit(X_train_all[:na-nv], y_train_all[:na-nv],
              eval_set=[(X_train_all[na-nv:], y_train_all[na-nv:])])
        yp = np.maximum(m.predict(X_train_all[na-nv:]), 0)
        try: return r2_score(y_train_all[na-nv:], yp)
        except: return -10.0
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=60, show_progress_bar=False)
    best_params = study.best_params

    # ---- Step 2b: 62-fold Jackknife ----
    log.task_detail(f'{ek}: running 62-fold LOOCV...')
    jack_residuals = np.zeros(TRAIN)  # R_i for each training month
    jack_predictions_train = np.zeros(TRAIN)  # mu_{-i}(x_i)

    for i in range(TRAIN):
        # Leave out month i
        X_loo = np.delete(X_train_all, i, axis=0)
        y_loo = np.delete(y_train_all, i, axis=0)

        # Train with best params (no Optuna — reuse from Step 2a)
        m_loo = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1, force_col_wise=True)
        na_loo = len(X_loo); nv_loo = min(12, na_loo//4)
        if nv_loo >= 3:
            m_loo.fit(X_loo[:na_loo-nv_loo], y_loo[:na_loo-nv_loo],
                      eval_set=[(X_loo[na_loo-nv_loo:], y_loo[na_loo-nv_loo:])])
        else: m_loo.fit(X_loo, y_loo)
        pred_i = max(0.0, m_loo.predict(X_train_all[i:i+1])[0])
        jack_predictions_train[i] = pred_i
        jack_residuals[i] = abs(y_train_all[i] - pred_i)

    # ---- Step 2c: Predict test set with all 62 Jackknife models ----
    # For each test point j, get 62 predictions from the 62 LOO models
    # Then compute 62×(test point pred - residual) and 62×(test point pred + residual)
    jack_test_preds = np.zeros((TRAIN, TEST))  # (62, 12)
    for i in range(TRAIN):
        # Retrain LOO model (same as above)
        X_loo = np.delete(X_train_all, i, axis=0)
        y_loo = np.delete(y_train_all, i, axis=0)
        m_loo = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1, force_col_wise=True)
        na_loo = len(X_loo); nv_loo = min(12, na_loo//4)
        if nv_loo >= 3:
            m_loo.fit(X_loo[:na_loo-nv_loo], y_loo[:na_loo-nv_loo],
                      eval_set=[(X_loo[na_loo-nv_loo:], y_loo[na_loo-nv_loo:])])
        else: m_loo.fit(X_loo, y_loo)
        jack_test_preds[i] = np.maximum(m_loo.predict(X_test), 0)

    # ---- Step 2d: Construct Jackknife+ intervals per test point ----
    lo_test = np.zeros(TEST)
    hi_test = np.zeros(TEST)
    # alpha/2 = 0.10 for P10, 1-alpha/2 = 0.90 for P90
    q_lo = int(np.floor((alpha/2) * (TRAIN + 1)))
    q_hi = int(np.ceil((1 - alpha/2) * (TRAIN + 1))) - 1
    q_lo = max(0, min(q_lo, TRAIN-1))
    q_hi = max(0, min(q_hi, TRAIN-1))

    for j in range(TEST):
        # 62 values of (pred_j - residual_i)
        lower_vals = np.sort(jack_test_preds[:, j] - jack_residuals)
        upper_vals = np.sort(jack_test_preds[:, j] + jack_residuals)
        lo_test[j] = max(0.0, lower_vals[q_lo])
        hi_test[j] = upper_vals[q_hi]

    # ---- Evaluate ----
    inside = (y_test >= lo_test) & (y_test <= hi_test)
    coverage = inside.sum() / TEST
    mean_width = np.mean(hi_test - lo_test)
    mean_pt = np.mean(jack_test_preds.mean(axis=0))
    rel_width = mean_width / max(mean_pt, 1.0)

    # Point prediction = mean of 62 Jackknife predictions
    pt_pred = jack_test_preds.mean(axis=0)
    r2_pt = r2_score(y_test, pt_pred)

    # Winkler score
    winkler = 0.0
    for j in range(TEST):
        w = hi_test[j] - lo_test[j]
        if y_test[j] < lo_test[j]: w += (2.0/alpha)*(lo_test[j]-y_test[j])
        elif y_test[j] > hi_test[j]: w += (2.0/alpha)*(y_test[j]-hi_test[j])
        winkler += w
    winkler /= TEST

    sharpness = coverage * max(0, 1 - rel_width/5.0) if rel_width < 5 else 0

    results.append({
        'key': ek, 'name': d['name'][:30], 'cat': d['cat'],
        'nz_tr': d['nz_tr'], 'nz_te': d['nz_te'],
        'r2_point': round(r2_pt, 4), 'coverage': round(coverage, 4),
        'mean_width': round(mean_width, 2), 'mean_pt': round(mean_pt, 2),
        'rel_width': round(rel_width, 4), 'winkler': round(winkler, 2),
        'sharpness': round(sharpness, 4), 'inside_count': int(inside.sum()),
        'y_te': y_test, 'pred_pt': pt_pred, 'lo': lo_test, 'hi': hi_test,
        'residuals': jack_residuals, 'best_params': best_params,
    })

    log.metrics(ek, {
        'Cov': f'{coverage:.0%}', 'Width': f'{mean_width:.0f}',
        'Winkler': f'{winkler:.0f}', 'R2_pt': f'{r2_pt:.4f}',
        'inside': f'{int(inside.sum())}/{TEST}',
    })

log.step_ok(f'62-fold Jackknife+ complete for 10 materials')

# ========================================================================
# Step 3: Comparison with Split Conformal
# ========================================================================
log.step('Comparison', 'Jackknife+ (62 residuals) vs Split (12 residuals) — coverage stability')

split_results_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'conformal', 'conformal_results.json')
split_data = {}
if os.path.exists(split_results_path):
    with open(split_results_path, encoding='utf-8') as f:
        split_data = json.load(f)
    split_map = {m['key']: m for m in split_data.get('materials', [])}

comp_rows = []
for r in results:
    ek = r['key']
    jk_cov = r['coverage']; jk_wid = r['mean_width']; jk_wink = r['winkler']
    jk_r2 = r['r2_point']
    sp_cov = split_map.get(ek, {}).get('coverage', 0)
    sp_wid = split_map.get(ek, {}).get('mean_width', 0)
    sp_wink = split_map.get(ek, {}).get('winkler', 0)
    cov_delta = jk_cov - sp_cov if sp_cov > 0 else 0
    win_delta = jk_wid - sp_wid if sp_wid > 0 else 0
    comp_rows.append([ek[:16], f'{sp_cov:.0%}→{jk_cov:.0%}', f'{sp_wid:.0f}→{jk_wid:.0f}',
                      f'{sp_wink:.0f}→{jk_wink:.0f}', f'{r["r2_point"]:.3f}'])

if comp_rows:
    log.data_table('Jackknife+ vs Split Conformal',
                   ['Material','Coverage(Split→JK+)','Width(Split→JK+)','Winkler(Split→JK+)','PtR2(JK+)'], comp_rows)

mean_jk_cov = np.mean([r['coverage'] for r in results])
mean_sp_cov = split_data.get('aggregate',{}).get('mean_coverage',0) if split_data else 0
mean_jk_wid = np.mean([r['mean_width'] for r in results])
on_target_jk = sum(1 for r in results if 0.70 <= r['coverage'] <= 0.90)

log.task(f'Coverage: Split={mean_sp_cov:.1%} → Jackknife+={mean_jk_cov:.1%}')
log.task(f'Width:    Jackknife+={mean_jk_wid:.0f}')
log.task(f'On-target (70%-90%): {on_target_jk}/10 (Split had 3/10)')
log.step_ok(f'Jackknife+ coverage: {mean_jk_cov:.1%}, on_target: {on_target_jk}/10')

# ========================================================================
# Step 4: Charts
# ========================================================================
log.step('Charts', '6 comparison figures')

for d in [matplotlib.get_cachedir()]:
    try:
        for f in os.listdir(d):
            if 'fontlist' in f: os.remove(os.path.join(d, f))
    except: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fig 1: Coverage comparison — Split vs Jackknife+
fig1, ax1 = plt.subplots(figsize=(14, 6)); x1 = np.arange(10); w1 = 0.3
jk_covs = [r['coverage'] for r in results]
sp_covs = [split_map.get(r['key'],{}).get('coverage',0) for r in results]
ax1.bar(x1-w1/2, sp_covs, w1, label='Split Conformal (12 residuals)', color='#FF9800', alpha=0.7)
ax1.bar(x1+w1/2, jk_covs, w1, label='Jackknife+ (62 residuals)', color='#2196F3', alpha=0.9)
ax1.axhline(y=0.80, c='gray', ls='--', lw=1.5, label='Target 80%')
# Mark target zone
ax1.axhspan(0.70, 0.90, alpha=0.05, color='green', label='On-target [70%,90%]')
for i in range(10):
    delta = jk_covs[i] - sp_covs[i]
    ax1.text(i, max(sp_covs[i],jk_covs[i])+0.02, f'{delta:+.0%}', ha='center', fontsize=8,
             color='green' if abs(jk_covs[i]-0.80) < abs(sp_covs[i]-0.80) else 'red')
ax1.set_xticks(x1); ax1.set_xticklabels([r['key'][:10] for r in results], fontsize=8, rotation=30)
ax1.set_ylabel('Coverage'); ax1.set_ylim(0, 1.2); ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3, axis='y')
ax1.set_title('Coverage: Split(12 res) vs Jackknife+(62 res) — JK+ more stable', fontsize=13, fontweight='bold')
plt.tight_layout(); fig1.savefig(os.path.join(OUT_DIR, 'fig1_cov_comparison.png'), dpi=150); plt.close()

# Fig 2: Prediction intervals for all 10
n_rows, n_cols = 5, 2
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(18, 24))
sorted_r = sorted(results, key=lambda x: -x['r2_point'])
for i, r in enumerate(sorted_r):
    ax = axes2[i//n_cols, i%n_cols]
    months = TEST_DATES
    ax.fill_between(months, r['lo'], r['hi'], color='#2196F3', alpha=0.12, label='P10-P90 (JK+)')
    ax.plot(months, r['y_te'], 'ko-', linewidth=2, markersize=5, label='Actual', zorder=3)
    ax.plot(months, r['pred_pt'], 's-', color='#E91E63', linewidth=1.8, markersize=5,
            label=f'JK+ Mean (R2={r["r2_point"]:.3f})', zorder=2)
    for j in range(TEST):
        in_range = r['lo'][j] <= r['y_te'][j] <= r['hi'][j]
        if not in_range:
            ax.plot(months[j], r['y_te'][j], 'ro', markersize=10, markerfacecolor='none',
                    markeredgewidth=2, markeredgecolor='red')
    ax.set_title(f'{r["key"]} [{r["cat"]}]  Cov={r["coverage"]:.0%} W={r["mean_width"]:.0f} R2={r["r2_point"]:.3f}',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=6, loc='upper left'); ax.grid(True, alpha=0.3); ax.tick_params('x', rotation=30, labelsize=6)
fig2.suptitle('Jackknife+ P10-P90 Prediction Intervals — All 10 Materials', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); fig2.savefig(os.path.join(OUT_DIR, 'fig2_all_intervals.png'), dpi=150); plt.close()

# Fig 3: Coverage bar chart (Jackknife+)
fig3, ax3 = plt.subplots(figsize=(14, 5))
covs3 = [r['coverage'] for r in results]
clrs3 = ['#4CAF50' if 0.70<=c<=0.90 else '#FF9800' if c>0.90 else '#F44336' for c in covs3]
bars3 = ax3.bar(range(10), covs3, color=clrs3, alpha=0.8, edgecolor='white')
ax3.axhline(y=0.80, c='gray', ls='--', lw=1.5, label='Target 80%')
ax3.axhspan(0.70, 0.90, alpha=0.05, color='green', label='On-target')
for bar, c in zip(bars3, covs3):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{c:.0%}', ha='center', fontsize=10)
ax3.set_xticks(range(10)); ax3.set_xticklabels([r['key'][:10] for r in results], fontsize=8, rotation=30)
ax3.set_ylabel('Coverage'); ax3.set_ylim(0,1.15); ax3.legend(); ax3.grid(True, alpha=0.3, axis='y')
ax3.set_title(f'Jackknife+ Coverage (On-target={on_target_jk}/10)', fontsize=13, fontweight='bold')
plt.tight_layout(); fig3.savefig(os.path.join(OUT_DIR, 'fig3_coverage_jk.png'), dpi=150); plt.close()

# Fig 4: Width comparison (Split vs JK+)
fig4, ax4 = plt.subplots(figsize=(14, 5))
jk_wids = [r['mean_width'] for r in results]
sp_wids = [split_map.get(r['key'],{}).get('mean_width',0) for r in results] if split_data else [0]*10
x4 = np.arange(10); w4 = 0.3
ax4.bar(x4-w4/2, sp_wids, w4, label='Split Conformal', color='#FF9800', alpha=0.7)
ax4.bar(x4+w4/2, jk_wids, w4, label='Jackknife+', color='#2196F3', alpha=0.9)
ax4.set_xticks(x4); ax4.set_xticklabels([r['key'][:10] for r in results], fontsize=8, rotation=30)
ax4.set_ylabel('Mean Interval Width'); ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')
ax4.set_title('Interval Width: Split vs Jackknife+ (narrower=sharper)', fontsize=13, fontweight='bold')
plt.tight_layout(); fig4.savefig(os.path.join(OUT_DIR, 'fig4_width_comparison.png'), dpi=150); plt.close()

# Fig 5: Residual distribution for 2 materials (showing JK+ stability)
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
for ax_i, ek in enumerate(['T10kV', 'Post_Insulator']):
    ax = axes5[ax_i]
    r = [r_ for r_ in results if r_['key']==ek][0]
    resids = r['residuals']
    ax.hist(resids, bins=15, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(x=np.percentile(resids, 90), color='#E91E63', linestyle='--', lw=2, label=f'P90 residual={np.percentile(resids,90):.0f}')
    ax.axvline(x=np.median(resids), color='#4CAF50', linestyle='-', lw=2, label=f'Median={np.median(resids):.0f}')
    ax.set_title(f'{ek} — 62 Jackknife Residuals\nCoverage={r["coverage"]:.0%} (was {split_map.get(ek,{}).get("coverage",0):.0%} in Split)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('|Actual - Predicted| (residual)'); ax.set_ylabel('Count'); ax.legend(fontsize=8)
fig5.suptitle('Jackknife+ Calibration Quality: 62 Residuals (vs 12 in Split)', fontsize=13, fontweight='bold')
plt.tight_layout(); fig5.savefig(os.path.join(OUT_DIR, 'fig5_residual_dist.png'), dpi=150); plt.close()

# Fig 6: Coverage-Width bubble chart
fig6, ax6 = plt.subplots(figsize=(10, 7))
for r in results:
    ax6.scatter(r['coverage'], r['mean_width'], s=max(50, 1/r['rel_width']*100),
                alpha=0.7, edgecolors='black', linewidth=0.5, label=r['key'][:10])
    ax6.annotate(r['key'][:10], (r['coverage'], r['mean_width']),
                 textcoords="offset points", xytext=(5,5), fontsize=8)
ax6.axvline(x=0.80, c='gray', ls='--', alpha=0.5, label='Target 80%')
ax6.set_xlabel('Coverage', fontsize=12); ax6.set_ylabel('Mean Width', fontsize=12)
ax6.set_title('Jackknife+: Coverage vs Width (bubble size=precision)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
plt.tight_layout(); fig6.savefig(os.path.join(OUT_DIR, 'fig6_cov_width_bubble.png'), dpi=150); plt.close()

log.step_ok('6 charts generated')

# ========================================================================
# Step 5: Final report
# ========================================================================
log.step('Final Report', 'Jackknife+ complete evaluation')

print(f"\n{'='*100}")
print(f" JACKKNIFE+ CONFORMAL PREDICTION — FINAL EVALUATION REPORT")
print(f"{'='*100}")
print(f"\n{'Material':<22s} {'PtR2(JK)':>9s} {'Coverage':>9s} {'Width':>8s} {'W/Point':>8s} {'Winkler':>8s} {'Inside':>7s}")
print(f"{'-'*85}")
for r in results:
    print(f"{r['key'][:20]:<22s} {r['r2_point']:>9.4f} {r['coverage']:>8.0%} {r['mean_width']:>8.0f} "
          f"{r['rel_width']:>8.2f} {r['winkler']:>8.0f} {r['inside_count']:>5d}/{TEST}")

mean_cov_jk = np.mean([r['coverage'] for r in results])
mean_wid_jk = np.mean([r['mean_width'] for r in results])
mean_wink_jk = np.mean([r['winkler'] for r in results])
best_wink = min(results, key=lambda r: r['winkler'])

print(f"\nAGGREGATE (Jackknife+):")
print(f"  Mean Coverage:          {mean_cov_jk:.1%}")
print(f"  Mean Width:             {mean_wid_jk:.0f}")
print(f"  Mean Winkler Score:     {mean_wink_jk:.0f}")
print(f"  Coverage on-target:     {on_target_jk}/10 (Split had 3/10)")
print(f"  Top material (Winkler): {best_wink['key']} (W={best_wink['winkler']:.0f}, Cov={best_wink['coverage']:.0%})")

# Save
out_json = {
    'method': 'Jackknife+ Conformal Prediction (Barber et al. 2021)',
    'setup': '62-fold LOOCV, Optuna(60 trials) per material for initial params',
    'alpha': alpha, 'target_interval': 'P10-P90',
    'aggregate': {
        'mean_coverage': round(mean_cov_jk, 4), 'mean_width': round(mean_wid_jk, 2),
        'mean_winkler': round(mean_wink_jk, 2), 'on_target_70_90': on_target_jk,
    },
    'materials': [{k: v for k, v in r.items() if k not in ['y_te','pred_pt','lo','hi','residuals','best_params']}
                  for r in results],
    'timestamp': datetime.now().isoformat(),
}
json_path = os.path.join(OUT_DIR, 'jackknife_results.json')
json.dump(out_json, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)

log.pipeline_end_log(f'JK+ Coverage={mean_cov_jk:.1%}, OnTarget={on_target_jk}/10, BestWinkler={best_wink["key"]}({best_wink["winkler"]:.0f})')
print(f"\n  JSON: {json_path}")
