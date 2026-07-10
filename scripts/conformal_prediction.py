"""
共形预测区间评估报告
================================================================
方法: Split Conformal Prediction (Vovk 2005)
  1. 用 Route A 最佳 LightGBM 点预测器做基础预测
  2. 训练集按 50/12 分割为 proper_train / calibration
  3. 在 calibration 上计算非一致性分数 (absolute residuals)
  4. 用 (1-α) 分位数确定预测区间半径
  5. 对测试集输出 P10-P90 区间

评估指标:
  - Coverage (% of actual values inside [P10, P90])
  - Mean Interval Width (区间宽度)
  - Interval Width / Point Prediction (相对宽度)
  - Sharpness (区间窄 + 覆盖高 = 最好的区间)
  - Winkler Score (区间预测的综合评分, 惩罚未覆盖+奖励窄区间)

10种高密度物资, 每种单独运行共形预测
================================================================
"""
import os, sys, sqlite3, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

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

log = StepLogger('Conformal')

# ========================================================================
# 0. Configuration
# ========================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'conformal')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN, TEST = 62, 12
CALIB_SIZE = 12  # last 12 months of training for calibration
PROPER_TRAIN = TRAIN - CALIB_SIZE  # 50

MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]
TEST_DATES = pd.date_range('2025-07-01', periods=12, freq='MS')

TARGETS = [
    ('AC_Arrester',        '%交流避雷器%',       '避雷器/绝缘子'),
    ('CVT',                '%电容式电压互感器%',  '其他'),
    ('Post_Insulator',     '%交流支柱绝缘子%',    '避雷器/绝缘子'),
    ('Breaker_Protect',    '%断路器保护%',        '断路器/组合电器'),
    ('Reactor_Protect',    '%电抗器保护%',        '保护/监控'),
    ('Line_Protect',       '%线路保护%',          '保护/监控'),
    ('T10kV',              '%10kV变压器%',        '变压器'),
    ('Transformer_Protect','%变压器保护%',        '变压器'),
    ('Busbar_Protect',     '%母线保护%',          '保护/监控'),
    ('GIS_500kV',          '%500kV%GIS%',         '开关柜/环网柜'),
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
    'Conformal Prediction: P10-P90 Intervals for 10 Materials',
    'Split Conformal (Vovk 2005): proper_train(50m) + calib(12m) + test(12m)'
)

# ========================================================================
# Step 1: Data loading
# ========================================================================
log.step('Data Loading', '10 materials + shared features + self-lag features')

ref = pd.read_excel(REF_XLSX, sheet_name=0)
COLS = [c for c in ref.columns if c not in ['日期', '需求量']]
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
X_shared = ref[ref['mk'].between('202005','202606')][COLS].values.astype(np.float64)
X_sh_tr, X_sh_te = X_shared[:TRAIN], X_shared[TRAIN:]

def build_self(y_tr, y_te):
    y_all = np.concatenate([y_tr, y_te])
    def extr(N, offset):
        F = np.zeros((N, 8))
        for i in range(N):
            t = offset + i
            F[i,0]=y_all[t-1]if t>=1 else 0.0;F[i,1]=y_all[t-2]if t>=2 else 0.0
            F[i,2]=y_all[t-3]if t>=3 else 0.0;F[i,3]=y_all[t-6]if t>=6 else 0.0
            F[i,4]=y_all[t-12]if t>=12 else 0.0
            w3=max(0,t-2);F[i,5]=np.mean(y_all[w3:t+1])if t>=1 else y_all[0]
            w6=max(0,t-5);F[i,6]=np.mean(y_all[w6:t+1])if t>=1 else y_all[0]
            last=-1
            for j in range(t-1,-1,-1):
                if y_all[j]>0:last=j;break
            F[i,7]=float(t-last)if last>=0 else 99.0
        return F
    return extr(TRAIN,0), extr(TEST,TRAIN)

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
    vals = np.array([dmap.get(m,0) for m in MONTHS], dtype=np.float64)
    L_tr, L_te = build_self(vals[:TRAIN], vals[TRAIN:])
    mat_data[ek] = {
        'name': best[0], 'cat': cat,
        'y_tr': vals[:TRAIN], 'y_te': vals[TRAIN:],
        'X_tr': np.column_stack([X_sh_tr, L_tr]),
        'X_te': np.column_stack([X_sh_te, L_te]),
        'nz_tr': (vals[:TRAIN]>0).sum(), 'nz_te': (vals[TRAIN:]>0).sum(),
    }
    log.task(f'  {ek:<22s} → {best[0][:35]} | train_nz={mat_data[ek]["nz_tr"]}/62')

db.close()
log.step_ok('10 materials loaded')

# ========================================================================
# Step 2: Optuna + Conformal Prediction per material
# ========================================================================
log.step('Conformal Prediction', f'Split: proper_train({PROPER_TRAIN}) + calib({CALIB_SIZE}) + test({TEST})')

def suggest(trial, space):
    pp = {}
    for pn, (pt, lo, hi) in space.items():
        if pt == 'int': pp[pn] = trial.suggest_int(pn, lo, hi)
        else: pp[pn] = trial.suggest_float(pn, lo, hi, log=True)
    return pp

results = []
alpha_target = 0.20  # P10-P90 = 80% interval => alpha = 0.20

for ek, d in mat_data.items():
    y_tr, y_te = d['y_tr'], d['y_te']
    X_tr, X_te = d['X_tr'], d['X_te']

    # ---- Split: proper_train (50) + calibration (12) ----
    X_proper, y_proper = X_tr[:PROPER_TRAIN], y_tr[:PROPER_TRAIN]
    X_calib,  y_calib  = X_tr[PROPER_TRAIN:], y_tr[PROPER_TRAIN:]
    na_p = len(X_proper); nv_p = min(CALIB_SIZE//2, na_p//4)

    # ---- Optuna on proper_train only ----
    def obj(trial):
        pp = suggest(trial, S2_SPACE)
        m = lgb.LGBMRegressor(**pp, random_state=42, verbose=-1, force_col_wise=True)
        if nv_p >= 3:
            m.fit(X_proper[:na_p-nv_p], y_proper[:na_p-nv_p],
                  eval_set=[(X_proper[na_p-nv_p:], y_proper[na_p-nv_p:])])
        else: m.fit(X_proper, y_proper)
        yp = np.maximum(m.predict(X_proper[na_p-nv_p:]), 0)
        try: return r2_score(y_proper[na_p-nv_p:], yp)
        except: return -10.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=60, show_progress_bar=False)

    # ---- Train final model on proper_train, predict on calibration ----
    m_final = lgb.LGBMRegressor(**study.best_params, random_state=42, verbose=-1, force_col_wise=True)
    m_final.fit(X_proper, y_proper)
    y_calib_pred = np.maximum(m_final.predict(X_calib), 0)

    # ---- Compute nonconformity scores on calibration ----
    residuals = np.abs(y_calib - y_calib_pred)
    scores = residuals.copy()

    # Conformal quantile: (1-alpha)-quantile of scores
    n_calib = len(scores)
    q_idx = int(np.ceil((1 - alpha_target) * (n_calib + 1))) - 1
    q_idx = min(q_idx, n_calib - 1)
    q_idx = max(q_idx, 0)
    scores_sorted = np.sort(scores)
    conformal_radius = scores_sorted[q_idx]

    # ---- Predict on test ----
    y_test_pred = np.maximum(m_final.predict(X_te), 0)
    lo = np.maximum(y_test_pred - conformal_radius, 0)
    hi = y_test_pred + conformal_radius

    # ---- Evaluate ----
    inside = (y_te >= lo) & (y_te <= hi)
    coverage = inside.sum() / TEST
    mean_width = np.mean(hi - lo)
    mean_point = np.mean(y_test_pred)
    rel_width = mean_width / max(mean_point, 1.0)
    r2_pt = r2_score(y_te, y_test_pred)

    # Winkler score (interval score for (1-alpha) prediction interval)
    # Winkler = width + (2/alpha)*(lo - actual) if actual < lo
    #                    + (2/alpha)*(actual - hi) if actual > hi
    winkler = 0.0
    for i in range(TEST):
        w = hi[i] - lo[i]  # base penalty = width
        if y_te[i] < lo[i]:
            w += (2.0 / alpha_target) * (lo[i] - y_te[i])
        elif y_te[i] > hi[i]:
            w += (2.0 / alpha_target) * (y_te[i] - hi[i])
        winkler += w
    winkler_mean = winkler / TEST

    # Sharpness: coverage * (1 - rel_width/max_rel_width) — higher is better
    sharpness = coverage * max(0, 1 - rel_width / 5.0) if rel_width < 5 else 0

    results.append({
        'key': ek, 'name': d['name'][:30], 'cat': d['cat'],
        'nz_tr': d['nz_tr'], 'nz_te': d['nz_te'],
        'r2_point': round(r2_pt, 4),
        'coverage': round(coverage, 4),
        'conformal_radius': round(float(conformal_radius), 2),
        'mean_width': round(mean_width, 2),
        'mean_point': round(mean_point, 2),
        'rel_width': round(rel_width, 4),
        'winkler': round(winkler_mean, 2),
        'sharpness': round(sharpness, 4),
        'inside_count': int(inside.sum()),
        'y_te': y_te, 'pred_pt': y_test_pred, 'lo': lo, 'hi': hi,
    })

    log.metrics(ek, {
        'Cov': f'{coverage:.0%}', 'Width': f'{mean_width:.0f}',
        'Width/Point': f'{rel_width:.2f}', 'Winkler': f'{winkler_mean:.0f}',
        'R2_pt': f'{r2_pt:.4f}', 'radius': f'{conformal_radius:.0f}',
    })

log.step_ok('10 conformal predictions complete')

# ========================================================================
# Step 3: Summary
# ========================================================================
log.step('Summary', 'Coverage, Width, Winkler Score, Sharpness')

summary_rows = []
for r in results:
    summary_rows.append([
        r['key'][:16], f"{r['coverage']:.0%}", f"{r['mean_width']:.0f}",
        f"{r['rel_width']:.2f}", f"{r['winkler']:.0f}",
        f"{r['sharpness']:.3f}", f"{r['r2_point']:.3f}",
    ])

log.data_table('Conformal Prediction Results',
               ['Material','Coverage','Width','W/Point','Winkler','Sharpness','R2_pt'],
               summary_rows)

# Aggregate stats
mean_cov = np.mean([r['coverage'] for r in results])
mean_wid = np.mean([r['mean_width'] for r in results])
mean_wink = np.mean([r['winkler'] for r in results])
on_target = sum(1 for r in results if 0.70 <= r['coverage'] <= 0.90)

log.task(f'Mean Coverage: {mean_cov:.1%} | Mean Width: {mean_wid:.0f} | Mean Winkler: {mean_wink:.0f}')
log.task(f'Coverage in [70%, 90%]: {on_target}/10 (ideal=10)')

log.step_ok(f'Summary done: coverage={mean_cov:.1%}, on_target={on_target}/10')

# ========================================================================
# Step 4: Charts
# ========================================================================
log.step('Charts', '5 figures for the report')

# Fix Chinese
for d in [matplotlib.get_cachedir()]:
    try:
        for f in os.listdir(d):
            if 'fontlist' in f: os.remove(os.path.join(d, f))
    except: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fig 1: Coverage vs Width scatter (the "efficient frontier")
fig1, ax1 = plt.subplots(figsize=(12, 7))
covs = [r['coverage'] for r in results]
wids = [r['mean_width'] for r in results]
names = [r['key'][:10] for r in results]
# Color by R2
r2s = [max(0, r['r2_point']) for r in results]
scatter = ax1.scatter(covs, wids, c=r2s, cmap='RdYlGn', s=150, alpha=0.8, edgecolors='black', linewidth=0.5)
for i, name in enumerate(names):
    ax1.annotate(name, (covs[i], wids[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax1.axvline(x=0.80, color='gray', linestyle='--', alpha=0.5, label='Target 80%')
ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.3, label='Width=100')
ax1.set_xlabel('Coverage (fraction in [P10,P90])', fontsize=12)
ax1.set_ylabel('Mean Interval Width', fontsize=12)
ax1.set_title(f'Conformal Prediction: Coverage vs Width\n(10 materials, each dot=1 material)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9); plt.colorbar(scatter, ax=ax1, label='Point R2')
ax1.grid(True, alpha=0.3)
plt.tight_layout(); fig1.savefig(os.path.join(OUT_DIR, 'fig1_coverage_vs_width.png'), dpi=150); plt.close()

# Fig 2: Prediction intervals for Top 4 materials (by R2)
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
top4 = sorted(results, key=lambda x: -x['r2_point'])[:4]
for i, r in enumerate(top4):
    ax = axes2[i//2, i%2]
    months = TEST_DATES
    ax.fill_between(months, r['lo'], r['hi'], color='#2196F3', alpha=0.15, label='P10-P90 Interval')
    ax.plot(months, r['y_te'], 'ko-', linewidth=2.5, markersize=7, label='Actual', zorder=3)
    ax.plot(months, r['pred_pt'], 's-', color='#E91E63', linewidth=2, markersize=6,
            label=f'Point Pred (R2={r["r2_point"]:.3f})', zorder=2)
    # Mark coverage
    for j in range(TEST):
        inside = r['lo'][j] <= r['y_te'][j] <= r['hi'][j]
        if not inside:
            ax.plot(months[j], r['y_te'][j], 'ro', markersize=12, markerfacecolor='none',
                    markeredgewidth=2.5, label='Outside' if j == 0 else '')
    ax.set_title(f"{r['key']} [{r['cat']}]\nCoverage={r['coverage']:.0%}, Width={r['mean_width']:.0f}, Winkler={r['winkler']:.0f}",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3); ax.tick_params('x', rotation=30)
fig2.suptitle('Conformal P10-P90 Prediction Intervals — Top 4 Materials', fontsize=14, fontweight='bold')
plt.tight_layout(); fig2.savefig(os.path.join(OUT_DIR, 'fig2_intervals_top4.png'), dpi=150); plt.close()

# Fig 3: Coverage bar chart (all 10)
fig3, ax3 = plt.subplots(figsize=(14, 5))
x3 = np.arange(10)
covs3 = [r['coverage'] for r in results]
clrs3 = ['#4CAF50' if 0.70 <= c <= 0.90 else '#FF9800' if c > 0.90 else '#F44336' for c in covs3]
bars3 = ax3.bar(x3, covs3, color=clrs3, alpha=0.8, edgecolor='white')
ax3.axhline(y=0.80, color='gray', linestyle='--', linewidth=1.5, label='Target 80%')
for bar, c in zip(bars3, covs3):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{c:.0%}', ha='center', fontsize=10)
ax3.set_xticks(x3); ax3.set_xticklabels([r['key'][:10] for r in results], fontsize=8, rotation=30)
ax3.set_ylabel('Coverage'); ax3.set_ylim(0, 1.15); ax3.legend()
ax3.set_title('Conformal Prediction: Coverage per Material (Target=80%)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); fig3.savefig(os.path.join(OUT_DIR, 'fig3_coverage_bars.png'), dpi=150); plt.close()

# Fig 4: Winkler Score ranking
fig4, ax4 = plt.subplots(figsize=(14, 5))
sorted_by_wink = sorted(results, key=lambda x: x['winkler'])
wink_vals = [r['winkler'] for r in sorted_by_wink]
wink_names = [r['key'][:12] for r in sorted_by_wink]
clrs4 = ['#4CAF50' if w < np.median(wink_vals) else '#FF9800' for w in wink_vals]
ax4.barh(range(10), wink_vals, color=clrs4, alpha=0.8)
for i, (w, r) in enumerate(zip(wink_vals, sorted_by_wink)):
    ax4.text(w+1, i, f'{w:.0f} (cov={r["coverage"]:.0%}, w={r["mean_width"]:.0f})', fontsize=8, va='center')
ax4.set_yticks(range(10)); ax4.set_yticklabels(wink_names, fontsize=9)
ax4.set_xlabel('Winkler Score (lower=better)'); ax4.set_title('Conformal Prediction: Winkler Score (lower=better interval)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')
plt.tight_layout(); fig4.savefig(os.path.join(OUT_DIR, 'fig4_winkler.png'), dpi=150); plt.close()

# Fig 5: Point R2 vs Interval Efficiency
fig5, ax5 = plt.subplots(figsize=(10, 6))
efficiency = [r['sharpness'] for r in results]
r2_vals = [max(-0.5, r['r2_point']) for r in results]
names5 = [r['key'][:10] for r in results]
sc = ax5.scatter(r2_vals, efficiency, c=range(10), cmap='tab10', s=120, alpha=0.8, edgecolors='black')
for i in range(10):
    ax5.annotate(names5[i], (r2_vals[i], efficiency[i]), textcoords="offset points", xytext=(5,5), fontsize=8)
ax5.set_xlabel('Point Prediction R2', fontsize=12)
ax5.set_ylabel('Interval Sharpness (coverage/(1+width))', fontsize=12)
ax5.set_title('Point Accuracy vs Interval Quality (higher+right = best)', fontsize=13, fontweight='bold')
ax5.axhline(y=np.mean(efficiency), color='gray', linestyle='--', alpha=0.5)
ax5.axvline(x=np.mean(r2_vals), color='gray', linestyle='--', alpha=0.5)
ax5.grid(True, alpha=0.3)
plt.tight_layout(); fig5.savefig(os.path.join(OUT_DIR, 'fig5_point_vs_interval.png'), dpi=150); plt.close()

log.step_ok('5 charts generated')

# ========================================================================
# Step 5: Detailed report for each material (text)
# ========================================================================
log.step('Detail Report', 'Per-material interval analysis')

for r in sorted(results, key=lambda x: -x['r2_point']):
    ek = r['key']
    log.task(f"[{ek}] Coverage={r['coverage']:.0%} "
             f"PointR2={r['r2_point']:.3f} "
             f"Width=[{r['mean_width']:.0f}] "
             f"ConformalRadius={r['conformal_radius']:.0f} "
             f"Inside={r['inside_count']}/{TEST}")

    # Show the test months where coverage failed
    failed_months = []
    for i in range(TEST):
        if not (r['lo'][i] <= r['y_te'][i] <= r['hi'][i]):
            month_str = f"2025-{7+i:02d}" if i < 6 else f"2026-{i-5:02d}"
            failed_months.append(f"{month_str}(actual={r['y_te'][i]:.0f}, "
                                f"interval=[{r['lo'][i]:.0f},{r['hi'][i]:.0f}])")
    if failed_months:
        log.task_detail(f'  FAILED months: {", ".join(failed_months)}')

log.step_ok('Detail report complete')

# ========================================================================
# Step 6: Comparison with pure point prediction
# ========================================================================
log.step('Final Comparison', 'Conformal Interval vs Point Prediction')

print(f"\n{'='*100}")
print(f" CONFORMAL PREDICTION — COMPLETE EVALUATION REPORT")
print(f"{'='*100}")
print(f"\n{'Material':<22s} {'PtR2':>7s} {'Coverage':>9s} {'Width':>8s} {'W/Point':>8s} {'Winkler':>8s} {'Sharp':>7s} {'Radius':>8s} {'Inside':>7s}")
print(f"{'-'*95}")
for r in results:
    print(f"{r['key'][:20]:<22s} {r['r2_point']:>7.4f} {r['coverage']:>8.0%} {r['mean_width']:>8.0f} "
          f"{r['rel_width']:>8.2f} {r['winkler']:>8.0f} {r['sharpness']:>7.3f} {r['conformal_radius']:>8.0f} {r['inside_count']:>5d}/{TEST}")

print(f"\nAGGREGATE:")
print(f"  Mean Coverage:          {mean_cov:.1%}")
print(f"  Mean Width:             {mean_wid:.0f}")
print(f"  Mean Winkler Score:     {mean_wink:.0f}")
print(f"  Coverage on-target:     {on_target}/10 (70%-90%)")
print(f"\n  Top material (interval): {min(results, key=lambda r: r['winkler'])['key']} "
      f"(Winkler={min(r['winkler'] for r in results):.0f})")
print(f"  Best coverage:           {max(results, key=lambda r: r['coverage'])['key']} "
      f"({max(r['coverage'] for r in results):.0%})")

# Save JSON
out_json = {
    'method': 'Split Conformal Prediction (Vovk 2005)',
    'setup': f'proper_train({PROPER_TRAIN}) + calibration({CALIB_SIZE}) + test({TEST})',
    'alpha': alpha_target,
    'aggregate': {
        'mean_coverage': round(mean_cov, 4), 'mean_width': round(mean_wid, 2),
        'mean_winkler': round(mean_wink, 2), 'on_target': on_target,
    },
    'materials': [{k: v for k, v in r.items() if k not in ['y_te','pred_pt','lo','hi']}
                  for r in results],
    'timestamp': datetime.now().isoformat(),
}
json_path = os.path.join(OUT_DIR, 'conformal_results.json')
json.dump(out_json, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)

log.pipeline_end_log(f'Coverage={mean_cov:.1%}, Width={mean_wid:.0f}, Winkler={mean_wink:.0f}, OnTarget={on_target}/10')
print(f"\n  日志: {log.get_log_path()}")
print(f"  JSON: {json_path}")
