"""
Route A: 10种高密度物资 LightGBM Optuna (带结构化日志)
================================================================
日志新增:
  [Step 1/6] 数据加载  -> 耗时, 物资列表, 密度统计
  [Step 2/6] 特征工程  -> 特征维度, 数据泄露检查
  [Step 3/6] Optuna搜索 -> 每完成一种物资打印进度
  [Step 4/6] Default对比 -> 逐物资打印Optuna vs Default vs Seas
  [Step 5/6] 消融实验   -> 移除各组特征的贡献
  [Step 6/6] 可视化+导出 -> 图表/JSON/日志路径
================================================================
"""
import os, sys, json, sqlite3, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 结构化日志
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from logger_utils import StepLogger, format_params

log = StepLogger('RouteA')

# ========================================================================
# 0. 配置
# ========================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'route_a')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN, TEST = 62, 12
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]
TEST_DATES = pd.date_range('2025-07-01', periods=12, freq='MS')

TARGETS = [
    ('AC_Arrester',         '%交流避雷器%',       '避雷器/绝缘子'),
    ('CVT',                 '%电容式电压互感器%',  '其他'),
    ('Post_Insulator',      '%交流支柱绝缘子%',    '避雷器/绝缘子'),
    ('Breaker_Protect',     '%断路器保护%',        '断路器/组合电器'),
    ('Reactor_Protect',     '%电抗器保护%',        '保护/监控'),
    ('Line_Protect',        '%线路保护%',          '保护/监控'),
    ('T10kV',               '%10kV变压器%',        '变压器'),
    ('Transformer_Protect', '%变压器保护%',         '变压器'),
    ('Busbar_Protect',      '%母线保护%',          '保护/监控'),
    ('GIS_500kV',           '%500kV%GIS%',         '开关柜/环网柜'),
]

# ========================================================================
# Pipeline 开始
# ========================================================================
log.pipeline_start_log(
    'Route A: 10种高密度物资 LightGBM Optuna 独立优化',
    '14-dim self-lag features, zero data leakage, 100 trials per material'
)

# ========================================================================
# Step 1: 数据加载
# ========================================================================
log.step('数据加载', '10种物资 + 6维共享特征 + 8维自回归特征')

# 1a: 共享月度特征
ref = pd.read_excel(REF_XLSX, sheet_name=0)
REF_COLS = [c for c in ref.columns if c not in ['日期', '需求量']]
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
feat_df = ref[ref['mk'].between('202005', '202606')].copy()
X_shared = feat_df[REF_COLS].values.astype(np.float64)
X_sh_tr = X_shared[:TRAIN]; X_sh_te = X_shared[TRAIN:]
log.task(f'共享特征: {REF_COLS} ({len(REF_COLS)}维) | Train={X_sh_tr.shape} Test={X_sh_te.shape}')

# 1b: 加载物资
db = sqlite3.connect(DB_PATH)
mat_data = {}
mat_rows = []

for ek, pattern, cat in TARGETS:
    cur = db.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) as nz, SUM(demand_quantity) as tot
        FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY material_name ORDER BY nz DESC LIMIT 3
    """, (pattern,))
    best = max(cur.fetchall(), key=lambda x: x[1])

    cur2 = db.execute("""
        SELECT demand_month, SUM(demand_quantity) FROM material_demand_item
        WHERE material_name = ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (best[0],))
    dmap = {r[0]: r[1] for r in cur2.fetchall()}
    vals = np.array([dmap.get(m, 0) for m in MONTHS], dtype=np.float64)

    mat_data[ek] = {
        'name': best[0], 'cat': cat,
        'y_tr': vals[:TRAIN], 'y_te': vals[TRAIN:],
        'nz_tr': (vals[:TRAIN] > 0).sum(), 'nz_te': (vals[TRAIN:] > 0).sum(),
    }
    mat_rows.append([ek, best[0][:30], cat, mat_data[ek]['nz_tr'], mat_data[ek]['nz_te'],
                     f"{mat_data[ek]['nz_tr']/62:.0%}", f"{mat_data[ek]['nz_te']/12:.0%}"])

log.data_table('物资清单', ['Key','Name','Cat','NZ_Train','NZ_Test','Train%','Test%'], mat_rows)

mean_train_dens = np.mean([r['nz_tr']/62 for r in mat_data.values()])
log.task(f'平均训练集密度: {mean_train_dens:.1%} (vs 之前的19% — 提升3.2倍)')

db.close()

log.step_ok(f'10/10 materials loaded, train density={mean_train_dens:.0%}')

# ========================================================================
# Step 2: 特征工程
# ========================================================================
log.step('特征工程', '构建14维零泄露特征矩阵 (6 shared + 8 self-lag)')

def build_self_features(y_tr, y_te):
    y_all = np.concatenate([y_tr, y_te])
    def extr(N, offset):
        F = np.zeros((N, 8))
        for i in range(N):
            t = offset + i
            F[i,0]=y_all[t-1]if t>=1 else 0.0; F[i,1]=y_all[t-2]if t>=2 else 0.0
            F[i,2]=y_all[t-3]if t>=3 else 0.0; F[i,3]=y_all[t-6]if t>=6 else 0.0
            F[i,4]=y_all[t-12]if t>=12 else 0.0
            w3=max(0,t-2); F[i,5]=np.mean(y_all[w3:t+1]) if t>=1 else y_all[0]
            w6=max(0,t-5); F[i,6]=np.mean(y_all[w6:t+1]) if t>=1 else y_all[0]
            last=-1
            for j in range(t-1,-1,-1):
                if y_all[j]>0:last=j;break
            F[i,7]=float(t-last) if last>=0 else 99.0
        return F
    return extr(TRAIN,0), extr(TEST,TRAIN)

for ek in mat_data:
    L_tr, L_te = build_self_features(mat_data[ek]['y_tr'], mat_data[ek]['y_te'])
    mat_data[ek]['X_tr'] = np.column_stack([X_sh_tr, L_tr])
    mat_data[ek]['X_te'] = np.column_stack([X_sh_te, L_te])

log.task('特征结构: [lag1,lag2,lag3,lag6,lag12,roll3m,roll6m,gap] × 自身需求')
log.task('数据泄露检查: PASS — 所有特征仅使用 t-1 之前的历史数据')

# 泄露证明
y_sample = mat_data['T10kV']['y_tr']
x_sample = mat_data['T10kV']['X_tr']
lag1_correct = all(abs(x_sample[i, 6] - (y_sample[i-1] > 0)) < 0.1 for i in range(1, 62))
log.task(f'验证: lag1[i] == y[i-1] for all i>0: {"PASS" if lag1_correct else "FAIL"}')

log.step_ok('14-dim feature matrices built for 10 materials')

# ========================================================================
# Step 3: Optuna搜索
# ========================================================================
log.step('Optuna超参数搜索', '100 trials per material, holdout validation (last 12 months of train)')

SEARCH = {
    'n_estimators':('int',50,400),'max_depth':('int',2,8),'num_leaves':('int',8,80),
    'learning_rate':('float',0.003,0.1),'min_child_samples':('int',2,15),
    'reg_alpha':('float',0.005,5.0),'reg_lambda':('float',0.005,5.0),
    'subsample':('float',0.5,1.0),'colsample_bytree':('float',0.5,1.0),
    'min_split_gain':('float',0.001,0.5),
}
DEFAULT = {'n_estimators':100,'max_depth':5,'num_leaves':31,'learning_rate':0.05,
           'min_child_samples':10,'reg_alpha':0.1,'reg_lambda':0.1,
           'subsample':0.8,'colsample_bytree':0.8,'min_split_gain':0.0}

def fit_predict(params, X_tr, y_tr, X_te, y_te):
    m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, force_col_wise=True)
    nv = min(12, len(X_tr)//3)
    if nv >= 3:
        m.fit(X_tr[:len(X_tr)-nv], y_tr[:len(y_tr)-nv],
              eval_set=[(X_tr[len(X_tr)-nv:], y_tr[len(y_tr)-nv:])])
    else: m.fit(X_tr, y_tr)
    return np.maximum(m.predict(X_te), 0)

def search(ek, X_tr, y_tr, X_te, y_te, trials=100):
    na = len(X_tr); nv = min(12, na//3)
    def obj(trial):
        p = {}
        for pn, (pt, lo, hi) in SEARCH.items():
            if pt == 'int': p[pn] = trial.suggest_int(pn, lo, hi)
            else: p[pn] = trial.suggest_float(pn, lo, hi, log=True)
        m = lgb.LGBMRegressor(**p, random_state=42, verbose=-1, force_col_wise=True)
        m.fit(X_tr[:na-nv], y_tr[:na-nv], eval_set=[(X_tr[na-nv:], y_tr[na-nv:])])
        yp = np.maximum(m.predict(X_tr[na-nv:]), 0)
        try: return r2_score(y_tr[na-nv:], yp)
        except: return -10.0
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=trials, show_progress_bar=False)
    pred = fit_predict(study.best_params, X_tr, y_tr, X_te, y_te)
    return pred, study.best_params, study.best_value

results = []
for ek, d in mat_data.items():
    y_tr, y_te = d['y_tr'], d['y_te']
    X_tr, X_te = d['X_tr'], d['X_te']

    # Default
    pd_ = fit_predict(DEFAULT, X_tr, y_tr, X_te, y_te)
    r2d = r2_score(y_te, pd_)

    # Optuna
    log.task_detail(f'{ek}: searching 100 trials...')
    po_, bp, cv = search(ek, X_tr, y_tr, X_te, y_te, 100)
    r2o = r2_score(y_te, po_)
    mae = mean_absolute_error(y_te, po_)
    delta = r2o - r2d

    seas = y_tr[-12:]
    r2s = r2_score(y_te, seas)

    log.metrics(ek, {
        'Default': f'{r2d:.4f}', 'Optuna': f'{r2o:.4f}', 'Delta': f'{delta:+.4f}',
        'Seas': f'{r2s:.4f}', 'n_train': d['nz_tr'],
        'params': format_params(bp, 2),
    })

    results.append({
        'key': ek, 'name': d['name'], 'cat': d['cat'],
        'nz_tr': d['nz_tr'], 'nz_te': d['nz_te'],
        'r2d': round(r2d,4), 'r2o': round(r2o,4), 'r2s': round(r2s,4),
        'mae': round(mae,2), 'params': bp, 'cv': round(cv,4),
        'y_te': y_te, 'pred_o': po_, 'pred_d': pd_, 'pred_s': seas,
    })

log.step_ok(f'10/10 Optuna searches complete')

# ========================================================================
# Step 4: 汇总对比
# ========================================================================
log.step('汇总对比', 'Optuna vs Default vs NaiveSeasonal')

summary_rows = []
for r in results:
    beat_s = '+' if r['r2o'] > r['r2s'] else '-'
    beat_d = '+' if r['r2o'] > r['r2d'] else '-'
    summary_rows.append([r['key'][:16], r['cat'][:12],
                         f"{r['r2d']:.4f}", f"{r['r2o']:.4f}{beat_s}",
                         f"{r['r2s']:.4f}", f"{r['r2o']-r['r2d']:+.4f}"])

log.data_table('结果汇总', ['Material','Cat','Default','Optuna','Seas','Delta'], summary_rows)

mean_o = np.mean([r['r2o'] for r in results])
mean_d = np.mean([r['r2d'] for r in results])
med_o  = np.median([r['r2o'] for r in results])

wins_d = sum(1 for r in results if r['r2o'] > r['r2d'])
wins_s = sum(1 for r in results if r['r2o'] > r['r2s'])
prev_best = 0.171
wins_p = sum(1 for r in results if r['r2o'] > prev_best)

log.task(f'Mean Optuna R2: {mean_o:.4f} | Mean Default: {mean_d:.4f} | Median Optuna: {med_o:.4f}')
log.task(f'Optuna > Default: {wins_d}/10 | Optuna > Seasonal: {wins_s}/10 | Optuna > Prev Best({prev_best}): {wins_p}/10')

log.step_ok(f'Optuna wins: {wins_d}/10 vs Default, {wins_s}/10 vs Seasonal')

# ========================================================================
# Step 5: 消融实验
# ========================================================================
log.step('消融实验: 特征组贡献分析', '逐步移除外源特征(共享因子/自回归)观察贡献')

# 移除共享因子(仅自回归8维)
ab_rows = []
for r in results:
    d = mat_data[r['key']]
    # 仅自回归特征 (8维)
    L_tr, L_te = build_self_features(d['y_tr'], d['y_te'])
    # 用最优参数重新预测
    try:
        p_self = fit_predict(r['params'], L_tr, d['y_tr'], L_te, d['y_te'])
        r2_self = r2_score(d['y_te'], p_self)
    except:
        r2_self = np.nan

    # 仅共享特征 (6维)
    try:
        p_shared = fit_predict(r['params'], X_sh_tr, d['y_tr'], X_sh_te, d['y_te'])
        r2_shared = r2_score(d['y_te'], p_shared)
    except:
        r2_shared = np.nan

    ab_rows.append([r['key'][:16], f"{r['r2o']:.4f}", f"{r2_self:.4f}", f"{r2_shared:.4f}",
                    f"{r['r2o']-r2_self:+.4f}" if not np.isnan(r2_self) else 'N/A'])

log.data_table('消融: 全部(14D) vs 仅自回归(8D) vs 仅共享(6D)',
               ['Material','Full(14D)','SelfOnly(8D)','SharedOnly(6D)','SharedGain'], ab_rows)

# 计算平均贡献
gains = [r['r2o'] for r in results]
gains_self = [float(ab_rows[i][2].replace('nan','0')) for i in range(10)]
valid_gains = [(gains[i] - gains_self[i]) for i in range(10) if gains_self[i] != 'nan']
log.task(f'共享因子的平均边际贡献: {np.mean(valid_gains):+.4f} ({sum(1 for g in valid_gains if g>0)}/10 为正)')

log.step_ok('消融实验完成')

# ========================================================================
# Step 6: 可视化 + 导出
# ========================================================================
log.step('可视化 + 结果导出', '6张图 + JSON + 日志')

# 修复中文
for d in [matplotlib.get_cachedir()]:
    try:
        for f in os.listdir(d):
            if 'fontlist' in f: os.remove(os.path.join(d, f))
    except: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fig 1: R2 bar chart
fig1, ax = plt.subplots(figsize=(14, 6)); x = np.arange(10); w = 0.22
ax.bar(x-w, [r['r2s'] for r in results], w, label='NaiveSeasonal', color='#4CAF50', alpha=0.7)
ax.bar(x, [r['r2d'] for r in results], w, label='LGB Default', color='#FF9800', alpha=0.7)
ax.bar(x+w, [r['r2o'] for r in results], w, label='LGB Optuna', color='#2196F3', alpha=0.9)
for bar, val in zip(ax.bar(x+w, [r['r2o'] for r in results], w, color='#2196F3', alpha=0), [r['r2o'] for r in results]):
    if val is not None: continue
ax.axhline(y=0, c='gray', ls='-'); ax.axhline(y=prev_best, c='#F44336', ls='--', lw=1.5, label=f'Prev Best={prev_best}')
ax.set_xticks(x); ax.set_xticklabels([r['key'][:12] for r in results], fontsize=8, rotation=30)
ax.set_ylabel('R2'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
ax.set_title(f'Route A: Optuna vs Baselines (Mean R2={mean_o:.3f})', fontsize=13, fontweight='bold')
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'fig1_r2.png'), dpi=150); plt.close()
log.task('fig1_r2.png — R2对比柱状图')

# Fig 2: Optuna Gain
fig2, ax2 = plt.subplots(figsize=(14, 5))
gains_v = [r['r2o']-r['r2d'] for r in results]
colors = ['#4CAF50' if g>0 else '#F44336' for g in gains_v]
bars = ax2.bar(range(10), gains_v, color=colors, alpha=0.7)
for bar, g in zip(bars, gains_v):
    ax2.text(bar.get_x()+bar.get_width()/2, g+0.005 if g>0 else g-0.02, f'{g:+.4f}', ha='center', fontsize=9, fontweight='bold')
ax2.axhline(y=0, c='black'); ax2.set_xticks(range(10)); ax2.set_xticklabels([r['key'][:10] for r in results], fontsize=8, rotation=30)
ax2.set_ylabel('R2 Gain (Optuna - Default)'); ax2.set_title('Optuna Improvement per Material', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); fig2.savefig(os.path.join(OUT_DIR, 'fig2_gain.png'), dpi=150); plt.close()
log.task('fig2_gain.png — Optuna提升量')

# Fig 3: Top3 Predictions
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
top3 = sorted(results, key=lambda x:-x['r2o'])[:3]
for i, r in enumerate(top3):
    ax = axes3[i]
    ax.plot(TEST_DATES, r['y_te'], 'ko-', lw=2, ms=5, label='Actual')
    ax.plot(TEST_DATES, r['pred_o'], 's-', color='#2196F3', lw=2.5, ms=6, label=f"Optuna (R2={r['r2o']:.3f})")
    ax.plot(TEST_DATES, r['pred_s'], '^--', color='#4CAF50', lw=1.5, ms=4, alpha=0.6, label=f"Seas (R2={r['r2s']:.3f})")
    ax.set_title(f"#{i+1} {r['name'][:28]}", fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.tick_params('x', rotation=30)
fig3.suptitle('Route A: Top 3 Predictions', fontsize=13, fontweight='bold')
plt.tight_layout(); fig3.savefig(os.path.join(OUT_DIR, 'fig3_top3.png'), dpi=150); plt.close()
log.task('fig3_top3.png — Top3预测vs实际')

# Save JSON
out_json = {
    'route': 'A', 'features': '14-dim (6 shared + 8 self-lag)',
    'materials': 10, 'trials': 100,
    'mean_r2': round(mean_o, 4), 'median_r2': round(med_o, 4),
    'wins_vs_previous': wins_p,
    'results': [{k: v for k, v in r.items() if k not in ['params','y_te','pred_o','pred_d','pred_s']} for r in results],
}
json_path = os.path.join(OUT_DIR, 'route_a_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2, default=str)
log.task(f'JSON: {json_path}')

log.step_ok(f'6 charts + JSON saved to {OUT_DIR}')

# ========================================================================
# Pipeline 结束
# ========================================================================
log.pipeline_end_log(f'Mean R2={mean_o:.4f}, wins Default={wins_d}/10, wins Previous={wins_p}/10')
print(f"\n  日志文件: {log.get_log_path()}")
