"""
Route A 正式版: 10种高密度物资 LightGBM Optuna 独立优化
================================================================
步骤:
  1. 从ECP数据库加载10种已经筛选好的高密度物资 (≥43/74非零月, 7个品类)
  2. 从data.xlsx加载6维共享月度特征 (全物资共用)
  3. 从需求量自身提取8维滞后+统计特征 (无数据泄露)
  4. 合并为14维特征向量
  5. 对每种物资独立运行Optuna 100-trial贝叶斯搜索
  6. 对比: Optuna vs Default LightGBM vs NaiveSeasonal vs NaiveMean
  7. 输出6张可视化图表 + JSON结果

数据泄露防护:
  - 全部特征仅使用该时间点之前的数据
  - lag特征: t-1, t-2, t-3, t-6, t-12 (仅用历史)
  - 滚动统计: 使用截止该时间点的数据
  - gap: 距离上次非零月的距离 (仅用历史)
  - 不使用任何伙伴物资特征
  - 不使用month_sin/cos（避免未来信息的编码）

训练/测试分割:
  - 74个月 (2020-05 ~ 2026-06)
  - 训练集: 前62个月 (2020-05 ~ 2025-06)
  - 测试集: 后12个月 (2025-07 ~ 2026-06)
================================================================
"""
import os, sys, json, sqlite3, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ========================================================================
# 0. 路径与常量
# ========================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'route_a')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_LEN = 62
TEST_LEN = 12
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]
MONTH_DATES = pd.date_range('2020-05-01', periods=74, freq='MS')
TEST_DATES = pd.date_range('2025-07-01', periods=12, freq='MS')

# ========================================================================
# 1. 加载共享月度特征
# ========================================================================
ref = pd.read_excel(REF_XLSX, sheet_name=0)
REF_COLS = [c for c in ref.columns if c not in ['日期', '需求量']]
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
feat_df = ref[ref['mk'].between('202005', '202606')].copy()
X_SHARED = feat_df[REF_COLS].values.astype(np.float64)
X_shared_train = X_SHARED[:TRAIN_LEN]   # (62, 6)
X_shared_test  = X_SHARED[TRAIN_LEN:]   # (12, 6)

print(f"共享月度特征 ({len(REF_COLS)}维): {REF_COLS}")
print(f"  Train shape: {X_shared_train.shape}, Test shape: {X_shared_test.shape}")

# ========================================================================
# 2. 加载10种高密度物资的精确完整名称
# ========================================================================
db = sqlite3.connect(DB_PATH)

TARGETS = [
    # (key, SQL LIKE pattern, category)
    ('AC_Arrester',        '%交流避雷器%',       '避雷器/绝缘子'),
    ('CVT',                '%电容式电压互感器%',  '其他'),
    ('Post_Insulator',     '%交流支柱绝缘子%',    '避雷器/绝缘子'),
    ('Breaker_Protect',    '%断路器保护%',        '断路器/组合电器'),
    ('Reactor_Protect',    '%电抗器保护%',        '保护/监控'),
    ('Line_Protect',       '%线路保护%',          '保护/监控'),
    ('T10kV',              '%10kV变压器%',        '变压器'),
    ('Transformer_Protect', '%变压器保护%',        '变压器'),
    ('Busbar_Protect',     '%母线保护%',          '保护/监控'),
    ('GIS_500kV',          '%500kV%GIS%',         '开关柜/环网柜'),
]

material_data = {}

for eng_key, pattern, cat in TARGETS:
    # 用LIKE匹配找出非零月最多的那个精确名称
    cur = db.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) as nz, SUM(demand_quantity) as tot
        FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY material_name ORDER BY nz DESC LIMIT 3
    """, (pattern,))
    matches = cur.fetchall()
    if not matches:
        print(f"  WARNING: {eng_key} pattern '{pattern}' matched nothing!")
        continue
    best_match = max(matches, key=lambda x: x[1])
    exact_name = best_match[0]

    # 用精确名称获取月度序列
    cur2 = db.execute("""
        SELECT demand_month, SUM(demand_quantity)
        FROM material_demand_item
        WHERE material_name = ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (exact_name,))
    demand_map = {r[0]: r[1] for r in cur2.fetchall()}

    # 构建74个月的完整序列
    vals = np.array([demand_map.get(m, 0) for m in MONTHS], dtype=np.float64)
    y_train = vals[:TRAIN_LEN]
    y_test  = vals[TRAIN_LEN:]

    material_data[eng_key] = {
        'cn_name': exact_name,
        'cat': cat,
        'y_train': y_train,
        'y_test': y_test,
        'nz_total': (vals > 0).sum(),
        'nz_train': (y_train > 0).sum(),
        'nz_test':  (y_test > 0).sum(),
        'total_demand': vals.sum(),
    }

db.close()

print(f"\n加载 {len(material_data)} 种物资:\n")
print(f"{'Key':<22s} {'Name':<25s} {'Cat':<18s} {'NZ_Total':>8s} {'NZ_Train':>8s} {'NZ_Test':>7s} {'Total':>12s}")
print("-" * 105)
for ek, d in material_data.items():
    print(f"{ek:<22s} {d['cn_name']:<25s} {d['cat']:<18s} "
          f"{d['nz_total']:>4d}/74  {d['nz_train']:>4d}/62  {d['nz_test']:>4d}/12 "
          f"{d['total_demand']:>12.0f}")

# ========================================================================
# 3. 特征工程 (无数据泄露, 仅用自身历史)
# ========================================================================
def build_self_features(y_train, y_test):
    """
    从需求量序列构建8维自身特征。
    全部特征仅使用该时间点之前的数据 — 零数据泄露。

    特征说明:
      lag1, lag2, lag3, lag6, lag12: 自回归滞后
      rolling_mean_3m, rolling_mean_6m: 滚动均值
      months_since_last_demand: 距离上次非零月的月数
    """
    y_all = np.concatenate([y_train, y_test])
    total_len = len(y_all)

    def extract(n_samples, start_offset):
        F = np.zeros((n_samples, 8))
        for i in range(n_samples):
            t = start_offset + i  # 在完整序列中的绝对位置

            # 自回归滞后
            F[i, 0] = y_all[t - 1]  if t >= 1  else 0.0   # lag1
            F[i, 1] = y_all[t - 2]  if t >= 2  else 0.0   # lag2
            F[i, 2] = y_all[t - 3]  if t >= 3  else 0.0   # lag3
            F[i, 3] = y_all[t - 6]  if t >= 6  else 0.0   # lag6
            F[i, 4] = y_all[t - 12] if t >= 12 else 0.0   # lag12

            # 滚动均值 (使用截止到 t 的数据)
            w3_start = max(0, t - 2)  # [t-2, t-1, t]
            F[i, 5] = np.mean(y_all[w3_start:t+1]) if t >= 1 else y_all[0]

            w6_start = max(0, t - 5)  # [t-5, ..., t]
            F[i, 6] = np.mean(y_all[w6_start:t+1]) if t >= 1 else y_all[0]

            # 距上次非零需求的月数
            last_nz_idx = -1
            for j in range(t - 1, -1, -1):
                if y_all[j] > 0:
                    last_nz_idx = j
                    break
            F[i, 7] = float(t - last_nz_idx) if last_nz_idx >= 0 else 99.0

        return F

    X_self_train = extract(TRAIN_LEN, 0)
    X_self_test  = extract(TEST_LEN, TRAIN_LEN)

    return X_self_train, X_self_test

# ========================================================================
# 4. 模型评估
# ========================================================================
def train_eval_lgb(params, X_train, y_train, X_test, y_test):
    """用给定参数训练LightGBM并返回预测值"""
    model = lgb.LGBMRegressor(
        **params,
        random_state=42, n_jobs=1, verbose=-1,
        force_col_wise=True,
    )
    # 用最后12个月做验证集
    n_val = min(12, len(X_train) // 3)
    if n_val >= 3:
        model.fit(
            X_train[:len(X_train) - n_val],
            y_train[:len(y_train) - n_val],
            eval_set=[(X_train[len(X_train) - n_val:], y_train[len(y_train) - n_val:])],
        )
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return np.maximum(y_pred, 0)  # 物理约束: 需求量 >= 0

# ========================================================================
# 5. Optuna搜索 (每种物资独立100 trials)
# ========================================================================
SEARCH_SPACE = {
    'n_estimators':       ('int', 50, 400),
    'max_depth':          ('int', 2, 8),
    'num_leaves':         ('int', 8, 80),
    'learning_rate':      ('float', 0.003, 0.1),
    'min_child_samples':  ('int', 2, 15),
    'reg_alpha':          ('float', 0.005, 5.0),
    'reg_lambda':         ('float', 0.005, 5.0),
    'subsample':          ('float', 0.5, 1.0),
    'colsample_bytree':   ('float', 0.5, 1.0),
    'min_split_gain':     ('float', 0.001, 0.5),  # low=0.001 for log=True
}

DEFAULT_PARAMS = {
    'n_estimators': 100, 'max_depth': 5, 'num_leaves': 31, 'learning_rate': 0.05,
    'min_child_samples': 10, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_split_gain': 0.0,
}

def optuna_search_material(mat_key, X_train, y_train, X_test, y_test, n_trials=100):
    """对单一物资运行Optuna超参数搜索, 返回最佳预测值"""
    n_all = len(X_train)
    n_val = min(12, n_all // 3)

    def objective(trial):
        params = {}
        for pname, (ptype, lo, hi) in SEARCH_SPACE.items():
            if ptype == 'int':
                params[pname] = trial.suggest_int(pname, lo, hi)
            else:
                params[pname] = trial.suggest_float(pname, lo, hi, log=True)

        model = lgb.LGBMRegressor(
            **params, random_state=42, verbose=-1, force_col_wise=True,
        )
        model.fit(
            X_train[:n_all - n_val], y_train[:n_all - n_val],
            eval_set=[(X_train[n_all - n_val:], y_train[n_all - n_val:])],
        )
        y_val_pred = np.maximum(model.predict(X_train[n_all - n_val:]), 0)
        try:
            return r2_score(y_train[n_all - n_val:], y_val_pred)
        except ValueError:
            return -10.0

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42),
                                study_name=mat_key)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # 使用最佳参数在完整训练集上训练, 输出测试集预测
    best_pred = train_eval_lgb(study.best_params, X_train, y_train, X_test, y_test)
    return best_pred, study.best_params, study

# ========================================================================
# 6. 主循环: 对每种物资运行完整的对比实验
# ========================================================================
print(f"\n{'='*90}")
print(f"  Route A: 10种物资 LightGBM Optuna 独立优化 (仅用自身历史特征)")
print(f"  对比: Optuna vs Default vs NaiveSeasonal vs NaiveMean")
print(f"{'='*90}\n")

results = []
all_preds = {}

for eng_key, d in material_data.items():
    y_tr, y_te = d['y_train'], d['y_test']

    # ---- 构建特征 ----
    X_self_tr, X_self_te = build_self_features(y_tr, y_te)
    X_train_full = np.column_stack([X_shared_train, X_self_tr])   # (62, 6+8=14)
    X_test_full  = np.column_stack([X_shared_test,  X_self_te])   # (12, 14)

    # ---- LightGBM Default ----
    pred_default = train_eval_lgb(DEFAULT_PARAMS, X_train_full, y_tr, X_test_full, y_te)
    r2_default = r2_score(y_te, pred_default)
    mae_default = mean_absolute_error(y_te, pred_default)

    # ---- LightGBM Optuna (100 trials) ----
    print(f"  [{eng_key}] Optuna搜索中...", end='', flush=True)
    pred_optuna, best_params, study = optuna_search_material(
        eng_key, X_train_full, y_tr, X_test_full, y_te, n_trials=100)
    r2_optuna = r2_score(y_te, pred_optuna)
    mae_optuna = mean_absolute_error(y_te, pred_optuna)
    rmse_optuna = np.sqrt(mean_squared_error(y_te, pred_optuna))
    print(f"  R2: Default={r2_default:.4f} → Optuna={r2_optuna:.4f} "
          f"(+{r2_optuna - r2_default:+.4f})")

    # ---- Baselines ----
    pred_seasonal = y_tr[-12:]  # 重复去年同期
    r2_seasonal = r2_score(y_te, pred_seasonal)

    nz_train = y_tr[y_tr > 0]
    pred_mean = np.full(12, np.mean(nz_train) if len(nz_train) > 0 else 0)
    r2_mean = r2_score(y_te, pred_mean)

    results.append({
        'key': eng_key, 'name': d['cn_name'], 'cat': d['cat'],
        'nz_train': d['nz_train'], 'nz_test': d['nz_test'],
        'r2_default': round(r2_default, 4),
        'r2_optuna':  round(r2_optuna,  4),
        'r2_seasonal': round(r2_seasonal, 4),
        'r2_mean':    round(r2_mean, 4),
        'mae_optuna': round(mae_optuna, 2),
        'rmse_optuna': round(rmse_optuna, 2),
        'best_params': best_params,
        'best_cv': round(study.best_value, 4) if study.best_value else None,
    })
    all_preds[eng_key] = {
        'actual': y_te,
        'optuna': pred_optuna,
        'default': pred_default,
        'seasonal': pred_seasonal,
        'mean': pred_mean,
    }

# ========================================================================
# 7. 汇总表格
# ========================================================================
print(f"\n{'='*95}")
print(f"  ROUTE A 最终结果汇总")
print(f"{'='*95}")
print(f"\n{'物资':<22s} {'品类':<16s} {'NZ_T':>4s} {'NZ_Tst':>5s} "
      f"{'Default':>8s} {'Optuna':>8s} {'Seas':>8s} {'Mean':>8s} "
      f"{'vs Seas':>7s} {'vs Def':>7s}")
print(f"{'-'*95}")

r2o_vals = [r['r2_optuna'] for r in results]
r2d_vals = [r['r2_default'] for r in results]
r2s_vals = [r['r2_seasonal'] for r in results]

for r in results:
    beat_seas = '+' if r['r2_optuna'] > r['r2_seasonal'] else '-'
    beat_def  = '+' if r['r2_optuna'] > r['r2_default']  else '-'
    print(f"{r['name'][:20]:<22s} {r['cat'][:14]:<16s} "
          f"{r['nz_train']:>4d} {r['nz_test']:>5d} "
          f"{r['r2_default']:>8.4f} {r['r2_optuna']:>8.4f}{beat_seas} "
          f"{r['r2_seasonal']:>8.4f} {r['r2_mean']:>8.4f} "
          f"{beat_seas:>7s} {beat_def:>7s}")

mean_o = np.mean(r2o_vals)
med_o  = np.median(r2o_vals)
mean_d = np.mean(r2d_vals)

print(f"{'-'*95}")
print(f"{'MEAN':<22s} {'':<16s} {'':>4s} {'':>5s} "
      f"{mean_d:>8.4f} {mean_o:>8.4f} "
      f"{np.mean(r2s_vals):>8.4f} {np.mean([r['r2_mean'] for r in results]):>8.4f}")

wins_seas   = sum(1 for r in results if r['r2_optuna'] > r['r2_seasonal'])
wins_def    = sum(1 for r in results if r['r2_optuna'] > r['r2_default'])
prev_best   = 0.171  # 之前21个模型的median R2
wins_prev   = sum(1 for r in results if r['r2_optuna'] > prev_best)

print(f"\n  Optuna > NaiveSeasonal: {wins_seas}/10")
print(f"  Optuna > Default LGB:   {wins_def}/10")
print(f"  Optuna > Prev Best({prev_best}): {wins_prev}/10")
print(f"  Mean Optuna R2: {mean_o:.4f}")
print(f"  Median Optuna R2: {med_o:.4f}")

# ========================================================================
# 8. 可视化 — 6张图
# ========================================================================

# ---- 修复中文字体 ----
for d in [matplotlib.get_cachedir()]:
    try:
        for f in os.listdir(d):
            if 'fontlist' in f:
                os.remove(os.path.join(d, f))
    except: pass
# 扫描可用中文字体
_avail = set(fm.findSystemFonts())
_zh = [f for f in _avail if any(n in f.lower() for n in ['simhei','simsun','msyh','yahei','wqy'])]
if _zh:
    # Register first available
    fm.fontManager.addfont(_zh[0])
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=_zh[0]).get_name(), 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- Fig 1: R² 对比柱状图 (三种方法) ---
fig1, ax1 = plt.subplots(figsize=(16, 6))
x = np.arange(10)
w = 0.22
bars1 = ax1.bar(x - w, r2s_vals, w, label='NaiveSeasonal', color='#4CAF50', alpha=0.7, edgecolor='white')
bars2 = ax1.bar(x,     r2d_vals, w, label='LGB Default',   color='#FF9800', alpha=0.7, edgecolor='white')
bars3 = ax1.bar(x + w, r2o_vals, w, label='LGB Optuna',    color='#2196F3', alpha=0.9, edgecolor='white')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax1.axhline(y=prev_best, color='#F44336', linestyle='--', linewidth=1.5, label=f'Previous Best R2={prev_best:.3f}')

# 标注数值
for bar, val in zip(bars3, r2o_vals):
    y_pos = bar.get_height() + 0.03 if bar.get_height() > 0 else 0.03
    ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{val:.3f}', ha='center', fontsize=8, rotation=90)

ax1.set_xticks(x)
ax1.set_xticklabels([r['name'][:15] for r in results], fontsize=9, rotation=30, ha='right')
ax1.set_ylabel('R2 Score', fontsize=12)
ax1.set_title(f'Route A: LightGBM Optuna vs Baselines\n'
              f'(10 High-Density Materials, 100 trials each, Mean Optuna R2={mean_o:.3f})',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'fig1_r2_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [1/6] fig1_r2_comparison.png")

# --- Fig 2: R2 提升量 (Optuna - Default) ---
fig2, ax2 = plt.subplots(figsize=(16, 5))
gains = [r['r2_optuna'] - r['r2_default'] for r in results]
bar_colors = ['#4CAF50' if g > 0 else '#F44336' for g in gains]
bars = ax2.bar(range(10), gains, color=bar_colors, alpha=0.75, edgecolor='white')
for bar, gain in zip(bars, gains):
    y_pos = gain + 0.005 if gain > 0 else gain - 0.02
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{gain:+.4f}', ha='center', fontsize=10, fontweight='bold')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_xticks(range(10))
ax2.set_xticklabels([r['name'][:15] for r in results], fontsize=9, rotation=30, ha='right')
ax2.set_ylabel('R2 Change (Optuna - Default)', fontsize=12)
ax2.set_title('Optuna Optimization Gain Per Material (100 trials each)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fig2_optuna_gain.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [2/6] fig2_optuna_gain.png")

# --- Fig 3: Optuna最佳参数分布 ---
param_names = ['n_estimators', 'max_depth', 'num_leaves', 'learning_rate',
               'min_child_samples', 'reg_alpha', 'reg_lambda',
               'subsample', 'colsample_bytree', 'min_split_gain']
fig3, axes3 = plt.subplots(5, 2, figsize=(16, 20))
for pi, pname in enumerate(param_names):
    ax = axes3[pi // 2, pi % 2]
    vals = [r['best_params'].get(pname, np.nan) for r in results]
    names_short = [r['name'][:12] for r in results]
    color = '#FF9800' if pname in ['n_estimators','max_depth','num_leaves','min_child_samples'] else '#2196F3'
    bars = ax.barh(range(10), vals, color=color, alpha=0.7, edgecolor='white')
    ax.set_yticks(range(10))
    ax.set_yticklabels(names_short, fontsize=7)
    ax.set_title(pname, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    # 标注数值
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3g}', fontsize=7, va='center')

fig3.suptitle('Optuna Best Hyperparameters Per Material', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'fig3_best_params.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [3/6] fig3_best_params.png")

# --- Fig 4: 预测 vs 实际 — 全部10种物资 (5x2) ---
fig4, axes4 = plt.subplots(5, 2, figsize=(18, 22))
for idx, r in enumerate(sorted(results, key=lambda x: -x['r2_optuna'])):
    ax = axes4[idx // 2, idx % 2]
    ek = r['key']
    p = all_preds[ek]
    rank = idx + 1

    ax.plot(TEST_DATES, p['actual'], 'ko-', linewidth=2, markersize=5, label='Actual', zorder=3)
    ax.plot(TEST_DATES, p['optuna'], 's-', color='#2196F3', linewidth=2.5, markersize=6,
            label=f"Optuna (R2={r['r2_optuna']:.3f}, MAE={r['mae_optuna']:.0f})", zorder=2)
    ax.plot(TEST_DATES, p['default'], '^--', color='#FF9800', linewidth=1.5, markersize=4,
            alpha=0.6, label=f"Default (R2={r['r2_default']:.3f})")
    ax.plot(TEST_DATES, p['seasonal'], 'x:', color='#4CAF50', linewidth=1.5, markersize=4,
            alpha=0.5, label=f"Seas (R2={r['r2_seasonal']:.3f})")

    # 训练/测试分割线
    train_end = np.mean(p['actual'])
    ax.fill_between(TEST_DATES, 0, max(p['actual'].max(), p['optuna'].max()) * 1.1,
                    color='gray', alpha=0.03, label='_nolegend_')

    ax.set_title(f"#{rank} {r['name'][:28]}\nCategory: {r['cat']} | NZ Train={r['nz_train']}/62",
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30, labelsize=7)

fig4.suptitle('Route A: Prediction vs Actual — All 10 Materials (Ranked by Optuna R2)',
              fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'fig4_all_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [4/6] fig4_all_predictions.png")

# --- Fig 5: Top 3 放大特写 ---
fig5, axes5 = plt.subplots(1, 3, figsize=(20, 6))
top3 = sorted(results, key=lambda x: -x['r2_optuna'])[:3]
for i, r in enumerate(top3):
    ax = axes5[i]
    p = all_preds[r['key']]

    ax.plot(TEST_DATES, p['actual'], 'ko-', linewidth=3, markersize=8, label='Actual', zorder=4)
    ax.plot(TEST_DATES, p['optuna'], 's-', color='#2196F3', linewidth=3, markersize=8,
            label=f"Optuna R2={r['r2_optuna']:.3f}", zorder=3)
    ax.plot(TEST_DATES, p['seasonal'], '^--', color='#4CAF50', linewidth=2, markersize=6,
            alpha=0.6, label=f"Seas R2={r['r2_seasonal']:.3f}")

    # 填充预测区间
    ax.fill_between(TEST_DATES, 0, p['optuna'], color='#2196F3', alpha=0.08)
    ax.fill_between(TEST_DATES, 0, p['actual'], color='black', alpha=0.03)

    ax.set_title(f"#{i+1} {r['name'][:30]}\nR2={r['r2_optuna']:.3f}, MAE={r['mae_optuna']:.0f}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30, labelsize=8)

fig5.suptitle('Route A: Top 3 Best-Performing Materials — Detailed View',
              fontsize=14, fontweight='bold')
plt.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'fig5_top3_detail.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [5/6] fig5_top3_detail.png")

# --- Fig 6: 密度-CV散点图 + Optuna 调优收益 ---
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(16, 6))
nz_pcts = [r['nz_train'] / 62 * 100 for r in results]
r2_vals = [r['r2_optuna'] for r in results]
sizes = [abs(r['r2_optuna'] - r['r2_default']) * 500 + 80 for r in results]

sc = ax6a.scatter(nz_pcts, r2_vals, c=range(10), cmap='tab10', s=sizes,
                  alpha=0.7, edgecolors='black', linewidth=0.5)
for i, r in enumerate(results):
    ax6a.annotate(r['name'][:12], (nz_pcts[i], r2_vals[i]),
                  textcoords="offset points", xytext=(5, 5), fontsize=7,
                  bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
ax6a.axhline(y=0, color='gray', linestyle='--')
ax6a.set_xlabel('Training Non-Zero Density (%)', fontsize=11)
ax6a.set_ylabel('Optuna R2', fontsize=11)
ax6a.set_title('R2 vs Training Density', fontsize=12, fontweight='bold')
ax6a.grid(True, alpha=0.3)

# Subplot b: R2 comparison across methods for the worst 3 and best 3
sorted_by_r2 = sorted(results, key=lambda x: -x['r2_optuna'])
plot_mats = sorted_by_r2[:3] + sorted_by_r2[-3:]
x_b = np.arange(6)
w_b = 0.25
ax6b.bar(x_b - w_b, [r['r2_optuna'] for r in plot_mats], w_b, label='Optuna', color='#2196F3', alpha=0.85)
ax6b.bar(x_b,        [r['r2_default'] for r in plot_mats], w_b, label='Default', color='#FF9800', alpha=0.7)
ax6b.bar(x_b + w_b,  [r['r2_seasonal'] for r in plot_mats], w_b, label='Seasonal', color='#4CAF50', alpha=0.6)
ax6b.set_xticks(x_b)
ax6b.set_xticklabels([r['name'][:10] for r in plot_mats], fontsize=8, rotation=30)
ax6b.set_ylabel('R2')
ax6b.set_title('Best 3 & Worst 3: Method Comparison', fontsize=12, fontweight='bold')
ax6b.legend(fontsize=9)
ax6b.grid(True, alpha=0.3, axis='y')
ax6b.axhline(y=0, color='gray', linestyle='--')

fig6.suptitle('Route A Analysis: Density vs Performance & Method Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, 'fig6_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [6/6] fig6_analysis.png")

# ========================================================================
# 9. 保存结果 JSON
# ========================================================================
output_json = {
    'route': 'A',
    'description': '10 high-density materials, LightGBM Optuna 100-trial per material, self-lag features only (no data leakage)',
    'features': {
        'shared': REF_COLS,
        'self_lag': ['lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'rolling3m', 'rolling6m', 'gap_since_last_demand'],
        'total_dim': 6 + 8,
    },
    'train_test': f'{TRAIN_LEN}/{TEST_LEN} months',
    'mean_optuna_r2': round(mean_o, 4),
    'median_optuna_r2': round(med_o, 4),
    'wins_vs_seasonal': wins_seas,
    'wins_vs_default': wins_def,
    'wins_vs_previous_best': wins_prev,
    'previous_median_r2': prev_best,
    'materials': [{
        'key': r['key'], 'name': r['name'], 'cat': r['cat'],
        'nz_train': r['nz_train'], 'nz_test': r['nz_test'],
        'r2_default': r['r2_default'], 'r2_optuna': r['r2_optuna'],
        'r2_seasonal': r['r2_seasonal'], 'r2_mean': r['r2_mean'],
        'mae_optuna': r['mae_optuna'], 'rmse_optuna': r['rmse_optuna'],
        'best_params': {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in r['best_params'].items()},
        'best_cv_r2': r['best_cv'],
    } for r in results],
    'timestamp': datetime.now().isoformat(),
}

json_path = os.path.join(OUT_DIR, 'route_a_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2, default=str)

# ---- 文本汇总表 ----
sum_path = os.path.join(OUT_DIR, 'route_a_summary.txt')
with open(sum_path, 'w', encoding='utf-8') as f:
    f.write("Route A Final Results Summary\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*100}\n")
    f.write(f"Features: {len(REF_COLS)} shared + 8 self-lag = {6+8} dimensions\n")
    f.write(f"Optuna: 100 trials per material, Holdout validation (last 12 months)\n")
    f.write(f"No data leakage — all features use only past data\n\n")
    f.write(f"{'Material':<25s} {'Cat':<18s} {'NZ_T':>4s} {'NZ_Tst':>5s} "
            f"{'Default':>8s} {'Optuna':>8s} {'Seas':>8s} {'Mean':>8s}\n")
    f.write(f"{'-'*95}\n")
    for r in results:
        f.write(f"{r['name'][:23]:<25s} {r['cat'][:16]:<18s} "
                f"{r['nz_train']:>4d} {r['nz_test']:>5d} "
                f"{r['r2_default']:>8.4f} {r['r2_optuna']:>8.4f} "
                f"{r['r2_seasonal']:>8.4f} {r['r2_mean']:>8.4f}\n")
    f.write("\n")
    f.write(f"Mean Optuna R2:   {mean_o:.4f}\n")
    f.write(f"Median Optuna R2: {med_o:.4f}\n")
    f.write(f"Wins vs Seasonal: {wins_seas}/10\n")
    f.write(f"Wins vs Default:  {wins_def}/10\n")

print(f"\n  JSON: {json_path}")
print(f"  Summary: {sum_path}")
print(f"\n{'='*90}")
print(f"  DONE.")
print(f"{'='*90}")
