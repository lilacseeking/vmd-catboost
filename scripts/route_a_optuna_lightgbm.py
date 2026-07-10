"""
Route A: High-density materials + LightGBM Optuna hyperparameter optimization
Builds 10 high-freq data sheets, runs Optuna per material, compares with previous R2

Features: monthly-level shared features from data.xlsx + demand from ECP database
Train/Test: 69 months / 12 months (matching the existing experiment split)
Materials: Top 10 by non-zero months, 7 categories
"""
import os, sys, json, sqlite3, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# ---- Paths ----
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'bidding-ecp-data', 'data', 'ecp_data.db')
REFERENCE_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'route_a')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load shared monthly features from reference data.xlsx ----
ref = pd.read_excel(REFERENCE_XLSX, sheet_name=0)
FEATURE_COLS = [c for c in ref.columns if c not in ['日期', '需求量']]
print(f"Shared monthly features: {FEATURE_COLS}")
print(f"Reference data: {ref.shape[0]} rows ({ref['日期'].iloc[0]} to {ref['日期'].iloc[-1]})")

# Date range: trim to match our materials (2020-05 to 2026-06 = 74 months)
# But reference starts from 2019-11. Start features from 2020-05.
ref['month_key'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
START_MONTH, END_MONTH = '202005', '202606'

# Build feature matrix for 74 months
feat_df = ref[ref['month_key'].between(START_MONTH, END_MONTH)].copy()
feat_X = feat_df[FEATURE_COLS].values.astype(np.float64)
TRAIN_LEN, TEST_LEN = 62, 12
X_train_shared = feat_X[:TRAIN_LEN]
X_test_shared = feat_X[TRAIN_LEN:]

print(f"Feature matrix: train={X_train_shared.shape}, test={X_test_shared.shape}")

# ---- Load 10 high-freq materials ----
db = sqlite3.connect(DB_PATH)
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= START_MONTH and f"{y}{m:02d}" <= END_MONTH]

# Top 10 materials for the experiment (already verified >=47 non-zero months)
MATERIALS_10 = [
    ('AC_Arrester', '%交流避雷器%', 'Insulator/Arrester'),
    ('CVT', '%电容式电压互感器%', 'Other'),
    ('Post_Insulator', '%交流支柱绝缘子%', 'Insulator/Arrester'),
    ('Breaker_Protect', '%断路器保护%', 'Breaker'),
    ('Reactor_Protect', '%电抗器保护%', 'Protection'),
    ('Line_Protect', '%线路保护%', 'Cable/Wire'),
    ('T10kV', '%10kV变压器%', 'Transformer'),
    ('Transformer_Protect', '%变压器保护%', 'Transformer'),
    ('Busbar_Protect', '%母线保护%', 'Cable/Wire'),
    ('GIS_500kV', '%500kV%GIS%', 'Switchgear'),
]

mat_data = {}
for eng_key, pattern, cat in MATERIALS_10:
    cur = db.execute("""
        SELECT material_name, demand_month, SUM(demand_quantity)
        FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month >= ? AND demand_month <= ?
        GROUP BY material_name, demand_month ORDER BY demand_month
    """, (pattern, START_MONTH, END_MONTH))

    # Group by material_name and pick the one with most non-zero months
    from collections import defaultdict
    mat_demands = defaultdict(dict)
    for name, dm, qty in cur.fetchall():
        mat_demands[name][dm] = qty

    # Pick best match (highest nz count)
    best_name, best_data = max(mat_demands.items(), key=lambda x: len(x[1]))
    vals = np.array([best_data.get(m, 0) for m in MONTHS])
    nz = (vals > 0).sum()
    nz_train = (vals[:TRAIN_LEN] > 0).sum()
    nz_test = (vals[TRAIN_LEN:] > 0).sum()

    mat_data[eng_key] = {
        'cn_name': best_name, 'cat': cat,
        'y_train': vals[:TRAIN_LEN], 'y_test': vals[TRAIN_LEN:],
        'nz': nz, 'nz_train': nz_train, 'nz_test': nz_test,
    }
    print(f"  {eng_key:<22s} [{cat:<20s}] {best_name[:40]:<40s} "
          f"nz={nz_train}T+{nz_test}T")

db.close()

# ==============================================================================
# Feature Engineering
# ==============================================================================
def build_lag_features(y_train, y_test, seq_len=12):
    """Build lag features: lag1, lag2, lag3, lag6, lag12 + rolling stats"""
    full = np.concatenate([y_train, y_test])
    n_feat = len(full)
    feats_train = np.zeros((len(y_train), 8))
    feats_test = np.zeros((len(y_test), 8))

    for i in range(len(y_train)):
        base = full[:TRAIN_LEN]
        idx = i
        feats_train[i, 0] = base[idx - 1] if idx >= 1 else 0
        feats_train[i, 1] = base[idx - 2] if idx >= 2 else 0
        feats_train[i, 2] = base[idx - 3] if idx >= 3 else 0
        feats_train[i, 3] = base[idx - 6] if idx >= 6 else 0
        feats_train[i, 4] = base[idx - 12] if idx >= 12 else 0
        start_3 = max(0, idx - 3)
        feats_train[i, 5] = np.mean(base[start_3:idx+1]) if idx >= 1 else base[0]
        start_6 = max(0, idx - 6)
        feats_train[i, 6] = np.mean(base[start_6:idx+1]) if idx >= 1 else base[0]
        # months since last non-zero
        last_nz = -1
        for j in range(idx - 1, -1, -1):
            if base[j] > 0:
                last_nz = j
                break
        feats_train[i, 7] = idx - last_nz if last_nz >= 0 else 99

    for i in range(len(y_test)):
        idx = TRAIN_LEN + i
        feats_test[i, 0] = full[idx - 1] if idx >= 1 else 0
        feats_test[i, 1] = full[idx - 2] if idx >= 2 else 0
        feats_test[i, 2] = full[idx - 3] if idx >= 3 else 0
        feats_test[i, 3] = full[idx - 6] if idx >= 6 else 0
        feats_test[i, 4] = full[idx - 12] if idx >= 12 else 0
        start_3 = max(0, idx - 3)
        feats_test[i, 5] = np.mean(full[start_3:idx+1]) if idx >= 1 else np.mean(y_train)
        start_6 = max(0, idx - 6)
        feats_test[i, 6] = np.mean(full[start_6:idx+1]) if idx >= 1 else np.mean(y_train)
        last_nz = -1
        for j in range(idx - 1, -1, -1):
            if full[j] > 0:
                last_nz = j
                break
        feats_test[i, 7] = idx - last_nz if last_nz >= 0 else 99

    return feats_train, feats_test

# ==============================================================================
# LightGBM Evaluation (single config)
# ==============================================================================
def eval_lgb(params, X_train, y_train, X_test, y_test):
    """Train and evaluate LightGBM with given params."""
    model = lgb.LGBMRegressor(
        **params,
        random_state=42, n_jobs=1, verbose=-1,
        force_col_wise=True,
    )
    # Use walk-forward validation (last 12 of training as validation)
    n_train = len(X_train)
    val_size = min(12, n_train // 4)
    if val_size > 0:
        model.fit(
            X_train[:n_train - val_size], y_train[:n_train - val_size],
            eval_set=[(X_train[n_train - val_size:], y_train[n_train - val_size:])],
        )
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # non-negative
    return y_pred, model

# ==============================================================================
# Optuna Search
# ==============================================================================
def optuna_search(mat_key, X_train, y_train, X_test, y_test, n_trials=100):
    """Run Optuna hyperparameter search for a material."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # TSS for validation
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'num_leaves': trial.suggest_int('num_leaves', 8, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
        }

        scores = []
        n_all = len(X_train)
        for train_idx, val_idx in tscv.split(X_train):
            if len(train_idx) < 5 or len(val_idx) < 2:
                continue
            model = lgb.LGBMRegressor(
                **params,
                random_state=42, verbose=-1,
                force_col_wise=True,
            )
            model.fit(X_train[train_idx], y_train[train_idx],
                      eval_set=[(X_train[val_idx], y_train[val_idx])])
            y_val_pred = np.maximum(model.predict(X_train[val_idx]), 0)
            try:
                r2 = r2_score(y_train[val_idx], y_val_pred)
            except:
                r2 = -10
            scores.append(r2)

        return np.mean(scores) if scores else -10

    study = optuna.create_study(direction='maximize', study_name=mat_key)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Final evaluation on test set
    best_params = study.best_params
    best_pred, best_model = eval_lgb(best_params, X_train, y_train, X_test, y_test)

    return best_pred, best_model, best_params, study.best_value

# ==============================================================================
# Main Experiment
# ==============================================================================
print(f"\n{'='*80}")
print(f"Route A: LightGBM Optuna — 10 High-Frequency Materials")
print(f"{'='*80}")

# Baselines to compare against (from previous experiment)
previous_best_model = {
    '火灾报警系统': ('SARIMA', 0.171), '线路在线监测装置': ('LightGBM-pure', -0.024),
    '一体化电源系统': ('LightGBM', 0.356), '变电在线监测装置': ('TwoStage', 0.083),
    '端子箱': ('LightGBM', 0.292),
}
prev_r2_values = [0.171, -0.024, 0.356, 0.083, 0.292]
prev_mean_r2 = np.mean(prev_r2_values)
prev_median_r2 = np.median(prev_r2_values)

results = []
all_predictions = {}

print("\n--- Running Optuna search per material (100 trials each) ---")
for eng_key, d in mat_data.items():
    cn_name = d['cn_name']
    y_train, y_test = d['y_train'], d['y_test']

    # Build lag features + shared monthly features
    lag_tr, lag_te = build_lag_features(y_train, y_test)
    X_train_full = np.column_stack([X_train_shared, lag_tr])
    X_test_full = np.column_stack([X_test_shared, lag_te])

    print(f"\n  [{eng_key}] {cn_name[:30]} ... ", end='', flush=True)

    # ---- LGB Default ----
    default_params = {'n_estimators': 100, 'max_depth': 6, 'num_leaves': 31,
                      'learning_rate': 0.05, 'min_child_samples': 10,
                      'reg_alpha': 0.1, 'reg_lambda': 0.1,
                      'subsample': 0.8, 'colsample_bytree': 0.8, 'min_split_gain': 0.0}
    pred_default, model_default = eval_lgb(default_params, X_train_full, y_train, X_test_full, y_test)
    r2_default = r2_score(y_test, pred_default)

    # ---- LGB Optuna ----
    try:
        pred_opt, model_opt, best_params, best_cv = optuna_search(
            eng_key, X_train_full, y_train, X_test_full, y_test, n_trials=100)
        r2_opt = r2_score(y_test, pred_opt)
    except Exception as e:
        print(f"Optuna failed ({e}), using default")
        pred_opt, best_params, best_cv, r2_opt = pred_default, default_params, np.nan, r2_default

    # ---- NaiveSeasonal ----
    naive_seas = y_train[-12:]
    r2_seas = r2_score(y_test, naive_seas)

    mae_opt = mean_absolute_error(y_test, pred_opt)
    rmse_opt = np.sqrt(mean_squared_error(y_test, pred_opt))

    print(f"R2: Default={r2_default:.4f}, Optuna={r2_opt:.4f}, NaiveSeas={r2_seas:.4f}")

    results.append({
        'key': eng_key, 'cn_name': cn_name, 'cat': d['cat'],
        'nz_train': d['nz_train'], 'nz_test': d['nz_test'],
        'r2_default': round(r2_default, 4), 'r2_optuna': round(r2_opt, 4),
        'r2_seasonal': round(r2_seas, 4),
        'mae_opt': round(mae_opt, 2), 'rmse_opt': round(rmse_opt, 2),
        'best_params': best_params, 'best_cv_r2': round(best_cv, 4) if not np.isnan(best_cv) else None,
    })
    all_predictions[eng_key] = {
        'y_test': y_test, 'pred_opt': pred_opt, 'pred_default': pred_default, 'pred_seas': naive_seas,
    }

# ==============================================================================
# Results Summary
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"RESULTS: LightGBM Optuna vs Default vs NaiveSeasonal vs Previous Best")
print(f"{'='*100}")
print(f"\n{'Material':<25s} {'Cat':<18s} {'NZ_T':>4s} {'Default':>8s} {'Optuna':>8s} {'Seas':>8s} {'PrevBest':>8s} {'Win':>6s}")
print(f"{'-'*100}")

wins_lgb = 0
wins_optuna = 0
optuna_r2_list = []
prev_r2_list = []

for r in results:
    opt_r2 = r['r2_optuna']
    optuna_r2_list.append(opt_r2)
    prev_r2_list.append(prev_mean_r2)  # for comparison

    # Find previous best from the old experiment for similar category
    # We use the overall previous best as reference
    better_than_prev = opt_r2 > prev_median_r2
    better_than_seas = opt_r2 > r['r2_seasonal']

    if better_than_prev: wins_lgb += 1
    if better_than_seas: wins_optuna += 1

    marker_prev = '+' if better_than_prev else '-'
    marker_seas = '+' if better_than_seas else '-'

    print(f"{r['key'][:23]:<25s} {r['cat'][:16]:<18s} {r['nz_train']:>3d} "
          f"{r['r2_default']:>8.4f} {opt_r2:>8.4f}{marker_seas} {r['r2_seasonal']:>8.4f} "
          f"{prev_median_r2:>8.4f}{marker_prev} {opt_r2 - r['r2_default']:>+5.3f}")

print(f"{'-'*100}")
mean_opt = np.mean(optuna_r2_list)
print(f"{'MEAN':<25s} {'':<18s} {'':>4s} {'':>8s} {mean_opt:>8.4f} "
      f"{np.mean([r['r2_seasonal'] for r in results]):>8.4f} "
      f"{prev_median_r2:>8.4f} ")

print(f"\nLightGBM Optuna > previous best: {wins_lgb}/10")
print(f"LightGBM Optuna > NaiveSeasonal: {wins_optuna}/10")

# ==============================================================================
# Visualizations
# ==============================================================================
print(f"\nGenerating charts...")

# Fig 1: R2 comparison bar chart
fig1, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(results))
w = 0.2
ax.bar(x - w, [r['r2_seasonal'] for r in results], w, label='NaiveSeasonal', color='#4CAF50', alpha=0.7)
ax.bar(x, [r['r2_default'] for r in results], w, label='LGB Default', color='#FF9800', alpha=0.7)
ax.bar(x + w, [r['r2_optuna'] for r in results], w, label='LGB Optuna', color='#2196F3', alpha=0.9)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=prev_median_r2, color='red', linestyle='--', linewidth=1.5, label=f'Previous Best Median={prev_median_r2:.3f}')
ax.set_xticks(x)
ax.set_xticklabels([r['key'][:12] for r in results], fontsize=8, rotation=30)
ax.set_ylabel('R2', fontsize=12)
ax.set_title(f'Route A: LightGBM Optuna vs Baselines\nMean Optuna R2={mean_opt:.4f}', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'route_a_r2_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Prediction vs Actual for top 4 materials
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
top4 = sorted(results, key=lambda x: -(x['r2_optuna']))[:4]
for i, r in enumerate(top4):
    ax = axes[i // 2, i % 2]
    eng_key = r['key']
    preds = all_predictions[eng_key]
    y_t = preds['y_test']
    months = pd.date_range('2025-07-01', periods=12, freq='MS')

    ax.plot(months, y_t, 'ko-', linewidth=2, markersize=6, label='Actual')
    ax.plot(months, preds['pred_opt'], 's-', color='#2196F3', linewidth=2.5, markersize=6, label=f'LGB Optuna (R2={r["r2_optuna"]:.3f})')
    ax.plot(months, preds['pred_default'], '^--', color='#FF9800', linewidth=1.5, markersize=5, alpha=0.6, label=f'LGB Default (R2={r["r2_default"]:.3f})')
    ax.plot(months, preds['pred_seas'], 'x:', color='#4CAF50', linewidth=1.5, markersize=5, alpha=0.6, label=f'NaiveSeas (R2={r["r2_seasonal"]:.3f})')

    ax.set_title(f"{r['key']} [{r['cat']}]", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

fig2.suptitle('Route A: Predictions vs Actual — Top 4 Materials (Optuna-tuned LightGBM)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'route_a_predictions_top4.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Optuna parameter importance per material
# Collect best params
param_names = ['n_estimators', 'max_depth', 'num_leaves', 'learning_rate', 'min_child_samples',
               'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree']
fig3, axes3 = plt.subplots(3, 3, figsize=(14, 10))
for pi, pname in enumerate(param_names):
    ax = axes3[pi // 3, pi % 3]
    vals = [r['best_params'].get(pname, np.nan) for r in results]
    if 'rate' in pname or 'reg_' in pname or 'col' in pname or 'sub' in pname:
        # Log-scale params
        ax.barh(range(10), vals, color='#2196F3', alpha=0.7)
    else:
        # Int params
        ax.barh(range(10), [int(v) for v in vals], color='#FF9800', alpha=0.7)
    ax.set_yticks(range(10))
    ax.set_yticklabels([r['key'][:10] for r in results], fontsize=6)
    ax.set_title(pname, fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

fig3.suptitle('Optuna Best Parameters per Material', fontsize=13, fontweight='bold')
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'route_a_optuna_params.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: R2 gain (Optuna - Default) for each material
fig4, ax4 = plt.subplots(figsize=(12, 5))
gains = [r['r2_optuna'] - r['r2_default'] for r in results]
colors = ['#4CAF50' if g > 0 else '#F44336' for g in gains]
bars = ax4.bar(range(10), gains, color=colors, alpha=0.7)
for bar, gain in zip(bars, gains):
    ax4.text(bar.get_x() + bar.get_width()/2, gain + 0.003 if gain > 0 else gain - 0.008,
             f'{gain:+.4f}', ha='center', fontsize=9)
ax4.axhline(y=0, color='gray', linewidth=1)
ax4.set_xticks(range(10))
ax4.set_xticklabels([r['key'][:12] for r in results], fontsize=8, rotation=30)
ax4.set_ylabel('R2 Improvement (Optuna - Default)', fontsize=12)
ax4.set_title('Optuna Optimization Gain per Material', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'route_a_gain_per_material.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"Charts saved to: {OUT_DIR}")

# ==============================================================================
# Save results JSON
# ==============================================================================
output_json = {
    'route': 'A',
    'description': 'High-density materials + LightGBM Optuna optimization',
    'method': '100-trial Optuna Bayesian search per material, TimeSeriesSplit CV',
    'materials': len(results),
    'mean_optuna_r2': round(mean_opt, 4),
    'wins_vs_seasonal': wins_optuna,
    'wins_vs_previous_best': wins_lgb,
    'previous_median_r2': round(prev_median_r2, 4),
    'per_material': results,
    'timestamp': datetime.now().isoformat(),
}
json_path = os.path.join(OUT_DIR, 'route_a_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2, default=str)

# ==============================================================================
# Final Summary Table (for report)
# ==============================================================================
print(f"\n\n{'='*120}")
print(f"FINAL COMPARISON TABLE — Route A Optuna LGB vs Previous Best")
print(f"{'='*120}")
print(f"\n{'Material':<25s} {'Cat':<18s} {'NZ_T':>4s} {'NZ_Tst':>5s} "
      f"{'LGB_Default':>10s} {'LGB_Optuna':>10s} {'NaiveSeas':>10s} "
      f"{'Prev_Best':>10s} {'Opt vs Prev':>10s}")

for r in results:
    opt_vs_prev = r['r2_optuna'] - prev_median_r2
    print(f"{r['cn_name'][:23]:<25s} {r['cat'][:16]:<18s} {r['nz_train']:>4d} {r['nz_test']:>5d} "
          f"{r['r2_default']:>10.4f} {r['r2_optuna']:>10.4f} {r['r2_seasonal']:>10.4f} "
          f"{prev_median_r2:>10.4f} {opt_vs_prev:>+9.4f}")

print(f"\n{'MEAN':<25s} {'':<18s} {'':>4s} {'':>5s} "
      f"{np.mean([r['r2_default'] for r in results]):>10.4f} "
      f"{mean_opt:>10.4f} "
      f"{np.mean([r['r2_seasonal'] for r in results]):>10.4f} "
      f"{prev_median_r2:>10.4f} "
      f"{mean_opt - prev_median_r2:>+9.4f}")

print(f"\n  JSON: {json_path}")
print(f"  Done!")
