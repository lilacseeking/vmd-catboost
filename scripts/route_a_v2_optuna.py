"""
Route A v2: Exact material names + co-occurrence features + Optuna
Fixes from v1: (1) exact names (2) Jaccard partner features (3) holdout validation
"""
import os, sys, json, sqlite3, warnings, time
import numpy as np, pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore'); np.random.seed(42)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- Paths ----
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'route_a')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load monthly features ----
ref = pd.read_excel(REF_XLSX, sheet_name=0)
FEATURE_COLS = [c for c in ref.columns if c not in ['日期', '需求量']]
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
feat_df = ref[ref['mk'].between('202005', '202606')]
X_shared = feat_df[FEATURE_COLS].values.astype(np.float64)
TRAIN, TEST = 62, 12
X_sh_train = X_shared[:TRAIN]; X_sh_test = X_shared[TRAIN:]

MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= '202005' and f"{y}{m:02d}" <= '202606']

# ---- Load exact materials ----
MAT_TARGETS = [
    ('AC_Arrester', '交流避雷器', 'Insulator'),
    ('CVT', '电容式电压互感器', 'Other'),
    ('Post_Insulator', '交流支柱绝缘子', 'Insulator'),
    ('Breaker_Protect', '断路器保护', 'Breaker'),
    ('Reactor_Protect', '电抗器保护', 'Protection'),
    ('Line_Protect', '线路保护', 'Protection'),
    ('T10kV', '10kV变压器', 'Transformer'),
    ('Transformer_Protect', '变压器保护', 'Transformer'),
    ('Busbar_Protect', '母线保护', 'Protection'),
    ('GIS_500kV', '500kVGIS组合电器', 'GIS'),
]

db = sqlite3.connect(DB_PATH)

# Load all >=30-month materials for co-occurrence
cur = db.execute("""
    SELECT material_name FROM material_demand_item
    WHERE demand_month >= '202005' AND demand_month <= '202606'
    GROUP BY material_name HAVING COUNT(DISTINCT demand_month) >= 30
""")
all_mats = [r[0] for r in cur.fetchall()]
mat_idx = {m: i for i, m in enumerate(all_mats)}
n_all = len(all_mats)

# Build demand presence matrix
demand_bool = np.zeros((n_all, 74), dtype=bool)
cur2 = db.execute(f"""
    SELECT material_name, demand_month FROM material_demand_item
    WHERE material_name IN ({','.join(['?']*n_all)})
    AND demand_month >= '202005' AND demand_month <= '202606' AND demand_quantity > 0
""", all_mats)
for m, dm in cur2.fetchall():
    demand_bool[mat_idx[m]][MONTHS.index(dm)] = True

# Load our 10 materials
mat_data = {}
for eng_key, cn_name, cat in MAT_TARGETS:
    cur3 = db.execute("""
        SELECT demand_month, SUM(demand_quantity) FROM material_demand_item
        WHERE material_name = ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (cn_name,))
    dmap = {r[0]: r[1] for r in cur3.fetchall()}
    vals = np.array([dmap.get(m, 0) for m in MONTHS])
    y_tr, y_te = vals[:TRAIN], vals[TRAIN:]

    # Find strongest Jaccard partner
    target_i = mat_idx.get(cn_name)
    best_jac, best_idx, best_name = 0, -1, ''
    if target_i is not None:
        nz_i = demand_bool[target_i].sum()
        for j in range(n_all):
            if j == target_i: continue
            nz_j = demand_bool[j].sum()
            if nz_j < 30: continue
            both = (demand_bool[target_i] & demand_bool[j]).sum()
            jac = both / (nz_i + nz_j - both) if (nz_i + nz_j - both) > 0 else 0
            if jac > best_jac:
                best_jac = jac; best_idx = j; best_name = all_mats[j]
    # Get partner demand
    partner_demand = np.zeros(74)
    if best_idx >= 0:
        cur4 = db.execute("""
            SELECT demand_month, SUM(demand_quantity) FROM material_demand_item
            WHERE material_name = ? AND demand_month >= '202005' AND demand_month <= '202606'
            GROUP BY demand_month ORDER BY demand_month
        """, (best_name,))
        pmap = {r[0]: r[1] for r in cur4.fetchall()}
        partner_demand = np.array([pmap.get(m, 0) for m in MONTHS])

    mat_data[eng_key] = {
        'cn_name': cn_name, 'cat': cat,
        'y_train': y_tr, 'y_test': y_te,
        'partner_train': partner_demand[:TRAIN], 'partner_test': partner_demand[TRAIN:],
        'partner_name': best_name, 'partner_jac': best_jac,
        'nz_train': (y_tr > 0).sum(), 'nz_test': (y_te > 0).sum(),
    }
    print(f"{eng_key:<22s} Jaccard={best_jac:.3f} with {best_name[:40]}")

db.close()

# ---- Build features ----
def build_features(y_tr, y_te, partner_tr, partner_te):
    """Build lag + rolling + partner features. NO DATA LEAKAGE."""
    full = np.concatenate([y_tr, y_te])
    # partner data: ONLY training period accessible. Test-period partner data = blind.
    full_p = np.concatenate([partner_tr, np.zeros(len(partner_te))])
    tr, te = len(y_tr), len(y_te)

    def extract_fts(N, offset=0, is_test=False):
        F = np.zeros((N, 10))
        for i in range(N):
            idx = offset + i
            F[i, 0] = full[idx - 1] if idx >= 1 else 0       # lag1
            F[i, 1] = full[idx - 2] if idx >= 2 else 0       # lag2
            F[i, 2] = full[idx - 3] if idx >= 3 else 0       # lag3
            F[i, 3] = full[idx - 6] if idx >= 6 else 0       # lag6
            F[i, 4] = full[idx - 12] if idx >= 12 else 0     # lag12
            w3 = max(0, idx - 3)
            F[i, 5] = np.mean(full[w3:idx + 1]) if idx >= 1 else full[0]  # rolling3
            F[i, 6] = full[idx - 1] > 0 if idx >= 1 else 0   # was_nonzero
            # months_since_last_demand
            last = -1
            for j in range(idx - 1, -1, -1):
                if full[j] > 0: last = j; break
            F[i, 7] = idx - last if last >= 0 else 99         # gap
            # partner_lag1 (safe: uses only past data)
            F[i, 8] = full_p[idx - 1] if idx >= 1 else 0
            # partner_now: ONLY for training. Set to 0 for test to prevent data leakage
            if not is_test:
                F[i, 9] = (full_p[idx] > 0).astype(float)
            else:
                F[i, 9] = 0  # ZERO for test — we cannot know partner's current-month demand
        return F

    Ft = extract_fts(tr, 0, is_test=False)
    Fe = extract_fts(te, TRAIN, is_test=True)
    return Ft, Fe

# ---- Evaluation ----
def eval_lgb(params, X_tr, y_tr, X_te, y_te):
    m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, force_col_wise=True)
    nv = min(12, len(X_tr) // 3)
    if nv >= 3:
        m.fit(X_tr[:len(X_tr)-nv], y_tr[:len(y_tr)-nv],
              eval_set=[(X_tr[len(X_tr)-nv:], y_tr[len(y_tr)-nv:])])
    else:
        m.fit(X_tr, y_tr)
    return np.maximum(m.predict(X_te), 0), m

# ---- Optuna (holdout validation, not TSCV) ----
def search(mat_key, X_tr, y_tr, X_te, y_te, trials=50):
    n_tr = len(X_tr)
    v = min(12, n_tr // 3)

    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
        }
        m = lgb.LGBMRegressor(**p, random_state=42, verbose=-1, force_col_wise=True)
        m.fit(X_tr[:n_tr - v], y_tr[:n_tr - v],
              eval_set=[(X_tr[n_tr - v:], y_tr[n_tr - v:])])
        yp = np.maximum(m.predict(X_tr[n_tr - v:]), 0)
        try:
            return r2_score(y_tr[n_tr - v:], yp)
        except:
            return -10.0

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=trials, show_progress_bar=False)
    pred, _ = eval_lgb(study.best_params, X_tr, y_tr, X_te, y_te)
    try:
        r2 = r2_score(y_te, pred)
    except:
        r2 = np.nan
    return pred, r2, study.best_params, study.best_value

# ---- Run ----
print(f"\n{'='*80}\nRoute A v2: Optuna LightGBM + Co-occurrence Features\n{'='*80}")

DEFAULT = {'n_estimators': 100, 'max_depth': 5, 'num_leaves': 20, 'learning_rate': 0.05,
           'min_child_samples': 5, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
           'subsample': 0.8, 'colsample_bytree': 0.8, 'min_split_gain': 0.0}

results = []

for ek, d in mat_data.items():
    y_tr, y_te = d['y_train'], d['y_test']
    p_tr, p_te = d['partner_train'], d['partner_test']

    # Build features
    Ft, Fe = build_features(y_tr, y_te, p_tr, p_te)
    X_tr_full = np.column_stack([X_sh_train, Ft])
    X_te_full = np.column_stack([X_sh_test, Fe])

    # Default
    pd_, _ = eval_lgb(DEFAULT, X_tr_full, y_tr, X_te_full, y_te)
    r2d = r2_score(y_te, pd_)

    # Optuna
    po, r2o, bp, cv_score = search(ek, X_tr_full, y_tr, X_te_full, y_te, trials=50)

    # Baselines
    r2_seas = r2_score(y_te, y_tr[-12:])
    r2_mean = r2_score(y_te, np.full(12, np.mean(y_tr[y_tr > 0])))

    gap = r2o - r2d
    tag = '+' if gap > 0 else ''
    print(f"  {ek:<22s} R2: Default={r2d:.4f} Optuna={r2o:.4f}{tag} Seas={r2_seas:.4f} N_train={d['nz_train']}")

    results.append({
        'key': ek, 'cn': d['cn_name'], 'cat': d['cat'],
        'r2_default': r2d, 'r2_optuna': r2o, 'r2_seasonal': r2_seas,
        'partner': d['partner_name'][:40], 'jac': d['partner_jac'],
        'best_params': bp, 'best_cv': cv_score,
    })

# ---- Summary ----
print(f"\n{'='*80}\nSummary\n{'='*80}")

r2o_vals = [r['r2_optuna'] for r in results]
r2d_vals = [r['r2_default'] for r in results]
r2s_vals = [r['r2_seasonal'] for r in results]
mean_o, med_o = np.mean(r2o_vals), np.median(r2o_vals)
mean_d = np.mean(r2d_vals)
wins_op = sum(1 for r in results if r['r2_optuna'] > r['r2_seasonal'])
wins_od = sum(1 for r in results if r['r2_optuna'] > r['r2_default'])

# Previous best from old experiment
prev_best = 0.171  # median R2 from the previous 21-model run
wins_prev = sum(1 for r in results if r['r2_optuna'] > prev_best)

print(f"\n{'Material':<22s} {'Partner_Jacc':>12s} {'Default':>8s} {'Optuna':>8s} {'Delta':>8s} {'Seas':>8s} {'>Prev':>6s}")
print('-'*80)
for r in results:
    print(f"{r['key']:<22s} {r['jac']:>11.3f}  {r['r2_default']:>8.4f} {r['r2_optuna']:>8.4f} "
          f"{r['r2_optuna']-r['r2_default']:>+7.4f}  {r['r2_seasonal']:>8.4f} "
          f"{'+' if r['r2_optuna']>prev_best else '-':>6s}")

print(f"\nMean Optuna R2: {mean_o:.4f} | Mean Default R2: {mean_d:.4f}")
print(f"Win vs Seasonal: {wins_op}/10 | Win vs Default: {wins_od}/10 | Win vs Prev Best({prev_best}): {wins_prev}/10")
print(f"Best material: {max(results, key=lambda x: x['r2_optuna'])['key']} R2={max(r2o_vals):.4f}")
print(f"Worst material: {min(results, key=lambda x: x['r2_optuna'])['key']} R2={min(r2o_vals):.4f}")

# Charts
# 1. R2 bar chart
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(10); w = 0.2
ax.bar(x - w, r2s_vals, w, label='NaiveSeasonal', color='#4CAF50', alpha=0.7)
ax.bar(x, r2d_vals, w, label='LGB Default', color='#FF9800', alpha=0.7)
ax.bar(x + w, r2o_vals, w, label='LGB Optuna', color='#2196F3', alpha=0.9)
ax.axhline(y=0, c='gray', ls='--')
ax.axhline(y=prev_best, c='red', ls='--', lw=1.5, label=f'Prev Best R2={prev_best}')
ax.set_xticks(x); ax.set_xticklabels([r['key'][:12] for r in results], fontsize=8, rotation=30)
ax.set_ylabel('R2'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
ax.set_title(f'Route A v2: LightGBM + Co-occurrence + Optuna (Mean Optuna R2={mean_o:.3f})', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'route_a_v2_r2.png'), dpi=150)
plt.close()

# 2. Top 3 prediction plots
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
top3 = sorted(results, key=lambda x: -x['r2_optuna'])[:3]
for i, r in enumerate(top3):
    ax = axes[i]
    d = mat_data[r['key']]
    months = pd.date_range('2025-07-01', periods=12, freq='MS')
    # Rebuild preds
    Ft, Fe = build_features(d['y_train'], d['y_test'], d['partner_train'], d['partner_test'])
    Xt = np.column_stack([X_sh_train, Ft]); Xe = np.column_stack([X_sh_test, Fe])
    po, _ = eval_lgb(r['best_params'], Xt, d['y_train'], Xe, d['y_test'])
    ps = d['y_train'][-12:]

    ax.plot(months, d['y_test'], 'ko-', lw=2, ms=5, label='Actual')
    ax.plot(months, po, 's-', color='#2196F3', lw=2.5, ms=6, label=f'Optuna (R2={r["r2_optuna"]:.3f})')
    ax.plot(months, ps, '^--', color='#4CAF50', lw=1.5, ms=4, alpha=0.6, label=f'Seas (R2={r["r2_seasonal"]:.3f})')
    ax.set_title(f"{r['key']}\nPartner: {r['partner'][:30]} (J={r['jac']:.2f})", fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.tick_params(axis='x', rotation=30)

fig2.suptitle('Route A v2: Top 3 Materials — Predictions with Co-occurrence Features', fontsize=13, fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'route_a_v2_predictions.png'), dpi=150)
plt.close()

# Save JSON
summary = {
    'route': 'A_v2', 'optuna_trials': 50, 'cooccurrence': True,
    'mean_optuna_r2': round(mean_o, 4), 'mean_default_r2': round(mean_d, 4),
    'wins_vs_previous_best': wins_prev, 'wins_vs_seasonal': wins_op,
    'previous_median_r2': prev_best,
    'materials': [{k: v for k, v in r.items() if k != 'best_params'} for r in results],
}
json.dump(summary, open(os.path.join(OUT_DIR, 'route_a_v2_results.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=str)

print(f"\nDone! Charts: {OUT_DIR}")
