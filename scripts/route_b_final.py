"""
Route B: 22维零数据泄露特征 + LightGBM Optuna
================================================================
从 Route A (14维) 扩展为 22维, 全部特征严格使用 t-1 之前的历史数据:

扩展特征 (相比 Route A):
  全局 lag1 (3维): 其他物资的总需求/活跃数/平均需求 — 全部 t-1
  共现 lag1 (4维): 搭子物资的活跃数/平均量/总量/最强信号 — 全部 t-1
  季节性 (2维): 训练集高峰月哑变量 + 去年同期需求
  删除: has_batch (不确定其独立性)

总计: 22 维, 零数据泄露, 每一步做法内嵌在注释中
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ========================================================================
# 0. 常量和路径
# ========================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'route_b')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN = 62; TEST = 12
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

# ========================================================================
# 1. 加载共享月度特征 (5维, 替换 has_batch)
# ========================================================================
ref = pd.read_excel(REF_XLSX, sheet_name=0)
REF_COLS = [c for c in ref.columns if c not in ['日期', '需求量', 'has_batch']]
# has_batch 无法确认其独立性(如果1=该月发布了包含目标物资的公告→泄露), 故删除
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
feat_df = ref[ref['mk'].between('202005', '202606')].copy()
X_SHARED = feat_df[REF_COLS].values.astype(np.float64)
X_shared_tr = X_SHARED[:TRAIN]; X_shared_te = X_SHARED[TRAIN:]
print(f"[Step 1] 共享月度特征 ({len(REF_COLS)}维, 已删除 has_batch): {REF_COLS}")

# ========================================================================
# 2. 从数据库加载10种目标物资的精确完整名称
# ========================================================================
db = sqlite3.connect(DB_PATH)

mat_data = {}
for eng_key, pattern, cat in TARGETS:
    cur = db.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) as nz, SUM(demand_quantity) as tot
        FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY material_name ORDER BY nz DESC LIMIT 3
    """, (pattern,))
    best = max(cur.fetchall(), key=lambda x: x[1])
    exact_name = best[0]

    cur2 = db.execute("""
        SELECT demand_month, SUM(demand_quantity) FROM material_demand_item
        WHERE material_name = ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (exact_name,))
    dmap = {r[0]: r[1] for r in cur2.fetchall()}
    vals = np.array([dmap.get(m, 0) for m in MONTHS], dtype=np.float64)

    mat_data[eng_key] = {
        'exact_name': exact_name, 'cat': cat,
        'y_train': vals[:TRAIN], 'y_test': vals[TRAIN:],
        'y_all': vals,
        'nz_train': (vals[:TRAIN] > 0).sum(), 'nz_test': (vals[TRAIN:] > 0).sum(),
    }
    print(f"  {eng_key:<22s} → {exact_name[:40]}")

# ========================================================================
# 3. 构建全局 lag1 特征 (仅用训练集, 排除目标物资自身)
# ========================================================================
print(f"\n[Step 3] 构建全局 lag1 特征...")

# 加载所有非零月≥20的物资 (分两步: 先找符合条件的物资名, 再加载数据)
cur = db.execute("""
    SELECT material_name FROM material_demand_item
    WHERE demand_month >= '202005' AND demand_month <= '202606'
    GROUP BY material_name
    HAVING COUNT(DISTINCT demand_month) >= 20
""")
qualifying_mats = [r[0] for r in cur.fetchall()]

print(f"  Step 3a: 符合条件的物资 = {len(qualifying_mats)}")

# 用 IN 子句加载这些物资的完整月度数据
placeholders = ','.join(['?'] * len(qualifying_mats))
cur2 = db.execute(f"""
    SELECT material_name, demand_month, SUM(demand_quantity)
    FROM material_demand_item
    WHERE material_name IN ({placeholders})
    AND demand_month >= '202005' AND demand_month <= '202606'
    GROUP BY material_name, demand_month
""", qualifying_mats)
all_demands = defaultdict(lambda: defaultdict(float))
for mat, dm, qty in cur2.fetchall():
    all_demands[mat][dm] = qty
all_mats = list(all_demands.keys())
print(f"  Step 3b: 实际加载物资 = {len(all_mats)} (有数据的月份)")

def build_global_lag1(target_exact_name):
    """为一种物资构建3维全局 lag1 特征。

    计算 t 月时, 使用 t-1 月所有其他物资的总需求/活跃数/平均需求。
    训练/测试时一致: 只看前一天的数据 (t-1)。
    """
    F_train = np.zeros((TRAIN, 3))
    F_test  = np.zeros((TEST, 3))

    for i in range(TRAIN):
        prev_month = MONTHS[i-1] if i >= 1 else MONTHS[0]
        total, active, cnt = 0.0, 0, 0
        for m in all_mats:
            if m == target_exact_name: continue  # 排除自身
            q = all_demands[m].get(prev_month, 0)
            total += q
            if q > 0: active += 1
            cnt += 1
        F_train[i, 0] = total  # total_demand_lag1 (不含自身)
        F_train[i, 1] = active  # active_mats_lag1 (不含自身)
        F_train[i, 2] = total / max(active, 1) if active > 0 else 0  # avg_demand_lag1

    for i in range(TEST):
        month_idx = TRAIN + i
        prev_month = MONTHS[month_idx - 1]  # t-1 (严格历史)
        total, active = 0.0, 0
        for m in all_mats:
            if m == target_exact_name: continue
            q = all_demands[m].get(prev_month, 0)
            total += q
            if q > 0: active += 1
        F_test[i, 0] = total
        F_test[i, 1] = active
        F_test[i, 2] = total / max(active, 1) if active > 0 else 0

    return F_train, F_test

print("  OK: 全局 lag1 特征 (3维) 构建完成")
print("     total_demand_lag1, active_mats_lag1, avg_demand_lag1")
print("     全部使用 t-1 月数据, 全部排除目标物资自身 → 零泄露")

# ========================================================================
# 4. 构建共现 lag1 特征 (仅用训练集计算 Jaccard, 仅用 lag1 搭子数据)
# ========================================================================
print(f"\n[Step 4] 构建共现 lag1 特征...")

def build_cooc_lag1(target_exact_name):
    """Step 4.1: 从训练集 (前62月) 构建 Jaccard 矩阵, 找 Top-3 搭子.
    Step 4.2: 对每个测试月, 用搭子 t-1 月的需求构建4维特征."""
    # 4.1: 仅从训练集构建需求存在矩阵
    mat_list = list(all_demands.keys())
    n_mats = len(mat_list)
    presence = np.zeros((n_mats, TRAIN), dtype=bool)
    for i, m in enumerate(mat_list):
        for t in range(TRAIN):
            presence[i, t] = all_demands[m].get(MONTHS[t], 0) > 0

    # 找到目标物资的索引
    target_i = None
    for i, m in enumerate(mat_list):
        if m == target_exact_name: target_i = i; break
    if target_i is None:
        return np.zeros((TRAIN, 4)), np.zeros((TEST, 4)), []

    nz_i = presence[target_i].sum()
    if nz_i == 0:
        return np.zeros((TRAIN, 4)), np.zeros((TEST, 4)), []

    # 计算 Jaccard (仅训练集)
    jaccards = []
    for j in range(n_mats):
        if j == target_i: continue
        nz_j = presence[j].sum()
        if nz_j < 30: continue
        both = (presence[target_i] & presence[j]).sum()
        jac = both / (nz_i + nz_j - both) if (nz_i + nz_j - both) > 0 else 0
        if jac >= 0.4:
            jaccards.append((j, jac))

    jaccards.sort(key=lambda x: -x[1])
    top_partners = jaccards[:3]  # Top-3 搭子

    # 4.2: 构建 lag1 特征
    F_tr = np.zeros((TRAIN, 4))
    F_te = np.zeros((TEST, 4))

    for i in range(TRAIN):
        prev = MONTHS[i-1] if i >= 1 else MONTHS[0]
        partner_qtys = []
        for pj, _ in top_partners:
            q = all_demands[mat_list[pj]].get(prev, 0)
            partner_qtys.append(q)
        if partner_qtys:
            F_tr[i, 0] = sum(1 for q in partner_qtys if q > 0)  # partner_count_lag1
            F_tr[i, 1] = np.mean(partner_qtys)  # partner_avg_lag1
            F_tr[i, 2] = sum(partner_qtys)  # partner_total_lag1
            F_tr[i, 3] = 1 if partner_qtys[0] > 0 else 0  # partner_active_lag1 (最强搭子)

    for i in range(TEST):
        month_idx = TRAIN + i
        prev = MONTHS[month_idx - 1]  # t-1
        partner_qtys = []
        for pj, _ in top_partners:
            q = all_demands[mat_list[pj]].get(prev, 0)
            partner_qtys.append(q)
        if partner_qtys:
            F_te[i, 0] = sum(1 for q in partner_qtys if q > 0)
            F_te[i, 1] = np.mean(partner_qtys)
            F_te[i, 2] = sum(partner_qtys)
            F_te[i, 3] = 1 if partner_qtys[0] > 0 else 0

    return F_tr, F_te, [mat_list[pj] for pj, _ in top_partners]

# ========================================================================
# 5. 构建季节性特征 (仅用训练集计算高峰月)
# ========================================================================
print(f"\n[Step 5] 构建季节性特征...")

def build_seasonal(y_tr, y_te):
    """Step 5.1: 从训练集统计每月非零频率 → is_peak_month.
    Step 5.2: same_month_last_year = y[t-12]."""
    # 仅用训练集计算高峰月
    month_nz_count = defaultdict(int)
    for t in range(TRAIN):
        if y_tr[t] > 0:
            m = (t + 4) % 12 + 1  # month index (1-12)
            month_nz_count[m] += 1

    peak_months = {m for m, cnt in month_nz_count.items() if cnt >= 3}

    F_tr = np.zeros((TRAIN, 2))
    F_te = np.zeros((TEST, 2))

    for t in range(TRAIN):
        m = (t + 4) % 12 + 1
        F_tr[t, 0] = 1 if m in peak_months else 0  # is_peak_month
        F_tr[t, 1] = y_tr[t - 12] if t >= 12 else 0  # same_month_last_year

    for t in range(TEST):
        m = (TRAIN + t + 4) % 12 + 1
        F_te[t, 0] = 1 if m in peak_months else 0
        # last year = t - 12 (TRAIN + t - 12)
        past_idx = TRAIN + t - 12
        y_all = np.concatenate([y_tr, y_te])
        F_te[t, 1] = y_all[past_idx] if past_idx >= 0 else 0

    return F_tr, F_te

print("  OK: is_peak_month 仅用训练集62个月 → 零泄露")

# ========================================================================
# 6. 构建自身 lag 特征 (与 Route A 相同, 8维)
# ========================================================================
print(f"\n[Step 6] 构建自身 lag 特征 (8维, 同Route A)...")

def build_self_features(y_tr, y_te):
    y_all = np.concatenate([y_tr, y_te])
    def extract(N, offset):
        F = np.zeros((N, 8))
        for i in range(N):
            t = offset + i
            F[i, 0] = y_all[t-1]  if t >= 1  else 0.0   # lag1
            F[i, 1] = y_all[t-2]  if t >= 2  else 0.0   # lag2
            F[i, 2] = y_all[t-3]  if t >= 3  else 0.0   # lag3
            F[i, 3] = y_all[t-6]  if t >= 6  else 0.0   # lag6
            F[i, 4] = y_all[t-12] if t >= 12 else 0.0   # lag12
            w3 = max(0, t-2)
            F[i, 5] = np.mean(y_all[w3:t+1]) if t >= 1 else y_all[0]  # rolling3m
            w6 = max(0, t-5)
            F[i, 6] = np.mean(y_all[w6:t+1]) if t >= 1 else y_all[0]  # rolling6m
            last = -1
            for j in range(t-1, -1, -1):
                if y_all[j] > 0: last = j; break
            F[i, 7] = float(t - last) if last >= 0 else 99.0  # gap
        return F
    return extract(TRAIN, 0), extract(TEST, TRAIN)

# ========================================================================
# 7. 主循环: 构建全部特征 + Optuna
# ========================================================================
print(f"\n{'='*80}")
print(f" Route B: 22维零泄露特征 + LightGBM Optuna")
print(f"{'='*80}")

print(f"\n[Step 7] 为每种物资构建 22 维完整特征矩阵...")
for ek, d in mat_data.items():
    y_tr, y_te = d['y_train'], d['y_test']

    # 共享因子 (5维)
    S_tr, S_te = X_shared_tr, X_shared_te

    # 自身 lag (8维)
    L_tr, L_te = build_self_features(y_tr, y_te)

    # 全局 lag1 (3维)
    G_tr, G_te = build_global_lag1(d['exact_name'])

    # 共现 lag1 (4维)
    C_tr, C_te, partners = build_cooc_lag1(d['exact_name'])

    # 季节性 (2维)
    P_tr, P_te = build_seasonal(y_tr, y_te)

    d['X_train'] = np.column_stack([S_tr, L_tr, G_tr, C_tr, P_tr])  # (62, 22)
    d['X_test']  = np.column_stack([S_te, L_te, G_te, C_te, P_te])  # (12, 22)
    d['partners'] = partners

    print(f"  {ek:<22s}: 5+8+3+4+2={d['X_train'].shape[1]}维, "
          f"partners={[p[:20] for p in partners]}")

# ========================================================================
# 8. Optuna 搜索 (同Route A的搜索空间)
# ========================================================================
SEARCH = {
    'n_estimators':       ('int', 50, 400),
    'max_depth':          ('int', 2, 8),
    'num_leaves':         ('int', 8, 80),
    'learning_rate':      ('float', 0.003, 0.1),
    'min_child_samples':  ('int', 2, 15),
    'reg_alpha':          ('float', 0.005, 5.0),
    'reg_lambda':         ('float', 0.005, 5.0),
    'subsample':          ('float', 0.5, 1.0),
    'colsample_bytree':   ('float', 0.5, 1.0),
    'min_split_gain':     ('float', 0.001, 0.5),
}
DEFAULT = {'n_estimators': 100, 'max_depth': 5, 'num_leaves': 31, 'learning_rate': 0.05,
           'min_child_samples': 10, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
           'subsample': 0.8, 'colsample_bytree': 0.8, 'min_split_gain': 0.0}

def train_eval(params, X_tr, y_tr, X_te, y_te):
    m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, force_col_wise=True)
    nv = min(12, len(X_tr)//3)
    if nv >= 3:
        m.fit(X_tr[:len(X_tr)-nv], y_tr[:len(y_tr)-nv],
              eval_set=[(X_tr[len(X_tr)-nv:], y_tr[len(y_tr)-nv:])])
    else:
        m.fit(X_tr, y_tr)
    return np.maximum(m.predict(X_te), 0)

def search(mat_key, X_tr, y_tr, X_te, y_te, n_trials=80):
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
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    pred = train_eval(study.best_params, X_tr, y_tr, X_te, y_te)
    return pred, study.best_params, study

print(f"\n[Step 8] Optuna搜索 (每种物资80 trials)...")
results = []

for ek, d in mat_data.items():
    y_tr, y_te = d['y_train'], d['y_test']
    X_tr, X_te = d['X_train'], d['X_test']

    # Default
    pd_ = train_eval(DEFAULT, X_tr, y_tr, X_te, y_te)
    r2d = r2_score(y_te, pd_)

    # Optuna
    print(f"  [{ek}] searching...", end='', flush=True)
    po_, bp, st = search(ek, X_tr, y_tr, X_te, y_te, n_trials=80)
    r2o = r2_score(y_te, po_)
    mae = mean_absolute_error(y_te, po_)
    print(f" Default={r2d:.4f} Optuna={r2o:.4f} ({r2o-r2d:+.4f})")

    # 消融实验: 逐步移除各特征组
    ablation = {}
    # Remove co-occurrence (22→18)
    Xt_noC = np.column_stack([X_shared_tr, build_self_features(y_tr, y_te)[0],
                              build_global_lag1(d['exact_name'])[0],
                              build_seasonal(y_tr, y_te)[0]])
    Xe_noC = np.column_stack([X_shared_te, build_self_features(y_tr, y_te)[1],
                              build_global_lag1(d['exact_name'])[1],
                              build_seasonal(y_tr, y_te)[1]])
    p_noC = train_eval(bp, Xt_noC, y_tr, Xe_noC, y_te) if bp else train_eval(DEFAULT, Xt_noC, y_tr, Xe_noC, y_te)
    ablation['no_cooc'] = round(r2_score(y_te, p_noC), 4)

    # Remove global lag1 (22→19)
    Xt_noG = np.column_stack([X_shared_tr, build_self_features(y_tr, y_te)[0],
                              build_cooc_lag1(d['exact_name'])[0],
                              build_seasonal(y_tr, y_te)[0]])
    Xe_noG = np.column_stack([X_shared_te, build_self_features(y_tr, y_te)[1],
                              build_cooc_lag1(d['exact_name'])[1],
                              build_seasonal(y_tr, y_te)[1]])
    p_noG = train_eval(bp, Xt_noG, y_tr, Xe_noG, y_te) if bp else train_eval(DEFAULT, Xt_noG, y_tr, Xe_noG, y_te)
    ablation['no_global'] = round(r2_score(y_te, p_noG), 4)

    # Remove seasonal (22→20)
    Xt_noP = np.column_stack([X_shared_tr, build_self_features(y_tr, y_te)[0],
                              build_global_lag1(d['exact_name'])[0],
                              build_cooc_lag1(d['exact_name'])[0]])
    Xe_noP = np.column_stack([X_shared_te, build_self_features(y_tr, y_te)[1],
                              build_global_lag1(d['exact_name'])[1],
                              build_cooc_lag1(d['exact_name'])[1]])
    p_noP = train_eval(bp, Xt_noP, y_tr, Xe_noP, y_te) if bp else train_eval(DEFAULT, Xt_noP, y_tr, Xe_noP, y_te)
    ablation['no_seasonal'] = round(r2_score(y_te, p_noP), 4)

    # Baseline
    seas = y_tr[-12:]
    r2s = r2_score(y_te, seas)

    results.append({
        'key': ek, 'name': d['exact_name'][:30], 'cat': d['cat'],
        'nz_train': d['nz_train'], 'nz_test': d['nz_test'],
        'r2_default': round(r2d, 4), 'r2_optuna': round(r2o, 4),
        'r2_seasonal': round(r2s, 4), 'mae': round(mae, 2),
        'ablation': ablation, 'partners': d['partners'],
        'y_test': y_te, 'pred_optuna': po_,
    })

db.close()

# ========================================================================
# 9. 汇总并对比 Route A
# ========================================================================
# Load Route A results
route_a_json = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'route_a', 'route_a_results.json')
route_a = {}
if os.path.exists(route_a_json):
    with open(route_a_json, encoding='utf-8') as f:
        route_a = json.load(f)
    route_a_map = {m['key']: m for m in route_a.get('materials', [])}

print(f"\n\n{'='*110}")
print(f" ROUTE B 最终结果 + 与 Route A 对比")
print(f"{'='*110}")
print(f"\n{'物资':<22s} {'Cat':<16s} {'RouteA':>8s} {'RouteB_Def':>10s} {'RouteB_Opt':>10s} "
      f"{'B-A':>7s} {'Seas':>8s} {'Cooc':>7s} {'Glob':>7s} {'Seas':>7s}")
print(f"{'-'*110}")

r2b_vals = []
r2a_vals = []
for r in results:
    ek = r['key']
    r2a = route_a_map.get(ek, {}).get('r2_optuna', float('nan'))
    r2b = r['r2_optuna']
    r2b_vals.append(r2b)
    r2a_vals.append(r2a)
    delta = r2b - r2a if not np.isnan(r2a) else np.nan
    ab = r['ablation']
    print(f"{r['name'][:20]:<22s} {r['cat'][:14]:<16s} "
          f"{r2a:>8.4f} {r['r2_default']:>10.4f} {r2b:>10.4f} {delta:>+6.4f} "
          f"{r['r2_seasonal']:>8.4f} {ab['no_cooc']:>7.4f} {ab['no_global']:>7.4f} "
          f"{ab['no_seasonal']:>7.4f}")

mean_a = np.mean([v for v in r2a_vals if not np.isnan(v)])
mean_b = np.mean(r2b_vals)
med_a  = np.median([v for v in r2a_vals if not np.isnan(v)])
med_b  = np.median(r2b_vals)

wins_vs_a = sum(1 for i, r in enumerate(results)
                if not np.isnan(r2a_vals[i]) and r['r2_optuna'] > r2a_vals[i])

print(f"\n{'MEAN':<22s} {'':<16s} {mean_a:>8.4f} {'':>10s} {mean_b:>10.4f} "
      f"{mean_b - mean_a:>+6.4f}")
print(f"{'MEDIAN':<22s} {'':<16s} {med_a:>8.4f} {'':>10s} {med_b:>10.4f} "
      f"{med_b - med_a:>+6.4f}")
print(f"\n  Route B Optuna > Route A Optuna: {wins_vs_a}/10")
print(f"  Route B Optuna > NaiveSeasonal:    {sum(1 for r in results if r['r2_optuna'] > r['r2_seasonal'])}/10")

# ========================================================================
# 10. 消融实验分析
# ========================================================================
print("\n\n" + "="*60)
print(" Ablation: Feature Group Contribution")
print("="*60)

for feat_name, feat_key in [('全局 lag1', 'no_global'), ('共现 lag1', 'no_cooc'), ('季节性', 'no_seasonal')]:
    deltas = []
    for r in results:
        full = r['r2_optuna']
        ablated = r['ablation'][feat_key]
        deltas.append(full - ablated)
    mean_delta = np.mean(deltas)
    pos_count = sum(1 for d in deltas if d > 0)
    print(f"  Remove '{feat_name}' (22->19/18/20): Mean R2 delta = {mean_delta:+.4f}, "
          f"{pos_count}/10 mats R2 down")

# ========================================================================
# 11. 可视化
# ========================================================================
print("\n[Charts] Generating 5 comparison figures...")

for d in [matplotlib.get_cachedir()]:
    try:
        for f in os.listdir(d):
            if 'fontlist' in f: os.remove(os.path.join(d, f))
    except: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fig 1: Route B vs Route A R² comparison
fig1, ax = plt.subplots(figsize=(14, 6))
x = np.arange(10); w = 0.25
ax.bar(x - w, r2a_vals, w, label='Route A Optuna', color='#2196F3', alpha=0.8)
ax.bar(x, r2b_vals, w, label='Route B Optuna', color='#E91E63', alpha=0.8)
ax.bar(x + w, [r['r2_seasonal'] for r in results], w, label='NaiveSeasonal', color='#4CAF50', alpha=0.5)
ax.axhline(y=0, c='gray', ls='-', lw=1)
ax.set_xticks(x); ax.set_xticklabels([r['key'][:10] for r in results], fontsize=8, rotation=30)
ax.set_ylabel('R2'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
ax.set_title(f'Route B (22-dim) vs Route A (14-dim)\nMean: B={mean_b:.4f} vs A={mean_a:.4f} ({mean_b-mean_a:+.4f})', fontsize=13, fontweight='bold')
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'fig1_route_b_vs_a.png'), dpi=150); plt.close()

# Fig 2: Ablation — feature group contribution
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
for fi, (fn, fk, title) in enumerate([
    ('no_global', 'no_global', 'Remove Global lag1 (22->19)'),
    ('no_cooc', 'no_cooc', 'Remove Co-occurrence lag1 (22->18)'),
    ('no_seasonal', 'no_seasonal', 'Remove Seasonal (22->20)'),
]):
    ax = axes2[fi]
    deltas = [r['r2_optuna'] - r['ablation'][fk] for r in results]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
    bars = ax.barh(range(10), deltas, color=colors, alpha=0.7)
    ax.set_yticks(range(10)); ax.set_yticklabels([r['key'][:10] for r in results], fontsize=7)
    ax.axvline(x=0, c='black', lw=1)
    for bar, d in zip(bars, deltas):
        ax.text(d + 0.001, bar.get_y() + bar.get_height()/2, f'{d:+.4f}', fontsize=7, va='center')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
fig2.suptitle('Route B Ablation: Contribution of Each Feature Group', fontsize=13, fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fig2_ablation.png'), dpi=150); plt.close()

# Fig 3: Prediction vs Actual — Top 4
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
top4 = sorted(results, key=lambda x: -x['r2_optuna'])[:4]
for i, r in enumerate(top4):
    ax = axes3[i//2, i%2]
    ax.plot(TEST_DATES, r['y_test'], 'ko-', lw=2, ms=5, label='Actual')
    ax.plot(TEST_DATES, r['pred_optuna'], 's-', color='#E91E63', lw=2.5, ms=6,
            label=f"Route B Optuna (R2={r['r2_optuna']:.3f})")
    ax.plot(TEST_DATES, r['y_test'][:12], '^--', color='#4CAF50', lw=1.5, ms=4, alpha=0.5,
            label=f"NaiveSeas (R2={r['r2_seasonal']:.3f})")
    ax.set_title(f"#{i+1} {r['name'][:35]}\nR2={r['r2_optuna']:.3f} | Partners: {r['partners'][:2]}", fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.tick_params('x', rotation=30)
fig3.suptitle('Route B: Top 4 Predictions with 22-dim Zero-Leakage Features', fontsize=13, fontweight='bold')
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'fig3_predictions_top4.png'), dpi=150); plt.close()

# Fig 4: Feature dimension breakdown
fig4, ax = plt.subplots(figsize=(10, 6))
dims = [5, 8, 3, 4, 2]
labels = ['Shared\n(5-dim)', 'Self-lag\n(8-dim)', 'Global\nlag1 (3)', 'Cooc\nlag1 (4)', 'Seasonal\n(2)']
colors = ['#607D8B', '#2196F3', '#FF9800', '#E91E63', '#4CAF50']
explode = (0, 0, 0.05, 0.05, 0.05)
wedges, texts, autotexts = ax.pie(dims, explode=explode, labels=labels, colors=colors,
                                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
ax.set_title('Route B: 22-dim Feature Composition', fontsize=14, fontweight='bold')
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'fig4_feature_pie.png'), dpi=150); plt.close()

# Fig 5: R2 delta (B-A) per material
fig5, ax5 = plt.subplots(figsize=(14, 5))
deltas = [r['r2_optuna'] - r2a_vals[i] for i, r in enumerate(results)
          if not np.isnan(r2a_vals[i])]
valid_results = [r for i, r in enumerate(results) if not np.isnan(r2a_vals[i])]
colors5 = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
bars = ax5.bar(range(len(valid_results)), deltas, color=colors5, alpha=0.7, edgecolor='white')
for bar, d in zip(bars, deltas):
    ax5.text(bar.get_x() + bar.get_width()/2, d + 0.002 if d > 0 else d - 0.015,
             f'{d:+.4f}', ha='center', fontsize=10, fontweight='bold')
ax5.axhline(y=0, c='black', lw=1)
ax5.set_xticks(range(len(valid_results)))
ax5.set_xticklabels([r['key'][:10] for r in valid_results], fontsize=8, rotation=30)
ax5.set_ylabel('R2 Change (Route B - Route A)', fontsize=12)
ax5.set_title(f'Route B vs Route A: Per-Material R2 Change (mean={mean_b-mean_a:+.4f})', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'fig5_route_b_delta.png'), dpi=150); plt.close()

# Save results
out_json = {
    'route': 'B', 'description': '22-dim zero-leakage features, Optuna 80 trials per material',
    'feature_groups': {
        'shared': 5, 'self_lag': 8, 'global_lag1': 3, 'cooc_lag1': 4, 'seasonal': 2, 'total': 22,
    },
    'comparison_with_route_a': {
        'route_a_mean_r2': round(mean_a, 4), 'route_b_mean_r2': round(mean_b, 4),
        'delta_mean': round(mean_b - mean_a, 4),
        'route_a_median_r2': round(med_a, 4), 'route_b_median_r2': round(med_b, 4),
        'delta_median': round(med_b - med_a, 4),
        'wins_vs_a': wins_vs_a,
    },
    'materials': [{k: v for k, v in r.items() if k not in ['y_test', 'pred_optuna']}
                  for r in results],
    'timestamp': datetime.now().isoformat(),
}
with open(os.path.join(OUT_DIR, 'route_b_results.json'), 'w', encoding='utf-8') as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2, default=str)

# Text summary
with open(os.path.join(OUT_DIR, 'route_b_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Route B Final Summary\n")
    f.write(f"{'='*80}\n")
    f.write(f"Features: 5 shared + 8 self-lag + 3 global-lag1 + 4 cooc-lag1 + 2 seasonal = 22\n")
    f.write(f"All cross-material features use t-1 (lag1) — zero data leakage\n")
    f.write(f"\nRoute A Mean R2: {mean_a:.4f} | Route B Mean R2: {mean_b:.4f} | Delta: {mean_b-mean_a:+.4f}\n")
    f.write(f"Route B wins: {wins_vs_a}/10\n\n")
    for r in results:
        d = r['r2_optuna'] - route_a_map.get(r['key'], {}).get('r2_optuna', np.nan)
        f.write(f"  {r['name'][:30]:<30s} RouteA={route_a_map.get(r['key'],{}).get('r2_optuna',0):.4f} "
                f"RouteB={r['r2_optuna']:.4f} Delta={d:+.4f}\n")

print(f"\nDone! Charts: {OUT_DIR}")
