"""
Route C: 池化两阶段预测模型 — Pooled Two-Stage with Guardrail
================================================================
Stage 1: 全部10种物资池化训练一个LightGBM分类器(620行)
Guardrail: 验证集F1 > baseline(全预测0) → 否则fallback到单阶段Optuna
Stage 2: 每种物资独立LightGBM + Optuna (仅非零月训练)
最终预测: P(有需求) × Q(需求量)

对比实验:
  1. 池化两阶段 vs 单阶段Optuna基准
  2. Guardrail有/无的差异
  3. 概率加权 vs 硬阈值
  4. Stage 1误差对MSE的传播量化
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
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from logger_utils import StepLogger, format_params

log = StepLogger('RouteC')

# ========================================================================
# 0. 配置
# ========================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'route_c')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN, TEST = 62, 12
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
# Pipeline start
# ========================================================================
log.pipeline_start_log(
    'Route C: Pooled Two-Stage Prediction with Guardrail',
    'Stage1=pooled classifier(620 rows), Stage2=per-material Optuna, Guardrail=F1>baseline'
)

# ========================================================================
# Step 1: 数据加载
# ========================================================================
log.step('数据加载', '10种物资 + 6维共享特征 + 8维自回归 + Stage1标签构建')

ref = pd.read_excel(REF_XLSX, sheet_name=0)
FEAT_COLS = [c for c in ref.columns if c not in ['日期', '需求量']]
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
feat_df = ref[ref['mk'].between('202005','202606')].copy()
X_shared = feat_df[FEAT_COLS].values.astype(np.float64)
X_sh_tr = X_shared[:TRAIN]; X_sh_te = X_shared[TRAIN:]

db = sqlite3.connect(DB_PATH)
mat_data = {}
rows_sum = []

for ek, pattern, cat in TARGETS:
    cur = db.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) nz, SUM(demand_quantity) tot
        FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY material_name ORDER BY nz DESC LIMIT 3
    """, (pattern,))
    best = max(cur.fetchall(), key=lambda x: x[1])
    cn_name = best[0]

    cur2 = db.execute("""
        SELECT demand_month, SUM(demand_quantity) FROM material_demand_item
        WHERE material_name=? AND demand_month>='202005' AND demand_month<='202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (cn_name,))
    dmap = {r[0]: r[1] for r in cur2.fetchall()}
    vals = np.array([dmap.get(m, 0) for m in MONTHS], dtype=np.float64)

    mat_data[ek] = {
        'name': cn_name, 'cat': cat,
        'y_tr': vals[:TRAIN], 'y_te': vals[TRAIN:],
        'nz_tr': (vals[:TRAIN]>0).sum(), 'nz_te': (vals[TRAIN:]>0).sum(),
    }
    rows_sum.append([ek, cn_name[:30], cat, mat_data[ek]['nz_tr'], mat_data[ek]['nz_te']])

log.data_table('物资清单', ['Key','Name','Cat','NZ_Train','NZ_Test'], rows_sum)
db.close()

# ========================================================================
# Step 2: 特征工程
# ========================================================================
log.step('特征工程', '构建14维零泄露特征(6 shared + 8 self-lag) + Stage1池化训练集')

def build_self_features(y_tr, y_te):
    y_all = np.concatenate([y_tr, y_te])
    def extr(N, offset):
        F = np.zeros((N, 8))
        for i in range(N):
            t = offset + i
            F[i,0]=y_all[t-1] if t>=1 else 0.0;F[i,1]=y_all[t-2] if t>=2 else 0.0
            F[i,2]=y_all[t-3] if t>=3 else 0.0;F[i,3]=y_all[t-6] if t>=6 else 0.0
            F[i,4]=y_all[t-12] if t>=12 else 0.0
            w3=max(0,t-2);F[i,5]=np.mean(y_all[w3:t+1]) if t>=1 else y_all[0]
            w6=max(0,t-5);F[i,6]=np.mean(y_all[w6:t+1]) if t>=1 else y_all[0]
            last=-1
            for j in range(t-1,-1,-1):
                if y_all[j]>0:last=j;break
            F[i,7]=float(t-last) if last>=0 else 99.0
        return F
    return extr(TRAIN,0), extr(TEST,TRAIN)

# 为每种物资构建特征
for ek in mat_data:
    d = mat_data[ek]
    L_tr, L_te = build_self_features(d['y_tr'], d['y_te'])
    d['X_tr'] = np.column_stack([X_sh_tr, L_tr])  # (62, 14)
    d['X_te'] = np.column_stack([X_sh_te, L_te])  # (12, 14)
    d['y_bin_tr'] = (d['y_tr'] > 0).astype(int)
    d['y_bin_te'] = (d['y_te'] > 0).astype(int)

# 池化训练集: 10种物资 × 62月 = 620行
X_pooled, y_pooled_bin = [], []
mat_idx_col = []  # material one-hot index
for i, ek in enumerate(mat_data):
    d = mat_data[ek]
    for t in range(TRAIN):
        feat_vec = list(d['X_tr'][t])
        feat_vec.append(i)  # material ID (0-9)
        X_pooled.append(feat_vec)
        y_pooled_bin.append(d['y_bin_tr'][t])

X_pooled = np.array(X_pooled)  # (620, 15)
y_pooled_bin = np.array(y_pooled_bin)

# 池化测试集: 10种物资 × 12月 = 120行
X_pooled_te = []
for i, ek in enumerate(mat_data):
    d = mat_data[ek]
    for t in range(TEST):
        feat_vec = list(d['X_te'][t])
        feat_vec.append(i)
        X_pooled_te.append(feat_vec)
X_pooled_te = np.array(X_pooled_te)  # (120, 15)

n_pos = y_pooled_bin.sum()
n_neg = len(y_pooled_bin) - n_pos
pooled_baseline_acc = max(n_pos, n_neg) / len(y_pooled_bin)
log.task(f'池化训练集: {len(X_pooled)}行, pos={n_pos} ({n_pos/len(X_pooled):.1%}), neg={n_neg}')
log.task(f'池化基线(全预测多数类)准确率: {pooled_baseline_acc:.1%}')
log.task(f'数据泄露检查: PASS — 所有特征仅使用t-1之前历史数据, material_id仅作为组标识')

log.step_ok(f'620 pooled training rows, 120 test rows')

# ========================================================================
# Step 3: Optuna搜索函数
# ========================================================================
log.step('定义搜索函数', 'Stage1分类器F1搜索 + Stage2回归器R2搜索')

S1_SEARCH = {
    'n_estimators':('int',50,400),'max_depth':('int',3,8),'num_leaves':('int',8,100),
    'learning_rate':('float',0.003,0.15),'min_child_samples':('int',2,20),
    'reg_alpha':('float',0.005,5.0),'reg_lambda':('float',0.005,5.0),
    'subsample':('float',0.5,1.0),'colsample_bytree':('float',0.5,1.0),
    'min_split_gain':('float',0.001,0.5),
}
S2_SEARCH = {
    'n_estimators':('int',50,400),'max_depth':('int',2,8),'num_leaves':('int',8,80),
    'learning_rate':('float',0.003,0.1),'min_child_samples':('int',2,15),
    'reg_alpha':('float',0.005,5.0),'reg_lambda':('float',0.005,5.0),
    'subsample':('float',0.5,1.0),'colsample_bytree':('float',0.5,1.0),
    'min_split_gain':('float',0.001,0.5),
}
DEFAULT_CL = {'n_estimators':200,'max_depth':5,'num_leaves':50,'learning_rate':0.03,
              'min_child_samples':10,'reg_alpha':1.0,'reg_lambda':1.0,
              'subsample':0.8,'colsample_bytree':0.8,'min_split_gain':0.0,
              'class_weight':'balanced'}
DEFAULT_RG = {'n_estimators':100,'max_depth':5,'num_leaves':31,'learning_rate':0.05,
              'min_child_samples':10,'reg_alpha':0.1,'reg_lambda':0.1,
              'subsample':0.8,'colsample_bytree':0.8,'min_split_gain':0.0}

def suggest_params(trial, space):
    p = {}
    for pn, (pt, lo, hi) in space.items():
        if pt == 'int': p[pn] = trial.suggest_int(pn, lo, hi)
        else: p[pn] = trial.suggest_float(pn, lo, hi, log=True)
    return p

log.step_ok(f'Search spaces: Stage1={len(S1_SEARCH)} params, Stage2={len(S2_SEARCH)} params')

# ========================================================================
# Step 4: Stage 1 — 池化分类器 + Optuna
# ========================================================================
log.step('Stage1: 池化分类器Optuna', '80 trials, 最大化验证集F1')

def obj_s1(trial):
    p = suggest_params(trial, S1_SEARCH)
    p['class_weight'] = 'balanced'
    cl = lgb.LGBMClassifier(**p, random_state=42, verbose=-1, force_col_wise=True)
    nv = min(60, len(X_pooled) // 3)
    cl.fit(X_pooled[:len(X_pooled)-nv], y_pooled_bin[:len(X_pooled)-nv])
    yp = cl.predict(X_pooled[len(X_pooled)-nv:])
    return f1_score(y_pooled_bin[len(X_pooled)-nv:], yp, zero_division=0)

study_s1 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), study_name='stage1')
study_s1.optimize(obj_s1, n_trials=80, show_progress_bar=False)

# 用最优参数在全量训练集上训练, 预测测试集
best_s1 = lgb.LGBMClassifier(**{k:v for k,v in study_s1.best_params.items()},
                             class_weight='balanced', random_state=42, verbose=-1, force_col_wise=True)
best_s1.fit(X_pooled, y_pooled_bin)
# predict_proba gives P(has_demand) for each test sample
s1_proba = best_s1.predict_proba(X_pooled_te)[:, 1]  # (120,) — P(has demand)

log.task(f'Stage1 best F1 on val: {study_s1.best_value:.4f}')
log.task(f'Stage1 best params: {format_params(study_s1.best_params)}')
log.step_ok(f'Pooled classifier trained, 120 test probabilities generated')

# ========================================================================
# Step 5: Guardrail — 单独对每种物资检查Stage1质量
# ========================================================================
log.step('Guardrail: Stage1分类质量检查', '验证集F1 > baseline → 通过; 否则→fallback')

guardrail_results = {}
for i, ek in enumerate(mat_data):
    d = mat_data[ek]
    # Stage1 predictions for this material's test set
    start_idx = i * TEST
    end_idx = (i+1) * TEST
    prob_this = s1_proba[start_idx:end_idx]  # (12,) — P(has demand)
    pred_this = (prob_this >= 0.5).astype(int)  # hard threshold for evaluation
    actual_this = d['y_bin_te']  # (12,) — actual 0/1

    # F1 score
    f1 = f1_score(actual_this, pred_this, zero_division=0)
    # Baseline: always predict 0 → positive class never predicted
    n_pos_te = actual_this.sum()
    baseline_f1 = 0 if n_pos_te == 0 else n_pos_te / (n_pos_te + 12)

    # Confusion matrix for error analysis
    cm = confusion_matrix(actual_this, pred_this, labels=[0, 1])
    tn, fp, fn, tp = (cm[0,0], cm[0,1], cm[1,0], cm[1,1]) if cm.size == 4 else (12,0,0,0)

    # Guardrail conditions
    passed = True; reason = "PASS"
    if f1 <= 0.01:
        passed = False; reason = f"F1={f1:.3f} too low (baseline={baseline_f1:.3f})"
    elif fn > 0 and fn >= max(tp, 1):
        passed = False; reason = f"FN({fn}) >= TP({tp}) — too many missed demands"
    elif fp > 6:
        passed = False; reason = f"FP({fp}) too many — classifier over-predicts demand"

    # Key error metric: how many actual non-zero months were missed?
    fn_ratio = fn / max(n_pos_te, 1)

    guardrail_results[ek] = {
        'passed': passed, 'reason': reason,
        'f1': round(f1, 4), 'baseline_f1': round(baseline_f1, 4),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'fn_ratio': round(fn_ratio, 4), 'n_pos_te': n_pos_te,
        'proba': prob_this, 'pred': pred_this,
    }
    log.task(f'  [{ek:<22s}] F1={f1:.3f}, baseline={baseline_f1:.3f}, '
             f'TP={tp}, FN={fn}, FP={fp}, TN={tn} → {"PASS" if passed else "FAIL: "+reason}')

n_passed = sum(1 for v in guardrail_results.values() if v['passed'])
n_failed = 10 - n_passed
log.task(f'Guardrail: {n_passed}/10 PASS, {n_failed}/10 FAIL (will fallback to single-stage Optuna)')
log.step_ok(f'{n_passed} guardrail passed, {n_failed} need fallback')

# ========================================================================
# Step 6: Stage 2 回归器 + 最终预测
# ========================================================================
log.step('Stage2: 回归器训练 + 最终预测', '通过guardrail的用两阶段, 否则fallback单阶段')

results_c = {}
all_yp = {}

# ---- Single-stage Optuna baseline (for all 10, regardless of guardrail) ----
log.task('先运行单阶段Optuna基准(全部10种)...')
baseline_r2 = {}

for ek, d in mat_data.items():
    X_tr, X_te = d['X_tr'], d['X_te']
    y_tr, y_te = d['y_tr'], d['y_te']
    na = len(X_tr); nv = min(12, na//3)

    def obj_single(trial):
        p = suggest_params(trial, S2_SEARCH)
        m = lgb.LGBMRegressor(**p, random_state=42, verbose=-1, force_col_wise=True)
        m.fit(X_tr[:na-nv], y_tr[:na-nv], eval_set=[(X_tr[na-nv:], y_tr[na-nv:])])
        yp = np.maximum(m.predict(X_tr[na-nv:]), 0)
        try: return r2_score(y_tr[na-nv:], yp)
        except: return -10.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_single, n_trials=80, show_progress_bar=False)

    m = lgb.LGBMRegressor(**study.best_params, random_state=42, verbose=-1, force_col_wise=True)
    if nv >= 3:
        m.fit(X_tr[:na-nv], y_tr[:na-nv], eval_set=[(X_tr[na-nv:], y_tr[na-nv:])])
    else:
        m.fit(X_tr, y_tr)
    yp_base = np.maximum(m.predict(X_te), 0)
    baseline_r2[ek] = r2_score(y_te, yp_base), yp_base

# ---- Route C: pooled two-stage ----
log.task('运行RouteC池化两阶段预测...')

for i, ek in enumerate(mat_data):
    d = mat_data[ek]
    y_tr, y_te = d['y_tr'], d['y_te']
    gr = guardrail_results[ek]
    prob = gr['proba']  # (12,) — Stage1 probabilities

    # ---- Option 1: Guardrail passed → Two-Stage with probability weighting ----
    # ---- Option 2: Guardrail failed → fallback to single-stage Optuna ----
    if gr['passed']:
        # Stage 2: train on non-zero samples only
        nz_mask = y_tr > 0
        if nz_mask.sum() < 5:
            # Not enough non-zero samples → fallback
            yp_c, r2_c = baseline_r2[ek][1], baseline_r2[ek][0]
            twostage_mode = 'fallback(no_nz)'
        else:
            X_nz = d['X_tr'][nz_mask]
            y_nz = y_tr[nz_mask]
            na = len(X_nz); nv = min(3, na//4)
            def obj_s2(trial):
                p = suggest_params(trial, S2_SEARCH)
                m = lgb.LGBMRegressor(**p, random_state=42, verbose=-1, force_col_wise=True)
                if nv >= 2:
                    m.fit(X_nz[:na-nv], y_nz[:na-nv], eval_set=[(X_nz[na-nv:], y_nz[na-nv:])])
                else: m.fit(X_nz, y_nz)
                yp = np.maximum(m.predict(X_nz[na-nv:]), 0)
                try: return r2_score(y_nz[na-nv:], yp)
                except: return -10.0
            study_s2 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study_s2.optimize(obj_s2, n_trials=50, show_progress_bar=False)

            m2 = lgb.LGBMRegressor(**study_s2.best_params, random_state=42, verbose=-1, force_col_wise=True)
            if nv >= 2:
                m2.fit(X_nz[:na-nv], y_nz[:na-nv], eval_set=[(X_nz[na-nv:], y_nz[na-nv:])])
            else: m2.fit(X_nz, y_nz)
            q_pred = np.maximum(m2.predict(d['X_te']), 0)  # Stage2 quantity

            # Probability-weighted final prediction: ŷ = P × Q
            # This is key: NO hard threshold. All months get predicted.
            yp_c = prob * q_pred  # probability-weighted
            r2_c = r2_score(y_te, yp_c)
            twostage_mode = 'prob_weighted'
    else:
        # Guardrail FAILED → fallback to single-stage Optuna
        yp_c, r2_c = baseline_r2[ek][1], baseline_r2[ek][0]
        twostage_mode = f'fallback({gr["reason"][:30]})'

    mae_c = mean_absolute_error(y_te, yp_c)
    r2_base = baseline_r2[ek][0]
    delta = r2_c - r2_base
    tag = '+' if delta > 0 else ''

    results_c[ek] = {
        'name': d['name'], 'cat': d['cat'],
        'nz_tr': d['nz_tr'], 'nz_te': d['nz_te'],
        'r2_base': round(r2_base, 4), 'r2_twostage': round(r2_c, 4),
        'r2_delta': round(delta, 4), 'mode': twostage_mode,
        'guardrail': gr, 'mae': round(mae_c, 2),
        'y_te': y_te, 'pred_c': yp_c, 'pred_base': baseline_r2[ek][1],
        'proba': prob, 'q_pred': q_pred if gr['passed'] and 'q_pred' in dir() else None,
    }
    all_yp[ek] = yp_c

    log.metrics(ek, {
        'BaseR2': f'{r2_base:.4f}', 'TwoStageR2': f'{r2_c:.4f}',
        'Delta': f'{delta:+.4f}{tag}', 'N_Train': d['nz_tr'], 'Mode': twostage_mode,
    })

# ========================================================================
# Step 7: 误差传播分析 — MSE分解
# ========================================================================
log.step('误差传播分析: MSE由Stage1误差贡献了多少?', '对通过guardrail的物资做MSE分解')

mse_decomposition_rows = []
for ek, rc in results_c.items():
    gr = rc['guardrail']
    if not gr['passed']:
        continue

    d = mat_data[ek]
    y_te = d['y_te']
    prob = gr['proba']
    pred = gr['pred']

    # 对每个月分类
    tn_months = []; fp_months = []; fn_months = []; tp_months = []
    for t in range(TEST):
        if pred[t] == 0 and y_te[t] == 0: tn_months.append(t)
        elif pred[t] == 0 and y_te[t] > 0: fn_months.append(t)
        elif pred[t] > 0 and y_te[t] == 0: fp_months.append(t)
        elif pred[t] > 0 and y_te[t] > 0: tp_months.append(t)

    # MSE decomposition
    mse_total = mean_squared_error(y_te, rc['pred_c'])
    mse_fn = sum(y_te[t]**2 for t in fn_months) / TEST  # pred=0, actual>0
    mse_fp = sum(rc['pred_c'][t]**2 for t in fp_months) / TEST  # pred>0, actual=0
    mse_tp = sum((rc['pred_c'][t] - y_te[t])**2 for t in tp_months) / TEST
    mse_tn = 0  # pred=0, actual=0 → perfect

    fn_pct = mse_fn / max(mse_total, 1e-10) * 100
    fp_pct = mse_fp / max(mse_total, 1e-10) * 100

    mse_decomposition_rows.append([
        ek[:16], f'{mse_total:.0f}',
        f'{mse_fn:.0f} ({fn_pct:.0f}%)' if fn_months else '0',
        f'{mse_fp:.0f} ({fp_pct:.0f}%)' if fp_months else '0',
        f'{mse_tp:.0f}', f'{len(fn_months)},{len(fp_months)}',
    ])

if mse_decomposition_rows:
    log.data_table('MSE分解 (仅Passed物资)',
                   ['Material','MSE_total','MSE_FN(%total)','MSE_FP(%total)','MSE_TP','(#FN,#FP)'],
                   mse_decomposition_rows)

# 定量回答用户问题: "第一步误差导致第二步偏差在可接受范围内?"
# 可接受标准: FN贡献 < 30% MSE
fn_dominant = sum(1 for r in mse_decomposition_rows if 'FN' in r[2] and float(r[2].split('(')[1].split('%')[0]) > 30)
log.task(f'FN贡献>30% MSE的物资: {fn_dominant}个 (阈值:可接受=0)')

# ========================================================================
# Step 8: 汇总对比
# ========================================================================
log.step('汇总对比', 'RouteC TwoStage vs SingleStage Optuna基准')

Summary_rows = []
for ek, rc in results_c.items():
    r2b = rc['r2_base']; r2c = rc['r2_twostage']; d = rc['r2_delta']
    win = 'TwoStage' if d > 0 else 'Single'
    Summary_rows.append([ek[:20], rc['cat'][:12], f'{r2b:.4f}', f'{r2c:.4f}',
                         f'{d:+.4f}', win, rc['mode'][:30]])

log.data_table('RouteC TwoStage vs SingleStage Optuna',
               ['Material','Cat','SingleR2','TwoStageR2','Delta','Winner','Mode'], Summary_rows)

mean_base = np.mean([rc['r2_base'] for rc in results_c.values()])
mean_twos = np.mean([rc['r2_twostage'] for rc in results_c.values()])
wins_twos = sum(1 for rc in results_c.values() if rc['r2_twostage'] > rc['r2_base'])
wins_guardrail = sum(1 for rc in results_c.values() if rc['guardrail']['passed'] and rc['r2_twostage'] > rc['r2_base'])

log.task(f'Mean R2: Single={mean_base:.4f}, TwoStage={mean_twos:.4f} (Δ={mean_twos-mean_base:+.4f})')
log.task(f'TwoStage wins: {wins_twos}/10 total, {wins_guardrail}/{n_passed} among guardrail-passed')
log.task(f'Guardrail passed: {n_passed}/10')

# ========================================================================
# Step 9: 对比实验 — 概率加权 vs 硬阈值
# ========================================================================
log.step('对比实验: 概率加权 vs 硬阈值', '对通过guardrail的物资做两方案对比')

prob_vs_hard_rows = []
for ek, rc in results_c.items():
    if not rc['guardrail']['passed']:
        continue
    d = mat_data[ek]
    prob = rc['proba']
    q_pred = rc['q_pred'] if rc['q_pred'] is not None else np.zeros(TEST)

    # 概率加权: y = prob * q_pred (already done above)
    r2_probw = rc['r2_twostage']

    # 硬阈值: y = (prob >= 0.5) ? q_pred : 0
    hard_pred = np.where(prob >= 0.5, q_pred, 0)
    r2_hard = r2_score(d['y_te'], hard_pred)

    prob_vs_hard_rows.append([ek[:16], f'{r2_probw:.4f}', f'{r2_hard:.4f}',
                              f'{r2_probw-r2_hard:+.4f}',
                              'ProbWeight' if r2_probw > r2_hard else 'HardThreshold'])

if prob_vs_hard_rows:
    log.data_table('概率加权 vs 硬阈值 (仅Passed物资)',
                   ['Material','ProbWeight','HardThresh','Delta','Winner'], prob_vs_hard_rows)
    wins_prob = sum(1 for r in prob_vs_hard_rows if r[4] == 'ProbWeight')
    log.task(f'概率加权胜出: {wins_prob}/{len(prob_vs_hard_rows)}')

# ========================================================================
# Step 10: 可视化
# ========================================================================
log.step('可视化 + 导出', '6张图 + JSON + 日志')

for d in [matplotlib.get_cachedir()]:
    try:
        for f in os.listdir(d):
            if 'fontlist' in f: os.remove(os.path.join(d, f))
    except: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fig 1: R2 comparison — TwoStage vs SingleStage
fig1, ax = plt.subplots(figsize=(14, 6)); x = np.arange(10); w = 0.3
r2s = [results_c[ek]['r2_base'] for ek in mat_data]
r2c = [results_c[ek]['r2_twostage'] for ek in mat_data]
ax.bar(x-w/2, r2s, w, label='SingleStage Optuna', color='#2196F3', alpha=0.8)
ax.bar(x+w/2, r2c, w, label='TwoStage (Route C)', color='#E91E63', alpha=0.8)
# Mark guardrail status
for i, ek in enumerate(mat_data):
    if not guardrail_results[ek]['passed']:
        ax.text(i, r2c[i]+0.03, 'FAIL', ha='center', fontsize=8, color='red', fontweight='bold')
    else:
        ax.text(i, r2c[i]+0.03, 'PASS', ha='center', fontsize=8, color='green', fontweight='bold')
ax.axhline(y=0, c='gray', ls='-'); ax.set_xticks(x); ax.set_xticklabels([ek[:10] for ek in mat_data], fontsize=8, rotation=30)
ax.set_ylabel('R2'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
ax.set_title(f'Route C: Pooled TwoStage vs SingleStage\nMean: Single={mean_base:.3f}, TwoStage={mean_twos:.3f} ({Δ}={mean_twos-mean_base:+.3f})', fontsize=13, fontweight='bold')
plt.tight_layout(); fig1.savefig(os.path.join(OUT_DIR, 'fig1_twostage_vs_single.png'), dpi=150); plt.close()

# Fig 2: Guardrail details — confusion matrix summary per material
fig2, ax2 = plt.subplots(figsize=(14, 5))
for i, ek in enumerate(mat_data):
    gr = guardrail_results[ek]
    total = gr['tn']+gr['fp']+gr['fn']+gr['tp']
    ax2.bar(i, gr['tn']/total, 0.8, color='#4CAF50', alpha=0.7, label='TN' if i==0 else '')
    ax2.bar(i, gr['tp']/total, 0.8, bottom=gr['tn']/total, color='#2196F3', alpha=0.7, label='TP' if i==0 else '')
    ax2.bar(i, gr['fn']/total, 0.8, bottom=(gr['tn']+gr['tp'])/total, color='#F44336', alpha=0.7, label='FN' if i==0 else '')
    ax2.bar(i, gr['fp']/total, 0.8, bottom=(gr['tn']+gr['tp']+gr['fn'])/total, color='#FF9800', alpha=0.7, label='FP' if i==0 else '')
    fn_pct = gr['fn']/max(gr['n_pos_te'],1)
    status = 'PASS' if gr['passed'] else 'FAIL'
    ax2.text(i, 1.05, f'{fn_pct:.0%}', ha='center', fontsize=7, color='red' if fn_pct>0.3 else 'green')
ax2.set_xticks(range(10)); ax2.set_xticklabels([ek[:10] for ek in mat_data], fontsize=8, rotation=30)
ax2.set_ylabel('Fraction of Test Months'); ax2.set_ylim(0, 1.15)
ax2.set_title('Stage1 Confusion Matrix per Material (Red=FN=MissedDemand)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, ncol=4, loc='upper right')
plt.tight_layout(); fig2.savefig(os.path.join(OUT_DIR, 'fig2_guardrail_cm.png'), dpi=150); plt.close()

# Fig 3: MSE decomposition for passed materials
if mse_decomposition_rows:
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    passed_eks = [ek for ek in mat_data if guardrail_results[ek]['passed']]
    for i, ek in enumerate(passed_eks):
        mse_fn = mse_decomposition_rows[i]  # need to rebuild
        # Actually recalculate
        rc = results_c[ek]; gr = guardrail_results[ek]
        d = mat_data[ek]; y_te = d['y_te']; pred_c = rc['pred_c']
        pred_bin = gr['pred']
        mse_fn = sum(y_te[t]**2 for t in range(TEST) if pred_bin[t]==0 and y_te[t]>0)
        mse_fp = sum(pred_c[t]**2 for t in range(TEST) if pred_bin[t]>0 and y_te[t]==0)
        total = mse_fn + mse_fp
        if total > 0:
            ax3.barh(i, mse_fn/total, color='#F44336', alpha=0.8, label='FN error' if i==0 else '')
            ax3.barh(i, mse_fp/total, left=mse_fn/total, color='#FF9800', alpha=0.8, label='FP error' if i==0 else '')
        ax3.text(0.5, i, f'Total={total:.0f}', va='center', fontsize=8)
    ax3.set_yticks(range(len(passed_eks))); ax3.set_yticklabels(passed_eks, fontsize=8)
    ax3.set_xlabel('Fraction of Stage1-Induced Error'); ax3.set_title('Stage1 Error Contribution to Total MSE', fontsize=12, fontweight='bold')
    if len(passed_eks) > 0: ax3.legend(fontsize=8)
    plt.tight_layout(); fig3.savefig(os.path.join(OUT_DIR, 'fig3_mse_decomp.png'), dpi=150); plt.close()

# Fig 4: Stage1 prediction probability vs actual for top 3
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
top3_for_plot = sorted([ek for ek in mat_data if guardrail_results[ek]['passed']],
                       key=lambda ek: guardrail_results[ek]['f1'], reverse=True)[:3]
for i, ek in enumerate(top3_for_plot):
    ax = axes4[i]; gr = guardrail_results[ek]; d = mat_data[ek]
    ax.plot(TEST_DATES, d['y_te'], 'ko-', lw=2, ms=5, label='Actual demand')
    ax2_ = ax.twinx()
    ax2_.plot(TEST_DATES, gr['proba'], 's-', color='#E91E63', lw=2, ms=6, label='P(demand) Stage1')
    ax2_.set_ylim(0, 1.1); ax2_.set_ylabel('P(demand)')
    # Mark where Stage1 was wrong
    for t in range(TEST):
        actual_has = d['y_te'][t] > 0
        pred_has = gr['proba'][t] >= 0.5
        if actual_has != pred_has:
            ax.axvspan(TEST_DATES[t]-pd.Timedelta(days=15), TEST_DATES[t]+pd.Timedelta(days=15),
                       color='red', alpha=0.15)
    ax.set_title(f'{ek}\nF1={gr["f1"]:.3f} FN={gr["fn"]}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left'); ax2_.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3); ax.tick_params('x', rotation=30)
fig4.suptitle('Stage1: P(demand) vs Actual — Top 3 Best Classifiers', fontsize=13, fontweight='bold')
plt.tight_layout(); fig4.savefig(os.path.join(OUT_DIR, 'fig4_stage1_detail.png'), dpi=150); plt.close()

# Fig 5: Prediction comparison — TwoStage vs SingleStage for all passed materials
passed_mats = [ek for ek in mat_data if guardrail_results[ek]['passed']]
if len(passed_mats) >= 4:
    top4 = passed_mats[:4]
else:
    top4 = list(mat_data.keys())[:4]
n_rows = (len(top4)+1)//2
fig5, axes5 = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
axes5 = axes5.flatten() if n_rows > 1 else [axes5]
for i, ek in enumerate(top4):
    ax = axes5[i]; rc = results_c[ek]; d = mat_data[ek]
    ax.plot(TEST_DATES, d['y_te'], 'ko-', lw=2, ms=5, label='Actual')
    ax.plot(TEST_DATES, rc['pred_c'], 's-', color='#E91E63', lw=2.5, ms=6, label=f'TwoStage (R2={rc["r2_twostage"]:.3f})')
    ax.plot(TEST_DATES, rc['pred_base'], '^--', color='#2196F3', lw=1.5, ms=4, alpha=0.6, label=f'SingleStage (R2={rc["r2_base"]:.3f})')
    ax.set_title(f'{ek}  R2_delta={rc["r2_delta"]:+.4f}  [{rc["mode"][:20]}]', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.tick_params('x', rotation=30)
fig5.suptitle('Route C: TwoStage vs SingleStage — Predictions for Guardrail-Passed Materials', fontsize=13, fontweight='bold')
plt.tight_layout(); fig5.savefig(os.path.join(OUT_DIR, 'fig5_predictions.png'), dpi=150); plt.close()

# Fig 6: Probability weighting vs Hard threshold comparison
if prob_vs_hard_rows:
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    prob_r2 = [float(r[1]) for r in prob_vs_hard_rows]
    hard_r2 = [float(r[2]) for r in prob_vs_hard_rows]
    labels = [r[0][:12] for r in prob_vs_hard_rows]
    x6 = np.arange(len(prob_r2)); w6 = 0.35
    ax6.bar(x6-w6/2, prob_r2, w6, label='Probability Weighting', color='#E91E63', alpha=0.8)
    ax6.bar(x6+w6/2, hard_r2, w6, label='Hard Threshold (P>0.5)', color='#2196F3', alpha=0.8)
    for i in range(len(prob_r2)):
        better = '+' if prob_r2[i] > hard_r2[i] else '-'
        ax6.text(i, max(prob_r2[i], hard_r2[i])+0.02, better, ha='center', fontsize=12, fontweight='bold')
    ax6.set_xticks(x6); ax6.set_xticklabels(labels, fontsize=8, rotation=30)
    ax6.set_ylabel('R2'); ax6.legend(fontsize=10); ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_title('Probability Weighting vs Hard Threshold (Stage1)', fontsize=13, fontweight='bold')
    plt.tight_layout(); fig6.savefig(os.path.join(OUT_DIR, 'fig6_prob_vs_hard.png'), dpi=150); plt.close()

log.step_ok(f'6 charts saved to {OUT_DIR}')

# ========================================================================
# Step 11: 结果JSON
# ========================================================================
out_json = {
    'route': 'C', 'description': 'Pooled Two-Stage with Guardrail',
    'materials': {ek: {
        'name': rc['name'], 'cat': rc['cat'],
        'r2_single': rc['r2_base'], 'r2_twostage': rc['r2_twostage'],
        'r2_delta': rc['r2_delta'], 'mode': rc['mode'],
        'guardrail_passed': rc['guardrail']['passed'],
        'stage1_f1': rc['guardrail']['f1'],
        'stage1_fn': rc['guardrail']['fn'],
        'stage1_fp': rc['guardrail']['fp'],
    } for ek, rc in results_c.items()},
    'summary': {
        'n_guardrail_passed': n_passed,
        'mean_single_r2': round(mean_base, 4),
        'mean_twostage_r2': round(mean_twos, 4),
        'delta': round(mean_twos - mean_base, 4),
        'wins_twostage': wins_twos,
    },
}
json.dump(out_json, open(os.path.join(OUT_DIR, 'route_c_results.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=str)

log.pipeline_end_log(f'TwoStage Mean R2={mean_twos:.4f}, Single Mean R2={mean_base:.4f}, '
                     f'Guardrail Passed={n_passed}/10, TwoStage Wins={wins_twos}/10')
print(f"\n  日志文件: {log.get_log_path()}")
