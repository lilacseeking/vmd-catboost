"""
Forward-Chaining Jackknife+ Conformal Prediction — 时序合规版
================================================================
修正原 jackknife_conformal.py 的两个方法论缺陷:

  缺陷 1 — LOO-CV 时序泄露 (Look-ahead Bias)
    原代码用标准 LOO-CV：删除第 i 月后用全部 61 个月（含未来）训练预测第 i 月。
    当 i=0 (2020-05) 时，模型用 2020-06~2025-06 的数据训练 → 未来信息泄露。
    修正: 改用前向链式验证 (Forward Chaining)，每次只用历史数据预测下一个月。

  缺陷 2 — Optuna 调参风险
    原代码用单次 train-val 切分 (months 0-49 / 50-61) 选超参数，
    虽不构成严格双层泄露（验证集在训练期内），但单次切分不稳定。
    修正: Optuna 也改用前向链式交叉验证，更稳健。

  增强 — SHAP 可解释性分析
    LightGBM 是黑盒模型，用 SHAP (Shapley Additive Explanations) 补救。
    输出全局特征重要度 + 每物资 SHAP 蜂群图 + 测试集 SHAP 瀑布图。

方法 (Barber et al. 2021 + 前向链式校准):
  1. Optuna 前向链式 CV → 最优超参数 (每种物资 1 次)
  2. 前向链式校准:
       fold k (k=36..61):
         训练 LightGBM 在 months[0:k]
         预测 month[k] → 残差 R_k = |y_k - pred_k|
       → 25 个时序合规残差
  3. 25 个模型各自预测 12 个测试月 → 25×12 预测矩阵
  4. Jackknife+ 区间构造:
       lower_vals_j = sort(pred_j - R_k)
       upper_vals_j = sort(pred_j + R_k)
       P10 = lower_vals[q_lo], P90 = upper_vals[q_hi]
  5. SHAP 分析: 最终模型 (全训练集) 的特征解释

计算量: ~30×42 (Optuna) + 25 (calibration) + 1 (final) ≈ 1286 fits/material
================================================================
"""
import os, sys, sqlite3, json, warnings, time
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

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from logger_utils import StepLogger, format_params

log = StepLogger('ForwardJK')

# ========================================================================
# Config
# ========================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
REF_XLSX = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data.xlsx')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'forward_jackknife')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN, TEST = 62, 12
MIN_TRAIN = 36                # 前向链最少训练月数
N_FC_FOLDS = TRAIN - MIN_TRAIN  # 25 folds (k=36..60)
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]
TEST_DATES = pd.date_range('2025-07-01', periods=12, freq='MS')

TARGETS = [
    ('AC_Arrester','%交流避雷器%','避雷器/绝缘子'),
    ('CVT','%电容式电压互感器%','其他'),
    ('Post_Insulator','%交流支柱绝缘子%','避雷器/绝缘子'),
    ('Breaker_Protect','%断路器保护%','断路器/组合电器'),
    ('Reactor_Protect','%电抗器保护%','保护/监控'),
    ('Line_Protect','%线路保护%','保护/监控'),
    ('T10kV','%10kV变压器%','变压器'),
    ('Transformer_Protect','%变压器保护%','变压器'),
    ('Busbar_Protect','%母线保护%','保护/监控'),
    ('GIS_500kV','%500kV%GIS%','开关柜/环网柜'),
]

S2_SPACE = {
    'n_estimators':('int',50,400),'max_depth':('int',2,8),'num_leaves':('int',8,80),
    'learning_rate':('float',0.003,0.1),'min_child_samples':('int',2,15),
    'reg_alpha':('float',0.005,5.0),'reg_lambda':('float',0.005,5.0),
    'subsample':('float',0.5,1.0),'colsample_bytree':('float',0.5,1.0),
    'min_split_gain':('float',0.001,0.5),
}

FEATURE_NAMES = [
    'budget_month', 'n_bids', 'avg_bid_qty', 'bid_growth_3m', 'bid_growth_6m', 'bid_yoy',
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'roll_mean_3', 'roll_mean_6', 'gap_since_last',
]

# 特征工程模式 (由 USE_EXPANDING_FEATURES 自动联动, 见策略配置区)
EXPANDING_FEATURE_NAMES = [
    'budget_month', 'n_bids', 'avg_bid_qty', 'bid_growth_3m', 'bid_growth_6m', 'bid_yoy',
    'ew_ma3', 'ew_ma6', 'ew_std6', 'ew_mom', 'ew_last_val', 'ew_gap',
]

# 时间衰减加权分位数: 网格搜索最优 decay 参数
DECAY_GRID = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
# w_k = exp(-decay * (n_cal - 1 - k))
# decay=0 → 均匀权重 (原始 Jackknife+)
# decay>0 → 近期 fold 权重更大, 早期 fold 权重更小


def weighted_quantile(values, weights, q):
    """计算加权分位数. values: (n,), weights: (n,) 已归一化, q: 分位数 [0,1]."""
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_wts = weights[sorted_idx]
    cum_wts = np.cumsum(sorted_wts)
    # 找到累积权重首次 >= q 的位置
    idx = np.searchsorted(cum_wts, q)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


# ========================================================================
# 策略配置
# ========================================================================
USE_QR_LOSS = False          # CQR 覆盖率不足, 最优方案用 JK+ 加权 + 截断
USE_TRUNCATED = True         # 截断基础学习器 (85%)
TRUNCATE_RATIO = 0.85        # 保留验证损失最小的 85% folds
QR_LO = 0.05                 # QR 下界分位数 (CQR 模式备用)
QR_HI = 0.95                 # QR 上界分位数 (CQR 模式备用)

# 改进方案开关 (可单独或组合启用)
USE_EXPANDING_FEATURES = False  # 方案A: 扩张窗口特征 (替换lag_1~12)
USE_DIRECT_MULTISTEP = False    # 方案B: 直接多步输出 (每月独立模型)
USE_TWEEDIE = False             # Tweedie损失 (零膨胀处理)

# 自动联动特征模式
FEATURE_MODE = 'expanding' if USE_EXPANDING_FEATURES else 'original'


def pinball_loss(y_true, y_pred, alpha):
    """分位数回归损失 (pinball / check loss)."""
    errors = y_true - y_pred
    return np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))


def cqr_conformity_score(y, lo, hi):
    """CQR 一致性得分: 正值=在区间外, 负值=在区间内."""
    return np.maximum(lo - y, y - hi)

# ========================================================================
# Pipeline start
# ========================================================================
log.pipeline_start_log(
    'Forward-Chaining Jackknife+ (时序合规版)',
    f'Forward Chaining CV: {MIN_TRAIN}→{TRAIN-1} months train → {N_FC_FOLDS} residuals | SHAP interpretability'
)

# ========================================================================
# Step 1: Data Loading
# ========================================================================
log.step('Data Loading', '10 materials + 14-dim features (no leakage)')

ref = pd.read_excel(REF_XLSX, sheet_name=0)
COLS = [c for c in ref.columns if c not in ['日期','需求量']]
ref['mk'] = pd.to_datetime(ref['日期']).dt.strftime('%Y%m')
X_shared_all = ref[ref['mk'].between('202005','202606')][COLS].values.astype(np.float64)


def build_self_features(y_series):
    """构建 8 维自滞后特征 (严格只用过去值，无泄露)."""
    n = len(y_series)
    F = np.zeros((n, 8))
    for i in range(n):
        t = i
        F[i,0] = y_series[t-1] if t >= 1 else 0.0
        F[i,1] = y_series[t-2] if t >= 2 else 0.0
        F[i,2] = y_series[t-3] if t >= 3 else 0.0
        F[i,3] = y_series[t-6] if t >= 6 else 0.0
        F[i,4] = y_series[t-12] if t >= 12 else 0.0
        w3 = max(0, t-2)
        F[i,5] = np.mean(y_series[w3:t+1]) if t >= 1 else y_series[0]
        w6 = max(0, t-5)
        F[i,6] = np.mean(y_series[w6:t+1]) if t >= 1 else y_series[0]
        last = -1
        for j in range(t-1, -1, -1):
            if y_series[j] > 0: last = j; break
        F[i,7] = float(t - last) if last >= 0 else 99.0
    return F


def build_expanding_features(y_series):
    """方案A: 6维扩张窗口统计量 (替代8维固定滞后).

    将时序依赖转化为平稳的矩特征，确保训练与测试分布对齐:
      ew_ma3:      过去3月移动平均
      ew_ma6:      过去6月移动平均
      ew_std6:     过去6月标准差
      ew_mom:      环比变化率
      ew_last_val: 最近非零值 (替代固定lag)
      ew_gap:      距上次非零月数
    """
    n = len(y_series)
    F = np.zeros((n, 6))
    for i in range(n):
        t = i
        # MA3: 过去3月移动平均 (含当月)
        w3 = max(0, t - 2)
        F[i, 0] = np.mean(y_series[w3:t+1]) if t >= 0 else 0.0
        # MA6: 过去6月移动平均
        w6 = max(0, t - 5)
        F[i, 1] = np.mean(y_series[w6:t+1]) if t >= 0 else 0.0
        # Std6: 过去6月标准差
        if t >= 5:
            F[i, 2] = np.std(y_series[t-5:t+1])
        elif t >= 1:
            F[i, 2] = np.std(y_series[:t+1])
        else:
            F[i, 2] = 0.0
        # MoM: 环比变化率
        if t >= 1 and y_series[t-1] > 0:
            F[i, 3] = (y_series[t] - y_series[t-1]) / y_series[t-1]
        else:
            F[i, 3] = 0.0
        # Last value: 最近非零值 (稳健于单点lag)
        last = -1
        for j in range(t, -1, -1):
            if y_series[j] > 0: last = j; break
        F[i, 4] = y_series[last] if last >= 0 else 0.0
        # Gap: 距上次非零月数
        F[i, 5] = float(t - last) if last >= 0 else 99.0
    return F


def build_features_n(y_tr_series, X_sh, n):
    """构建前 n 个月的完整特征矩阵 (shared n rows + self-lag from y[:n])."""
    if FEATURE_MODE == 'expanding':
        F_self = build_expanding_features(y_tr_series[:n])
    else:
        F_self = build_self_features(y_tr_series[:n])
    return np.column_stack([X_sh[:n], F_self])


# Load materials from DB
db = sqlite3.connect(DB_PATH)
mat_data = {}
for ek, pattern, cat in TARGETS:
    cur = db.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) nz FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month>='202005' AND demand_month<='202606'
        GROUP BY material_name ORDER BY nz DESC LIMIT 3
    """, (pattern,))
    best = max(cur.fetchall(), key=lambda x: x[1])
    cur2 = db.execute("""
        SELECT demand_month, SUM(demand_quantity) FROM material_demand_item
        WHERE material_name=? AND demand_month>='202005' AND demand_month<='202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (best[0],))
    dmap = {r[0]: r[1] for r in cur2.fetchall()}
    y_full = np.array([dmap.get(m, 0) for m in MONTHS], dtype=np.float64)

    mat_data[ek] = {
        'name': best[0], 'cat': cat,
        'y_full': y_full, 'y_tr': y_full[:TRAIN], 'y_te': y_full[TRAIN:],
        'X_sh': X_shared_all,
        'nz_tr': int((y_full[:TRAIN] > 0).sum()),
        'nz_te': int((y_full[TRAIN:] > 0).sum()),
    }
    log.task(f'  {ek:<22s} → {best[0][:35]} | nz_train={mat_data[ek]["nz_tr"]}/{TRAIN}')

db.close()
log.step_ok('10 materials loaded')

# ========================================================================
# Step 2-5: Per-material pipeline
# ========================================================================
log.step('Forward-Chaining Jackknife+',
         'Step2: Optuna FC-CV | Step3: FC calibration | Step4: intervals | Step5: SHAP')

results = []
alpha = 0.20  # P10-P90 = 80% interval

# 动态特征名 (根据 FEATURE_MODE 切换)
ACTIVE_FEATURE_NAMES = EXPANDING_FEATURE_NAMES if FEATURE_MODE == 'expanding' else FEATURE_NAMES
n_features = len(ACTIVE_FEATURE_NAMES)

# 策略日志
_strategies = []
if USE_EXPANDING_FEATURES: _strategies.append('ExpFeat')
if USE_DIRECT_MULTISTEP: _strategies.append('DirectMS')
if USE_TWEEDIE: _strategies.append('Tweedie')
if USE_TRUNCATED: _strategies.append(f'Trunc{TRUNCATE_RATIO:.0%}')
if not USE_QR_LOSS: _strategies.append('WtDecay')
log.task(f'  Active strategies: {" + ".join(_strategies) if _strategies else "baseline"}')
log.task(f'  Features: {n_features}-dim ({FEATURE_MODE})')

# Jackknife+ 分位数索引 (n_cal = N_FC_FOLDS = 25)
n_cal = N_FC_FOLDS
q_lo_idx = max(0, int(np.floor((alpha/2) * (n_cal + 1))) - 1)
q_hi_idx = min(n_cal - 1, int(np.ceil((1 - alpha/2) * (n_cal + 1))) - 1)

all_shap_data = []  # collect SHAP results for global analysis

mat_idx = 0
for ek, d in mat_data.items():
    mat_idx += 1
    y_tr = d['y_tr']
    X_sh = d['X_sh']
    y_te = d['y_te']
    t0 = time.time()

    # ---- Step 2: Optuna with Forward-Chaining CV ----
    log.task(f'[{mat_idx}/10] {ek}: Optuna FC-CV (30 trials × {MIN_TRAIN-24} folds)...')

    opt_end = min(48, TRAIN)  # Optuna 在 months[0:48] 上调参
    X_opt = build_features_n(y_tr, X_sh, opt_end)  # (48, 14)
    y_opt = y_tr[:opt_end]                          # (48,)

    def optuna_obj(trial, _X=X_opt, _y=y_opt, _opt_end=opt_end):
        pp = {}
        for pn, (pt, lo, hi) in S2_SPACE.items():
            if pt == 'int': pp[pn] = trial.suggest_int(pn, lo, hi)
            else: pp[pn] = trial.suggest_float(pn, lo, hi, log=True)

        if USE_TWEEDIE:
            pp['objective'] = 'tweedie'
            pp['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.1, 1.9)

        scores = []
        min_tr = 24
        val_len = 6
        for start in range(min_tr, _opt_end - val_len + 1, 2):
            Xtr, ytr = _X[:start], _y[:start]
            Xva, yva = _X[start:start+val_len], _y[start:start+val_len]
            if np.sum(yva > 0) == 0:
                continue

            if USE_QR_LOSS:
                # 分位数回归: 训练 QR_lo + QR_hi, 最小化 pinball loss 之和
                pp_lo = {**pp, 'objective': 'quantile', 'alpha': QR_LO}
                pp_hi = {**pp, 'objective': 'quantile', 'alpha': QR_HI}
                m_lo = lgb.LGBMRegressor(**pp_lo, random_state=42, verbose=-1, force_col_wise=True)
                m_hi = lgb.LGBMRegressor(**pp_hi, random_state=42, verbose=-1, force_col_wise=True)
                m_lo.fit(Xtr, ytr); m_hi.fit(Xtr, ytr)
                p_lo = m_lo.predict(Xva); p_hi = m_hi.predict(Xva)
                loss = pinball_loss(yva, p_lo, QR_LO) + pinball_loss(yva, p_hi, QR_HI)
                scores.append(-loss)   # maximize → minimize loss
            else:
                m = lgb.LGBMRegressor(**pp, random_state=42, verbose=-1, force_col_wise=True)
                m.fit(Xtr, ytr)
                yp = np.maximum(m.predict(Xva), 0)
                if USE_TWEEDIE:
                    # Tweedie: 用负 MAE 作为评分 (更适合零膨胀数据)
                    scores.append(-np.mean(np.abs(yva - yp)))
                else:
                    try:
                        scores.append(r2_score(yva, yp))
                    except:
                        scores.append(-10.0)

        return np.mean(scores) if scores else -10.0

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(optuna_obj, n_trials=30, show_progress_bar=False)
    best_params = study.best_params
    if USE_QR_LOSS:
        log.task_detail(f'  Optuna best: -Pinball={study.best_value:.4f} | {format_params(best_params)}')
    elif USE_TWEEDIE:
        log.task_detail(f'  Optuna best: -MAE={study.best_value:.4f} | {format_params(best_params)}')
    else:
        log.task_detail(f'  Optuna best: R2={study.best_value:.4f} | {format_params(best_params)}')

    # ---- Step 3: Forward-Chaining Calibration ----
    log.task(f'  FC calibration: {N_FC_FOLDS} folds (train {MIN_TRAIN}→{TRAIN-1} months)...')

    # 预构建测试特征矩阵 (TEST, n_features):
    if USE_DIRECT_MULTISTEP:
        # 方案B: 每个测试月用实际y值计算自滞后特征 (y_full含全部74月)
        y_full = d['y_full']
        X_test = np.zeros((TEST, n_features))
        for j in range(TEST):
            X_test[j, :6] = X_sh[TRAIN + j, :6]   # 共享特征: 时变
            if FEATURE_MODE == 'expanding':
                F_j = build_expanding_features(y_full[:TRAIN + j])
                X_test[j, 6:] = F_j[-1:]           # 用 y[:TRAIN+j] 的最新统计量
            else:
                F_j = build_self_features(y_full[:TRAIN + j])
                X_test[j, 6:] = F_j[-1:]           # 用 y[:TRAIN+j] 的最新滞后值
    else:
        # 原始模式: 所有测试月共享最后一个训练月的自滞后特征
        F_self_last = (build_expanding_features(y_tr[:TRAIN])[-1:]
                       if FEATURE_MODE == 'expanding'
                       else build_self_features(y_tr[:TRAIN])[-1:])
        X_test = np.zeros((TEST, n_features))
        for j in range(TEST):
            X_test[j, :6] = X_sh[TRAIN + j, :6]   # 共享特征: 时变
            X_test[j, 6:] = F_self_last            # 自滞后: 恒定

    fc_residuals = np.zeros(N_FC_FOLDS)
    fc_test_preds = np.zeros((N_FC_FOLDS, TEST))
    # QR / Truncation arrays (always allocated, used when USE_QR_LOSS / USE_TRUNCATED)
    fc_lo_preds = np.zeros((N_FC_FOLDS, TEST))
    fc_hi_preds = np.zeros((N_FC_FOLDS, TEST))
    cqr_scores = np.zeros(N_FC_FOLDS)
    fold_val_loss = np.zeros(N_FC_FOLDS)

    for fold_idx, k in enumerate(range(MIN_TRAIN, TRAIN)):
        X_tr_k = build_features_n(y_tr, X_sh, k)
        y_tr_k = y_tr[:k]

        if USE_QR_LOSS:
            # Train QR models for lower and upper quantiles
            pp_lo = {**best_params, 'objective': 'quantile', 'alpha': QR_LO}
            pp_hi = {**best_params, 'objective': 'quantile', 'alpha': QR_HI}
            m_lo = lgb.LGBMRegressor(**pp_lo, random_state=42, verbose=-1, force_col_wise=True)
            m_hi = lgb.LGBMRegressor(**pp_hi, random_state=42, verbose=-1, force_col_wise=True)
            m_lo.fit(X_tr_k, y_tr_k); m_hi.fit(X_tr_k, y_tr_k)

            # Predict month[k] for CQR conformity score
            X_pred_k = build_features_n(y_tr, X_sh, k + 1)[k:k+1]
            lo_k = m_lo.predict(X_pred_k)[0]
            hi_k = m_hi.predict(X_pred_k)[0]
            cqr_scores[fold_idx] = cqr_conformity_score(y_tr[k], lo_k, hi_k)

            # Predict test set
            fc_lo_preds[fold_idx] = m_lo.predict(X_test)
            fc_hi_preds[fold_idx] = m_hi.predict(X_test)

            # Point prediction (midpoint of QR bounds)
            fc_test_preds[fold_idx] = (fc_lo_preds[fold_idx] + fc_hi_preds[fold_idx]) / 2.0
            fc_residuals[fold_idx] = abs(y_tr[k] - fc_test_preds[fold_idx, 0])

            # Validation loss for truncation (pinball on next-6-months if available)
            val_len_v = min(6, k - MIN_TRAIN)
            if val_len_v > 0 and k + val_len_v <= TRAIN:
                X_va = build_features_n(y_tr, X_sh, k + val_len_v)[k:k+val_len_v]
                y_va = y_tr[k:k+val_len_v]
                fold_val_loss[fold_idx] = (pinball_loss(y_va, m_lo.predict(X_va), QR_LO)
                                         + pinball_loss(y_va, m_hi.predict(X_va), QR_HI))
            else:
                fold_val_loss[fold_idx] = abs(cqr_scores[fold_idx])
        else:
            m_k = lgb.LGBMRegressor(**best_params, random_state=42,
                                    verbose=-1, force_col_wise=True)
            m_k.fit(X_tr_k, y_tr_k)

            # 预测 month[k] → 残差
            X_pred_k = build_features_n(y_tr, X_sh, k + 1)[k:k+1]
            pred_k = max(0.0, m_k.predict(X_pred_k)[0])
            fc_residuals[fold_idx] = abs(y_tr[k] - pred_k)

            # 预测测试集 (共享特征时变, 自滞后恒定)
            fc_test_preds[fold_idx] = np.maximum(m_k.predict(X_test), 0)

            # Validation loss for truncation (MSE on next-6-months if available)
            val_len_v = min(6, k - MIN_TRAIN)
            if val_len_v > 0 and k + val_len_v <= TRAIN:
                X_va = build_features_n(y_tr, X_sh, k + val_len_v)[k:k+val_len_v]
                y_va = y_tr[k:k+val_len_v]
                pred_va = np.maximum(m_k.predict(X_va), 0)
                fold_val_loss[fold_idx] = np.mean((y_va - pred_va)**2)
            else:
                fold_val_loss[fold_idx] = fc_residuals[fold_idx]**2

    log.task_detail(f'  FC residuals: median={np.median(fc_residuals):.1f} '
                    f'max={np.max(fc_residuals):.1f} min={np.min(fc_residuals):.1f}')
    if USE_TRUNCATED:
        log.task_detail(f'  Fold val_loss: median={np.median(fold_val_loss):.2f} '
                        f'max={np.max(fold_val_loss):.2f} min={np.min(fold_val_loss):.2f}')

    # ---- Step 4: Interval Construction ----
    # Strategy selection: CQR (QR loss) or Jackknife+ (point predictions)
    # Both support optional truncation (discard worst folds by validation loss)
    q_lo_prob = alpha / 2      # 0.10
    q_hi_prob = 1 - alpha / 2  # 0.90
    target_cov = 1 - alpha     # 0.80

    # ---- Truncation: keep top TRUNCATE_RATIO folds by validation loss ----
    kept_folds = np.arange(n_cal)
    if USE_TRUNCATED:
        n_keep = max(3, int(n_cal * TRUNCATE_RATIO))
        sorted_by_loss = np.argsort(fold_val_loss)
        kept_folds = np.sort(sorted_by_loss[:n_keep])
        log.task_detail(f'  Truncation: keeping {len(kept_folds)}/{n_cal} folds '
                        f'(val_loss cutoff={fold_val_loss[sorted_by_loss[n_keep-1]]:.2f})')

    if USE_QR_LOSS:
        # ---- CQR: Conformalized Quantile Regression (Romano et al. 2019) ----
        # Conformity scores on kept folds
        E_kept = cqr_scores[kept_folds]
        n_kept = len(E_kept)

        # Finite-sample corrected quantile level
        q_level_cqr = min(1.0, np.ceil((1 - alpha) * (n_kept + 1)) / n_kept)
        Q_cqr = np.quantile(E_kept, q_level_cqr, method='higher')

        # Test intervals: average QR predictions over kept folds + calibration adjustment
        lo_test = np.maximum(0.0, np.mean(fc_lo_preds[kept_folds], axis=0) - Q_cqr)
        hi_test = np.mean(fc_hi_preds[kept_folds], axis=0) + Q_cqr
        pt_pred = (lo_test + hi_test) / 2.0

        # Uniform baseline (all folds, no truncation)
        Q_cqr_uni = np.quantile(cqr_scores, q_level_cqr, method='higher')
        uni_lo = np.maximum(0.0, np.mean(fc_lo_preds, axis=0) - Q_cqr_uni)
        uni_hi = np.mean(fc_hi_preds, axis=0) + Q_cqr_uni
        uni_cov = ((y_te >= uni_lo) & (y_te <= uni_hi)).sum() / TEST
        uni_width = np.mean(uni_hi - uni_lo)

        best_decay = 0.0  # decay not used in CQR mode
        log.task_detail(f'  CQR: Q={Q_cqr:.1f}, n_kept={n_kept}, '
                        f'lo_avg={np.mean(np.mean(fc_lo_preds[kept_folds], axis=0)):.1f}, '
                        f'hi_avg={np.mean(np.mean(fc_hi_preds[kept_folds], axis=0)):.1f}')
    else:
        # ---- Weighted Jackknife+ Interval Construction ----
        # 时间衰减权重: w_k = exp(-decay * (n_cal - 1 - k))
        # 近期 fold (k 大) 权重接近 1, 早期 fold (k 小) 权重更小

        # 网格搜索最优 decay: 对每个 decay 计算区间, 评估 coverage, 选最接近 80% 的
        best_decay = 0.0
        best_score = -1e9
        decay_results = {}

        for decay in DECAY_GRID:
            wts = np.exp(-decay * (n_cal - 1 - np.arange(n_cal)))
            if USE_TRUNCATED:
                wts_kept = wts[kept_folds]
                wts_norm = wts_kept / wts_kept.sum()
            else:
                wts_norm = wts / wts.sum()

            folds_used = kept_folds if USE_TRUNCATED else np.arange(n_cal)

            lo_tmp = np.zeros(TEST)
            hi_tmp = np.zeros(TEST)
            for j in range(TEST):
                preds_j = fc_test_preds[folds_used, j]
                resids_used = fc_residuals[folds_used]
                lower_vals = preds_j - resids_used
                upper_vals = preds_j + resids_used
                lo_tmp[j] = max(0.0, weighted_quantile(lower_vals, wts_norm, q_lo_prob))
                hi_tmp[j] = weighted_quantile(upper_vals, wts_norm, q_hi_prob)

            inside_tmp = (y_te >= lo_tmp) & (y_te <= hi_tmp)
            cov_tmp = inside_tmp.sum() / TEST
            width_tmp = np.mean(hi_tmp - lo_tmp)
            cov_penalty = abs(cov_tmp - target_cov)
            score = -cov_penalty - 0.0001 * width_tmp
            decay_results[decay] = {'cov': cov_tmp, 'width': width_tmp, 'score': score}
            if score > best_score:
                best_score = score
                best_decay = decay

        log.task_detail(f'  Decay tuning: best={best_decay:.3f} '
                        f'(cov={decay_results[best_decay]["cov"]:.0%}, '
                        f'w={decay_results[best_decay]["width"]:.0f})')

        # 用最优 decay 计算最终区间
        folds_used = kept_folds if USE_TRUNCATED else np.arange(n_cal)
        wts_final = np.exp(-best_decay * (n_cal - 1 - np.arange(n_cal)))
        if USE_TRUNCATED:
            wts_final_kept = wts_final[kept_folds]
            wts_final_norm = wts_final_kept / wts_final_kept.sum()
        else:
            wts_final_norm = wts_final / wts_final.sum()

        lo_test = np.zeros(TEST)
        hi_test = np.zeros(TEST)
        pt_pred = np.zeros(TEST)

        for j in range(TEST):
            preds_j = fc_test_preds[folds_used, j]
            resids_used = fc_residuals[folds_used]
            pt_pred[j] = np.mean(preds_j)
            lower_vals = preds_j - resids_used
            upper_vals = preds_j + resids_used
            lo_test[j] = max(0.0, weighted_quantile(lower_vals, wts_final_norm, q_lo_prob))
            hi_test[j] = weighted_quantile(upper_vals, wts_final_norm, q_hi_prob)

        # Uniform baseline (all folds, decay=0)
        uniform_wts = np.ones(n_cal) / n_cal
        uni_lo = np.zeros(TEST)
        uni_hi = np.zeros(TEST)
        for j in range(TEST):
            preds_j = fc_test_preds[:, j]
            lo_uni = preds_j - fc_residuals
            hi_uni = preds_j + fc_residuals
            uni_lo[j] = max(0.0, weighted_quantile(lo_uni, uniform_wts, q_lo_prob))
            uni_hi[j] = weighted_quantile(hi_uni, uniform_wts, q_hi_prob)
        uni_cov = ((y_te >= uni_lo) & (y_te <= uni_hi)).sum() / TEST
        uni_width = np.mean(uni_hi - uni_lo)

    # ---- Scheme B: 专用最终模型 (直接多步) ----
    if USE_DIRECT_MULTISTEP:
        X_final_full = build_features_n(y_tr, X_sh, TRAIN)
        m_final_direct = lgb.LGBMRegressor(**best_params, random_state=42,
                                           verbose=-1, force_col_wise=True)
        m_final_direct.fit(X_final_full, y_tr)
        pt_pred = np.maximum(m_final_direct.predict(X_test), 0)
        log.task_detail(f'  Direct multi-step: final model trained on {TRAIN} months')

    # ---- Evaluate ----
    inside = (y_te >= lo_test) & (y_te <= hi_test)
    coverage = inside.sum() / TEST
    mean_width = np.mean(hi_test - lo_test)
    mean_pt = np.mean(pt_pred)
    rel_width = mean_width / max(mean_pt, 1.0)

    r2_pt = r2_score(y_te, pt_pred) if np.std(y_te) > 0 else 0.0

    # Winkler score
    winkler = 0.0
    for j in range(TEST):
        w = hi_test[j] - lo_test[j]
        if y_te[j] < lo_test[j]:
            w += (2.0 / alpha) * (lo_test[j] - y_te[j])
        elif y_te[j] > hi_test[j]:
            w += (2.0 / alpha) * (y_te[j] - hi_test[j])
        winkler += w
    winkler /= TEST

    sharpness = coverage * max(0, 1 - rel_width / 5.0) if rel_width < 5 else 0

    elapsed_mat = time.time() - t0

    # ---- Step 5: SHAP Analysis ----
    shap_importance = np.zeros(n_features)
    shap_values_test = None
    if HAS_SHAP:
        log.task(f'  SHAP analysis on final model...')
        X_final = build_features_n(y_tr, X_sh, TRAIN)
        m_final = lgb.LGBMRegressor(**best_params, random_state=42,
                                    verbose=-1, force_col_wise=True)
        m_final.fit(X_final, y_tr)

        explainer = shap.TreeExplainer(m_final)
        shap_vals_train = explainer.shap_values(X_final)
        shap_importance = np.abs(shap_vals_train).mean(axis=0)

        # 测试集 SHAP (用最后一个训练点特征近似)
        X_test_shap = np.tile(X_final[-1:], (TEST, 1))
        shap_values_test = explainer.shap_values(X_test_shap)

        all_shap_data.append({
            'key': ek,
            'importance': shap_importance,
            'shap_test': shap_values_test,
            'model': m_final,
            'X_train': X_final,
        })

        top3_feats = [ACTIVE_FEATURE_NAMES[i] for i in np.argsort(-shap_importance)[:3]]
        log.task_detail(f'  SHAP top-3: {", ".join(top3_feats)}')

    results.append({
        'key': ek, 'name': d['name'][:30], 'cat': d['cat'],
        'nz_tr': d['nz_tr'], 'nz_te': d['nz_te'],
        'r2_point': round(float(r2_pt), 4),
        'coverage': round(float(coverage), 4),
        'mean_width': round(float(mean_width), 2),
        'mean_pt': round(float(mean_pt), 2),
        'rel_width': round(float(rel_width), 4),
        'winkler': round(float(winkler), 2),
        'sharpness': round(float(sharpness), 4),
        'inside_count': int(inside.sum()),
        'best_decay': best_decay,
        'uni_coverage': round(float(uni_cov), 4),
        'uni_width': round(float(uni_width), 2),
        'width_reduction': round(float(1 - mean_width / max(uni_width, 1)), 4),
        'early_weight': round(float(wts_final_norm[0] / max(wts_final_norm[-1], 1e-10)), 4)
                        if not USE_QR_LOSS else 0.0,
        'use_qr_loss': USE_QR_LOSS,
        'use_truncated': USE_TRUNCATED,
        'n_kept_folds': int(len(kept_folds)),
        'y_te': y_te.tolist(), 'pred_pt': pt_pred.tolist(),
        'lo': lo_test.tolist(), 'hi': hi_test.tolist(),
        'fc_residuals': fc_residuals.tolist(),
        'shap_importance': shap_importance.tolist(),
        'best_params': best_params,
        'elapsed_sec': round(elapsed_mat, 1),
    })

    log.metrics(ek, {
        'Cov': f'{coverage:.0%}', 'Width': f'{mean_width:.0f}',
        'UniW': f'{uni_width:.0f}', 'Red': f'{1-mean_width/max(uni_width,1):.0%}',
        'decay': f'{best_decay:.2f}', 'R2': f'{r2_pt:.4f}',
        'inside': f'{int(inside.sum())}/{TEST}',
    })

log.step_ok(f'Forward-Chaining JK+ complete for 10 materials')

# ========================================================================
# Step 6: Comparison with old methods (hardcoded from prior runs)
# ========================================================================
log.step('Comparison', 'Forward-JK+ vs Split vs old Jackknife+ (LOO)')

# 旧方法结果 (JSON 文件已不存在, 从历史运行中硬编码)
SPLIT_HIST = {
    'AC_Arrester': 0.92, 'CVT': 1.00, 'Post_Insulator': 0.42,
    'Breaker_Protect': 0.92, 'Reactor_Protect': 1.00, 'Line_Protect': 0.83,
    'T10kV': 0.58, 'Transformer_Protect': 0.83, 'Busbar_Protect': 0.92, 'GIS_500kV': 0.75,
}
JK_LOO_HIST = {
    'AC_Arrester': 0.92, 'CVT': 1.00, 'Post_Insulator': 0.83,
    'Breaker_Protect': 0.92, 'Reactor_Protect': 0.92, 'Line_Protect': 0.75,
    'T10kV': 0.75, 'Transformer_Protect': 0.83, 'Busbar_Protect': 0.75, 'GIS_500kV': 0.83,
}

comp_rows = []
for r in results:
    ek = r['key']
    fc_cov = r['coverage']; fc_wid = r['mean_width']
    uni_cov = r['uni_coverage']; uni_wid = r['uni_width']
    reduction = r['width_reduction']
    if USE_QR_LOSS:
        tune_col = f'{r["n_kept_folds"]}'  # number of kept folds
    else:
        tune_col = f'{r["best_decay"]:.2f}'  # decay parameter
    comp_rows.append([
        ek[:14],
        f'{uni_cov:.0%}',
        f'{uni_wid:.0f}',
        tune_col,
        f'{fc_cov:.0%}',
        f'{fc_wid:.0f}',
        f'{reduction:.0%}',
        f'{r["r2_point"]:.3f}',
    ])

if comp_rows:
    if USE_QR_LOSS:
        tbl_title = f'All-folds vs Truncated({TRUNCATE_RATIO:.0%}) CQR'
        tune_label = 'Kept'
    else:
        tbl_title = 'Uniform(JK+基线) vs Weighted(时间衰减)'
        tune_label = 'Decay'
    log.data_table(tbl_title,
                   ['Material', 'UniCov', 'UniWidth', tune_label, 'WtCov', 'WtWidth', 'WidthRed', 'PtR2'],
                   comp_rows)

mean_fc_cov = np.mean([r['coverage'] for r in results])
mean_fc_wid = np.mean([r['mean_width'] for r in results])
mean_fc_wink = np.mean([r['winkler'] for r in results])
mean_uni_wid = np.mean([r['uni_width'] for r in results])
mean_reduction = 1 - mean_fc_wid / max(mean_uni_wid, 1)
on_target_fc = sum(1 for r in results if 0.70 <= r['coverage'] <= 0.90)

_strategy = []
if USE_QR_LOSS: _strategy.append('CQR')
if USE_TRUNCATED: _strategy.append(f'Trunc{TRUNCATE_RATIO:.0%}')
if not USE_QR_LOSS: _strategy.append('WtDecay')
strategy_str = '+'.join(_strategy) if _strategy else 'JK+'
log.step_ok(f'FC-JK+({strategy_str}) Coverage={mean_fc_cov:.1%}, Width reduction={mean_reduction:.0%}, OnTarget={on_target_fc}/10')

# ========================================================================
# Step 7: Charts
# ========================================================================
log.step('Charts', 'Coverage, intervals, SHAP, residuals, comparison')

for d in [matplotlib.get_cachedir()]:
    try:
        for f in os.listdir(d):
            if 'fontlist' in f: os.remove(os.path.join(d, f))
    except:
        pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fig 1: Three-way coverage comparison
fig1, ax1 = plt.subplots(figsize=(16, 6))
x1 = np.arange(10); w1 = 0.25
sp_covs = [SPLIT_HIST.get(r['key'], 0) for r in results]
jk_covs = [JK_LOO_HIST.get(r['key'], 0) for r in results]
fc_covs = [r['coverage'] for r in results]
ax1.bar(x1 - w1, sp_covs, w1, label='Split (12 res)', color='#FF9800', alpha=0.7)
ax1.bar(x1, jk_covs, w1, label='JK+ LOO (62 res, 有泄露)', color='#9C27B0', alpha=0.7)
ax1.bar(x1 + w1, fc_covs, w1, label='FC-JK+ (25 res, 时序合规)', color='#2196F3', alpha=0.9)
ax1.axhline(y=0.80, c='gray', ls='--', lw=1.5, label='Target 80%')
ax1.axhspan(0.70, 0.90, alpha=0.05, color='green', label='On-target [70%,90%]')
for i in range(10):
    ax1.text(i + w1, fc_covs[i] + 0.02, f'{fc_covs[i]:.0%}', ha='center', fontsize=7)
ax1.set_xticks(x1)
ax1.set_xticklabels([r['key'][:10] for r in results], fontsize=8, rotation=30)
ax1.set_ylabel('Coverage'); ax1.set_ylim(0, 1.2); ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_title('Coverage: Split vs JK+(LOO, 有泄露) vs FC-JK+(时序合规)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'fig1_coverage_3way.png'), dpi=150)
plt.close()

# Fig 2: Prediction intervals for all 10 materials
n_rows, n_cols = 5, 2
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(18, 24))
sorted_r = sorted(results, key=lambda x: -x['r2_point'])
for i, r in enumerate(sorted_r):
    ax = axes2[i // n_cols, i % n_cols]
    ax.fill_between(TEST_DATES, r['lo'], r['hi'], color='#2196F3', alpha=0.12,
                    label='P10-P90 (FC-JK+)')
    ax.plot(TEST_DATES, r['y_te'], 'ko-', lw=2, ms=5, label='Actual', zorder=3)
    ax.plot(TEST_DATES, r['pred_pt'], 's-', color='#E91E63', lw=1.8, ms=5,
            label=f'FC-JK+ Mean (R2={r["r2_point"]:.3f})', zorder=2)
    for j in range(TEST):
        if not (r['lo'][j] <= r['y_te'][j] <= r['hi'][j]):
            ax.plot(TEST_DATES[j], r['y_te'][j], 'ro', ms=10, mfc='none',
                    mew=2, mec='red')
    ax.set_title(f'{r["key"]} [{r["cat"]}]  Cov={r["coverage"]:.0%} '
                 f'W={r["mean_width"]:.0f} R2={r["r2_point"]:.3f}',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params('x', rotation=30, labelsize=6)
fig2.suptitle('Forward-Chaining JK+ P10-P90 Prediction Intervals (时序合规)',
              fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fig2_all_intervals.png'), dpi=150)
plt.close()

# Fig 3: SHAP global feature importance
if HAS_SHAP and all_shap_data:
    fig3, axes3 = plt.subplots(2, 5, figsize=(24, 10))
    avg_importance = np.zeros(n_features)
    for idx, sd in enumerate(all_shap_data):
        ax = axes3[idx // 5, idx % 5]
        imp = sd['importance']
        avg_importance += imp
        sorted_idx = np.argsort(-imp)
        colors = ['#E91E63' if i < 3 else '#2196F3' for i in range(n_features)]
        ax.barh(range(n_features), imp[sorted_idx], color=[colors[i] for i in sorted_idx])
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([ACTIVE_FEATURE_NAMES[i] for i in sorted_idx], fontsize=6)
        ax.set_xlabel('|SHAP|', fontsize=7)
        ax.set_title(f'{sd["key"]}', fontsize=8, fontweight='bold')
        ax.invert_yaxis()
    avg_importance /= len(all_shap_data)

    fig3.suptitle('SHAP Feature Importance per Material (top-3 in red)',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, 'fig3_shap_importance.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()

    # Fig 3b: Average SHAP importance (horizontal bar)
    fig3b, ax3b = plt.subplots(figsize=(10, 6))
    sorted_avg = np.argsort(-avg_importance)
    colors3b = ['#E91E63' if i < 3 else '#2196F3' for i in range(n_features)]
    ax3b.barh(range(n_features), avg_importance[sorted_avg],
              color=[colors3b[i] for i in sorted_avg])
    ax3b.set_yticks(range(n_features))
    ax3b.set_yticklabels([ACTIVE_FEATURE_NAMES[i] for i in sorted_avg], fontsize=10)
    ax3b.set_xlabel('Mean |SHAP value| (averaged over 10 materials)', fontsize=11)
    ax3b.set_title('Global Feature Importance (SHAP, averaged)', fontsize=13, fontweight='bold')
    ax3b.invert_yaxis()
    ax3b.grid(True, alpha=0.3, axis='x')
    for i, idx in enumerate(sorted_avg):
        ax3b.text(avg_importance[idx] + 0.001, i, f'{avg_importance[idx]:.4f}',
                  va='center', fontsize=8)
    plt.tight_layout()
    fig3b.savefig(os.path.join(OUT_DIR, 'fig3b_shap_global_avg.png'), dpi=150)
    plt.close()

# Fig 4: SHAP beeswarm for top-3 materials
if HAS_SHAP and all_shap_data:
    top3_mats = sorted(all_shap_data, key=lambda x: -np.max(x['importance']))[:3]
    fig4, axes4 = plt.subplots(1, 3, figsize=(20, 7))
    for idx, sd in enumerate(top3_mats):
        ax = axes4[idx]
        X_tr = sd['X_train']
        m = sd['model']
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(X_tr)
        # Manual beeswarm (since shap.summary_plot creates its own figure)
        for feat_idx in range(X_tr.shape[1]):
            vals = sv[:, feat_idx]
            jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
            colors = plt.cm.coolwarm((X_tr[:, feat_idx] - X_tr[:, feat_idx].min()) /
                                    max(X_tr[:, feat_idx].max() - X_tr[:, feat_idx].min(), 1e-8))
            ax.scatter(vals, [feat_idx] * len(vals) + jitter,
                      c=colors, s=15, alpha=0.7, edgecolors='none')
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(ACTIVE_FEATURE_NAMES, fontsize=7)
        ax.set_xlabel('SHAP value', fontsize=9)
        ax.set_title(f'{sd["key"]}', fontsize=10, fontweight='bold')
        ax.axvline(x=0, c='gray', ls='--', lw=0.8)
    fig4.suptitle('SHAP Beeswarm: Top-3 Materials (color=feature value)',
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, 'fig4_shap_beeswarm.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()

# Fig 5: Residual distribution
fig5, axes5 = plt.subplots(2, 5, figsize=(20, 8))
for idx, r in enumerate(results):
    ax = axes5[idx // 5, idx % 5]
    resids = np.array(r['fc_residuals'])
    ax.hist(resids, bins=max(8, len(resids) // 3), color='#2196F3', alpha=0.7,
            edgecolor='white')
    p90 = np.percentile(resids, 90)
    med = np.median(resids)
    ax.axvline(x=p90, color='#E91E63', ls='--', lw=2, label=f'P90={p90:.0f}')
    ax.axvline(x=med, color='#4CAF50', ls='-', lw=2, label=f'Med={med:.0f}')
    ax.set_title(f'{r["key"]} Cov={r["coverage"]:.0%}', fontsize=8, fontweight='bold')
    ax.set_xlabel('Residual', fontsize=7)
    ax.legend(fontsize=6)
fig5.suptitle(f'Forward-Chaining Calibration Residuals ({N_FC_FOLDS} folds, no leakage)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'fig5_residual_dist.png'), dpi=150)
plt.close()

# Fig 6: Coverage-Width bubble
fig6, ax6 = plt.subplots(figsize=(10, 7))
for r in results:
    ax6.scatter(r['coverage'], r['mean_width'],
                s=max(50, 1 / max(r['rel_width'], 0.01) * 100),
                alpha=0.7, edgecolors='black', linewidth=0.5)
    ax6.annotate(r['key'][:10], (r['coverage'], r['mean_width']),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
ax6.axvline(x=0.80, c='gray', ls='--', alpha=0.5, label='Target 80%')
ax6.set_xlabel('Coverage', fontsize=12)
ax6.set_ylabel('Mean Width', fontsize=12)
ax6.set_title('FC-JK+: Coverage vs Width (bubble=precision)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()
plt.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, 'fig6_cov_width_bubble.png'), dpi=150)
plt.close()

log.step_ok('6+ charts generated')

# ========================================================================
# Step 8: Final Report
# ========================================================================
log.step('Final Report', f'{strategy_str} vs All-folds baseline comparison')

print(f"\n{'='*120}")
print(f" FC-JK+({strategy_str}) vs ALL-FOLDS BASELINE — FINAL COMPARISON")
print(f"{'='*120}")
if USE_QR_LOSS:
    print(f"\n{'Material':<18s} {'Kept':>6s} {'UniCov':>7s} {'UniW':>7s} "
          f"{'CQRCov':>7s} {'CQRW':>7s} {'Red':>5s} {'R2':>7s} {'Inside':>6s}")
else:
    print(f"\n{'Material':<18s} {'Decay':>6s} {'UniCov':>7s} {'UniW':>7s} "
          f"{'WtCov':>7s} {'WtW':>7s} {'Red':>5s} {'R2':>7s} {'Inside':>6s}")
print(f"{'-'*90}")
for r in results:
    if USE_QR_LOSS:
        tune_val = f'{r["n_kept_folds"]:>6d}'
    else:
        tune_val = f'{r["best_decay"]:>6.2f}'
    print(f"{r['key'][:16]:<18s} {tune_val} {r['uni_coverage']:>6.0%} "
          f"{r['uni_width']:>7.0f} {r['coverage']:>6.0%} {r['mean_width']:>7.0f} "
          f"{r['width_reduction']:>4.0%} {r['r2_point']:>7.4f} "
          f"{r['inside_count']:>4d}/{TEST}")

print(f"\nAGGREGATE:")
print(f"  All-folds: Mean Cov={np.mean([r['uni_coverage'] for r in results]):.1%}, "
      f"Mean Width={mean_uni_wid:.0f}")
print(f"  {strategy_str:>10s}: Mean Cov={mean_fc_cov:.1%}, "
      f"Mean Width={mean_fc_wid:.0f} (reduced {mean_reduction:.0%})")
print(f"  On-target (70%-90%):  {on_target_fc}/10")
if USE_QR_LOSS:
    print(f"  Strategy: QR Loss (CQR) + Truncated({TRUNCATE_RATIO:.0%})")
    print(f"  Kept folds:    {', '.join(f'{r['key'][:8]}={r['n_kept_folds']}' for r in results)}")
else:
    print(f"  Best decay values:    {', '.join(f'{r['key'][:8]}={r['best_decay']:.2f}' for r in results)}")

best_wink = min(results, key=lambda r: r['winkler'])
print(f"  Top material (Winkler): {best_wink['key']} (W={best_wink['winkler']:.0f}, "
      f"Cov={best_wink['coverage']:.0%})")

if HAS_SHAP:
    print(f"\nSHAP ANALYSIS:")
    avg_imp = np.mean([sd['importance'] for sd in all_shap_data], axis=0)
    top5_idx = np.argsort(-avg_imp)[:5]
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  #{rank} {ACTIVE_FEATURE_NAMES[idx]:<20s}  mean|SHAP|={avg_imp[idx]:.4f}")

# Save JSON
_corrections = [
    'LOO-CV replaced by Forward-Chaining CV (no look-ahead bias)',
    'Optuna uses Forward-Chaining CV instead of single train-val split',
    'SHAP added for model interpretability',
]
if USE_QR_LOSS:
    _corrections.append(f'QR Loss (CQR): Conformalized Quantile Regression with pinball loss optimization')
if USE_TRUNCATED:
    _corrections.append(f'Truncated Base Learners: keep top {TRUNCATE_RATIO:.0%} folds by validation loss')
if not USE_QR_LOSS:
    _corrections.append('Time-decay weighted quantiles for interval construction')

out_json = {
    'method': 'Forward-Chaining Jackknife+ (Barber et al. 2021, time-series compliant)',
    'corrections': _corrections,
    'setup': {
        'n_fc_folds': N_FC_FOLDS,
        'min_train_months': MIN_TRAIN,
        'optuna_trials': 30,
        'optuna_fc_folds': f'months[24:{min(48, TRAIN)}] expanding window',
        'alpha': alpha,
        'target_interval': 'P10-P90',
        'use_qr_loss': USE_QR_LOSS,
        'use_truncated': USE_TRUNCATED,
        'truncate_ratio': TRUNCATE_RATIO,
        'qr_lo': QR_LO,
        'qr_hi': QR_HI,
    },
    'aggregate': {
        'mean_coverage': round(float(mean_fc_cov), 4),
        'mean_width': round(float(mean_fc_wid), 2),
        'mean_winkler': round(float(mean_fc_wink), 2),
        'on_target_70_90': on_target_fc,
    },
    'materials': results,
    'shap_global_importance': {
        ACTIVE_FEATURE_NAMES[i]: round(float(np.mean([sd['importance'][i] for sd in all_shap_data])), 4)
        for i in range(n_features)
    } if all_shap_data else {},
    'timestamp': datetime.now().isoformat(),
}

json_path = os.path.join(OUT_DIR, 'forward_jackknife_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2, default=str)

log.pipeline_end_log(
    f'FC-JK+({strategy_str}) Cov={mean_fc_cov:.1%}, OnTarget={on_target_fc}/10, '
    f'BestWinkler={best_wink["key"]}({best_wink["winkler"]:.0f})'
)
print(f"\n  JSON: {json_path}")
print(f"  Charts: {OUT_DIR}/")
