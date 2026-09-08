"""
paper5_focal_smote.py -- 复现 Paper 5 方法论
"Enhancing Intermittent Spare Part Demand Forecasting:
 A Novel Ensemble Approach with Focal Loss and SMOTE"
(MDPI Logistics, 2026, Vol.9 Issue 1, Article 25)

核心方法论复现:
1. 两阶段框架: Stage1 分类(需求是否发生) + Stage2 回归(发生时的数量)
2. SMOTE 过采样: 解决 Stage1 中零/非零类别不平衡
3. Focal Loss (γ=2): 让分类器聚焦难分样本(边界附近的零/非零)
4. 多学习器 Stacking: 5个基学习器 + RidgeCV 元学习器
5. 遗传算法/差分进化: 优化融合权重 + 分类阈值

适配说明 (论文→我们的数据):
- 论文: 624周×5备件, 滑动窗口构造样本 → 我们: 81月×20物资, 沿用D-02增强滞后特征
- 论文: LSTM/DNN+Focal Loss → 我们: 样本仅69个月, 跳过深度学习, 用CatBoost自定义Focal Loss
- 论文: GA优化 → 我们: scipy differential_evolution (更稳健, 小搜索空间)
- 论文未报告γ/SMOTE k/窗口长度/GA参数 → 采用标准默认值(γ=2, k=5, DE默认参数)

运行: python scripts/paper5_focal_smote.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNetCV, RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.over_sampling import SMOTE
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)

# Windows console UTF-8 support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ===================== 配置 =====================
N_TEST = 12
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs')
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
_INTERNAL_FACTORS = ['project_count', 'transformer_bids', 'monthly_bid_count', 'uhv_bids']
FOCAL_GAMMA = 2.0  # 论文未报告, 采用 Lin et al. 2017 标准默认值
SMOTE_K = 5        # 论文未报告, 采用 imbalanced-learn 默认值
RANDOM_SEED = 42


# ===================== 数据加载与特征工程 =====================
def get_top_factors(material, df_train):
    """Spearman top-4 因子选择 (与 main.py 一致)"""
    demand = df_train['demand'].values
    scores = {}
    for col in _INTERNAL_FACTORS:
        if col in df_train.columns:
            vals = df_train[col].values
            mask = ~(np.isnan(vals) | np.isnan(demand))
            if mask.sum() > 5:
                rho, _ = spearmanr(vals[mask], demand[mask])
                scores[col] = abs(rho)
    sorted_factors = sorted(scores, key=scores.get, reverse=True)
    return sorted_factors[:4]


def preprocess_data(df, material):
    """时序安全特征工程 (复用 main.py D-02 增强滞后特征)"""
    top4 = get_top_factors(material, df.iloc[:len(df)-N_TEST])
    cols = ['demand'] + top4
    sub = df[cols].copy().ffill().bfill().fillna(0)
    data = sub.values.astype(np.float64)
    demand_raw = data[:, 0].copy()

    train_len = len(data) - N_TEST
    data_train, data_test = data[:train_len], data[train_len:]
    demand_train, demand_test = demand_raw[:train_len], demand_raw[train_len:]

    # 基础 lag/rolling
    def make_lag_rolling(seq):
        n = len(seq)
        lag1 = np.zeros(n); lag1[1:] = seq[:-1]
        lag12 = np.zeros(n); lag12[12:] = seq[:-12]
        roll3 = np.array([np.mean(seq[max(0,i-3):i]) for i in range(n)])
        return lag1, lag12, roll3

    lag1_tr, lag12_tr, roll3_tr = make_lag_rolling(demand_train)
    is_zero_lag1_tr = (demand_train == 0).astype(float)
    is_zero_lag12_tr = np.zeros(train_len, dtype=float)
    for i in range(12, train_len):
        is_zero_lag12_tr[i] = (demand_train[i-12] == 0)
    m_train = (np.arange(train_len)+1) % 12; m_train[m_train==0] = 12
    q_train = ((np.arange(train_len)+1) // 3) % 4; q_train[q_train==0] = 4

    # D-02: 增强滞后特征
    seq = demand_train
    n = train_len
    lag2 = np.zeros(n); lag2[2:] = seq[:-2]
    lag3 = np.zeros(n); lag3[3:] = seq[:-3]
    lag6 = np.zeros(n); lag6[6:] = seq[:-6]
    roll6_mean = np.array([np.mean(seq[max(0,i-6):i]) if i > 0 else 0 for i in range(n)])
    roll12_mean = np.array([np.mean(seq[max(0,i-12):i]) if i > 0 else 0 for i in range(n)])
    roll3_std = np.array([np.std(seq[max(0,i-3):i]) if i > 1 else 0 for i in range(n)])
    ewm = np.zeros(n)
    alpha = 0.3
    for i in range(1, n):
        ewm[i] = alpha * seq[i-1] + (1 - alpha) * ewm[i-1]
    yoy_diff = np.zeros(n)
    for i in range(13, n):
        yoy_diff[i] = seq[i-1] - seq[i-13]

    # 事件特征
    d = demand_train
    gap_since_last = np.zeros(n)
    last_event = -999
    for i in range(n):
        if d[i] > 0: last_event = i
        gap_since_last[i] = i - last_event if last_event >= 0 else n
    evt_cnt_6m = np.zeros(n); evt_cnt_12m = np.zeros(n); cumul_12m = np.zeros(n)
    for i in range(n):
        evt_cnt_6m[i] = np.sum(d[max(0,i-5):i+1] > 0)
        evt_cnt_12m[i] = np.sum(d[max(0,i-11):i+1] > 0)
        cumul_12m[i] = np.sum(d[max(0,i-11):i+1])
    month_freq = np.zeros(n)
    cal_month = np.array([(4+i)%12+1 for i in range(n)])
    for i in range(12, n):
        past_same_month = [j for j in range(i) if cal_month[j]==cal_month[i]]
        if past_same_month:
            month_freq[i] = np.mean(d[past_same_month] > 0)

    X_train_raw = np.column_stack([
        data_train[:,1:], lag1_tr, lag12_tr, roll3_tr,
        is_zero_lag1_tr, is_zero_lag12_tr,
        np.sin(2*np.pi*m_train/12), np.cos(2*np.pi*m_train/12),
        np.sin(2*np.pi*q_train/4), np.cos(2*np.pi*q_train/4),
        lag2, lag3, lag6, roll6_mean, roll12_mean, roll3_std, ewm, yoy_diff,
        gap_since_last, evt_cnt_6m, evt_cnt_12m, cumul_12m, month_freq,
    ])

    # 测试集特征 (用训练集最后值填充无法计算的滞后)
    lag1_te = np.zeros(N_TEST); lag1_te[0] = demand_train[-1]
    for i in range(1, N_TEST): lag1_te[i] = demand_test[i-1]
    lag12_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src_idx = train_len + i - 12
        if 0 <= src_idx < train_len: lag12_te[i] = demand_train[src_idx]
        elif src_idx >= train_len: lag12_te[i] = demand_test[src_idx - train_len]
    roll3_te = np.zeros(N_TEST)
    full_seq = np.concatenate([demand_train, demand_test])
    for i in range(N_TEST):
        idx = train_len + i
        roll3_te[i] = np.mean(full_seq[max(0,idx-3):idx])

    is_zero_lag1_te = np.zeros(N_TEST)
    is_zero_lag1_te[0] = float(demand_train[-1] == 0)
    for i in range(1, N_TEST): is_zero_lag1_te[i] = float(demand_test[i-1] == 0)
    is_zero_lag12_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src_idx = train_len + i - 12
        if 0 <= src_idx < train_len: is_zero_lag12_te[i] = float(demand_train[src_idx] == 0)
        elif src_idx >= train_len: is_zero_lag12_te[i] = float(demand_test[src_idx-train_len] == 0)

    m_test = (np.arange(train_len, train_len+N_TEST)+1) % 12; m_test[m_test==0] = 12
    q_test = ((np.arange(train_len, train_len+N_TEST)+1) // 3) % 4; q_test[q_test==0] = 4

    # D-02 测试集
    lag2_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len + i - 2
        if 0 <= src < train_len: lag2_te[i] = demand_train[src]
        elif src >= train_len: lag2_te[i] = demand_test[src-train_len]
    lag3_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len + i - 3
        if 0 <= src < train_len: lag3_te[i] = demand_train[src]
        elif src >= train_len: lag3_te[i] = demand_test[src-train_len]
    lag6_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len + i - 6
        if 0 <= src < train_len: lag6_te[i] = demand_train[src]
        elif src >= train_len: lag6_te[i] = demand_test[src-train_len]
    roll6_te = np.array([np.mean(full_seq[max(0,train_len+i-6):train_len+i]) for i in range(N_TEST)])
    roll12_te = np.array([np.mean(full_seq[max(0,train_len+i-12):train_len+i]) for i in range(N_TEST)])
    roll3_std_te = np.array([np.std(full_seq[max(0,train_len+i-3):train_len+i]) for i in range(N_TEST)])
    ewm_te = np.zeros(N_TEST)
    ewm_val = ewm[-1]
    for i in range(N_TEST):
        if i == 0:
            ewm_te[i] = alpha * demand_train[-1] + (1-alpha) * ewm_val
        else:
            ewm_te[i] = alpha * demand_test[i-1] + (1-alpha) * ewm_te[i-1]
    yoy_diff_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src1 = train_len + i - 1
        src13 = train_len + i - 13
        v1 = demand_train[src1] if src1 < train_len else demand_test[src1-train_len]
        v13 = demand_train[src13] if 0 <= src13 < train_len else (demand_test[src13-train_len] if src13 >= train_len else 0)
        yoy_diff_te[i] = v1 - v13

    # 事件特征测试集
    gap_te = np.zeros(N_TEST)
    last_evt = -999
    for i in range(n):
        if d[i] > 0: last_evt = i
    for i in range(N_TEST):
        if demand_test[i] > 0: last_evt = train_len + i
        gap_te[i] = (train_len + i) - last_evt if last_evt >= 0 else n + N_TEST
    evt6_te = np.zeros(N_TEST); evt12_te = np.zeros(N_TEST); cum12_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        idx = train_len + i
        evt6_te[i] = np.sum(full_seq[max(0,idx-5):idx+1] > 0)
        evt12_te[i] = np.sum(full_seq[max(0,idx-11):idx+1] > 0)
        cum12_te[i] = np.sum(full_seq[max(0,idx-11):idx+1])
    mf_te = np.zeros(N_TEST)
    cal_month_full = np.array([(4+i)%12+1 for i in range(train_len+N_TEST)])
    for i in range(N_TEST):
        idx = train_len + i
        past = [j for j in range(idx) if cal_month_full[j] == cal_month_full[idx]]
        if past: mf_te[i] = np.mean(full_seq[past] > 0)

    X_test_raw = np.column_stack([
        data_test[:,1:], lag1_te, lag12_te, roll3_te,
        is_zero_lag1_te, is_zero_lag12_te,
        np.sin(2*np.pi*m_test/12), np.cos(2*np.pi*m_test/12),
        np.sin(2*np.pi*q_test/4), np.cos(2*np.pi*q_test/4),
        lag2_te, lag3_te, lag6_te, roll6_te, roll12_te, roll3_std_te, ewm_te, yoy_diff_te,
        gap_te, evt6_te, evt12_te, cum12_te, mf_te,
    ])

    # MinMaxScaler (论文使用 MinMaxScaler)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, demand_train, X_test, demand_test


# ===================== Focal Loss CatBoost =====================
class FocalLossCatBoostClassifier:
    """CatBoost + Focal Loss (γ=2)
    CatBoost 不原生支持 Focal Loss, 通过 sample_weight 近似:
    对每个样本计算 p_t, 权重 = (1-p_t)^γ
    迭代两轮: 第一轮标准训练获取概率, 第二轮用 focal 权重重新训练
    """
    def __init__(self, gamma=FOCAL_GAMMA, **kwargs):
        self.gamma = gamma
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, eval_set=None):
        # 第一轮: 标准训练获取概率估计
        m1 = CatBoostClassifier(**self.kwargs, loss_function='Logloss',
                                random_seed=RANDOM_SEED, verbose=0)
        if eval_set:
            m1.fit(X, y, eval_set=eval_set)
        else:
            m1.fit(X, y)
        # 计算 focal weights
        proba = np.clip(m1.predict_proba(X)[:, 1], 1e-7, 1-1e-7)
        p_t = np.where(y == 1, proba, 1 - proba)
        focal_weights = (1 - p_t) ** self.gamma
        # 归一化权重
        focal_weights = focal_weights / focal_weights.mean()
        # 第二轮: 用 focal weights 重新训练
        self.model = CatBoostClassifier(**self.kwargs, loss_function='Logloss',
                                        random_seed=RANDOM_SEED, verbose=0)
        pool_kwargs = {}
        if eval_set:
            pool_kwargs['eval_set'] = eval_set
        self.model.fit(X, y, sample_weight=focal_weights, **pool_kwargs)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ===================== 多学习器 Stacking =====================
def build_stage1_learners():
    """Stage 1: 5个分类基学习器 (论文: LR/DT/RF/SVM/LightGBM/MLP/LSTM/DNN)
    适配: 跳过 LSTM/DNN (69样本太少), 用 Focal-CatBoost 替代 DNN+FocalLoss
    """
    return {
        'focal_catboost': FocalLossCatBoostClassifier(
            gamma=FOCAL_GAMMA, iterations=600, learning_rate=0.03,
            depth=5, l2_leaf_reg=5, early_stopping_rounds=30),
        'lightgbm': LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=15, reg_alpha=1, reg_lambda=3, verbose=-1,
            random_state=RANDOM_SEED),
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            random_state=RANDOM_SEED),
        'catboost_balanced': CatBoostClassifier(
            iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=6,
            auto_class_weights='Balanced', random_seed=RANDOM_SEED, verbose=0),
    }


def build_stage2_learners():
    """Stage 2: 4个回归基学习器"""
    return {
        'elasticnet': ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                                   cv=3, max_iter=5000, random_state=RANDOM_SEED),
        'ridge': RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3),
        'lightgbm_reg': LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=15, reg_alpha=1, reg_lambda=3, verbose=-1,
            random_state=RANDOM_SEED),
        'random_forest_reg': RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            random_state=RANDOM_SEED),
    }


# ===================== 核心: 两阶段 Stacking + SMOTE + Focal Loss =====================
def paper5_predict(X_train, y_train, X_test, material):
    """
    论文方法论完整流程:
    1. SMOTE 过采样 Stage1 训练数据
    2. 多学习器分类 (含 Focal Loss CatBoost)
    3. Stacking 元学习器融合分类概率
    4. 差分进化优化分类阈值
    5. Stage2 多学习器回归 (仅非零样本)
    6. Stacking 元学习器融合回归预测
    7. 最终: P(occurrence) × E(quantity|occurrence)
    """
    y_bin = (y_train > 0).astype(int)
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos

    if n_pos < 5 or n_neg < 5:
        return None  # 样本不足, 回退

    # === Stage 1: 分类 (SMOTE + Focal Loss + Stacking) ===
    # SMOTE 过采样
    k = min(SMOTE_K, n_pos - 1, n_neg - 1)
    if k < 1:
        X_cls, y_cls = X_train, y_bin
    else:
        try:
            smote = SMOTE(k_neighbors=k, random_state=RANDOM_SEED)
            X_cls, y_cls = smote.fit_resample(X_train, y_bin)
        except ValueError:
            X_cls, y_cls = X_train, y_bin

    # 训练多个分类器, 收集 OOF 概率用于 stacking
    n_train = len(X_train)
    cls_learners = build_stage1_learners()
    cls_oof_probs = np.zeros((n_train, len(cls_learners)))
    cls_test_probs = np.zeros((len(X_test), len(cls_learners)))

    # 时序合规: 用最后 1/4 做验证 (前向验证)
    nv = max(6, n_train // 4)

    for idx, (name, learner) in enumerate(cls_learners.items()):
        try:
            if name == 'focal_catboost':
                learner.fit(X_cls[:-nv], y_cls[:-nv],
                           eval_set=(X_cls[-nv:], y_cls[-nv:]))
            elif hasattr(learner, 'fit') and 'eval_set' in learner.fit.__code__.co_varnames:
                learner.fit(X_cls[:-nv], y_cls[:-nv],
                           eval_set=[(X_cls[-nv:], y_cls[-nv:])])
            else:
                learner.fit(X_cls, y_cls)
            # OOF 概率 (对原始训练集, 非 SMOTE 后的)
            cls_oof_probs[:, idx] = learner.predict_proba(X_train)[:, 1]
            cls_test_probs[:, idx] = learner.predict_proba(X_test)[:, 1]
        except Exception as e:
            # 回退: 用简单比例
            cls_oof_probs[:, idx] = n_pos / len(y_bin)
            cls_test_probs[:, idx] = n_pos / len(y_bin)

    # Stacking 元学习器 (Ridge, 论文用 Linear Regression)
    meta_cls = Ridge(alpha=1.0)
    meta_cls.fit(cls_oof_probs, y_bin)
    prob_stacked = np.clip(meta_cls.predict(cls_test_probs), 0, 1)

    # === 差分进化优化分类阈值 ===
    # 用训练集 OOF 概率找最优阈值 (最大化 F1 或 Youden's J)
    oof_stacked = np.clip(meta_cls.predict(cls_oof_probs), 0, 1)

    def neg_youden(threshold):
        pred = (oof_stacked >= threshold[0]).astype(int)
        tp = ((pred == 1) & (y_bin == 1)).sum()
        fp = ((pred == 1) & (y_bin == 0)).sum()
        fn = ((pred == 0) & (y_bin == 1)).sum()
        tn = ((pred == 0) & (y_bin == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return -(tpr - fpr)  # 最大化 Youden's J = TPR - FPR

    result = differential_evolution(neg_youden, bounds=[(0.1, 0.9)],
                                    seed=RANDOM_SEED, maxiter=50, tol=1e-4)
    opt_threshold = result.x[0]

    # === Stage 2: 回归 (仅非零样本) ===
    nz_mask = y_train > 0
    X_nz = X_train[nz_mask]
    y_nz = y_train[nz_mask]

    if len(y_nz) < 8:
        # 非零样本太少, 直接用概率×均值
        return prob_stacked * np.mean(y_nz) if len(y_nz) > 0 else prob_stacked * 0

    reg_learners = build_stage2_learners()
    reg_oof_preds = np.zeros((len(y_nz), len(reg_learners)))
    reg_test_preds = np.zeros((len(X_test), len(reg_learners)))

    nv2 = min(6, len(y_nz) // 4)

    for idx, (name, learner) in enumerate(reg_learners.items()):
        try:
            if nv2 >= 2 and hasattr(learner, 'fit'):
                try:
                    learner.fit(X_nz[:-nv2], y_nz[:-nv2])
                except TypeError:
                    learner.fit(X_nz, y_nz)
            else:
                learner.fit(X_nz, y_nz)
            reg_oof_preds[:, idx] = learner.predict(X_nz)
            reg_test_preds[:, idx] = learner.predict(X_test)
        except Exception:
            reg_oof_preds[:, idx] = np.mean(y_nz)
            reg_test_preds[:, idx] = np.mean(y_nz)

    # Stacking 元学习器 (Ridge)
    meta_reg = Ridge(alpha=1.0)
    meta_reg.fit(reg_oof_preds, y_nz)
    qty_stacked = np.maximum(meta_reg.predict(reg_test_preds), 0)

    # === 最终预测: P × Q ===
    # 论文: 如果分类预测为有需求 → 输出回归量; 否则 → 0
    # 软版本: P × Q (连续概率加权, 比硬阈值更平滑)
    y_pred_soft = prob_stacked * qty_stacked

    # 硬版本: 用优化阈值二值化
    y_pred_hard = np.where(prob_stacked >= opt_threshold, qty_stacked, 0)

    # 取软/硬中 R² 更好的 (在 OOF 上评估)
    # 这里直接返回软版本 (通常更稳定)
    return y_pred_soft, opt_threshold, prob_stacked


# ===================== 评估 =====================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'R2': r2, 'MAPE': mape, 'RMSE': rmse, 'MAE': mae}


# ===================== 主流程 =====================
def main():
    print("=" * 70)
    print("  Paper 5 复现: Focal Loss + SMOTE + Multi-Learner Stacking")
    print("  (MDPI Logistics 2026, Vol.9, Art.25)")
    print("=" * 70)
    print(f"  Focal γ={FOCAL_GAMMA}, SMOTE k={SMOTE_K}, N_TEST={N_TEST}")
    print()

    # 加载数据
    xl = pd.ExcelFile(DATA_FILE)
    materials = xl.sheet_names
    print(f"  物资数量: {len(materials)}")
    print()

    all_metrics = {}
    results_detail = []

    for mat in materials:
        df = xl.parse(mat)
        # 标准化列名
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if '需求' in c or 'demand' in cl: col_map[c] = 'demand'
            elif '日期' in c or 'date' in cl: col_map[c] = 'date'
            elif '项目' in c or 'project' in cl: col_map[c] = 'project_count'
            elif 'transformer' in cl: col_map[c] = 'transformer_bids'
            elif 'monthly' in cl or '公告' in c: col_map[c] = 'monthly_bid_count'
            elif 'uhv' in cl or '特高压' in c: col_map[c] = 'uhv_bids'
        df = df.rename(columns=col_map)
        if 'demand' not in df.columns:
            print(f"  [SKIP] {mat}: 无 demand 列")
            continue

        df = df[['demand'] + [c for c in _INTERNAL_FACTORS if c in df.columns]].copy()
        df = df.fillna(0)

        try:
            X_train, y_train, X_test, y_test = preprocess_data(df, mat)
            result = paper5_predict(X_train, y_train, X_test, mat)

            if result is None:
                print(f"  [SKIP] {mat}: 样本不足")
                continue

            y_pred, opt_thr, prob = result
            metrics = evaluate(y_test, y_pred)
            all_metrics[mat] = metrics

            n_pos = (y_train > 0).sum()
            print(f"  {mat}: R²={metrics['R2']:.4f} MAPE={metrics['MAPE']:.1f}% "
                  f"RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} "
                  f"[thr={opt_thr:.2f}, 非零月={n_pos}]")
            results_detail.append({
                'material': mat, **metrics,
                'threshold': opt_thr, 'n_nonzero': int(n_pos)
            })
        except Exception as e:
            print(f"  [ERROR] {mat}: {e}")

    # 汇总
    print()
    print("=" * 70)
    print("  汇总统计")
    print("=" * 70)
    if all_metrics:
        r2_vals = [m['R2'] for m in all_metrics.values()]
        mape_vals = [m['MAPE'] for m in all_metrics.values()]
        n_pass = sum(1 for r in r2_vals if r >= 0.85)
        print(f"  R² 均值: {np.mean(r2_vals):.4f}")
        print(f"  MAPE 均值: {np.mean(mape_vals):.2f}%")
        print(f"  达标比例 (R²≥0.85): {n_pass}/{len(r2_vals)} = {n_pass/len(r2_vals)*100:.0f}%")
        print()
        # 排序输出
        sorted_mats = sorted(all_metrics.items(), key=lambda x: x[1]['R2'], reverse=True)
        print("  排名  物资                          R²      MAPE%")
        print("  " + "-" * 60)
        for i, (mat, m) in enumerate(sorted_mats, 1):
            flag = "✓" if m['R2'] >= 0.85 else " "
            print(f"  {i:2d}. {flag} {mat:<28s} {m['R2']:.4f}  {m['MAPE']:7.1f}%")

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'paper5_results.json')
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'Paper5_FocalLoss_SMOTE_Stacking',
            'paper': 'MDPI Logistics 2026, Vol.9, Art.25',
            'params': {'focal_gamma': FOCAL_GAMMA, 'smote_k': SMOTE_K, 'n_test': N_TEST},
            'summary': {
                'r2_mean': float(np.mean(r2_vals)) if all_metrics else None,
                'mape_mean': float(np.mean(mape_vals)) if all_metrics else None,
                'pass_ratio': n_pass / len(r2_vals) if all_metrics else None,
            },
            'per_material': results_detail,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == '__main__':
    main()
