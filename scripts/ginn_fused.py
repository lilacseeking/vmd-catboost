"""
ginn_fused.py -- GINN融合策略: 灰色先验特征 + ElasticNet-2S两阶段框架
"Grey-Informed Two-Stage Framework (GI-2S)"

融合逻辑:
- 纯GINN(MLP)在69样本上失败 → 保留灰色系统理论的"先验信息"
- 将GM(1,1)的预测值、灰色方程残差、分数阶AGO作为额外特征
- 嫁接到已验证有效的两阶段框架: CatBoost分类 + ElasticNetCV回归
- ElasticNet的L1正则自动筛选: 灰色特征有用则保留, 无用则归零

创新点保留:
1. 灰色微分方程 dX⁽¹⁾/dt + a·X⁽¹⁾ = b 作为物理先验 (非纯数据驱动)
2. 分数阶r-AGO捕捉长程记忆 (r∈(0,1)优化)
3. 灰色发展系数a作为局部趋势指示器
4. 两阶段零膨胀处理 (分类+回归)

与纯GINN的区别:
- 不用MLP (69样本撑不起) → 用ElasticNet (L1+L2天然适配小样本高维)
- 灰色信息从"损失函数约束"变为"信息特征" → 更灵活, 不强制服从
- 保留两阶段架构 → 已验证R²=0.7241的成熟框架

运行: python scripts/ginn_fused.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNetCV
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from scipy.special import gamma as gamma_fn

warnings.filterwarnings('ignore')
np.random.seed(42)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ===================== 配置 =====================
N_TEST = 12
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs')
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
_INTERNAL_FACTORS = ['project_count', 'transformer_bids', 'monthly_bid_count', 'uhv_bids']
RANDOM_SEED = 42


# ===================== GM(1,1) 灰色模型 =====================
def gm11_fit(x0):
    """GM(1,1) OLS估计 a, b"""
    n = len(x0)
    if n < 4:
        return -0.01, np.mean(x0) if len(x0) > 0 else 0
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = x0[1:]
    try:
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        return params[0], params[1]
    except:
        return -0.01, np.mean(x0)


def gm11_fitted_values(x0, a, b):
    """GM(1,1) 拟合值序列 (与原始序列等长)"""
    n = len(x0)
    if abs(a) < 1e-10:
        return np.full(n, np.mean(x0))
    x1_hat = np.zeros(n)
    x1_hat[0] = x0[0]
    for k in range(1, n):
        x1_hat[k] = (x0[0] - b/a) * np.exp(-a * k) + b/a
    fitted = np.zeros(n)
    fitted[0] = x0[0]
    for k in range(1, n):
        fitted[k] = x1_hat[k] - x1_hat[k-1]
    return np.maximum(fitted, 0)


def gm11_predict_future(x0_last, a, b, n_pred):
    """GM(1,1) 外推预测"""
    if abs(a) < 1e-10:
        return np.full(n_pred, x0_last)
    preds = np.zeros(n_pred)
    # 从序列末尾继续
    n0 = 1  # 相对起点
    x1_prev = x0_last
    for k in range(1, n_pred + 1):
        x1_k = (x0_last - b/a) * np.exp(-a * k) + b/a
        preds[k-1] = x1_k - x1_prev
        x1_prev = x1_k
    return np.maximum(preds, 0)


def fractional_ago_sequence(x0, r):
    """分数阶r-AGO序列"""
    n = len(x0)
    xr = np.zeros(n)
    for k in range(n):
        for j in range(k+1):
            coeff = gamma_fn(k - j + r) / (gamma_fn(k - j + 1) * gamma_fn(r))
            xr[k] += coeff * x0[j]
    return xr


def compute_grey_features(demand_seq, window=12):
    """计算灰色先验特征序列 (滑动窗口GM(1,1))

    对每个时间点t, 用前window个点拟合GM(1,1), 输出:
    - gm_fitted: GM(1,1)对当前点的拟合值
    - gm_residual: 实际值 - GM(1,1)拟合值 (灰色方程残差)
    - grey_a: 局部发展系数 (趋势方向)
    - fago: 分数阶AGO值 (平滑趋势)
    """
    n = len(demand_seq)
    gm_fitted = np.zeros(n)
    gm_residual = np.zeros(n)
    grey_a_arr = np.zeros(n)
    fago_arr = np.zeros(n)

    # 最优分数阶 (用前20个点搜索)
    best_r = 0.7
    search_end = min(20, n)
    if search_end >= 5:
        y_sub = np.maximum(demand_seq[:search_end], 1e-6)
        best_smooth = float('inf')
        for r in [0.3, 0.5, 0.7, 0.9, 1.0]:
            try:
                fago_sub = fractional_ago_sequence(y_sub, r)
                smooth = np.std(np.diff(fago_sub))
                if smooth < best_smooth:
                    best_smooth = smooth
                    best_r = r
            except:
                continue

    # 滑动窗口GM(1,1)
    for t in range(n):
        start = max(0, t - window)
        seg = demand_seq[start:t+1]
        seg_pos = np.maximum(seg, 0)

        if len(seg_pos) >= 4 and seg_pos.sum() > 0:
            a, b = gm11_fit(seg_pos)
            fitted = gm11_fitted_values(seg_pos, a, b)
            gm_fitted[t] = fitted[-1]
            gm_residual[t] = demand_seq[t] - fitted[-1]
            grey_a_arr[t] = a
        else:
            gm_fitted[t] = np.mean(seg_pos) if len(seg_pos) > 0 else 0
            gm_residual[t] = 0
            grey_a_arr[t] = 0

    # 分数阶AGO (全序列)
    y_pos = np.maximum(demand_seq, 1e-6)
    try:
        fago_full = fractional_ago_sequence(y_pos, best_r)
        # 归一化到与原始序列同量级
        if fago_full[-1] > 0:
            fago_arr = fago_full * (np.mean(demand_seq[demand_seq > 0]) / np.mean(fago_full)) if (demand_seq > 0).any() else fago_full
        else:
            fago_arr = np.zeros(n)
    except:
        fago_arr = np.zeros(n)

    return gm_fitted, gm_residual, grey_a_arr, fago_arr, best_r


# ===================== 特征工程 =====================
def get_top_factors(material, df_train):
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
    """特征工程: D-02增强滞后 + 灰色先验特征"""
    top4 = get_top_factors(material, df.iloc[:len(df)-N_TEST])
    cols = ['demand'] + top4
    sub = df[cols].copy().ffill().bfill().fillna(0)
    data = sub.values.astype(np.float64)
    demand_raw = data[:, 0].copy()

    train_len = len(data) - N_TEST
    data_train, data_test = data[:train_len], data[train_len:]
    demand_train, demand_test = demand_raw[:train_len], demand_raw[train_len:]

    n = train_len
    seq = demand_train

    # === 基础特征 (与main.py一致) ===
    lag1 = np.zeros(n); lag1[1:] = seq[:-1]
    lag12 = np.zeros(n); lag12[12:] = seq[:-12]
    roll3 = np.array([np.mean(seq[max(0,i-3):i]) for i in range(n)])
    is_zero_lag1 = (seq == 0).astype(float)
    m_train = (np.arange(n)+1) % 12; m_train[m_train==0] = 12
    q_train = ((np.arange(n)+1) // 3) % 4; q_train[q_train==0] = 4

    # D-02 增强滞后
    lag2 = np.zeros(n); lag2[2:] = seq[:-2]
    lag3 = np.zeros(n); lag3[3:] = seq[:-3]
    lag6 = np.zeros(n); lag6[6:] = seq[:-6]
    roll6 = np.array([np.mean(seq[max(0,i-6):i]) if i > 0 else 0 for i in range(n)])
    roll12 = np.array([np.mean(seq[max(0,i-12):i]) if i > 0 else 0 for i in range(n)])
    roll3_std = np.array([np.std(seq[max(0,i-3):i]) if i > 1 else 0 for i in range(n)])
    ewm = np.zeros(n)
    alpha = 0.3
    for i in range(1, n):
        ewm[i] = alpha * seq[i-1] + (1-alpha) * ewm[i-1]
    yoy_diff = np.zeros(n)
    for i in range(13, n):
        yoy_diff[i] = seq[i-1] - seq[i-13]

    # 事件特征
    gap = np.zeros(n); last_evt = -999
    for i in range(n):
        if seq[i] > 0: last_evt = i
        gap[i] = i - last_evt if last_evt >= 0 else n
    evt6 = np.array([np.sum(seq[max(0,i-5):i+1] > 0) for i in range(n)])
    evt12 = np.array([np.sum(seq[max(0,i-11):i+1] > 0) for i in range(n)])
    cum12 = np.array([np.sum(seq[max(0,i-11):i+1]) for i in range(n)])

    # === 灰色先验特征 (GINN核心创新) ===
    gm_fitted_tr, gm_res_tr, grey_a_tr, fago_tr, r_opt = compute_grey_features(seq, window=12)

    X_train_raw = np.column_stack([
        data_train[:,1:], lag1, lag12, roll3, is_zero_lag1,
        np.sin(2*np.pi*m_train/12), np.cos(2*np.pi*m_train/12),
        np.sin(2*np.pi*q_train/4), np.cos(2*np.pi*q_train/4),
        lag2, lag3, lag6, roll6, roll12, roll3_std, ewm, yoy_diff,
        gap, evt6, evt12, cum12,
        # 灰色先验特征 (4个新特征)
        gm_fitted_tr, gm_res_tr, grey_a_tr, fago_tr,
    ])

    # === 测试集特征 ===
    full_seq = np.concatenate([demand_train, demand_test])
    # 对全序列计算灰色特征, 取测试部分
    gm_fitted_full, gm_res_full, grey_a_full, fago_full, _ = compute_grey_features(full_seq, window=12)
    gm_fitted_te = gm_fitted_full[train_len:]
    gm_res_te = gm_res_full[train_len:]
    grey_a_te = grey_a_full[train_len:]
    fago_te = fago_full[train_len:]

    lag1_te = np.zeros(N_TEST); lag1_te[0] = seq[-1]
    for i in range(1, N_TEST): lag1_te[i] = demand_test[i-1]
    lag12_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len+i-12
        if 0 <= src < train_len: lag12_te[i] = seq[src]
        elif src >= train_len: lag12_te[i] = demand_test[src-train_len]
    roll3_te = np.array([np.mean(full_seq[max(0,train_len+i-3):train_len+i]) for i in range(N_TEST)])
    is_zero_te = np.zeros(N_TEST)
    is_zero_te[0] = float(seq[-1] == 0)
    for i in range(1, N_TEST): is_zero_te[i] = float(demand_test[i-1] == 0)
    m_test = (np.arange(train_len, train_len+N_TEST)+1) % 12; m_test[m_test==0] = 12
    q_test = ((np.arange(train_len, train_len+N_TEST)+1) // 3) % 4; q_test[q_test==0] = 4

    lag2_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len+i-2
        if 0 <= src < train_len: lag2_te[i] = seq[src]
        elif src >= train_len: lag2_te[i] = demand_test[src-train_len]
    lag3_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len+i-3
        if 0 <= src < train_len: lag3_te[i] = seq[src]
        elif src >= train_len: lag3_te[i] = demand_test[src-train_len]
    lag6_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len+i-6
        if 0 <= src < train_len: lag6_te[i] = seq[src]
        elif src >= train_len: lag6_te[i] = demand_test[src-train_len]
    roll6_te = np.array([np.mean(full_seq[max(0,train_len+i-6):train_len+i]) for i in range(N_TEST)])
    roll12_te = np.array([np.mean(full_seq[max(0,train_len+i-12):train_len+i]) for i in range(N_TEST)])
    roll3_std_te = np.array([np.std(full_seq[max(0,train_len+i-3):train_len+i]) for i in range(N_TEST)])
    ewm_te = np.zeros(N_TEST)
    ewm_val = ewm[-1]
    for i in range(N_TEST):
        if i == 0: ewm_te[i] = alpha*seq[-1] + (1-alpha)*ewm_val
        else: ewm_te[i] = alpha*demand_test[i-1] + (1-alpha)*ewm_te[i-1]
    yoy_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        s1 = train_len+i-1; s13 = train_len+i-13
        v1 = seq[s1] if s1 < train_len else demand_test[s1-train_len]
        v13 = seq[s13] if 0 <= s13 < train_len else (demand_test[s13-train_len] if s13 >= train_len else 0)
        yoy_te[i] = v1 - v13
    gap_te = np.zeros(N_TEST)
    last_evt_te = -999
    for i in range(n):
        if seq[i] > 0: last_evt_te = i
    for i in range(N_TEST):
        if demand_test[i] > 0: last_evt_te = train_len + i
        gap_te[i] = (train_len+i) - last_evt_te if last_evt_te >= 0 else n+N_TEST
    evt6_te = np.array([np.sum(full_seq[max(0,train_len+i-5):train_len+i+1] > 0) for i in range(N_TEST)])
    evt12_te = np.array([np.sum(full_seq[max(0,train_len+i-11):train_len+i+1] > 0) for i in range(N_TEST)])
    cum12_te = np.array([np.sum(full_seq[max(0,train_len+i-11):train_len+i+1]) for i in range(N_TEST)])

    X_test_raw = np.column_stack([
        data_test[:,1:], lag1_te, lag12_te, roll3_te, is_zero_te,
        np.sin(2*np.pi*m_test/12), np.cos(2*np.pi*m_test/12),
        np.sin(2*np.pi*q_test/4), np.cos(2*np.pi*q_test/4),
        lag2_te, lag3_te, lag6_te, roll6_te, roll12_te, roll3_std_te, ewm_te, yoy_te,
        gap_te, evt6_te, evt12_te, cum12_te,
        # 灰色先验特征
        gm_fitted_te, gm_res_te, grey_a_te, fago_te,
    ])

    # 处理NaN/Inf
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, demand_train, X_test, demand_test, r_opt


# ===================== 两阶段预测 (CatBoost分类 + ElasticNet回归) =====================
def two_stage_predict(X_train, y_train, X_test, material):
    """GI-2S: 灰色信息增强的两阶段预测"""
    y_bin = (y_train > 0).astype(int)
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos

    if n_pos < 5 or n_neg < 5:
        return np.full(len(X_test), np.mean(y_train))

    # Stage 1: CatBoost 分类器
    cls = CatBoostClassifier(
        iterations=600, learning_rate=0.03, depth=5, l2_leaf_reg=5,
        loss_function='Logloss', early_stopping_rounds=30,
        random_seed=RANDOM_SEED, verbose=0)
    nv = max(6, len(y_train) // 4)
    cls.fit(X_train[:-nv], y_bin[:-nv], eval_set=(X_train[-nv:], y_bin[-nv:]))
    prob = np.clip(cls.predict_proba(X_test)[:, 1], 0, 1)

    # Stage 2: ElasticNetCV 回归 (仅非零样本)
    nz = y_train > 0
    X_nz = X_train[nz]
    y_nz = y_train[nz]

    if len(y_nz) < 8:
        return prob * np.mean(y_nz)

    reg = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=3, max_iter=5000, random_state=RANDOM_SEED)
    reg.fit(X_nz, y_nz)
    qty = np.maximum(reg.predict(X_test), 0)

    # 最终: P × Q
    return prob * qty


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
    print("  GI-2S: Grey-Informed Two-Stage Framework")
    print("  融合: GINN灰色先验特征 + ElasticNet-2S两阶段")
    print("  (基于 Neurocomputing 2025 GINN 方法论适配)")
    print("=" * 70)
    print(f"  灰色特征: GM(1,1)拟合值 + 灰色残差 + 发展系数a + 分数阶AGO")
    print(f"  Stage1: CatBoost分类 | Stage2: ElasticNetCV回归")
    print(f"  N_TEST={N_TEST}")
    print()

    xl = pd.ExcelFile(DATA_FILE)
    materials = xl.sheet_names
    print(f"  物资数量: {len(materials)}")
    print()

    all_metrics = {}
    results_detail = []

    for mat in materials:
        df = xl.parse(mat)
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
            continue

        df = df[['demand'] + [c for c in _INTERNAL_FACTORS if c in df.columns]].copy()
        df = df.fillna(0)

        try:
            X_train, y_train, X_test, y_test, r_opt = preprocess_data(df, mat)
            y_pred = two_stage_predict(X_train, y_train, X_test, mat)
            metrics = evaluate(y_test, y_pred)
            all_metrics[mat] = metrics

            n_pos = (y_train > 0).sum()
            print(f"  {mat}: R²={metrics['R2']:.4f} MAPE={metrics['MAPE']:.1f}% "
                  f"RMSE={metrics['RMSE']:.2f} [r={r_opt:.1f} 非零={n_pos}]")
            results_detail.append({'material': mat, **metrics, 'r_opt': r_opt, 'n_nonzero': int(n_pos)})
        except Exception as e:
            print(f"  [ERROR] {mat}: {e}")

    # 汇总
    print()
    print("=" * 70)
    print("  GI-2S 汇总统计")
    print("=" * 70)
    if all_metrics:
        r2_vals = [m['R2'] for m in all_metrics.values()]
        mape_vals = [m['MAPE'] for m in all_metrics.values()]
        n_pass85 = sum(1 for r in r2_vals if r >= 0.85)
        n_pass80 = sum(1 for r in r2_vals if r >= 0.80)
        print(f"  R² 均值: {np.mean(r2_vals):.4f}")
        print(f"  MAPE 均值: {np.mean(mape_vals):.2f}%")
        print(f"  达标 R²≥0.85: {n_pass85}/{len(r2_vals)} = {n_pass85/len(r2_vals)*100:.0f}%")
        print(f"  达标 R²≥0.80: {n_pass80}/{len(r2_vals)} = {n_pass80/len(r2_vals)*100:.0f}%")
        print()
        # 与基线对比
        print("  对比基线 (ElasticNet-2S无灰色特征): R²=0.7241, 达标47%")
        delta = np.mean(r2_vals) - 0.7241
        print(f"  GI-2S vs 基线: ΔR² = {delta:+.4f}")
        print()
        sorted_mats = sorted(all_metrics.items(), key=lambda x: x[1]['R2'], reverse=True)
        print("  排名  物资                          R²      MAPE%")
        print("  " + "-" * 60)
        for i, (mat, m) in enumerate(sorted_mats, 1):
            flag = "✓" if m['R2'] >= 0.85 else ("~" if m['R2'] >= 0.80 else " ")
            print(f"  {i:2d}. {flag} {mat:<28s} {m['R2']:.4f}  {m['MAPE']:7.1f}%")

    # 保存
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ginn_fused_results.json')
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'GI-2S_Grey_Informed_Two_Stage',
            'paper': 'Based on GINN (Neurocomputing 2025) + ElasticNet-2S',
            'innovation': 'Grey system prior as informative features + two-stage zero-inflated framework',
            'summary': {
                'r2_mean': float(np.mean(r2_vals)) if all_metrics else None,
                'mape_mean': float(np.mean(mape_vals)) if all_metrics else None,
                'pass_85': n_pass85 / len(r2_vals) if all_metrics else None,
                'pass_80': n_pass80 / len(r2_vals) if all_metrics else None,
                'delta_vs_baseline': float(delta) if all_metrics else None,
            },
            'per_material': results_detail,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == '__main__':
    main()
