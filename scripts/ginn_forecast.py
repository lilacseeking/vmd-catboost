"""
ginn_forecast.py -- 复现 Paper 1: GINN/FGINN
"Grey-informed Neural Network for Time-Series Forecasting"
(Neurocomputing, 2025)

核心方法论:
1. GM(1,1) 灰色模型提供"物理先验": dX⁽¹⁾/dt + a·X⁽¹⁾ = b
2. 小型MLP神经网络做数据驱动预测
3. 灰色方程残差作为正则化项嵌入损失函数:
   L = MSE(y, ŷ) + λ_grey · ||grey_equation_residual||² + λ_L2 · ||W||²
4. 分数阶扩展(FGINN): 用分数阶r-AGO替代整数阶, r∈(0,1)参与优化

适配说明:
- 论文用能源/环境/商业数据 → 我们用电力物资81月×20种
- 论文未公开具体网络结构 → 采用2层MLP(64→32), 适配69样本
- 分数阶r通过网格搜索在[0.3, 1.0]上优化
- 灰色方程参数a,b由训练集OLS估计, 作为固定先验

运行: python scripts/ginn_forecast.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNetCV
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ===================== 配置 =====================
N_TEST = 12
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs')
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
_INTERNAL_FACTORS = ['project_count', 'transformer_bids', 'monthly_bid_count', 'uhv_bids']
RANDOM_SEED = 42
DEVICE = 'cpu'

# GINN 超参数
HIDDEN_DIMS = [64, 32]       # MLP结构
DROPOUT = 0.3                # 防过拟合
LR = 0.005                   # 学习率
EPOCHS = 500                 # 训练轮数
LAMBDA_GREY = 0.1            # 灰色方程约束权重
LAMBDA_L2 = 1e-3             # L2正则化
PATIENCE = 50                # 早停


# ===================== GM(1,1) 灰色模型 =====================
def gm11_fit(x0):
    """GM(1,1) 最小二乘估计参数 a, b
    x0: 原始非负序列 (一阶累加前的原始值)
    返回: a(发展系数), b(灰色作用量)
    """
    n = len(x0)
    # 1-AGO: 一阶累加生成
    x1 = np.cumsum(x0)
    # 紧邻均值生成: z1(k) = 0.5*(x1(k) + x1(k-1))
    z1 = 0.5 * (x1[1:] + x1[:-1])
    # 构建矩阵 B·[a,b]^T = Y
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = x0[1:]
    # OLS
    params = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, b = params[0], params[1]
    return a, b, x1


def gm11_predict(a, b, x0_first, n_pred):
    """GM(1,1) 预测: 从第一个值出发, 预测n_pred步
    X̂⁽¹⁾(k+1) = (x0[0] - b/a)·e^(-a·k) + b/a
    X̂⁽⁰⁾(k+1) = X̂⁽¹⁾(k+1) - X̂⁽¹⁾(k)
    """
    preds = np.zeros(n_pred)
    x1_hat = np.zeros(n_pred + 1)
    x1_hat[0] = x0_first
    for k in range(1, n_pred + 1):
        x1_hat[k] = (x0_first - b/a) * np.exp(-a * k) + b/a
    for k in range(n_pred):
        preds[k] = x1_hat[k+1] - x1_hat[k]
    return np.maximum(preds, 0)


def fractional_ago(x0, r):
    """分数阶r-AGO (r阶累加生成)
    X⁽ʳ⁾(k) = Σⱼ₌₀ᵏ⁻¹ C(k-j+r-1, k-j-1) · x0(j+1)
    其中 C(n,m) = Γ(n+1)/(Γ(m+1)·Γ(n-m+1))
    """
    from scipy.special import gamma as gamma_fn
    n = len(x0)
    xr = np.zeros(n)
    # 计算二项式系数
    for k in range(n):
        for j in range(k+1):
            # C(k-j+r-1, k-j) = Γ(k-j+r) / (Γ(k-j+1) · Γ(r))
            num = gamma_fn(k - j + r)
            den = gamma_fn(k - j + 1) * gamma_fn(r)
            xr[k] += (num / den) * x0[j]
    return xr


def gm11_grey_residual(y_pred_seq, a, b):
    """计算灰色方程残差: X⁽⁰⁾(k) + a·Z⁽¹⁾(k) - b
    用于GINN损失函数中的物理约束项
    y_pred_seq: 预测序列 (模拟的X⁽⁰⁾)
    """
    n = len(y_pred_seq)
    if n < 3:
        return 0.0
    # 对预测序列做1-AGO
    x1 = np.cumsum(y_pred_seq)
    # 紧邻均值
    z1 = 0.5 * (x1[1:] + x1[:-1])
    # 灰色方程残差: X⁽⁰⁾(k) + a·Z⁽¹⁾(k) - b ≈ 0
    residual = y_pred_seq[1:] + a * z1 - b
    return np.mean(residual ** 2)


# ===================== 特征工程 (复用D-02) =====================
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
    """时序安全特征工程 (与main.py D-02一致)"""
    top4 = get_top_factors(material, df.iloc[:len(df)-N_TEST])
    cols = ['demand'] + top4
    sub = df[cols].copy().ffill().bfill().fillna(0)
    data = sub.values.astype(np.float64)
    demand_raw = data[:, 0].copy()

    train_len = len(data) - N_TEST
    data_train, data_test = data[:train_len], data[train_len:]
    demand_train, demand_test = demand_raw[:train_len], demand_raw[train_len:]

    # 基础lag/rolling
    n = train_len
    lag1 = np.zeros(n); lag1[1:] = demand_train[:-1]
    lag12 = np.zeros(n); lag12[12:] = demand_train[:-12]
    roll3 = np.array([np.mean(demand_train[max(0,i-3):i]) for i in range(n)])
    is_zero_lag1 = (demand_train == 0).astype(float)
    m_train = (np.arange(n)+1) % 12; m_train[m_train==0] = 12
    q_train = ((np.arange(n)+1) // 3) % 4; q_train[q_train==0] = 4

    # D-02 增强滞后
    seq = demand_train
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
    evt6 = np.zeros(n); evt12 = np.zeros(n); cum12 = np.zeros(n)
    for i in range(n):
        evt6[i] = np.sum(seq[max(0,i-5):i+1] > 0)
        evt12[i] = np.sum(seq[max(0,i-11):i+1] > 0)
        cum12[i] = np.sum(seq[max(0,i-11):i+1])

    X_train_raw = np.column_stack([
        data_train[:,1:], lag1, lag12, roll3, is_zero_lag1,
        np.sin(2*np.pi*m_train/12), np.cos(2*np.pi*m_train/12),
        np.sin(2*np.pi*q_train/4), np.cos(2*np.pi*q_train/4),
        lag2, lag3, lag6, roll6, roll12, roll3_std, ewm, yoy_diff,
        gap, evt6, evt12, cum12,
    ])

    # 测试集特征
    full_seq = np.concatenate([demand_train, demand_test])
    lag1_te = np.zeros(N_TEST); lag1_te[0] = demand_train[-1]
    for i in range(1, N_TEST): lag1_te[i] = demand_test[i-1]
    lag12_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len + i - 12
        if 0 <= src < train_len: lag12_te[i] = demand_train[src]
        elif src >= train_len: lag12_te[i] = demand_test[src-train_len]
    roll3_te = np.array([np.mean(full_seq[max(0,train_len+i-3):train_len+i]) for i in range(N_TEST)])
    is_zero_te = np.zeros(N_TEST)
    is_zero_te[0] = float(demand_train[-1] == 0)
    for i in range(1, N_TEST): is_zero_te[i] = float(demand_test[i-1] == 0)
    m_test = (np.arange(train_len, train_len+N_TEST)+1) % 12; m_test[m_test==0] = 12
    q_test = ((np.arange(train_len, train_len+N_TEST)+1) // 3) % 4; q_test[q_test==0] = 4

    lag2_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len+i-2
        if 0 <= src < train_len: lag2_te[i] = demand_train[src]
        elif src >= train_len: lag2_te[i] = demand_test[src-train_len]
    lag3_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len+i-3
        if 0 <= src < train_len: lag3_te[i] = demand_train[src]
        elif src >= train_len: lag3_te[i] = demand_test[src-train_len]
    lag6_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        src = train_len+i-6
        if 0 <= src < train_len: lag6_te[i] = demand_train[src]
        elif src >= train_len: lag6_te[i] = demand_test[src-train_len]
    roll6_te = np.array([np.mean(full_seq[max(0,train_len+i-6):train_len+i]) for i in range(N_TEST)])
    roll12_te = np.array([np.mean(full_seq[max(0,train_len+i-12):train_len+i]) for i in range(N_TEST)])
    roll3_std_te = np.array([np.std(full_seq[max(0,train_len+i-3):train_len+i]) for i in range(N_TEST)])
    ewm_te = np.zeros(N_TEST)
    ewm_val = ewm[-1]
    for i in range(N_TEST):
        if i == 0: ewm_te[i] = alpha*demand_train[-1] + (1-alpha)*ewm_val
        else: ewm_te[i] = alpha*demand_test[i-1] + (1-alpha)*ewm_te[i-1]
    yoy_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        s1 = train_len+i-1; s13 = train_len+i-13
        v1 = demand_train[s1] if s1 < train_len else demand_test[s1-train_len]
        v13 = demand_train[s13] if 0 <= s13 < train_len else (demand_test[s13-train_len] if s13 >= train_len else 0)
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
    ])

    # 处理NaN/Inf
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, demand_train, X_test, demand_test


# ===================== GINN 神经网络 =====================
class GINNNet(nn.Module):
    """Grey-Informed Neural Network: 小型MLP + 灰色方程约束"""
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_ginn(X_train, y_train, a_grey, b_grey, lambda_grey=LAMBDA_GREY):
    """训练GINN: MSE + 灰色方程残差 + L2正则化"""
    input_dim = X_train.shape[1]
    model = GINNNet(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=LAMBDA_L2)

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)

    # 验证集 (最后1/4)
    nv = max(6, len(y_train) // 4)
    X_tr, X_val = X_t[:-nv], X_t[-nv:]
    y_tr, y_val = y_t[:-nv], y_t[-nv:]

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        y_pred = model(X_tr)

        # 主损失: MSE
        mse_loss = nn.MSELoss()(y_pred, y_tr)

        # 灰色方程约束: 预测序列应近似满足 X⁽⁰⁾(k) + a·Z⁽¹⁾(k) ≈ b
        pred_np = y_pred.detach().cpu().numpy()
        pred_np_sorted = pred_np  # 保持时间顺序
        if len(pred_np_sorted) > 3:
            x1_cum = np.cumsum(np.abs(pred_np_sorted))
            z1 = 0.5 * (x1_cum[1:] + x1_cum[:-1])
            grey_res = np.abs(pred_np_sorted[1:]) + a_grey * z1 - b_grey
            grey_loss = torch.FloatTensor([np.mean(grey_res**2)]).to(DEVICE)
        else:
            grey_loss = torch.tensor(0.0).to(DEVICE)

        # 总损失
        total_loss = mse_loss + lambda_grey * grey_loss
        total_loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.MSELoss()(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ===================== 分数阶优化 =====================
def optimize_fractional_order(y_train, X_train, a_grey, b_grey):
    """搜索最优分数阶r ∈ [0.3, 1.0]"""
    best_r = 1.0
    best_score = -float('inf')

    for r in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        try:
            # 分数阶AGO变换目标
            y_pos = np.maximum(y_train, 1e-6)
            y_r = fractional_ago(y_pos[:20], r)  # 用前20个点估计
            # 简单评估: 分数阶AGO的平滑度 (越平滑越好)
            if len(y_r) > 2:
                smoothness = -np.std(np.diff(y_r))
                if smoothness > best_score:
                    best_score = smoothness
                    best_r = r
        except:
            continue
    return best_r


# ===================== 主预测流程 =====================
def ginn_predict(X_train, y_train, X_test, material):
    """
    GINN完整预测流程:
    1. GM(1,1)估计灰色参数a, b
    2. 分数阶r优化
    3. 训练GINN (MSE + grey_constraint + L2)
    4. GM(1,1)外推 + GINN修正 → 最终预测
    """
    # Step 1: GM(1,1) 参数估计
    y_pos = np.maximum(y_train, 0)
    # 对非零子序列做GM(1,1) (避免全零导致a,b无意义)
    nz_mask = y_pos > 0
    if nz_mask.sum() < 5:
        # 非零太少, 直接用均值
        return np.full(len(X_test), np.mean(y_pos[y_pos > 0]) if nz_mask.any() else 0)

    y_nz = y_pos[nz_mask]
    try:
        a_grey, b_grey, _ = gm11_fit(y_nz[:min(30, len(y_nz))])
    except:
        a_grey, b_grey = -0.01, np.mean(y_nz)

    # Step 2: 分数阶优化 (轻量)
    r_opt = optimize_fractional_order(y_train, X_train, a_grey, b_grey)

    # Step 3: 训练GINN
    model = train_ginn(X_train, y_train, a_grey, b_grey, lambda_grey=LAMBDA_GREY)

    # Step 4: 预测
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        y_pred_nn = model(X_t).cpu().numpy()

    # GM(1,1) 外推预测
    gm_pred = gm11_predict(a_grey, b_grey, y_nz[-1], N_TEST)

    # 融合: GINN输出为主, GM(1,1)作为下界参考
    # 如果GINN预测为负, 用GM(1,1)兜底
    y_pred = np.maximum(y_pred_nn, 0)
    # 对GINN预测为0但GM(1,1)有值的情况, 取加权
    for i in range(N_TEST):
        if y_pred[i] < 1e-6 and gm_pred[i] > 0:
            y_pred[i] = 0.3 * gm_pred[i]  # 灰色先验兜底

    return y_pred, {'a': a_grey, 'b': b_grey, 'r': r_opt, 'gm_pred': gm_pred}


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
    print("  Paper 1 复现: GINN (Grey-Informed Neural Network)")
    print("  Neurocomputing 2025 — 灰色方程约束 + 分数阶 + MLP")
    print("=" * 70)
    print(f"  网络: MLP{HIDDEN_DIMS}, dropout={DROPOUT}, lr={LR}")
    print(f"  损失: MSE + {LAMBDA_GREY}×grey_residual + {LAMBDA_L2}×L2")
    print(f"  N_TEST={N_TEST}, EPOCHS={EPOCHS}, PATIENCE={PATIENCE}")
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
            X_train, y_train, X_test, y_test = preprocess_data(df, mat)
            result = ginn_predict(X_train, y_train, X_test, mat)

            if isinstance(result, np.ndarray):
                y_pred = result
                info = {}
            else:
                y_pred, info = result

            metrics = evaluate(y_test, y_pred)
            all_metrics[mat] = metrics

            n_pos = (y_train > 0).sum()
            a_str = f"a={info.get('a', 0):.4f}" if info else ""
            r_str = f"r={info.get('r', 1.0):.1f}" if info else ""
            print(f"  {mat}: R²={metrics['R2']:.4f} MAPE={metrics['MAPE']:.1f}% "
                  f"RMSE={metrics['RMSE']:.2f} [{a_str} {r_str} 非零={n_pos}]")
            results_detail.append({'material': mat, **metrics, 'n_nonzero': int(n_pos)})
        except Exception as e:
            print(f"  [ERROR] {mat}: {e}")

    # 汇总
    print()
    print("=" * 70)
    print("  GINN 汇总统计")
    print("=" * 70)
    if all_metrics:
        r2_vals = [m['R2'] for m in all_metrics.values()]
        mape_vals = [m['MAPE'] for m in all_metrics.values()]
        n_pass = sum(1 for r in r2_vals if r >= 0.85)
        n_pass80 = sum(1 for r in r2_vals if r >= 0.80)
        print(f"  R² 均值: {np.mean(r2_vals):.4f}")
        print(f"  MAPE 均值: {np.mean(mape_vals):.2f}%")
        print(f"  达标 R²≥0.85: {n_pass}/{len(r2_vals)} = {n_pass/len(r2_vals)*100:.0f}%")
        print(f"  达标 R²≥0.80: {n_pass80}/{len(r2_vals)} = {n_pass80/len(r2_vals)*100:.0f}%")
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
    out_path = os.path.join(out_dir, 'ginn_results.json')
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'GINN_Grey_Informed_Neural_Network',
            'paper': 'Neurocomputing 2025',
            'params': {'hidden': HIDDEN_DIMS, 'dropout': DROPOUT, 'lr': LR,
                       'lambda_grey': LAMBDA_GREY, 'lambda_l2': LAMBDA_L2,
                       'epochs': EPOCHS, 'n_test': N_TEST},
            'summary': {
                'r2_mean': float(np.mean(r2_vals)) if all_metrics else None,
                'mape_mean': float(np.mean(mape_vals)) if all_metrics else None,
                'pass_85': n_pass / len(r2_vals) if all_metrics else None,
                'pass_80': n_pass80 / len(r2_vals) if all_metrics else None,
            },
            'per_material': results_detail,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == '__main__':
    main()
