"""
main.py —— 配电网物资需求预测
三种模型对比: CatBoost / VMD-CatBoost / VMD-LSTM-CatBoost
三类物资: 10KV电缆(cable) / 柱上变压器台成套设备(transformer) / 10kv交流避雷器(arrester)
Python 3.12

运行: python main.py
输出: 控制台评估指标表 + outputs/figures/ 目录下所有图表
"""
import os, sys, warnings, json, logging, io
from datetime import datetime
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from vmdpy import VMD
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ---- 抑制 matplotlib Agg 后端 Windows 字体权限错误 ----
# 某些系统字体文件（如 C:\Windows\Fonts\ 下）有权限限制，
# matplotlib 尝试读取时会产生 PermissionError，表现为控制台大量
# "Exception ignored in: 'read_from_file_callback'" 输出。
# 该错误不会影响图表正常生成，但污染控制台输出。
# 解决方案：(1) 删除旧字体缓存 (2) 重建并限制可用字体列表

# 清除旧的字体缓存以强制重建
_font_cache_dir = matplotlib.get_cachedir()
for _fname in os.listdir(_font_cache_dir):
    if _fname.startswith('fontlist'):
        _cache_path = os.path.join(_font_cache_dir, _fname)
        try:
            os.remove(_cache_path)
        except OSError:
            pass

# 仅加载可访问的 TrueType 字体，跳过有权限问题的字体文件
_font_dirs = set(fm.findSystemFonts())
_accessible_fonts = []
for _fp in _font_dirs:
    try:
        with open(_fp, 'rb') as _f:
            _f.read(4)
        _accessible_fonts.append(_fp)
    except (PermissionError, OSError):
        pass

# 使用过滤后的字体列表重建字体管理器
fm.fontManager = fm.FontManager()
for _fp in _accessible_fonts:
    try:
        fm.fontManager.addfont(_fp)
    except Exception:
        pass

# 设置中文字体 —— 有 SimHei 用 SimHei，否则退回 DejaVu Sans
_zh_fonts = [f for f in _accessible_fonts if any(
    name in f.lower() for name in ['simhei', 'simsun', 'msyh', 'yahei', 'wqy'])
]
if not _zh_fonts:
    _zh_fonts = [f for f in _accessible_fonts if 'dejavu' in f.lower()]

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 抑制底层 C 扩展在字体回调中抛出的 "Exception ignored" 噪音
# 当 matplotlib Agg 后端渲染文字时，尝试读取某些受限制的系统字体文件
# 会触发 PermissionError。fontmanager 重建后基本不会再发生，此处作为兜底。
@contextmanager
def suppress_font_stderr():
    if sys.platform != 'win32':
        yield
        return
    _orig_stderr = sys.stderr
    class _Filter(io.StringIO):
        def write(self, s):
            if isinstance(s, str) and ('Permission' in s or 'read_from_file_callback' in s):
                return len(s)  # 静默吞掉该条消息
            _orig_stderr.write(s)
            return len(s) if isinstance(s, str) else 0
    sys.stderr = _Filter()
    try:
        yield
    finally:
        sys.stderr = _orig_stderr

# Windows console UTF-8 support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ===================== 全局配置 =====================
MATERIALS = ['cable', 'transformer', 'arrester']
MATERIAL_LABELS = {'cable': '10KV电缆', 'transformer': '柱上变压器台成套设备', 'arrester': '10kv交流避雷器'}
FACTOR_NAMES = ['load_growth', 'investment', 'history_demand', 'equipment_cost',
                'typhoon_count', 'lightning_count']
FACTOR_LABELS = {'load_growth': '负荷增长量(分)', 'investment': '工程投资量',
                 'history_demand': '历史需求量', 'equipment_cost': '设备进价成本(万元)',
                 'typhoon_count': '台风(分)', 'lightning_count': '雷击(分)'}
VMD_K = 5
VMD_ALPHA = 2000
SEQ_LEN = 6
RANDOM_SEED = 42
OUTPUT_DIR = 'outputs/figures'
LOG_DIR = 'outputs/logs'
DATA_DIR = 'inputs/data'
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- 日志系统 ----
log_filename = os.path.join(LOG_DIR, f'main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logger = logging.getLogger('vmd_catboost')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_filename, encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'))

logger.addHandler(fh)

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 1. 数据加载/生成 =====================
SHEET_NAMES = {'cable': '10KV电缆', 'transformer': '柱上变压器台成套设备', 'arrester': '10kv交流避雷器'}

# Excel 列名中英文映射
COLUMN_CN = {
    'date': '日期',
    'demand': '需求量',
    'load_growth': '负荷增长量(分)',
    'investment': '工程投资量',
    'history_demand': '历史需求量',
    'equipment_cost': '设备进价成本(万元)',
    'typhoon_count': '台风(分)',
    'lightning_count': '雷击(分)',
}
COLUMN_EN = {v: k for k, v in COLUMN_CN.items()}


def _generate_all_data(months):
    """根据数据生成要求和影响因素说明生成三种物资的模拟数据，确保'统一'因素共享"""
    n = len(months)
    t = np.arange(n)
    np.random.seed(RANDOM_SEED)

    # ===== 统一因素（同一月份所有相关物资共享） =====
    # 负荷增长量: 统一 (all 3 materials)，范围 0.00-1.00，夏季峰值最高
    summer = np.clip(np.sin(np.pi * ((t % 12) - 3) / 6), 0, 1)
    load_growth = 0.15 + summer * 0.7 + np.random.randn(n) * 0.05
    load_growth = np.clip(np.round(load_growth, 3), 0, 1)

    # 雷击: 统一 (all 3 materials)，范围 0.00-1.00 分，5-9月峰值最高
    in_lightning = np.isin(t % 12, [4, 5, 6, 7, 8])
    lightning_count = np.where(in_lightning,
                               np.random.uniform(0.3, 1.0, n),
                               np.random.uniform(0, 0.3, n))
    lightning_count = np.clip(np.round(lightning_count, 3), 0, 1)

    # 台风: 二进制——仅 2023-07(索引18) 和 2024-08(索引31) 为 1
    typhoon_count = np.zeros(n)
    typhoon_count[18] = 1
    typhoon_count[31] = 1

    data_dict = {}

    # ====================================================================
    # 10KV电缆
    # 需求: 冬季(12-2月)为0；非0时 15-25 (10千米)
    # 因子: investment(#1), load_growth(#2), equipment_cost(#3), lightning_count(#4)
    # ====================================================================
    cable_low = np.isin(t % 12, [0, 1, 11])
    cable_raw = 20 + np.sin(2 * np.pi * t / 12) * 3 + 0.3 * t + np.random.randn(n) * 1.5
    cable_demand = np.where(cable_low, 0, np.clip(np.round(cable_raw), 15, 25))
    cable_demand = np.maximum(cable_demand, 0)

    cable_inv_zero = np.isin(t % 12, [0, 4, 8])
    cable_investment = np.where(cable_inv_zero, 0,
                                np.round(np.random.uniform(15, 25, n)))

    cable_cost = 4.5 + np.random.randn(n) * 1.2 + np.sin(2 * np.pi * t / 12) * 1.0
    cable_cost = np.clip(np.round(cable_cost, 1), 2, 7)

    data_dict['cable'] = pd.DataFrame({
        'date': months,
        'demand': cable_demand,
        'load_growth': load_growth,
        'investment': cable_investment,
        'equipment_cost': cable_cost,
        'lightning_count': lightning_count,
    })

    # ====================================================================
    # 柱上变压器台成套设备
    # 需求: 冬季(12-2月)为0；非0时 5-15 套
    # 因子: load_growth(#1), investment(#2), history_demand(#3), equipment_cost(#4)
    # ====================================================================
    trans_low = np.isin(t % 12, [0, 1, 11])
    trans_raw = 10 + np.sin(2 * np.pi * t / 12) * 2.5 + 0.15 * t + np.random.randn(n) * 1.0
    trans_demand = np.where(trans_low, 0, np.clip(np.round(trans_raw), 5, 15))
    trans_demand = np.maximum(trans_demand, 0)

    trans_inv_zero = np.isin(t % 12, [1, 5, 9])
    trans_investment = np.where(trans_inv_zero, 0,
                                np.round(np.random.uniform(5, 20, n)))

    trans_history = np.roll(trans_demand, 1)
    trans_history[0] = 0

    trans_cost = 8.0 + np.random.randn(n) * 0.8 + np.sin(2 * np.pi * t / 12) * 0.8
    trans_cost = np.clip(np.round(trans_cost, 1), 6, 10)

    data_dict['transformer'] = pd.DataFrame({
        'date': months,
        'demand': trans_demand,
        'load_growth': load_growth,
        'investment': trans_investment,
        'history_demand': trans_history,
        'equipment_cost': trans_cost,
    })

    # ====================================================================
    # 10kv交流避雷器
    # 需求: 干季(11-3月)为0；雨季 60-100 台
    # 因子: investment(#1), load_growth(#2), lightning_count(#3), typhoon_count(#4)
    # ====================================================================
    arr_low = np.isin(t % 12, [0, 1, 2, 10, 11])
    arr_raw = 80 + np.sin(2 * np.pi * t / 12 + 1) * 12 + 0.08 * t + np.random.randn(n) * 3
    arr_demand = np.where(arr_low, 0, np.clip(np.round(arr_raw), 60, 100))
    arr_demand = np.maximum(arr_demand, 0)

    arr_inv_zero = np.isin(t % 12, [0, 3, 7])
    arr_investment = np.where(arr_inv_zero, 0,
                              np.round(np.random.uniform(50, 150, n)))

    data_dict['arrester'] = pd.DataFrame({
        'date': months,
        'demand': arr_demand,
        'load_growth': load_growth,
        'investment': arr_investment,
        'lightning_count': lightning_count,
        'typhoon_count': typhoon_count,
    })

    return data_dict


def load_or_generate_data():
    """加载数据：优先读取 inputs/data/data.xlsx（每物资一个sheet），不存在则生成并保存后读取"""
    if os.path.exists(DATA_FILE):
        logger.info(f"读取已有数据文件: {DATA_FILE}")
        data_dict = {}
        for material in MATERIALS:
            sheet = SHEET_NAMES[material]
            df = pd.read_excel(DATA_FILE, sheet_name=sheet)
            df.rename(columns=COLUMN_EN, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            data_dict[material] = df
            logger.info(f"  Sheet[{sheet}]: {len(df)} 条, demand范围=[{df['demand'].min():.2f}, {df['demand'].max():.2f}]")
        return data_dict

    logger.info("数据文件不存在，根据数据生成要求生成模拟数据...")
    months = pd.date_range('2022-01-01', periods=36, freq='MS')
    data_dict = _generate_all_data(months)

    with pd.ExcelWriter(DATA_FILE, engine='openpyxl') as writer:
        for material in MATERIALS:
            df = data_dict[material]
            sheet = SHEET_NAMES[material]
            df.rename(columns=COLUMN_CN).to_excel(writer, sheet_name=sheet, index=False)
            logger.info(f"  生成 Sheet[{sheet}]: {len(df)} 条, demand范围=[{df['demand'].min():.2f}, {df['demand'].max():.2f}]")

    logger.info(f"数据已保存至: {os.path.abspath(DATA_FILE)}")
    return data_dict


# ===================== 2. Top-4 影响因子（预确定） =====================
def get_top_factors(material):
    """返回 top-4 影响因子，按排名顺序"""
    mapping = {
        'cable':        ['investment', 'load_growth', 'equipment_cost', 'lightning_count'],
        'transformer':  ['load_growth', 'investment', 'history_demand', 'equipment_cost'],
        'arrester':     ['investment', 'load_growth', 'lightning_count', 'typhoon_count'],
    }
    return mapping[material]


# ===================== 3. 数据预处理 =====================
def preprocess_data(df, material):
    """MinMax归一化 + 时序分割(前24月train, 后12月test)"""
    top4 = get_top_factors(material)
    cols = ['demand'] + top4
    data = df[cols].values.astype(np.float64)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    train_raw, test_raw = data_scaled[:24], data_scaled[24:]
    y_train = train_raw[:, 0].copy()
    X_train_factors = train_raw[:, 1:].copy()
    y_test = test_raw[:, 0].copy()
    X_test_factors = test_raw[:, 1:].copy()

    # 保留原始尺度（反归一化用）
    demand_scaler = MinMaxScaler()
    demand_scaler.fit(df[['demand']].values.astype(np.float64))
    y_train_orig = df['demand'].values[:24].copy()
    y_test_orig = df['demand'].values[24:].copy()

    return X_train_factors, y_train, X_test_factors, y_test, scaler, demand_scaler


# ===================== 4. VMD 分解 =====================
def vmd_decompose_full(signal, K=VMD_K):
    """对完整需求量序列进行VMD分解，返回所有IMF和残差/模态索引"""
    u, u_hat, omega = VMD(signal, VMD_ALPHA, 0, K, 0, 1, 1e-7)
    residual_idx = int(np.argmin(np.abs(omega[-1])))
    modal_indices = [i for i in range(K) if i != residual_idx]
    return u, u_hat, omega, residual_idx, modal_indices


# ===================== 5. LSTM 模型定义 =====================
class MultiFeatureLSTM(nn.Module):
    """多特征LSTM: 残差分量+4因子 → 预测值"""
    def __init__(self, input_size=5, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class SingleFeatureLSTM(nn.Module):
    """单特征LSTM: 单个模态分量 → 预测值"""
    def __init__(self, hidden_size=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def create_sequences(data, seq_len=SEQ_LEN):
    """构建时间窗口序列 X:(n, seq_len, features), y:(n,)"""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    X, y_list = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y_list.append(data[i + seq_len, 0])
    return np.array(X), np.array(y_list)


def train_lstm_model(model, X, y, epochs=200, patience=30, lr=0.01):
    """训练LSTM模型，返回训练好的模型"""
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.FloatTensor(y).to(DEVICE)

    if isinstance(model, MultiFeatureLSTM):
        arch = (f"MultiFeatureLSTM | input_size=5, hidden_size=32, num_layers=1, dropout=0.2 | "
                f"CNN/池化层: 无(本模型不使用卷积/池化)")
        train_cfg = (f"optimizer=Adam, lr={lr}, epochs={epochs}, patience={patience}, "
                     f"loss=MSELoss, batch_size=full_batch(全批次), device={DEVICE} | "
                     f"训练样本数(train_samples)={len(X)}, 序列长度(seq_len)={X.shape[1]}")
        logger.info(f"  [LSTM架构] {arch}")
        logger.info(f"  [训练配置] {train_cfg}")
    elif isinstance(model, SingleFeatureLSTM):
        arch = (f"SingleFeatureLSTM | hidden_size=16, num_layers=1, dropout=0.3 | "
                f"CNN/池化层: 无(本模型不使用卷积/池化)")
        train_cfg = (f"optimizer=Adam, lr={lr}, epochs={epochs}, patience={patience}, "
                     f"loss=MSELoss, batch_size=full_batch(全批次), device={DEVICE} | "
                     f"训练样本数(train_samples)={len(X)}, 序列长度(seq_len)={X.shape[1]}")
        logger.debug(f"  [LSTM架构] {arch}")
        logger.debug(f"  [训练配置] {train_cfg}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t).squeeze()
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ===================== 6. 模型一: CatBoost =====================
def run_catboost(X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler):
    """模型一: 直接使用4个影响因子，CatBoost回归预测"""
    logger.info("  [CatBoost超参数] iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5, "
                 f"loss_function=RMSE, early_stopping_rounds=30, random_seed={RANDOM_SEED} | "
                 f"输入特征数(input_features)={X_train_factors.shape[1]}")
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5,
        loss_function='RMSE', early_stopping_rounds=30,
        random_seed=RANDOM_SEED, verbose=0
    )
    model.fit(X_train_factors, y_train,
              eval_set=(X_train_factors[-6:], y_train[-6:]))

    y_pred = model.predict(X_test_factors)
    importance = model.get_feature_importance()

    # 反归一化
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, model


# ===================== 7. 模型二: VMD-CatBoost =====================
def run_vmd_catboost(X_train_factors, y_train, X_test_factors, y_test,
                     demand_full, material, demand_scaler):
    """模型二: VMD分解需求量 → 全部分量+4因子 → CatBoost端到端预测"""
    # VMD 分解完整需求量序列
    u, _, omega, _, _ = vmd_decompose_full(demand_full)

    # 分割 IMFs 为 train/test
    imfs_train = u[:, :24].T   # (24, 5)
    imfs_test = u[:, 24:].T    # (12, 5)

    # 拼接特征: IMFs + 4个影响因子
    X_train_full = np.column_stack([imfs_train, X_train_factors])
    X_test_full = np.column_stack([imfs_test, X_test_factors])

    model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5,
        loss_function='RMSE', early_stopping_rounds=30,
        random_seed=RANDOM_SEED, verbose=0
    )
    model.fit(X_train_full, y_train,
              eval_set=(X_train_full[-6:], y_train[-6:]))

    y_pred = model.predict(X_test_full)
    importance = model.get_feature_importance()

    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, omega, u, model


# ===================== 8. 模型三: VMD-LSTM-CatBoost =====================
def run_vmd_lstm_catboost(X_train_factors, y_train, X_test_factors, y_test,
                          demand_full, material, demand_scaler):
    """模型三: VMD → 残差多特征LSTM + 4模态单特征LSTM → CatBoost融合"""
    seq_len = SEQ_LEN
    top4 = get_top_factors(material)

    # 1. VMD 分解完整需求量
    u, _, omega, residual_idx, modal_indices = vmd_decompose_full(demand_full)
    logger.debug(f"    VMD分解: 残差=IMF{residual_idx+1}, 模态={[f'IMF{i+1}' for i in modal_indices]}")

    # 分割IMF为train/test
    u_train = u[:, :24]  # (5, 24)
    u_test = u[:, 24:]   # (5, 12)

    lstm_preds_train = []
    lstm_preds_test = []

    # 2. 残差分量 → 多特征LSTM
    residual_train = u_train[residual_idx]  # (24,)
    residual_test = u_test[residual_idx]    # (12,)

    # 构建多特征输入：残差 + 4因子
    residual_features_train = np.column_stack([
        residual_train, X_train_factors[:, 0], X_train_factors[:, 1],
        X_train_factors[:, 2], X_train_factors[:, 3]
    ])
    residual_features_test = np.column_stack([
        residual_test, X_test_factors[:, 0], X_test_factors[:, 1],
        X_test_factors[:, 2], X_test_factors[:, 3]
    ])

    X_r, y_r = create_sequences(residual_features_train, seq_len)
    # 测试序列需要包含训练集尾部以构建窗口
    residual_full_seq = np.concatenate([residual_train[-seq_len:], residual_test])
    factor_full_seqs = []
    for j in range(4):
        factor_full_seqs.append(np.concatenate([X_train_factors[-seq_len:, j], X_test_factors[:, j]]))
    residual_features_full = np.column_stack([residual_full_seq] + factor_full_seqs)
    X_r_test, _ = create_sequences(residual_features_full, seq_len)

    mf_model = MultiFeatureLSTM(input_size=5, hidden_size=32, dropout=0.2)
    mf_model = train_lstm_model(mf_model, X_r, y_r)

    mf_model.eval()
    with torch.no_grad():
        pred_r_train = mf_model(torch.FloatTensor(X_r).to(DEVICE)).cpu().numpy().flatten()
        pred_r_test = mf_model(torch.FloatTensor(X_r_test).to(DEVICE)).cpu().numpy().flatten()
    lstm_preds_train.append(pred_r_train)
    lstm_preds_test.append(pred_r_test)

    # 3. 4个模态分量 → 单特征LSTM
    for idx in modal_indices:
        modal_train = u_train[idx]
        modal_test = u_test[idx]
        modal_full = np.concatenate([modal_train[-seq_len:], modal_test])

        X_m, y_m = create_sequences(modal_train.reshape(-1, 1), seq_len)
        X_m_test, _ = create_sequences(modal_full.reshape(-1, 1), seq_len)

        sf_model = SingleFeatureLSTM(hidden_size=16, dropout=0.3)
        sf_model = train_lstm_model(sf_model, X_m, y_m)

        sf_model.eval()
        with torch.no_grad():
            pred_m_train = sf_model(torch.FloatTensor(X_m).to(DEVICE)).cpu().numpy().flatten()
            pred_m_test = sf_model(torch.FloatTensor(X_m_test).to(DEVICE)).cpu().numpy().flatten()
        lstm_preds_train.append(pred_m_train)
        lstm_preds_test.append(pred_m_test)

    # 4. CatBoost 融合
    # 输入：5个LSTM预测值 + 4个影响因子 = 9维
    fusion_train = np.column_stack(lstm_preds_train + [X_train_factors[seq_len:]])
    fusion_test = np.column_stack(lstm_preds_test + [X_test_factors])

    fusion_model = CatBoostRegressor(
        iterations=300, learning_rate=0.03, depth=3, l2_leaf_reg=5,
        loss_function='RMSE', early_stopping_rounds=20,
        random_seed=RANDOM_SEED, verbose=0
    )
    fusion_model.fit(fusion_train, y_train[seq_len:],
                     eval_set=(fusion_train[-3:], y_train[seq_len:][-3:]))

    y_pred_fusion = fusion_model.predict(fusion_test)
    importance = fusion_model.get_feature_importance()

    # 对齐测试集长度
    effective_test_len = len(y_pred_fusion)
    y_test_aligned = y_test[-effective_test_len:]

    y_test_orig = demand_scaler.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred_fusion.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, omega, u, fusion_model


# ===================== 9. 模型评估 =====================
def evaluate_model(y_true, y_pred):
    """计算 MSE/RMSE/MAE/R²"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': round(mse, 4), 'RMSE': round(rmse, 4),
            'MAE': round(mae, 4), 'R2': round(r2, 4)}


# ===================== 10. 可视化 =====================
def plot_prediction_comparison(all_results):
    """预测对比曲线：每种物资一张图，含三模型预测 vs 真实值"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    months = pd.date_range('2024-01-01', periods=12, freq='MS')
    colors = {'CatBoost': '#2196F3', 'VMD-CatBoost': '#4CAF50', 'VMD-LSTM-CatBoost': '#FF5722'}
    markers = {'CatBoost': 'o', 'VMD-CatBoost': '^', 'VMD-LSTM-CatBoost': 'D'}
    y_labels = {'cable': '需求量 (10千米)', 'transformer': '需求量 (套)', 'arrester': '需求量 (台)'}

    for ax_idx, material in enumerate(MATERIALS):
        ax = axes[ax_idx]
        results = all_results[material]
        # 对齐：模型三的测试集可能短于12个月（LSTM窗口效应）
        test_len = len(results['CatBoost']['y_test'])
        x = months[:test_len]

        ax.plot(x, results['CatBoost']['y_test'][:len(x)], color='black', marker='o',
                linestyle='solid', label='真实值', markersize=5, linewidth=2)

        # 收集各模型 R² 用于标题
        r2_parts = []
        for model_name in ['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost']:
            if model_name in results:
                pred = results[model_name]['y_pred']
                pred_x = months[:len(pred)]
                ax.plot(pred_x, pred, color=colors[model_name], marker=markers[model_name],
                        linestyle='solid', label=model_name,
                        markersize=4, alpha=0.85)
                r2 = results[model_name]['metrics']['R2']
                r2_parts.append(f'{model_name} R²={r2:.3f}')

        title = f'{MATERIAL_LABELS[material]}\n({", ".join(r2_parts)})'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel(y_labels[material])
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'prediction_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 预测对比图 → {path}")


def plot_vmd_decomposition(demand_full, u, omega, material):
    """VMD分解可视化：原始信号+5个IMF分量"""
    fig, axes = plt.subplots(VMD_K + 1, 1, figsize=(14, 10))
    t = np.arange(len(demand_full))

    # 原始信号
    axes[0].plot(t, demand_full, 'k-', linewidth=1.5)
    axes[0].set_title(f'{MATERIAL_LABELS[material]} — 原始需求量序列', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('需求量')
    axes[0].grid(True, alpha=0.3)

    # 各IMF分量
    final_freqs = omega[-1]
    for i in range(VMD_K):
        axes[i + 1].plot(t, u[i], linewidth=1)
        axes[i + 1].set_ylabel(f'IMF{i+1}\n(f={final_freqs[i]:.3f})')
        axes[i + 1].grid(True, alpha=0.3)
        if i == VMD_K - 1:
            axes[i + 1].set_xlabel('月份序号')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'vmd_decomposition_{material}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] VMD分解图({material}) → {path}")


def plot_feature_importance(importance_dict, material):
    """特征重要性条形图（每种物资的CatBoost和VMD-CatBoost）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (model_name, imp, feat_names) in enumerate([
        ('CatBoost', importance_dict.get('catboost_imp'),
         get_top_factors(material)),
        ('VMD-CatBoost', importance_dict.get('vmd_catboost_imp'),
         [f'IMF{i+1}' for i in range(VMD_K)] + get_top_factors(material))
    ]):
        ax = axes[ax_idx]
        if imp is not None:
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp)))
            ax.barh(range(len(imp)), imp, color=colors, edgecolor='navy', alpha=0.85)
            ax.set_yticks(range(len(imp)))
            ax.set_yticklabels(feat_names)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} — {MATERIAL_LABELS[material]}', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, '无特征重要性数据', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'feature_importance_{material}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 特征重要性({material}) → {path}")


def plot_metrics_comparison(all_metrics):
    """模型指标对比：分组柱状图（各物资各模型的四项指标）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_names = ['MSE', 'RMSE', 'MAE', 'R2']
    model_names = ['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost']
    colors = ['#2196F3', '#4CAF50', '#FF5722']

    for ax_idx, metric in enumerate(metric_names):
        ax = axes[ax_idx // 2, ax_idx % 2]
        x = np.arange(len(MATERIALS))
        width = 0.25

        for i, model_name in enumerate(model_names):
            values = []
            for material in MATERIALS:
                vals = all_metrics[material].get(model_name, {})
                values.append(vals.get(metric, 0))
            bars = ax.bar(x + i * width, values, width, label=model_name,
                          color=colors[i], alpha=0.85, edgecolor='white')

            # 数值标注
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([MATERIAL_LABELS[m] for m in MATERIALS])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('模型评估指标对比', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 指标对比图 → {path}")


def print_metrics_table(all_metrics):
    """打印评估指标汇总表（同时输出到控制台和日志）"""
    lines = []
    lines.append("")
    lines.append("=" * 120)
    lines.append("评估指标汇总表")
    lines.append("=" * 120)
    header = f"{'物资':<16s} {'模型':<22s} {'MSE':>12s} {'RMSE':>12s} {'MAE':>12s} {'R^2':>12s}"
    lines.append(header)
    lines.append("-" * 120)

    for material in MATERIALS:
        for i, model_name in enumerate(['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost']):
            metrics = all_metrics[material].get(model_name, {})
            if metrics:
                if i == 0:
                    lines.append(f"{MATERIAL_LABELS[material]:<16s} {model_name:<22s} "
                                 f"{metrics['MSE']:>12.4f} {metrics['RMSE']:>12.4f} "
                                 f"{metrics['MAE']:>12.4f} {metrics['R2']:>12.4f}")
                else:
                    lines.append(f"{'':<16s} {model_name:<22s} "
                                 f"{metrics['MSE']:>12.4f} {metrics['RMSE']:>12.4f} "
                                 f"{metrics['MAE']:>12.4f} {metrics['R2']:>12.4f}")
        lines.append("-" * 120)
    lines.append("")

    for line in lines:
        logger.info(line)


# ===================== 11. 主函数 =====================
def main():
    # Step 1: 加载数据
    logger.info("=" * 70)
    logger.info("  配电网物资需求预测 —— VMD-CatBoost 模型对比实验")
    logger.info("  物资: 10KV电缆 / 柱上变压器台成套设备 / 10kv交流避雷器")
    logger.info("  模型: CatBoost / VMD-CatBoost / VMD-LSTM-CatBoost")
    logger.info("=" * 70)
    logger.info(f"  日志文件: {log_filename}")
    logger.info("[1/6] 加载数据...")
    data_dict = load_or_generate_data()

    # Step 2: 初始化结果容器
    all_results = {}
    all_metrics = {}

    # Step 3+4：建模与可视化（抑制 Windows 字体权限错误噪音）
    with suppress_font_stderr():
        # Step 3: 对三种物资分别建模
        for material in MATERIALS:
            label = MATERIAL_LABELS[material]
            logger.info("")
            logger.info(f"[2/6] 处理 {label} ({material})...")
            df = data_dict[material]
            top4 = get_top_factors(material)
            logger.info(f"  Top-4 影响因子: {top4}")

            X_train_factors, y_train, X_test_factors, y_test, scaler, demand_scaler = \
                preprocess_data(df, material)
            demand_full_scaled = np.concatenate([y_train, y_test])
            logger.debug(f"  训练集: {len(y_train)}月, 测试集: {len(y_test)}月")

            all_results[material] = {}
            all_metrics[material] = {}

            # --- 模型一: CatBoost ---
            logger.info(f"  [3/6] 模型一: CatBoost...")
            y_pred_1, y_test_1, imp_1, model_1 = run_catboost(
                X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler)
            metrics_1 = evaluate_model(y_test_1, y_pred_1)
            all_results[material]['CatBoost'] = {
                'y_pred': y_pred_1, 'y_test': y_test_1, 'metrics': metrics_1}
            all_metrics[material]['CatBoost'] = metrics_1
            logger.info(f"        MSE={metrics_1['MSE']:.4f} RMSE={metrics_1['RMSE']:.4f} "
                         f"MAE={metrics_1['MAE']:.4f} R^2={metrics_1['R2']:.4f}")

            # --- 模型二: VMD-CatBoost ---
            logger.info(f"  [4/6] 模型二: VMD-CatBoost...")
            y_pred_2, y_test_2, imp_2, omega_2, u_2, model_2 = run_vmd_catboost(
                X_train_factors, y_train, X_test_factors, y_test,
                demand_full_scaled, material, demand_scaler)
            metrics_2 = evaluate_model(y_test_2, y_pred_2)
            all_results[material]['VMD-CatBoost'] = {
                'y_pred': y_pred_2, 'y_test': y_test_2, 'metrics': metrics_2}
            all_metrics[material]['VMD-CatBoost'] = metrics_2
            logger.info(f"        MSE={metrics_2['MSE']:.4f} RMSE={metrics_2['RMSE']:.4f} "
                         f"MAE={metrics_2['MAE']:.4f} R^2={metrics_2['R2']:.4f}")

            # VMD 分解可视化
            plot_vmd_decomposition(demand_full_scaled, u_2, omega_2, material)

            # --- 模型三: VMD-LSTM-CatBoost ---
            logger.info(f"  [5/6] 模型三: VMD-LSTM-CatBoost...")
            y_pred_3, y_test_3, imp_3, omega_3, u_3, model_3 = run_vmd_lstm_catboost(
                X_train_factors, y_train, X_test_factors, y_test,
                demand_full_scaled, material, demand_scaler)
            metrics_3 = evaluate_model(y_test_3, y_pred_3)
            all_results[material]['VMD-LSTM-CatBoost'] = {
                'y_pred': y_pred_3, 'y_test': y_test_3, 'metrics': metrics_3}
            all_metrics[material]['VMD-LSTM-CatBoost'] = metrics_3
            logger.info(f"        MSE={metrics_3['MSE']:.4f} RMSE={metrics_3['RMSE']:.4f} "
                         f"MAE={metrics_3['MAE']:.4f} R^2={metrics_3['R2']:.4f}")

            # 特征重要性图
            imp_dict = {
                'catboost_imp': imp_1,
                'vmd_catboost_imp': imp_2,
                'vmd_lstm_catboost_imp': imp_3,
            }
            plot_feature_importance(imp_dict, material)

        # Step 4: 评估汇总
        logger.info("")
        logger.info("[6/6] 汇总评估与可视化...")
        print_metrics_table(all_metrics)

        # 保存指标 JSON
        metrics_json = {}
        for material in MATERIALS:
            metrics_json[material] = {}
            for model_name in ['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost']:
                if model_name in all_metrics[material]:
                    metrics_json[material][model_name] = all_metrics[material][model_name]
        json_path = os.path.join(OUTPUT_DIR, 'metrics_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)
        logger.info(f"  [文件] 指标JSON → {json_path}")

        # 预测对比图
        plot_prediction_comparison(all_results)

        # 指标对比柱状图
        plot_metrics_comparison(all_metrics)

    # 最佳模型识别（纯日志输出，无需抑制）
    logger.info("")
    logger.info("=" * 70)
    logger.info("  模型性能排序 (按R^2)")
    logger.info("=" * 70)
    for material in MATERIALS:
        sorted_models = sorted(
            [(k, v['R2']) for k, v in all_metrics[material].items()],
            key=lambda x: x[1], reverse=True)
        best = sorted_models[0]
        logger.info(f"  {MATERIAL_LABELS[material]}: "
                     f"[1st] {best[0]} (R^2={best[1]:.4f})  "
                     f"[2nd] {sorted_models[1][0]} (R^2={sorted_models[1][1]:.4f})  "
                     f"[3rd] {sorted_models[2][0]} (R^2={sorted_models[2][1]:.4f})")

    logger.info(f"")
    logger.info(f"所有图表已保存至: {os.path.abspath(OUTPUT_DIR)}/")
    logger.info(f"日志文件: {os.path.abspath(log_filename)}")
    logger.info("完成!")


if __name__ == '__main__':
    main()
