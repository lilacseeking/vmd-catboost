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
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
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
                'typhoon_count', 'lightning_count', 'rainstorm_count']
FACTOR_LABELS = {'load_growth': '负荷增长量(分)', 'investment': '工程投资量',
                 'history_demand': '历史需求量', 'equipment_cost': '设备进价成本(万元)',
                 'typhoon_count': '台风(分)', 'lightning_count': '雷击(分)',
                 'rainstorm_count': '暴雨(分)'}
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
    'rainstorm_count': '暴雨(分)',
}
COLUMN_EN = {v: k for k, v in COLUMN_CN.items()}


def _generate_all_data(months):
    """根据数据生成要求和影响因素说明生成三种物资的模拟数据，确保'统一'因素共享"""
    n = len(months)
    t = np.arange(n)
    np.random.seed(RANDOM_SEED)

    # ===== 统一因素（同一月份所有相关物资共享） =====
    # 负荷增长量: 统一，范围 0.00-1.00，夏季峰值最高，增加随机年际波动
    summer = np.clip(np.sin(np.pi * ((t % 12) - 3) / 6), 0, 1)
    load_growth_base = 0.15 + summer * 0.7
    load_growth = load_growth_base + np.random.randn(n) * 0.08 + np.sin(np.arange(n) * 0.15) * 0.05
    load_growth = np.clip(np.round(load_growth, 3), 0, 1)

    # 雷击: 统一，范围 0.00-1.00 分，5-9月峰值最高，增加过渡月份平滑值
    in_lightning = np.isin(t % 12, [4, 5, 6, 7, 8])
    lightning_count = np.where(in_lightning,
                               np.random.uniform(0.3, 1.0, n),
                               np.random.uniform(0, 0.4, n))  # 过渡月有低速值
    lightning_count = np.clip(np.round(lightning_count, 3), 0, 1)

    # 台风: 连续值(台风影响天数)，6-10月台风季，范围 0.00-1.00
    in_typhoon = np.isin(t % 12, [5, 6, 7, 8, 9])
    typhoon_seasonal = np.where(in_typhoon,
                                np.random.uniform(0.2, 1.0, n),
                                np.random.uniform(0, 0.15, n))
    typhoon_count = np.clip(np.round(typhoon_seasonal + np.random.randn(n) * 0.1, 3), 0, 1)

    # 暴雨: 统一，范围 0.00-1.00 分，3-8月暴雨集中期较高
    in_rainstorm = np.isin(t % 12, [2, 3, 4, 5, 6, 7])
    rainstorm_count = np.where(in_rainstorm,
                               np.random.uniform(0.3, 1.0, n),
                               np.random.uniform(0, 0.35, n))
    rainstorm_count = np.clip(np.round(rainstorm_count, 3), 0, 1)

    data_dict = {}

    # ====================================================================
    # 10KV电缆
    # 需求: 冬季(12-2月)为0；非0时 10-30 (10千米)，增大波动性
    # 因子: investment(#1), history_demand(#2), load_growth(#3), equipment_cost(#4)
    # ====================================================================
    cable_low = np.isin(t % 12, [0, 1, 11])
    cable_noise = np.random.randn(n) * 5 + np.sin(np.arange(n) * 0.3) * 2  # 年际波动
    cable_raw = 20 + np.sin(2 * np.pi * t / 12) * 4 + 0.2 * t + cable_noise
    cable_demand = np.where(cable_low, 0, np.clip(np.round(cable_raw), 10, 30))
    cable_demand = np.maximum(cable_demand, 0)

    cable_inv_zero = np.isin(t % 12, [0, 4, 8])
    cable_investment = np.where(cable_inv_zero,
                                np.round(np.random.uniform(0, 8, n)),  # 零值月也有小额投资
                                np.round(np.random.uniform(12, 28, n)))
    # 历史需求量 = 滞后一期 + 噪声（非完全等价）
    cable_history = np.roll(cable_demand, 1) * (1 + np.random.randn(n) * 0.08)
    cable_history = np.clip(np.round(cable_history), 0, None)
    cable_history[0] = 0

    cable_cost = 4.5 + np.random.randn(n) * 1.2 + np.sin(2 * np.pi * t / 12) * 1.0
    cable_cost = np.clip(np.round(cable_cost, 1), 2, 7)

    data_dict['cable'] = pd.DataFrame({
        'date': months,
        'demand': cable_demand,
        'load_growth': load_growth,
        'investment': cable_investment,
        'history_demand': cable_history,
        'equipment_cost': cable_cost,
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
    trans_investment = np.where(trans_inv_zero,
                                np.round(np.random.uniform(0, 7, n)),
                                np.round(np.random.uniform(5, 22, n)))

    trans_history = np.roll(trans_demand, 1) * (1 + np.random.randn(n) * 0.08)
    trans_history = np.clip(np.round(trans_history), 0, None)
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
    # 需求: 干季(11-3月)为0；雨季双峰: 5月(雷雨季开始)+8月(台风+雷暴高发)
    # 因子: lightning_count(#1), typhoon_count(#2), rainstorm_count(#3), load_growth(#4)
    # ====================================================================
    arr_low = np.isin(t % 12, [0, 1, 2, 10, 11])
    # 双高斯峰结构: 5月(μ=4)和8月(μ=7)
    month_in_year = t % 12
    peak_may = np.exp(-0.5 * ((month_in_year - 4) / 1.2) ** 2)  # 5月 峰值
    peak_aug = np.exp(-0.5 * ((month_in_year - 7) / 1.0) ** 2)  # 8月 峰值(更高)
    seasonal_dual = peak_may * 18 + peak_aug * 22  # 8月峰值高于5月
    arr_raw = 70 + seasonal_dual + 0.08 * t + np.random.randn(n) * 4
    arr_demand = np.where(arr_low, 0, np.clip(np.round(arr_raw), 55, 105))
    arr_demand = np.maximum(arr_demand, 0)

    arr_inv_zero = np.isin(t % 12, [0, 3, 7])
    arr_investment = np.where(arr_inv_zero,
                              np.round(np.random.uniform(5, 30, n)),
                              np.round(np.random.uniform(50, 150, n)))

    data_dict['arrester'] = pd.DataFrame({
        'date': months,
        'demand': arr_demand,
        'load_growth': load_growth,
        'investment': arr_investment,
        'lightning_count': lightning_count,
        'typhoon_count': typhoon_count,
        'rainstorm_count': rainstorm_count,
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
            # 验证所需因子列是否存在（top-4 或 FACTOR_NAMES 变更后可能缺失）
            required = ['demand'] + get_top_factors(material)
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.warning(f"  Sheet[{sheet}] 缺少列 {missing}，数据文件版本过旧，删除并重新生成...")
                os.remove(DATA_FILE)
                return load_or_generate_data()
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
    """返回 top-4 影响因子，按排名顺序（与论文斯皮尔曼分析一致）"""
    mapping = {
        'cable':        ['investment', 'history_demand', 'load_growth', 'equipment_cost'],
        'transformer':  ['load_growth', 'investment', 'history_demand', 'equipment_cost'],
        'arrester':     ['lightning_count', 'typhoon_count', 'rainstorm_count', 'load_growth'],
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

    return X_train_factors, y_train, X_test_factors, y_test, demand_scaler


# ===================== 4. VMD 分解 =====================
def vmd_decompose_full(signal, K=VMD_K):
    """对需求量序列进行VMD分解，返回所有IMF和残差/模态索引"""
    u, u_hat, omega = VMD(signal, VMD_ALPHA, 0, K, 0, 1, 1e-7)
    residual_idx = int(np.argmin(np.abs(omega[-1])))
    modal_indices = [i for i in range(K) if i != residual_idx]
    return u, u_hat, omega, residual_idx, modal_indices


def extrapolate_imfs(imfs_train, n_test, method='persistence'):
    """将训练集IMF外推至测试集长度（避免Look-Ahead Bias）

    VMD 仅对训练集需求量进行分解，测试期IMF通过外推获得：
    - persistence: 重复训练集最后一个IMF值（朴素但无数据泄露）
    - 这确保了模型在测试期的预测只依赖训练期已知信息
    """
    if method == 'persistence':
        last = imfs_train[-1]  # (K,)
        return np.tile(last, (n_test, 1))
    raise ValueError(f"Unknown method: {method}")


# ===================== 5. LSTM 模型定义 =====================
class MultiFeatureLSTM(nn.Module):
    """多特征LSTM: 残差分量+4因子 → 预测值（小样本过拟合优化）"""
    def __init__(self, input_size=5, hidden_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SingleFeatureLSTM(nn.Module):
    """单特征LSTM: 单个模态分量 → 预测值（小样本过拟合优化）"""
    def __init__(self, hidden_size=4):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data, seq_len=SEQ_LEN):
    """构建时间窗口序列 X:(n, seq_len, features), y:(n,)"""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    X, y_list = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y_list.append(data[i + seq_len, 0])
    return np.array(X), np.array(y_list)


def train_lstm_model(model, X, y, epochs=400, patience=50, lr=0.005, weight_decay=1e-4):
    """训练LSTM模型（小样本过拟合优化），返回训练好的模型"""
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.FloatTensor(y).to(DEVICE)

    if isinstance(model, MultiFeatureLSTM):
        arch = (f"MultiFeatureLSTM | input_size=5, hidden_size=6, num_layers=1, dropout=0(已移除) | "
                f"CNN/池化层: 无(本模型不使用卷积/池化)")
        train_cfg = (f"optimizer=Adam, lr={lr}, weight_decay={weight_decay}, epochs={epochs}, patience={patience}, "
                     f"loss=MSELoss, batch_size=full_batch(全批次), device={DEVICE} | "
                     f"训练样本数(train_samples)={len(X)}, 序列长度(seq_len)={X.shape[1]}")
        logger.info(f"  [LSTM架构] {arch}")
        logger.info(f"  [训练配置] {train_cfg}")
    elif isinstance(model, SingleFeatureLSTM):
        arch = (f"SingleFeatureLSTM | hidden_size=4, num_layers=1, dropout=0(已移除) | "
                f"CNN/池化层: 无(本模型不使用卷积/池化)")
        train_cfg = (f"optimizer=Adam, lr={lr}, weight_decay={weight_decay}, epochs={epochs}, patience={patience}, "
                     f"loss=MSELoss, batch_size=full_batch(全批次), device={DEVICE} | "
                     f"训练样本数(train_samples)={len(X)}, 序列长度(seq_len)={X.shape[1]}")
        logger.debug(f"  [LSTM架构] {arch}")
        logger.debug(f"  [训练配置] {train_cfg}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                logger.debug(f"  [LSTM收敛] epoch={epoch+1}, best_loss={best_loss:.6f}")
                break
    else:
        logger.debug(f"  [LSTM收敛] epoch={epochs}(max), best_loss={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


# ===================== 6. 模型一: CatBoost =====================
def run_catboost(X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler):
    """模型一: 直接使用4个影响因子，CatBoost回归预测"""
    # 避雷器数据含大量零值(41.7%), 需要更强拟合能力
    iters = 800 if material == 'arrester' else 500
    depth = 6 if material == 'arrester' else 4
    lr = 0.02 if material == 'arrester' else 0.03
    logger.info(f"  [CatBoost超参数] iterations={iters}, learning_rate={lr}, depth={depth}, l2_leaf_reg=5, "
                 f"loss_function=RMSE, early_stopping_rounds=30, random_seed={RANDOM_SEED} | "
                 f"输入特征数(input_features)={X_train_factors.shape[1]}")
    model = CatBoostRegressor(
        iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=5,
        loss_function='RMSE', early_stopping_rounds=30,
        random_seed=RANDOM_SEED, verbose=0
    )
    # 时序验证集: 前18月训练, 后6月(第19-24月)作为早停验证
    n_val = min(6, len(y_train) // 4)
    X_tr, X_val = X_train_factors[:-n_val], X_train_factors[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    y_pred = model.predict(X_test_factors)
    importance = model.get_feature_importance()

    # 反归一化
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, model


# ===================== 7. 模型二: VMD-CatBoost =====================
def run_vmd_catboost(X_train_factors, y_train, X_test_factors, y_test,
                     material, demand_scaler):
    """模型二: VMD(仅训练集) → IMF外推 → 全部分量+4因子 → CatBoost"""
    # VMD 仅对训练集需求量进行分解，避免 Look-Ahead Bias
    u, _, omega, _, _ = vmd_decompose_full(y_train)  # (5, 24)

    imfs_train = u.T   # (24, 5)
    # 测试期 IMF 通过 persistence 外推（无未来数据泄露）
    imfs_test = extrapolate_imfs(imfs_train, len(y_test))

    # 拼接特征: IMFs + 4个影响因子
    X_train_full = np.column_stack([imfs_train, X_train_factors])
    X_test_full = np.column_stack([imfs_test, X_test_factors])

    model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5,
        loss_function='RMSE', early_stopping_rounds=30,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_val = min(6, len(y_train) // 4)
    X_tr, X_val = X_train_full[:-n_val], X_train_full[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    y_pred = model.predict(X_test_full)
    importance = model.get_feature_importance()

    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, omega, u, model


# ===================== 8. 模型三: VMD-LSTM-CatBoost =====================
def run_vmd_lstm_catboost(X_train_factors, y_train, X_test_factors, y_test,
                          material, demand_scaler):
    """模型三: VMD(仅训练集) → 残差多特征LSTM + 4模态单特征LSTM → CatBoost融合"""
    seq_len = SEQ_LEN
    top4 = get_top_factors(material)

    # 1. VMD 仅对训练集分解，避免 Look-Ahead Bias
    u, _, omega, residual_idx, modal_indices = vmd_decompose_full(y_train)  # (5, 24)
    logger.debug(f"    VMD分解: 残差=IMF{residual_idx+1}, 模态={[f'IMF{i+1}' for i in modal_indices]}")

    # 测试期 IMF 通过 persistence 外推
    imfs_test_ext = extrapolate_imfs(u.T, len(y_test))  # (12, 5)
    u_train = u  # (5, 24)
    u_test = imfs_test_ext.T  # (5, 12)

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

    mf_model = MultiFeatureLSTM(input_size=5, hidden_size=6)
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

        sf_model = SingleFeatureLSTM(hidden_size=4)
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
    n_fusion_val = min(6, len(y_train[seq_len:]) // 3)
    fusion_model.fit(fusion_train, y_train[seq_len:],
                     eval_set=(fusion_train[-n_fusion_val:], y_train[seq_len:][-n_fusion_val:]))

    y_pred_fusion = fusion_model.predict(fusion_test)
    importance = fusion_model.get_feature_importance()

    # 对齐测试集长度
    effective_test_len = len(y_pred_fusion)
    y_test_aligned = y_test[-effective_test_len:]

    y_test_orig = demand_scaler.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred_fusion.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, omega, u, fusion_model


# ===================== 9. 模型四: VMD-LSTM（直接求和） =====================
def run_vmd_lstm_direct_sum(X_train_factors, y_train, X_test_factors, y_test,
                            material, demand_scaler):
    """模型四: VMD → 5个LSTM预测各分量 → 直接求和（无CatBoost融合层）

    消融实验: 对比 VMD-LSTM 与 VMD-LSTM-CatBoost，验证 CatBoost 融合层的必要性。
    VMD 分解满足 Σ(IMF_i) = 原始信号，直接求和有物理依据。
    """
    seq_len = SEQ_LEN
    top4 = get_top_factors(material)

    # 1. VMD 仅对训练集分解
    u, _, omega, residual_idx, modal_indices = vmd_decompose_full(y_train)
    logger.debug(f"    VMD分解: 残差=IMF{residual_idx+1}, 模态={[f'IMF{i+1}' for i in modal_indices]}")

    u_train = u  # (5, 24)
    imfs_test_ext = extrapolate_imfs(u.T, len(y_test))
    u_test = imfs_test_ext.T  # (5, 12)

    lstm_preds_train = []
    lstm_preds_test = []

    # 2. 残差分量 → 多特征LSTM
    residual_train = u_train[residual_idx]
    residual_features_train = np.column_stack([
        residual_train, X_train_factors[:, 0], X_train_factors[:, 1],
        X_train_factors[:, 2], X_train_factors[:, 3]
    ])
    residual_full_seq = np.concatenate([residual_train[-seq_len:], u_test[residual_idx]])
    factor_full_seqs = [np.concatenate([X_train_factors[-seq_len:, j], X_test_factors[:, j]])
                        for j in range(4)]
    residual_features_full = np.column_stack([residual_full_seq] + factor_full_seqs)
    residual_features_test = np.column_stack([
        u_test[residual_idx], X_test_factors[:, 0], X_test_factors[:, 1],
        X_test_factors[:, 2], X_test_factors[:, 3]
    ])

    X_r, y_r = create_sequences(residual_features_train, seq_len)
    X_r_test, _ = create_sequences(residual_features_full, seq_len)

    mf_model = MultiFeatureLSTM(input_size=5, hidden_size=6)
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
        modal_full = np.concatenate([modal_train[-seq_len:], u_test[idx]])

        X_m, y_m = create_sequences(modal_train.reshape(-1, 1), seq_len)
        X_m_test, _ = create_sequences(modal_full.reshape(-1, 1), seq_len)

        sf_model = SingleFeatureLSTM(hidden_size=4)
        sf_model = train_lstm_model(sf_model, X_m, y_m)

        sf_model.eval()
        with torch.no_grad():
            pred_m_train = sf_model(torch.FloatTensor(X_m).to(DEVICE)).cpu().numpy().flatten()
            pred_m_test = sf_model(torch.FloatTensor(X_m_test).to(DEVICE)).cpu().numpy().flatten()
        lstm_preds_train.append(pred_m_train)
        lstm_preds_test.append(pred_m_test)

    # 4. 直接求和（VMD 重构特性: ΣIMF = 原始信号）
    y_pred_sum_train = np.sum(lstm_preds_train, axis=0)
    y_pred_sum_test = np.sum(lstm_preds_test, axis=0)

    effective_test_len = len(y_pred_sum_test)
    y_test_aligned = y_test[-effective_test_len:]

    y_test_orig = demand_scaler.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred_sum_test.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, None, omega, u, None


# ===================== 10. 模型五: VMD-SVR =====================
def run_vmd_svr(X_train_factors, y_train, X_test_factors, y_test,
                material, demand_scaler):
    """模型五: VMD(仅训练集)分解 + SVR核方法端到端预测"""
    u, _, omega, _, _ = vmd_decompose_full(y_train)
    imfs_train = u.T
    imfs_test = extrapolate_imfs(imfs_train, len(y_test))
    X_train_full = np.column_stack([imfs_train, X_train_factors])
    X_test_full = np.column_stack([imfs_test, X_test_factors])

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto', 0.01, 0.1],
                  'epsilon': [0.01, 0.05, 0.1, 0.2]}
    svr = SVR(kernel='rbf')
    grid = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error',
                        n_jobs=1, verbose=0)
    grid.fit(X_train_full, y_train)
    logger.info(f"  [SVR最优参数] C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}, "
                 f"epsilon={grid.best_params_['epsilon']}")

    y_pred = grid.predict(X_test_full)
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    return y_pred_orig, y_test_orig, None, omega, u, grid


# ===================== 11. 模型评估 =====================
def evaluate_model(y_true, y_pred):
    """计算 MSE/RMSE/MAE/R²"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': round(mse, 4), 'RMSE': round(rmse, 4),
            'MAE': round(mae, 4), 'R2': round(r2, 4)}


# ===================== 10. 可视化 =====================
def plot_prediction_comparison(all_results, material):
    """预测对比曲线：单种物资独立成图，三模型预测 vs 真实值"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    months = pd.date_range('2024-01-01', periods=12, freq='MS')
    colors = {'CatBoost': '#2196F3', 'VMD-CatBoost': '#4CAF50', 'VMD-LSTM-CatBoost': '#FF5722', 'VMD-LSTM': '#795548', 'VMD-SVR': '#9C27B0'}
    markers = {'CatBoost': 'o', 'VMD-CatBoost': '^', 'VMD-LSTM-CatBoost': 'D', 'VMD-LSTM': 'v', 'VMD-SVR': 's'}
    y_units = {'cable': '(10千米)', 'transformer': '(套)', 'arrester': '(台)'}

    results = all_results[material]
    test_len = len(results['CatBoost']['y_test'])
    x = months[:test_len]

    ax.plot(x, results['CatBoost']['y_test'][:len(x)], color='black', marker='o',
            linestyle='solid', label='真实值', markersize=5, linewidth=2)

    for model_name in ['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost', 'VMD-LSTM', 'VMD-SVR']:
        if model_name in results:
            pred = results[model_name]['y_pred']
            pred_x = months[:len(pred)]
            ax.plot(pred_x, pred, color=colors[model_name], marker=markers[model_name],
                    linestyle='solid', label=model_name,
                    markersize=4, alpha=0.85)

    ax.set_xlabel('日期')
    ax.set_ylabel(f'{MATERIAL_LABELS[material]}需求量{y_units[material]}')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'prediction_comparison_{material}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 预测对比图({material}) → {path}")


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
    """特征重要性条形图（每种物资的CatBoost / VMD-CatBoost / VMD-LSTM-CatBoost）"""
    top4 = get_top_factors(material)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax_idx, (model_name, imp, feat_names) in enumerate([
        ('CatBoost', importance_dict.get('catboost_imp'), top4),
        ('VMD-CatBoost', importance_dict.get('vmd_catboost_imp'),
         [f'IMF{i+1}' for i in range(VMD_K)] + top4),
        ('VMD-LSTM-CatBoost', importance_dict.get('vmd_lstm_catboost_imp'),
         ['LSTM残差'] + [f'LSTM模态{i+1}' for i in range(VMD_K - 1)] + top4),
    ]):
        ax = axes[ax_idx]
        if imp is not None and len(imp) > 0:
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
    model_names = ['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost', 'VMD-LSTM', 'VMD-SVR']
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#795548', '#9C27B0']

    for ax_idx, metric in enumerate(metric_names):
        ax = axes[ax_idx // 2, ax_idx % 2]
        x = np.arange(len(MATERIALS))
        width = 0.16

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
        ax.set_xticks(x + width * 2)
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
        for i, model_name in enumerate(['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost', 'VMD-LSTM', 'VMD-SVR']):
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
    logger.info("  模型: CatBoost / VMD-CatBoost / VMD-LSTM-CatBoost / VMD-LSTM / VMD-SVR")
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

            X_train_factors, y_train, X_test_factors, y_test, demand_scaler = \
                preprocess_data(df, material)
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
                material, demand_scaler)
            metrics_2 = evaluate_model(y_test_2, y_pred_2)
            all_results[material]['VMD-CatBoost'] = {
                'y_pred': y_pred_2, 'y_test': y_test_2, 'metrics': metrics_2}
            all_metrics[material]['VMD-CatBoost'] = metrics_2
            logger.info(f"        MSE={metrics_2['MSE']:.4f} RMSE={metrics_2['RMSE']:.4f} "
                         f"MAE={metrics_2['MAE']:.4f} R^2={metrics_2['R2']:.4f}")

            # VMD 分解可视化（仅展示训练集部分）
            plot_vmd_decomposition(y_train, u_2, omega_2, material)

            # --- 模型三: VMD-LSTM-CatBoost ---
            logger.info(f"  [5/6] 模型三: VMD-LSTM-CatBoost...")
            y_pred_3, y_test_3, imp_3, omega_3, u_3, model_3 = run_vmd_lstm_catboost(
                X_train_factors, y_train, X_test_factors, y_test,
                material, demand_scaler)
            metrics_3 = evaluate_model(y_test_3, y_pred_3)
            all_results[material]['VMD-LSTM-CatBoost'] = {
                'y_pred': y_pred_3, 'y_test': y_test_3, 'metrics': metrics_3}
            all_metrics[material]['VMD-LSTM-CatBoost'] = metrics_3
            logger.info(f"        MSE={metrics_3['MSE']:.4f} RMSE={metrics_3['RMSE']:.4f} "
                         f"MAE={metrics_3['MAE']:.4f} R^2={metrics_3['R2']:.4f}")

            # --- 模型四: VMD-LSTM（直接求和消融实验）---
            logger.info(f"  [6/7] 模型四: VMD-LSTM(直接求和)...")
            y_pred_4, y_test_4, imp_4, omega_4, u_4, model_4 = run_vmd_lstm_direct_sum(
                X_train_factors, y_train, X_test_factors, y_test,
                material, demand_scaler)
            metrics_4 = evaluate_model(y_test_4, y_pred_4)
            all_results[material]['VMD-LSTM'] = {
                'y_pred': y_pred_4, 'y_test': y_test_4, 'metrics': metrics_4}
            all_metrics[material]['VMD-LSTM'] = metrics_4
            logger.info(f"        MSE={metrics_4['MSE']:.4f} RMSE={metrics_4['RMSE']:.4f} "
                         f"MAE={metrics_4['MAE']:.4f} R^2={metrics_4['R2']:.4f}")

            # --- 模型五: VMD-SVR ---
            logger.info(f"  [7/8] 模型五: VMD-SVR...")
            y_pred_5, y_test_5, imp_5, omega_5, u_5, model_5 = run_vmd_svr(
                X_train_factors, y_train, X_test_factors, y_test,
                material, demand_scaler)
            metrics_5 = evaluate_model(y_test_5, y_pred_5)
            all_results[material]['VMD-SVR'] = {
                'y_pred': y_pred_5, 'y_test': y_test_5, 'metrics': metrics_5}
            all_metrics[material]['VMD-SVR'] = metrics_5
            logger.info(f"        MSE={metrics_5['MSE']:.4f} RMSE={metrics_5['RMSE']:.4f} "
                         f"MAE={metrics_5['MAE']:.4f} R^2={metrics_5['R2']:.4f}")

            # 特征重要性图
            imp_dict = {
                'catboost_imp': imp_1,
                'vmd_catboost_imp': imp_2,
                'vmd_lstm_catboost_imp': imp_3,
            }
            plot_feature_importance(imp_dict, material)

        # Step 4: 评估汇总
        logger.info("")
        logger.info("[8/8] 汇总评估与可视化...")
        print_metrics_table(all_metrics)

        # 保存指标 JSON
        metrics_json = {}
        for material in MATERIALS:
            metrics_json[material] = {}
            for model_name in ['CatBoost', 'VMD-CatBoost', 'VMD-LSTM-CatBoost', 'VMD-LSTM', 'VMD-SVR']:
                if model_name in all_metrics[material]:
                    metrics_json[material][model_name] = all_metrics[material][model_name]
        json_path = os.path.join(OUTPUT_DIR, 'metrics_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)
        logger.info(f"  [文件] 指标JSON → {json_path}")

        # 预测对比图（每种物资独立成图）
        for material in MATERIALS:
            plot_prediction_comparison(all_results, material)

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
