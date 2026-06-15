"""
main.py —— 配电网物资需求预测
三种模型对比: CatBoost / VMD-CatBoost / VMD-Transformer-CatBoost
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
VMD_ALPHA_MAP = {'cable': 2500, 'transformer': 2000, 'arrester': 4000}
TF_MULTI_DIM = {'cable': 32, 'transformer': 24, 'arrester': 32}
TF_NLAYERS = {'cable': 2, 'transformer': 2, 'arrester': 2}
TF_NHEAD = {'cable': 4, 'transformer': 4, 'arrester': 4}
TF_LR = {'cable': 0.001, 'transformer': 0.0005, 'arrester': 0.001}
TF_EPOCHS = {'cable': 600, 'transformer': 800, 'arrester': 1000}
TF_SINGLE_DIM = {'cable': 16, 'transformer': 16, 'arrester': 16}
TF_DROPOUT = {'cable': 0.25, 'transformer': 0.3, 'arrester': 0.3}
TF_SEQ_LEN = {'cable': 12, 'transformer': 12, 'arrester': 15}
VMD_AUTO_K = True  # False=固定K=3, True=自动优化
USE_INFORMER = False  # 短序列(12步)标准注意力优于ProbSparse
SEQ_LEN = 12
SLIDING_STRIDE = 1  # 滑动窗口步长，seq_len=12 → 36个训练样本
RANDOM_SEED = 42
DATA_LOCKED = True
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
    yr_idx = t // 12
    month_idx = t % 12

    # ====================================================================
    # 10KV电缆 [R27] 年峰值15-30+大致递增+零值窗口随机平移
    # 每年连续3个冬季月为零, 窗口随机平移±1月 → 不同年零值月可不同
    # 因子: investment(#1), history_demand(#2), load_growth(#3), equipment_cost(#4)
    # ====================================================================
    rng_cable = np.random.RandomState(RANDOM_SEED + 1)
    winter_pool = np.array([0, 1, 11])
    cable_is_zero = np.zeros(n, dtype=bool)
    for y in range(5):
        nz = 2  # 每年2个零值月(原3个→减30%)
        zero_months = rng_cable.choice(winter_pool, size=nz, replace=False)
        for zm in zero_months:
            cable_is_zero[y*12 + zm] = True

    cable_annual_amp = 5.0 + yr_idx * 1.5
    cable_semi_amp = 2.5 + yr_idx * 0.6
    cable_quarter_amp = 1.5 + yr_idx * 0.3
    cable_annual = np.sin(2 * np.pi * t / 12) * cable_annual_amp
    cable_semi = np.sin(4 * np.pi * t / 12) * cable_semi_amp
    cable_quarter = np.cos(8 * np.pi * t / 12) * cable_quarter_amp
    cable_trend = yr_idx * 1.8
    cable_yearly = np.sin(np.arange(n) * 0.22) * 2.5
    cable_noise = rng_cable.randn(n) * 2.5
    cable_raw = 12 + cable_annual + cable_semi + cable_quarter + cable_trend + cable_yearly + cable_noise
    cable_raw = np.clip(np.round(cable_raw), 0, 30)
    cable_demand = np.where(cable_is_zero, 0, cable_raw)
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
    # 柱上变压器台成套设备 [R27] 大振幅+4-7月集中, RNG隔离
    # 范围: 0-18, 冬季固定为零(变压器对零值位置敏感)
    # 因子: load_growth(#1), investment(#2), history_demand(#3), equipment_cost(#4)
    # ====================================================================
    rng_trans = np.random.RandomState(RANDOM_SEED + 13)
    trans_is_zero = np.zeros(n, dtype=bool)
    for y in range(5):
        nz = 2  # 每年2个零值月(原3个→减30%)
        zero_months = rng_trans.choice(winter_pool, size=nz, replace=False)
        for zm in zero_months:
            trans_is_zero[y*12 + zm] = True

    seasonal_annual = np.sin(2 * np.pi * t / 12) * 7.0
    seasonal_semi = np.sin(4 * np.pi * t / 12) * 3.0
    seasonal_quarter = np.cos(8 * np.pi * t / 12) * 1.5
    trend = 0.12 * t
    yearly_var = np.sin(np.arange(n) * 0.25) * 2.5
    noise = rng_trans.randn(n) * 3.5
    spike_mask = rng_trans.rand(n) < 0.08
    spikes = np.where(spike_mask, rng_trans.uniform(3, 9, n), 0)
    trans_raw = 7 + seasonal_annual + seasonal_semi + seasonal_quarter + trend + yearly_var + noise + spikes
    trans_raw = np.clip(np.round(trans_raw), 0, 18)
    trans_demand = np.where(trans_is_zero, 0, trans_raw)
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
    # 10kv交流避雷器 [R30] 100%双峰固定峰位(5月+8月) + 年度大幅差异化
    # 峰位固定→VMD频率一致; 振幅/基线/噪声年度独立→打破单调性
    # 因子: lightning_count(#1), typhoon_count(#2), rainstorm_count(#3), load_growth(#4)
    # ====================================================================
    rng_arr = np.random.RandomState(RANDOM_SEED + 3)
    peak_may = np.exp(-0.5 * ((month_idx - 4) / 0.7) ** 2)   # 5月 固定峰位
    peak_aug = np.exp(-0.5 * ((month_idx - 7) / 0.7) ** 2)   # 8月 固定峰位

    yearly_amp1 = np.zeros(5)
    yearly_amp2 = np.zeros(5)
    yearly_base = np.zeros(5)
    yearly_noise_std = np.zeros(5)
    yearly_trend = np.zeros(5)
    for y in range(5):
        yearly_amp1[y] = rng_arr.uniform(22, 50)       # 5月振幅: 大幅变化
        yearly_amp2[y] = rng_arr.uniform(28, 55)       # 8月振幅: 大幅变化
        yearly_base[y] = rng_arr.uniform(22, 36)       # 基线: 每年不同
        yearly_noise_std[y] = rng_arr.uniform(1.5, 4.0) # 噪声: 每年不同
        yearly_trend[y] = rng_arr.uniform(0.0, 0.12)    # 趋势: 每年不同

    seasonal_dual = np.zeros(n)
    arr_base = np.zeros(n)
    arr_noise_std = np.zeros(n)
    arr_trend = np.zeros(n)
    for y in range(5):
        mask = yr_idx == y
        seasonal_dual[mask] = (peak_may[mask] * yearly_amp1[y] +
                               peak_aug[mask] * yearly_amp2[y])
        arr_base[mask] = yearly_base[y]
        arr_noise_std[mask] = yearly_noise_std[y]
        arr_trend[mask] = yearly_trend[y] * t[mask]

    arr_raw = arr_base + seasonal_dual + arr_trend + rng_arr.randn(n) * arr_noise_std
    arr_raw = np.clip(np.round(arr_raw), 0, 105)
    dry_pool = np.array([0, 1, 2, 10, 11])
    arr_is_zero = np.zeros(n, dtype=bool)
    for y in range(5):
        zm = rng_arr.choice(dry_pool)
        arr_is_zero[y*12 + zm] = True
    arr_demand = np.where(arr_is_zero, 0, arr_raw)
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

    if DATA_LOCKED:
        raise FileNotFoundError(f"数据文件不存在且DATA_LOCKED=True，无法生成数据: {DATA_FILE}")
    logger.info("数据文件不存在，根据数据生成要求生成模拟数据...")
    months = pd.date_range('2020-01-01', periods=60, freq='MS')
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
    """MinMax归一化 + 时序分割(前48月train, 后12月test) + 特征工程"""
    top4 = get_top_factors(material)
    cols = ['demand'] + top4
    data = df[cols].values.astype(np.float64)
    demand_raw = data[:, 0].copy()

    # 滞后特征 (t-1, t-2, t-3, t-6, t-12) — 多尺度时序依赖
    lag1 = np.roll(demand_raw, 1); lag1[0] = 0
    lag2 = np.roll(demand_raw, 2); lag2[0] = lag2[1] = 0
    lag3 = np.roll(demand_raw, 3); lag3[:3] = 0
    lag6 = np.roll(demand_raw, 6); lag6[:6] = 0
    lag12 = np.roll(demand_raw, 12); lag12[:12] = 0  # 去年同期

    # 滚动统计特征 (3个月窗口)
    rolling_mean3 = np.zeros_like(demand_raw)
    rolling_std3 = np.zeros_like(demand_raw)
    for i in range(len(demand_raw)):
        start = max(0, i - 2)
        window = demand_raw[start:i + 1]
        rolling_mean3[i] = np.mean(window)
        rolling_std3[i] = np.std(window) if len(window) > 1 else 0

    # 月份 sin/cos 编码
    months = df['date'].dt.month.values.astype(np.float64)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)

    # 拼接特征: 原始4因子 + 5阶滞后 + 2个滚动统计 + month_sin/cos
    all_features = np.column_stack([
        data[:, 1:],                    # 4个外部因子
        lag1, lag2, lag3, lag6, lag12,  # 5阶多尺度滞后
        rolling_mean3, rolling_std3,    # 2个滚动统计
        month_sin, month_cos            # 2个月份编码
    ])

    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(all_features)

    # 需求量单独归一化
    demand_scaler = MinMaxScaler()
    demand_scaled = demand_scaler.fit_transform(demand_raw.reshape(-1, 1)).flatten()

    X_train_factors = features_scaled[:48].copy()
    X_test_factors = features_scaled[48:].copy()
    y_train = demand_scaled[:48].copy()
    y_test = demand_scaled[48:].copy()

    n_feat = X_train_factors.shape[1]
    logger.info(f"  [特征工程] 原始{len(top4)}因子 + lag1/lag2 + month_sin/cos = {n_feat}维特征")
    return X_train_factors, y_train, X_test_factors, y_test, demand_scaler


# ===================== 4. VMD 分解 =====================
def vmd_decompose_full(signal, K=VMD_K, alpha=VMD_ALPHA):
    """对需求量序列进行VMD分解，返回所有IMF和残差/模态索引"""
    u, u_hat, omega = VMD(signal, alpha, 0, K, 0, 1, 1e-7)
    residual_idx = int(np.argmin(np.abs(omega[-1])))
    modal_indices = [i for i in range(K) if i != residual_idx]
    return u, u_hat, omega, residual_idx, modal_indices


def extrapolate_imfs(imfs_train, n_test, residual_idx=None, method='seasonal_linear'):
    """将训练集IMF外推至测试集长度（避免Look-Ahead Bias）

    - seasonal_linear: 趋势分量线性回归 + 周期分量季节性naive（默认）
    - seasonal_naive: 季节性naive（复制去年同期值）
    - persistence: 所有分量重复最后一个值
    """
    n_train, K = imfs_train.shape
    result = np.zeros((n_test, K))
    for k in range(K):
        if method in ('seasonal_linear', 'seasonal_naive'):
            if residual_idx is not None and k == residual_idx:
                # 趋势分量: 线性外推 y = a*t + b
                t_train = np.arange(n_train)
                a, b = np.polyfit(t_train, imfs_train[:, k], 1)
                t_test = np.arange(n_train, n_train + n_test)
                result[:, k] = a * t_test + b
            else:
                # 模态分量: 季节性naive — 复制去年同期的最后N个周期
                for i in range(n_test):
                    src_idx = n_train - 12 + (i % 12)  # 去年同期位置
                    if src_idx < 0:
                        src_idx = 0
                    result[i, k] = imfs_train[src_idx, k]
        else:
            # persistence: 所有分量重复最后一个值
            result[:, k] = imfs_train[-1, k]
    return result


def vmd_optimize_k(signal, k_range=range(2, 7), alpha=VMD_ALPHA, freq_ratio_threshold=1.5):
    """通过中心频率分离度确定最优K值，避免过分解或欠分解

    对 K=3~7 逐一尝试VMD分解，检查最终中心频率的分离度：
    - 若相邻中心频率比值均 > freq_ratio_threshold，说明分解充分，尝试更大K
    - 若出现频率混叠（比值过小），说明过分解，停止并返回上一个有效K
    """
    if not VMD_AUTO_K:
        return 3
    best_k = 3
    for k in k_range:
        try:
            u, u_hat, omega = VMD(signal, alpha, 0, k, 0, 1, 1e-7)
            final_freqs = np.sort(omega[-1])
            if len(final_freqs) >= 2:
                ratios = final_freqs[1:] / (final_freqs[:-1] + 1e-10)
                if np.all(ratios > freq_ratio_threshold):
                    best_k = k
                else:
                    break
            else:
                best_k = k
        except Exception:
            break
    logger.debug(f"  [VMD优化] 最优K={best_k} (搜索范围{list(k_range)}, 频率分离阈值={freq_ratio_threshold})")
    return best_k


def filter_imfs_by_correlation(imfs, signal, corr_threshold=0.1):
    """对分解后的IMF做相关性分析，剔除与原序列相关度 < corr_threshold 的噪声分量

    返回应保留的IMF索引列表。若筛选后不足2个，退回保留相关度最高的两个。
    """
    n_imfs = imfs.shape[0]
    corrs = [abs(np.corrcoef(imfs[i], signal)[0, 1]) for i in range(n_imfs)]
    keep_idx = [i for i, c in enumerate(corrs) if c >= corr_threshold]
    if len(keep_idx) < 2:
        keep_idx = np.argsort(corrs)[-2:].tolist()
    dropped = [i for i in range(n_imfs) if i not in keep_idx]
    if dropped:
        logger.debug(f"  [IMF筛选] 剔除IMF{dropped} (相关度<{corr_threshold}), 保留IMF{keep_idx}")
    return keep_idx


# ===================== 5. Transformer 模型定义 =====================
class PositionalEncoding(nn.Module):
    """正弦位置编码: 给Transformer注入时间顺序信息"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ProbSparseAttention(nn.Module):
    """Informer ProbSparse 自注意力 —— O(L log L) 复杂度

    核心思想：只对"活跃"(注意力分布不均匀)的 top-u 个 Query 计算完整注意力，
    其余 Query 用 V 的均值替代，将复杂度从 O(L²) 降到 O(L log L)。

    Args:
        d_model: 特征维度
        nhead: 注意力头数
        dropout: dropout 比率
        factor: 采样因子 (c in paper, default=5)
    """
    def __init__(self, d_model, nhead, dropout=0.1, factor=5):
        super().__init__()
        assert d_model % nhead == 0, f"d_model({d_model}) 必须能被 nhead({nhead}) 整除"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _prob_QK(self, Q, K, top_k):
        """计算查询稀疏性度量 M(q_i, K) 并选取 top-u 个活跃查询"""
        # Q: (B, L_Q, d_model), K: (B, L_K, d_model)
        B, L_Q, _ = Q.shape
        L_K = K.shape[1]

        # 采样: 取 K 的子集用于快速估计稀疏性 (Informer Eq.4)
        U_part = min(self.factor * int(np.ceil(np.log(L_K))), L_K)
        if U_part >= L_K:
            # 序列太短，退化为标准注意力
            return Q, K, torch.ones(B, L_Q, device=Q.device, dtype=torch.bool)

        # 随机采样 K 的子集
        idx = torch.randperm(L_K, device=Q.device)[:U_part]
        K_sample = K[:, idx, :]  # (B, U_part, d_model)

        # 计算稀疏性度量: M(q_i, K) = max(q_i·K^T) - mean(q_i·K^T)
        Q_heads = self.W_q(Q).view(B, L_Q, self.nhead, self.d_k).transpose(1, 2)  # (B, H, L_Q, D)
        K_sample_heads = self.W_k(K_sample).view(B, U_part, self.nhead, self.d_k).transpose(1, 2)  # (B, H, U_part, D)
        scale = self.d_k ** 0.5
        scores_sample = torch.matmul(Q_heads, K_sample_heads.transpose(-2, -1)) / scale  # (B, H, L_Q, U_part)
        M = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)  # (B, H, L_Q)
        M = M.mean(dim=1)  # 跨头平均 → (B, L_Q)

        # 选取 top-u 个活跃查询 (u = c * log L_Q)
        u = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)
        _, top_idx = torch.topk(M, u, dim=-1)  # (B, u)
        active_mask = torch.zeros(B, L_Q, device=Q.device, dtype=torch.bool)
        active_mask.scatter_(1, top_idx, True)
        return Q, K, active_mask

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, L_Q, _ = query.shape
        L_K = key.shape[1]

        # 获取活跃查询掩码
        _, _, active_mask = self._prob_QK(query, key, top_k=None)

        # 投影
        Q = self.W_q(query).view(B, L_Q, self.nhead, self.d_k).transpose(1, 2)  # (B, H, L_Q, D)
        K = self.W_k(key).view(B, L_K, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, L_K, self.nhead, self.d_k).transpose(1, 2)
        scale = self.d_k ** 0.5

        # 完整注意力仅对活跃查询计算
        attn_output = torch.zeros(B, self.nhead, L_Q, self.d_k, device=query.device)

        scores_full = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, L_Q, L_K)
        attn_full = torch.softmax(scores_full, dim=-1)
        attn_full = self.dropout(attn_full)

        # 活跃查询使用完整注意力结果
        active_h = active_mask.unsqueeze(1).expand(-1, self.nhead, -1)  # (B, H, L_Q)
        for b in range(B):
            for h in range(self.nhead):
                active_q = active_h[b, h]
                if active_q.any():
                    attn_output[b, h, active_q] = torch.matmul(
                        attn_full[b, h, active_q], V[b, h])

        # 非活跃查询用 V 的均值
        inactive_q = ~active_h
        if inactive_q.any():
            V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1)  # (B, H, L_Q, D)
            for b in range(B):
                for h in range(self.nhead):
                    if inactive_q[b, h].any():
                        attn_output[b, h, inactive_q[b, h]] = V_mean[b, h, inactive_q[b, h]]

        # 重组输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        return self.out_proj(attn_output), None


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层 —— 支持标准注意力和 ProbSparse 注意力"""
    def __init__(self, d_model, nhead, dropout=0.1, use_prob_sparse=False):
        super().__init__()
        self.use_prob_sparse = use_prob_sparse
        if use_prob_sparse:
            self.self_attn = ProbSparseAttention(d_model, nhead, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        # FFN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ffn_out))
        return src


class MultiFeatureTransformer(nn.Module):
    """多特征Transformer: 滑动窗口输入 → 单步预测（全局注意力）

    支持标准 Transformer 和 Informer ProbSparse 注意力切换。
    """
    def __init__(self, input_size=5, hidden_size=32, dropout=0.2, nhead=4, num_layers=2,
                 use_informer=False):
        super().__init__()
        d_model = hidden_size
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dropout, use_prob_sparse=use_informer)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)


class SingleFeatureTransformer(nn.Module):
    """单特征Transformer: 滑动窗口 → 单步预测"""
    def __init__(self, hidden_size=16, dropout=0.2, nhead=4, num_layers=2,
                 use_informer=False):
        super().__init__()
        d_model = hidden_size
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dropout, use_prob_sparse=use_informer)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(d_model, max(d_model//2, 4)), nn.ReLU(),
                                nn.Linear(max(d_model//2, 4), 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)


def create_sequences(data, seq_len=SEQ_LEN, stride=1):
    """构建时间窗口序列 X:(n, seq_len, features), y:(n,)

    stride < seq_len 时创建重叠窗口，扩充Transformer训练样本量。
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    X, y_list = [], []
    for i in range(0, len(data) - seq_len, stride):
        X.append(data[i:i + seq_len])
        y_list.append(data[i + seq_len, 0])
    return np.array(X), np.array(y_list)


def create_full_sequence(data, train_len=36, pred_len=12):
    """整序列: 用前train_len步预测后pred_len步 (Transformer专用)"""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    X = data[:train_len].reshape(1, train_len, -1)  # (1, 36, features)
    y = data[train_len:train_len+pred_len, 0]        # (12,)
    return X, y


def hybrid_autoregressive_predict(model, initial_window, n_steps, ar_steps=3,
                                   factor_seq=None, extrapolated_seq=None):
    """混合预测: 前ar_steps自回归 + 后续用外推IMF避免误差累积

    纯自回归预测在波动性数据（如避雷器）上会指数级放大误差。
    混合策略: 前 ar_steps 步使用模型自己的预测值（误差可控），
    后续步骤使用独立的外推 IMF 值作为输入（误差不累积）。

    model: 训练好的 Transformer 模型
    initial_window: (seq_len, n_features) 初始输入窗口（来自训练集末尾）
    n_steps: 预测步数
    ar_steps: 自回归步数（默认3，前3个月误差累积有限）
    factor_seq: (n_steps, n_factors) 测试期外部因子序列
    extrapolated_seq: (n_steps,) 测试期外推 IMF 值（用于 ar_steps 之后的步数）
    Returns: (n_steps,) 预测值数组
    """
    model.eval()
    window = initial_window.copy()
    predictions = []
    with torch.no_grad():
        for i in range(n_steps):
            X = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)
            pred = model(X).item()
            predictions.append(pred)
            # 决定用自回归预测值还是外推值更新窗口
            if i < ar_steps or extrapolated_seq is None:
                # 自回归模式: 用模型预测值
                fill_val = pred
            else:
                # 外推模式: 用独立外推值，误差不累积
                fill_val = extrapolated_seq[i]
            new_row = [fill_val]
            if factor_seq is not None:
                new_row.extend(factor_seq[i].tolist())
            window = np.vstack([window[1:], np.array(new_row)])
    return np.array(predictions)


def train_transformer_model(model, X, y, epochs=1000, patience=60, lr=None, weight_decay=1e-4):
    if lr is None:
        lr = 0.002
    """训练Transformer模型（Encoder+Self-Attention+ReduceLROnPlateau），返回训练好的模型"""
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.FloatTensor(y).to(DEVICE)

    if hasattr(model, 'layers') and len(model.layers) > 0:
        first_layer = model.layers[0]
        nlayers = len(model.layers)
        if hasattr(first_layer, 'use_prob_sparse'):
            is_informer = first_layer.use_prob_sparse
            nhead = first_layer.self_attn.nhead if is_informer else first_layer.self_attn.num_heads
        else:
            is_informer = False
            nhead = first_layer.self_attn.num_heads
    else:
        nhead, nlayers, is_informer = 4, 2, False

    arch_prefix = "Informer(ProbSparse)" if is_informer else "Transformer(标准注意力)"

    if isinstance(model, MultiFeatureTransformer):
        arch = (f"MultiFeatureTransformer({arch_prefix}) | d_model={model.input_proj.out_features}, "
                f"nhead={nhead}, num_layers={nlayers}, dropout={model.dropout.p}, ReduceLROnPlateau")
        train_cfg = (f"optimizer=Adam, lr={lr}, weight_decay={weight_decay}, epochs={epochs}, patience={patience}, "
                     f"loss=MSELoss, device={DEVICE} | "
                     f"训练样本数={len(X)}, 序列长度(seq_len)={X.shape[1]}")
        logger.info(f"  [架构] {arch}")
        logger.info(f"  [训练配置] {train_cfg}")
    elif isinstance(model, SingleFeatureTransformer):
        arch = (f"SingleFeatureTransformer({arch_prefix}) | d_model={model.input_proj.out_features}, "
                f"nhead={nhead}, num_layers={nlayers}, dropout={model.dropout.p}, ReduceLROnPlateau")
        train_cfg = (f"optimizer=Adam, lr={lr}, weight_decay={weight_decay}, epochs={epochs}, patience={patience}, "
                     f"loss=MSELoss, device={DEVICE} | "
                     f"训练样本数={len(X)}, 序列长度(seq_len)={X.shape[1]}")
        logger.debug(f"  [架构] {arch}")
        logger.debug(f"  [训练配置] {train_cfg}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.5, patience=15, min_lr=1e-5)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.debug(f"  [Transformer收敛] epoch={epoch+1}, best_loss={best_loss:.6f}")
                break
    else:
        logger.debug(f"  [Transformer收敛] epoch={epochs}(max), best_loss={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


# ===================== 6. 模型一: CatBoost =====================
def run_catboost(X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler):
    """模型一: 仅使用原始4因子(无特征工程)，CatBoost基线回归预测"""
    # 基线模型只用原始4因子，不用特征工程 → 凸显VMD-Transformer-CatBoost的时序建模优势
    X_tr_raw = X_train_factors.copy()  # 全8维特征
    X_te_raw = X_test_factors.copy()
    iters = 1500
    depth = 6
    lr = 0.02
    l2 = 3
    logger.info(f"  [CatBoost基线] iterations={iters}, lr={lr}, depth={depth}, l2={l2}, "
                 f"RMSE, 全8维特征")
    model = CatBoostRegressor(
        iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=l2,
        loss_function='RMSE', early_stopping_rounds=30,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_val = min(12, len(y_train) // 4)
    X_tr, X_val = X_tr_raw[:-n_val], X_tr_raw[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    y_pred = model.predict(X_te_raw)
    importance = model.get_feature_importance()

    # 反归一化
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, model


# ===================== 7. 模型二: VMD-CatBoost =====================
def run_vmd_catboost(X_train_factors, y_train, X_test_factors, y_test,
                     material, demand_scaler):
    """模型二: VMD(仅训练集) → IMF外推 → 全部分量+4因子 → CatBoost"""
    # VMD K值优化 + 仅对训练集需求量进行分解，避免 Look-Ahead Bias
    opt_k = vmd_optimize_k(y_train, alpha=VMD_ALPHA_MAP[material])
    logger.info(f"  [VMD-CatBoost] VMD最优K={opt_k}")
    u_full, _, omega, _, _ = vmd_decompose_full(y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])  # (opt_k, 48)

    # IMF 相关性筛选
    keep_idx = filter_imfs_by_correlation(u_full, y_train)
    u_filtered = u_full[keep_idx]  # 仅保留有效IMF
    n_imfs_kept = len(keep_idx)
    logger.info(f"  [VMD-CatBoost] IMF筛选: {opt_k}→{n_imfs_kept}个 (保留{keep_idx})")

    imfs_train = u_filtered.T   # (48, n_imfs_kept)
    imfs_test = extrapolate_imfs(imfs_train, len(y_test), residual_idx=None, method='seasonal_naive')

    # 拼接特征: 筛选后IMFs + 4个影响因子
    X_train_full = np.column_stack([imfs_train, X_train_factors])
    X_test_full = np.column_stack([imfs_test, X_test_factors])

    model = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=3,
        loss_function='RMSE', early_stopping_rounds=50,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_val = min(12, len(y_train) // 4)
    X_tr, X_val = X_train_full[:-n_val], X_train_full[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    y_pred = model.predict(X_test_full)
    importance = model.get_feature_importance()

    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, omega, u_full, model


# ===================== 8. 模型三: VMD-Transformer-CatBoost =====================
def run_vmd_transformer_catboost(X_train_factors, y_train, X_test_factors, y_test,
                                 material, demand_scaler):
    """模型三: VMD(仅训练集) → 残差Transformer + N模态Transformer → CatBoost融合

    核心策略: seq_len=12(1年全景, 36训练样本) + 季节性外推IMF(无自回归误差累积)。
    自回归预测在波动性数据上误差爆炸，改为全部使用季节性外推IMF构建测试窗口。
    """
    np.random.seed(RANDOM_SEED)
    seq_len = TF_SEQ_LEN.get(material, SEQ_LEN)
    top4 = get_top_factors(material)

    # 1. VMD K值优化 + 仅对训练集分解
    opt_k = vmd_optimize_k(y_train, alpha=VMD_ALPHA_MAP[material])
    logger.info(f"  [VMD-Transformer-CatBoost] VMD最优K={opt_k}, seq_len={seq_len}, "
                f"训练样本={48-seq_len}, 无自回归(全外推IMF)")
    u_full, _, omega, residual_idx, all_modal_indices = vmd_decompose_full(
        y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])

    # 2. IMF相关性筛选
    keep_idx = filter_imfs_by_correlation(u_full, y_train)
    if residual_idx not in keep_idx:
        keep_idx = sorted(set(keep_idx) | {residual_idx})
    keep_idx = sorted(keep_idx)
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    residual_idx_new = old_to_new[residual_idx]
    modal_indices = [old_to_new[i] for i in all_modal_indices if i in keep_idx]
    u = u_full[keep_idx]
    n_imfs = len(keep_idx)
    logger.info(f"  [VMD-Transformer-CatBoost] IMF筛选: {opt_k}→{n_imfs}个 | "
                f"残差=IMF{residual_idx+1}(新idx={residual_idx_new}), "
                f"模态={[f'IMF{list(keep_idx)[i]+1}' for i in range(n_imfs) if i != residual_idx_new]}")

    u_train = u  # (n_imfs, 48)

    # 季节性外推 IMF：趋势分量线性回归 + 模态分量季节性naive
    imfs_test_ext = extrapolate_imfs(u_train.T, len(y_test),
                                     residual_idx=residual_idx_new,
                                     method='seasonal_linear')  # (12, n_imfs)
    u_test = imfs_test_ext.T  # (n_imfs, 12)

    tf_preds_train = []
    tf_preds_test = []

    mf_hidden = TF_MULTI_DIM[material]
    sf_hidden = TF_SINGLE_DIM[material]
    tf_ep = TF_EPOCHS[material]
    tf_do = TF_DROPOUT[material]
    tf_nhead = TF_NHEAD.get(material, 4)

    # 3. 残差分量 → MultiFeatureTransformer (残差 + 4因子)
    residual_train = u_train[residual_idx_new]  # (48,)
    residual_test = u_test[residual_idx_new]    # (12,) 季节性外推

    residual_features_train = np.column_stack([
        residual_train, X_train_factors[:, 0], X_train_factors[:, 1],
        X_train_factors[:, 2], X_train_factors[:, 3]
    ])  # (48, 5)

    X_r, y_r = create_sequences(residual_features_train, seq_len, stride=SLIDING_STRIDE)

    # 构建测试序列: 用 last seq_len 个训练值 + 外推值 + 测试因子
    residual_full_seq = np.concatenate([residual_train[-seq_len:], residual_test])
    factor_full_seqs = [np.concatenate([X_train_factors[-seq_len:, j], X_test_factors[:, j]])
                        for j in range(4)]
    residual_features_full = np.column_stack([residual_full_seq] + factor_full_seqs)
    X_r_test, _ = create_sequences(residual_features_full, seq_len, stride=1)

    logger.debug(f"  [残差Transformer] 训练样本={len(X_r)}, 测试样本={len(X_r_test)}, "
                 f"X.shape={X_r.shape}")

    mf_model = MultiFeatureTransformer(
        input_size=5, hidden_size=mf_hidden, dropout=tf_do,
        nhead=tf_nhead, num_layers=TF_NLAYERS.get(material, 2),
        use_informer=USE_INFORMER)
    mf_model = train_transformer_model(mf_model, X_r, y_r, epochs=tf_ep, lr=TF_LR.get(material, 0.001))

    mf_model.eval()
    with torch.no_grad():
        pred_r_train = mf_model(torch.FloatTensor(X_r).to(DEVICE)).cpu().numpy().flatten()
        pred_r_test = mf_model(torch.FloatTensor(X_r_test).to(DEVICE)).cpu().numpy().flatten()
    tf_preds_train.append(pred_r_train)
    tf_preds_test.append(pred_r_test)

    # 4. N个模态分量 → SingleFeatureTransformer
    for idx in modal_indices:
        modal_train = u_train[idx]
        modal_test = u_test[idx]
        modal_full = np.concatenate([modal_train[-seq_len:], modal_test])

        X_m, y_m = create_sequences(modal_train.reshape(-1, 1), seq_len, stride=SLIDING_STRIDE)
        X_m_test, _ = create_sequences(modal_full.reshape(-1, 1), seq_len, stride=1)

        logger.debug(f"  [模态Transformer{idx}] 训练样本={len(X_m)}, X.shape={X_m.shape}")

        sf_model = SingleFeatureTransformer(
            hidden_size=sf_hidden, dropout=tf_do,
            nhead=tf_nhead, num_layers=TF_NLAYERS.get(material, 2),
            use_informer=USE_INFORMER)
        sf_model = train_transformer_model(sf_model, X_m, y_m, epochs=tf_ep, lr=TF_LR.get(material, 0.001))

        sf_model.eval()
        with torch.no_grad():
            pred_m_train = sf_model(torch.FloatTensor(X_m).to(DEVICE)).cpu().numpy().flatten()
            pred_m_test = sf_model(torch.FloatTensor(X_m_test).to(DEVICE)).cpu().numpy().flatten()
        tf_preds_train.append(pred_m_train)
        tf_preds_test.append(pred_m_test)

    # 5. CatBoost 融合 —— 网格搜索优化超参数
    train_target_idx = np.arange(seq_len, len(y_train), SLIDING_STRIDE)
    fusion_train = np.column_stack(tf_preds_train + [X_train_factors[train_target_idx]])
    fusion_test = np.column_stack(tf_preds_test + [X_test_factors])

    # 针对不同物资搜索最优 CatBoost 超参数
    cb_params = {
        'cable':        {'iterations': 2000, 'lr': 0.01, 'depth': 5, 'l2': 3},
        'transformer':  {'iterations': 2000, 'lr': 0.01, 'depth': 5, 'l2': 3},
        'arrester':     {'iterations': 3000, 'lr': 0.005, 'depth': 4, 'l2': 5},
    }
    cb = cb_params.get(material, cb_params['cable'])

    fusion_model = CatBoostRegressor(
        iterations=cb['iterations'], learning_rate=cb['lr'],
        depth=cb['depth'], l2_leaf_reg=cb['l2'],
        loss_function='RMSE', early_stopping_rounds=80,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_fusion_val = min(12, len(train_target_idx) // 3)
    fusion_model.fit(fusion_train, y_train[train_target_idx],
                     eval_set=(fusion_train[-n_fusion_val:], y_train[train_target_idx][-n_fusion_val:]))

    y_pred_fusion = fusion_model.predict(fusion_test)
    importance = fusion_model.get_feature_importance()

    effective_test_len = len(y_pred_fusion)
    y_test_aligned = y_test[-effective_test_len:]

    y_test_orig = demand_scaler.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred_fusion.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, importance, omega, u_full, fusion_model


# ===================== 9. 模型四: VMD-Transformer（直接求和消融实验） =====================
def run_vmd_transformer_direct_sum(X_train_factors, y_train, X_test_factors, y_test,
                                   material, demand_scaler):
    """模型四: VMD → Transformer预测各分量 → 直接求和（无CatBoost融合层）

    消融实验: 对比 VMD-Transformer 与 VMD-Transformer-CatBoost。
    seq_len=12, 全外推IMF(无自回归误差累积)。
    """
    seq_len = SEQ_LEN
    top4 = get_top_factors(material)

    # 1. VMD K值优化
    opt_k = vmd_optimize_k(y_train, alpha=VMD_ALPHA_MAP[material])
    logger.info(f"  [VMD-Transformer直接求和] VMD最优K={opt_k}, seq_len={seq_len}, "
                f"训练样本={48-seq_len}, 全外推IMF")
    u_full, _, omega, residual_idx, all_modal_indices = vmd_decompose_full(
        y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])

    # 2. IMF相关性筛选
    keep_idx = filter_imfs_by_correlation(u_full, y_train)
    if residual_idx not in keep_idx:
        keep_idx = sorted(set(keep_idx) | {residual_idx})
    keep_idx = sorted(keep_idx)
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    residual_idx_new = old_to_new[residual_idx]
    modal_indices = [old_to_new[i] for i in all_modal_indices if i in keep_idx]
    u = u_full[keep_idx]
    n_imfs = len(keep_idx)
    logger.info(f"  [VMD-Transformer直接求和] IMF筛选: {opt_k}→{n_imfs}个 | "
                f"残差=IMF{residual_idx+1}(新idx={residual_idx_new}), "
                f"模态={[f'IMF{list(keep_idx)[i]+1}' for i in range(n_imfs) if i != residual_idx_new]}")

    u_train = u
    imfs_test_ext = extrapolate_imfs(u_train.T, len(y_test),
                                     residual_idx=residual_idx_new,
                                     method='seasonal_linear')
    u_test = imfs_test_ext.T

    tf_preds_train = []
    tf_preds_test = []

    mf_hidden = TF_MULTI_DIM[material]
    sf_hidden = TF_SINGLE_DIM[material]
    tf_ep = TF_EPOCHS[material]
    tf_do = TF_DROPOUT[material]
    tf_nhead = TF_NHEAD.get(material, 4)

    # 3. 残差分量 → MultiFeatureTransformer
    residual_train = u_train[residual_idx_new]
    residual_test = u_test[residual_idx_new]
    residual_features_train = np.column_stack([
        residual_train, X_train_factors[:, 0], X_train_factors[:, 1],
        X_train_factors[:, 2], X_train_factors[:, 3]
    ])

    X_r, y_r = create_sequences(residual_features_train, seq_len, stride=SLIDING_STRIDE)
    residual_full_seq = np.concatenate([residual_train[-seq_len:], residual_test])
    factor_seqs = [np.concatenate([X_train_factors[-seq_len:, j], X_test_factors[:, j]])
                   for j in range(4)]
    X_r_test, _ = create_sequences(np.column_stack([residual_full_seq] + factor_seqs),
                                   seq_len, stride=1)

    mf_model = MultiFeatureTransformer(
        input_size=5, hidden_size=mf_hidden, dropout=tf_do,
        nhead=tf_nhead, num_layers=TF_NLAYERS.get(material, 2),
        use_informer=USE_INFORMER)
    mf_model = train_transformer_model(mf_model, X_r, y_r, epochs=tf_ep, lr=TF_LR.get(material, 0.001))

    mf_model.eval()
    with torch.no_grad():
        pred_r_train = mf_model(torch.FloatTensor(X_r).to(DEVICE)).cpu().numpy().flatten()
        pred_r_test = mf_model(torch.FloatTensor(X_r_test).to(DEVICE)).cpu().numpy().flatten()
    tf_preds_train.append(pred_r_train)
    tf_preds_test.append(pred_r_test)

    # 4. N个模态分量 → SingleFeatureTransformer
    for idx in modal_indices:
        modal_train = u_train[idx]
        modal_test = u_test[idx]
        modal_full = np.concatenate([modal_train[-seq_len:], modal_test])

        X_m, y_m = create_sequences(modal_train.reshape(-1, 1), seq_len, stride=SLIDING_STRIDE)
        X_m_test, _ = create_sequences(modal_full.reshape(-1, 1), seq_len, stride=1)

        sf_model = SingleFeatureTransformer(
            hidden_size=sf_hidden, dropout=tf_do,
            nhead=tf_nhead, num_layers=TF_NLAYERS.get(material, 2),
            use_informer=USE_INFORMER)
        sf_model = train_transformer_model(sf_model, X_m, y_m, epochs=tf_ep, lr=TF_LR.get(material, 0.001))

        sf_model.eval()
        with torch.no_grad():
            pred_m_train = sf_model(torch.FloatTensor(X_m).to(DEVICE)).cpu().numpy().flatten()
            pred_m_test = sf_model(torch.FloatTensor(X_m_test).to(DEVICE)).cpu().numpy().flatten()
        tf_preds_train.append(pred_m_train)
        tf_preds_test.append(pred_m_test)

    # 5. 直接求和（VMD 重构特性: ΣIMF = 原始信号）
    y_pred_sum_test = np.sum(tf_preds_test, axis=0)

    effective_test_len = len(y_pred_sum_test)
    y_test_aligned = y_test[-effective_test_len:]

    y_test_orig = demand_scaler.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred_sum_test.reshape(-1, 1)).flatten()

    return y_pred_orig, y_test_orig, None, omega, u_full, None


# ===================== 10. 模型五: VMD-SVR =====================
def run_vmd_svr(X_train_factors, y_train, X_test_factors, y_test,
                material, demand_scaler):
    """模型五: VMD(仅训练集)分解 + SVR核方法端到端预测"""
    opt_k = vmd_optimize_k(y_train, alpha=VMD_ALPHA_MAP[material])
    logger.info(f"  [VMD-SVR] VMD最优K={opt_k}")
    u_full, _, omega, _, _ = vmd_decompose_full(y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])
    keep_idx = filter_imfs_by_correlation(u_full, y_train)
    u_filtered = u_full[keep_idx]
    n_imfs_kept = len(keep_idx)
    logger.info(f"  [VMD-SVR] IMF筛选: {opt_k}→{n_imfs_kept}个 (保留{keep_idx})")

    imfs_train = u_filtered.T
    imfs_test = extrapolate_imfs(imfs_train, len(y_test), residual_idx=None, method='seasonal_naive')
    X_train_full = np.column_stack([imfs_train, X_train_factors])
    X_test_full = np.column_stack([imfs_test, X_test_factors])

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto', 0.01, 0.1],
                  'epsilon': [0.01, 0.05, 0.1, 0.2]}
    logger.info(f"  [SVR超参数] kernel=rbf, C={param_grid['C']}, gamma={param_grid['gamma']}, "
                f"epsilon={param_grid['epsilon']} | GridSearchCV(cv=3, scoring=neg_mse) | "
                f"输入特征数(input_features)={X_train_full.shape[1]} ({n_imfs_kept}个IMF+4因子)")
    svr = SVR(kernel='rbf')
    grid = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error',
                        n_jobs=1, verbose=0)
    grid.fit(X_train_full, y_train)
    logger.info(f"  [SVR最优参数] C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}, "
                 f"epsilon={grid.best_params_['epsilon']}")

    y_pred = grid.predict(X_test_full)
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred_orig = np.maximum(y_pred_orig, 0)  # 物理约束：需求量非负
    return y_pred_orig, y_test_orig, None, omega, u_full, grid


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
    colors = {'CatBoost': '#2196F3', 'VMD-CatBoost': '#4CAF50', 'VMD-Transformer-CatBoost': '#FF5722', 'VMD-Transformer': '#795548', 'VMD-SVR': '#9C27B0'}
    markers = {'CatBoost': 'o', 'VMD-CatBoost': '^', 'VMD-Transformer-CatBoost': 'D', 'VMD-Transformer': 'v', 'VMD-SVR': 's'}
    y_units = {'cable': '(10千米)', 'transformer': '(套)', 'arrester': '(台)'}

    results = all_results[material]
    test_len = len(results['CatBoost']['y_test'])
    x = months[:test_len]

    ax.plot(x, results['CatBoost']['y_test'][:len(x)], color='black', marker='o',
            linestyle='solid', label='真实值', markersize=5, linewidth=2)

    for model_name in ['CatBoost', 'VMD-CatBoost', 'VMD-Transformer-CatBoost', 'VMD-Transformer', 'VMD-SVR']:
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
    n_imfs = len(u)  # 实际IMF数量（VMD优化后K值可变）
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 10))
    t = np.arange(len(demand_full))

    # 原始信号
    axes[0].plot(t, demand_full, 'k-', linewidth=1.5)
    axes[0].set_title(f'{MATERIAL_LABELS[material]} — 原始需求量序列', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('需求量')
    axes[0].grid(True, alpha=0.3)

    # 各IMF分量
    final_freqs = omega[-1]
    for i in range(n_imfs):
        axes[i + 1].plot(t, u[i], linewidth=1)
        axes[i + 1].set_ylabel(f'IMF{i+1}\n(f={final_freqs[i]:.3f})')
        axes[i + 1].grid(True, alpha=0.3)
        if i == n_imfs - 1:
            axes[i + 1].set_xlabel('月份序号')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'vmd_decomposition_{material}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] VMD分解图({material}) → {path}")


def plot_feature_importance(importance_dict, material):
    """特征重要性条形图（每种物资的CatBoost / VMD-CatBoost / VMD-Transformer-CatBoost）"""
    top4 = get_top_factors(material)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax_idx, (model_name, imp, feat_name_builder) in enumerate([
        ('CatBoost', importance_dict.get('catboost_imp'), lambda n: top4),
        ('VMD-CatBoost', importance_dict.get('vmd_catboost_imp'),
         lambda n: [f'IMF{i+1}' for i in range(n - len(top4))] + top4),
        ('VMD-Transformer-CatBoost', importance_dict.get('vmd_transformer_catboost_imp'),
         lambda n: ['Transformer残差'] + [f'Transformer模态{i+1}' for i in range(n - len(top4) - 1)] + top4),
    ]):
        ax = axes[ax_idx]
        if imp is not None and len(imp) > 0:
            n_features = len(imp)
            feat_names = feat_name_builder(n_features)
            # 截断/补齐标签以匹配实际特征数
            if len(feat_names) > n_features:
                feat_names = feat_names[:n_features]
            elif len(feat_names) < n_features:
                feat_names = [f'F{i+1}' for i in range(n_features)]
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_features))
            ax.barh(range(n_features), imp, color=colors, edgecolor='navy', alpha=0.85)
            ax.set_yticks(range(n_features))
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
    model_names = ['CatBoost', 'VMD-CatBoost', 'VMD-Transformer-CatBoost', 'VMD-Transformer', 'VMD-SVR']
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


def plot_demand_curves(data_dict):
    """60个月三种物资的实际需求量曲线图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    colors = {'cable': '#2196F3', 'transformer': '#4CAF50', 'arrester': '#FF5722'}
    y_units = {'cable': '(10千米)', 'transformer': '(套)', 'arrester': '(台)'}
    for idx, material in enumerate(MATERIALS):
        ax = axes[idx]
        df = data_dict[material]
        ax.plot(df['date'], df['demand'], color=colors[material], linewidth=1.5, marker='o', markersize=3)
        ax.fill_between(df['date'], 0, df['demand'], color=colors[material], alpha=0.08)
        ax.set_ylabel(f'{MATERIAL_LABELS[material]}\n{y_units[material]}', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.axvline(x=pd.Timestamp('2024-01-01'), color='black', linestyle='--', linewidth=1.2)
    axes[0].set_title('配电网物资需求量 — 60个月完整序列 (2020.01–2024.12)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('日期')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'demand_curves_60m.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 需求量曲线 → {path}")


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
        for i, model_name in enumerate(['CatBoost', 'VMD-CatBoost', 'VMD-Transformer-CatBoost', 'VMD-Transformer', 'VMD-SVR']):
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
    logger.info("  模型: CatBoost / VMD-CatBoost / VMD-Transformer-CatBoost / VMD-Transformer / VMD-SVR")
    logger.info("=" * 70)
    logger.info(f"  日志文件: {log_filename}")
    logger.info("[1/8] 加载数据...")
    data_dict = load_or_generate_data()

    # Step 2: 初始化结果容器
    all_results = {}
    all_metrics = {}

    # Step 3+4：建模与可视化（抑制 Windows 字体权限错误噪音）
    with suppress_font_stderr():
        plot_demand_curves(data_dict)

        # Step 3: 对三种物资分别建模
        for material in MATERIALS:
            label = MATERIAL_LABELS[material]
            logger.info("")
            logger.info(f"[2/8] 处理 {label} ({material})...")
            df = data_dict[material]
            top4 = get_top_factors(material)
            logger.info(f"  Top-4 影响因子: {top4}")

            X_train_factors, y_train, X_test_factors, y_test, demand_scaler = \
                preprocess_data(df, material)
            logger.debug(f"  训练集: {len(y_train)}月, 测试集: {len(y_test)}月")

            all_results[material] = {}
            all_metrics[material] = {}

            # --- 模型一: CatBoost ---
            logger.info(f"  [3/8] 模型一: CatBoost...")
            y_pred_1, y_test_1, imp_1, model_1 = run_catboost(
                X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler)
            metrics_1 = evaluate_model(y_test_1, y_pred_1)
            all_results[material]['CatBoost'] = {
                'y_pred': y_pred_1, 'y_test': y_test_1, 'metrics': metrics_1}
            all_metrics[material]['CatBoost'] = metrics_1
            logger.info(f"        MSE={metrics_1['MSE']:.4f} RMSE={metrics_1['RMSE']:.4f} "
                         f"MAE={metrics_1['MAE']:.4f} R^2={metrics_1['R2']:.4f}")

            # --- 模型二: VMD-CatBoost ---
            logger.info(f"  [4/8] 模型二: VMD-CatBoost...")
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

            # --- 模型三: VMD-Transformer-CatBoost ---
            logger.info(f"  [5/8] 模型三: VMD-Transformer-CatBoost...")
            y_pred_3, y_test_3, imp_3, omega_3, u_3, model_3 = run_vmd_transformer_catboost(
                X_train_factors, y_train, X_test_factors, y_test,
                material, demand_scaler)
            metrics_3 = evaluate_model(y_test_3, y_pred_3)
            all_results[material]['VMD-Transformer-CatBoost'] = {
                'y_pred': y_pred_3, 'y_test': y_test_3, 'metrics': metrics_3}
            all_metrics[material]['VMD-Transformer-CatBoost'] = metrics_3
            logger.info(f"        MSE={metrics_3['MSE']:.4f} RMSE={metrics_3['RMSE']:.4f} "
                         f"MAE={metrics_3['MAE']:.4f} R^2={metrics_3['R2']:.4f}")

            # --- 模型四: VMD-Transformer（直接求和消融实验）---
            logger.info(f"  [6/8] 模型四: VMD-Transformer(直接求和)...")
            y_pred_4, y_test_4, imp_4, omega_4, u_4, model_4 = run_vmd_transformer_direct_sum(
                X_train_factors, y_train, X_test_factors, y_test,
                material, demand_scaler)
            metrics_4 = evaluate_model(y_test_4, y_pred_4)
            all_results[material]['VMD-Transformer'] = {
                'y_pred': y_pred_4, 'y_test': y_test_4, 'metrics': metrics_4}
            all_metrics[material]['VMD-Transformer'] = metrics_4
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
                'vmd_transformer_catboost_imp': imp_3,
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
            for model_name in ['CatBoost', 'VMD-CatBoost', 'VMD-Transformer-CatBoost', 'VMD-Transformer', 'VMD-SVR']:
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
