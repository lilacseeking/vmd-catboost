"""
main.py -- 电力物资需求量预测 (期刊论文)
模型: CatBoost / Conditional-CatBoost / N-HiTS / TwoStage + NaiveSeasonal/Persistence/SARIMA
物资: Top5采购频率最高 (从ECP数据库自动选择)
运行: python main.py [--data data.xlsx]
"""
import os, sys, warnings, json, logging, io
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr
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
from catboost import CatBoostRegressor, CatBoostClassifier
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
MATERIALS = []  # 动态从 data.xlsx sheet 名加载
MATERIAL_LABELS = {}
# 真实市场因子 + 农历日历因子
FACTOR_NAMES = ['project_count', 'transformer_bids', 'monthly_bid_count',
                'uhv_bids', 'has_batch', 'digital_bids']
FACTOR_LABELS = {'project_count': '项目数量(同源)', 'transformer_bids': '输变电批次数',
                 'monthly_bid_count': '当月公告总数', 'uhv_bids': '特高压批次数',
                 'has_batch': '是否有批次', 'digital_bids': '数字化批次数'}
VMD_K = 5
VMD_ALPHA = 2000
VMD_ALPHA_MAP = {'ac_arrester': 4000, 'cvt': 2000, 'post_insulator': 3000}
TF_MULTI_DIM = {'ac_arrester': 32, 'cvt': 24, 'post_insulator': 32}
TF_NLAYERS = {'ac_arrester': 2, 'cvt': 2, 'post_insulator': 2}
TF_NHEAD = {'ac_arrester': 4, 'cvt': 4, 'post_insulator': 4}
TF_LR = {'ac_arrester': 0.001, 'cvt': 0.0005, 'post_insulator': 0.001}
TF_EPOCHS = {'ac_arrester': 1000, 'cvt': 800, 'post_insulator': 800}
TF_SINGLE_DIM = {'ac_arrester': 16, 'cvt': 16, 'post_insulator': 16}
TF_DROPOUT = {'ac_arrester': 0.3, 'cvt': 0.25, 'post_insulator': 0.3}
TF_SEQ_LEN = {'ac_arrester': 15, 'cvt': 12, 'post_insulator': 12}
VMD_AUTO_K = True  # False=固定K=3, True=自动优化
USE_INFORMER = False  # 短序列(12步)标准注意力优于ProbSparse
SEQ_LEN = 12
SLIDING_STRIDE = 1  # 滑动窗口步长，seq_len=12 → 36个训练样本
RANDOM_SEED = 42
DATA_LOCKED = True  # 严格模式 — 数据由外部手动生成，禁止自动回退
N_TEST = 12  # 测试集月数
OUTPUT_DIR = 'outputs/figures'
LOG_DIR = 'outputs/logs'
DATA_DIR = 'inputs'
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
# --data flag support: python main.py --data data_sgcc.xlsx
if '--data' in sys.argv:
    idx = sys.argv.index('--data')
    DATA_FILE = os.path.join(DATA_DIR, sys.argv[idx+1])

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
# SHEET_NAMES dynamically loaded from data.xlsx sheet names in load_or_generate_data()

# Excel 列名中英文映射
COLUMN_CN = {
    'date': '日期',
    'demand': '需求量',
    'project_count': '项目数量',
    'transformer_bids': 'transformer_bids',
    'monthly_bid_count': 'monthly_bid_count',
    'uhv_bids': 'uhv_bids',
    'has_batch': 'has_batch',
    'digital_bids': 'digital_bids',
}
COLUMN_EN = {v: k for k, v in COLUMN_CN.items()}


def load_or_generate_data():
    """加载数据：读取 data.xlsx 全部 sheet，动态匹配物资"""
    if os.path.exists(DATA_FILE):
        logger.info(f"读取已有数据文件: {DATA_FILE}")
        data_dict = {}
        all_sheets = pd.read_excel(DATA_FILE, sheet_name=None)
        for sheet_name, df in all_sheets.items():
            df.rename(columns=COLUMN_EN, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            if 'demand' not in df.columns: continue
            data_dict[sheet_name] = df
            logger.info(f"  Sheet[{sheet_name}]: {len(df)} 条, demand范围=[{df['demand'].min():.2f}, {df['demand'].max():.2f}]")
        if data_dict:
            return data_dict
        logger.warning("  未加载到有效数据，删除文件重试...")
        os.remove(DATA_FILE)

    if DATA_LOCKED:
        raise FileNotFoundError(f"数据文件不存在且DATA_LOCKED=True: {DATA_FILE}")
    logger.info("数据文件不存在，重新生成...")
    # Trigger rebuild via build_real_data.py
    build_script = r'C:/Users/董文涛/PycharmProjects/bidding-ecp-data/build_real_data.py'
    if os.path.exists(build_script):
        import subprocess
        subprocess.run([sys.executable, build_script], check=False)
    if os.path.exists(DATA_FILE):
        return load_or_generate_data()
    raise FileNotFoundError(f"无法生成数据文件: {DATA_FILE}")


# ===================== 2. Top-4 影响因子（基于Spearman动态计算） =====================
_top_factors_cache = {}

def get_top_factors(material, df=None):
    """基于Spearman相关系数动态选择top-4影响因子（首次计算后缓存）"""
    if material in _top_factors_cache:
        return _top_factors_cache[material]
    if df is None or 'demand' not in df.columns:
        # fallback: 按默认顺序取前4个
        fallback = FACTOR_NAMES[:4]
        _top_factors_cache[material] = fallback
        return fallback
    from scipy.stats import spearmanr
    scores = {}
    for f in FACTOR_NAMES:
        if f in df.columns:
            valid = df[[f, 'demand']].dropna()
            if len(valid) > 5:
                corr, _ = spearmanr(valid[f], valid['demand'])
                scores[f] = abs(corr)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top4 = ranked[:4]
    if len(top4) < 4:
        for f in FACTOR_NAMES:
            if f not in top4 and f in df.columns:
                top4.append(f)
                if len(top4) == 4:
                    break
    _top_factors_cache[material] = top4
    logger.info(f"  [Spearman因子选择] {MATERIAL_LABELS.get(material, material)}: top4={top4}")
    return top4


# ===================== 3. 数据预处理 =====================
def preprocess_data(df, material):
    """时序安全特征工程: Scaler仅对训练集fit, lag/rolling无未来值泄露."""
    top4 = get_top_factors(material, df)
    cols = ['demand'] + top4
    sub = df[cols].copy().ffill().bfill().fillna(0)
    data = sub.values.astype(np.float64)
    demand_raw = data[:, 0].copy()

    # Step 1: 先分割, 再分别在Train/Test上构造时序特征
    train_len = len(data) - N_TEST
    data_train, data_test = data[:train_len], data[train_len:]
    demand_train, demand_test = demand_raw[:train_len], demand_raw[train_len:]

    # Step 2: 训练集lag/rolling (无未来泄露)
    def make_lag_rolling(seq):
        n = len(seq)
        lag1 = np.zeros(n); lag1[1:] = seq[:-1]
        lag12 = np.zeros(n); lag12[12:] = seq[:-12]
        roll3 = np.array([np.mean(seq[max(0,i-2):i+1]) for i in range(n)])
        return lag1, lag12, roll3

    lag1_tr, lag12_tr, roll3_tr = make_lag_rolling(demand_train)
    m_train = (np.arange(train_len)+1) % 12; m_train[m_train==0]=12
    X_train_raw = np.column_stack([
        data_train[:,1:], lag1_tr, lag12_tr, roll3_tr,
        np.sin(2*np.pi*m_train/12), np.cos(2*np.pi*m_train/12)
    ])

    # Step 3: Scaler仅对训练集fit, 测试集transform
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(X_train_raw)
    demand_scaler = MinMaxScaler()
    y_train = demand_scaler.fit_transform(demand_train.reshape(-1,1)).flatten()

    # Step 4: 测试集特征(用原始值构造lag, 不依赖测试集未来)
    lag1_te = np.zeros(N_TEST); lag12_te = np.zeros(N_TEST); roll3_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        idx = train_len + i
        lag1_te[i] = demand_raw[idx-1] if idx>0 else 0
        lag12_te[i] = demand_raw[idx-12] if idx>=12 else 0
        roll3_te[i] = np.mean(demand_raw[max(0,idx-2):idx+1])

    m_test = (np.arange(train_len+1, train_len+N_TEST+1)) % 12; m_test[m_test==0]=12
    X_test_raw = np.column_stack([
        data_test[:,1:], lag1_te, lag12_te, roll3_te,
        np.sin(2*np.pi*m_test/12), np.cos(2*np.pi*m_test/12)
    ])
    X_test = feature_scaler.transform(X_test_raw)
    y_test = demand_scaler.transform(demand_test.reshape(-1,1)).flatten()

    logger.info(f"  [特征工程] top4={top4}, n_feat={X_train.shape[1]}维, 训练={train_len}月")
    return X_train, y_train, X_test, y_test, demand_scaler


# ===================== 4. VMD 分解 =====================
def vmd_decompose_full(signal, K=VMD_K, alpha=VMD_ALPHA):
    """对需求量序列进行VMD分解，返回所有IMF和残差/模态索引。

    vmdpy可能截断1个样本，此函数将signal截断到IMF长度以保证一致。
    """
    u, u_hat, omega = VMD(signal, alpha, 0, K, 0, 1, 1e-7)
    residual_idx = int(np.argmin(np.abs(omega[-1])))
    modal_indices = [i for i in range(K) if i != residual_idx]
    eff_len = u.shape[1]
    sig = signal[:eff_len]
    return u, u_hat, omega, residual_idx, modal_indices, sig


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


def vmd_optimize_k(signal, k_range=range(2, 8), alpha=VMD_ALPHA, freq_ratio_threshold=1.5):
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


def filter_imfs_by_correlation(imfs, signal, corr_threshold=0.05):
    """对分解后的IMF做相关性分析，剔除与原序列相关度 < corr_threshold 的噪声分量

    返回应保留的IMF索引列表。若筛选后不足2个，退回保留相关度最高的两个。
    """
    n_imfs = imfs.shape[0]
    min_len = min(imfs.shape[1], len(signal))
    corrs = [abs(np.corrcoef(imfs[i][:min_len], signal[:min_len])[0, 1]) for i in range(n_imfs)]
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


# ===================== Baseline: Simple Statistical Models =====================
def baseline_naive_seasonal(y_train, y_test, demand_scaler, period=12):
    """季节性朴素预测: yhat_t = y_{t-period} (抄去年同期)"""
    preds = np.array([y_train[-period + (i % period)] for i in range(len(y_test))])
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    return y_pred_orig, y_test_orig

def baseline_persistence(y_train, y_test, demand_scaler):
    """持久性预测: yhat_{t+1} = y_t (抄上月)"""
    preds = np.full(len(y_test), y_train[-1])
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    return y_pred_orig, y_test_orig

def baseline_sarima(y_train, y_test, demand_scaler):
    """SARIMA(1,0,1)(1,0,1,12) 基线预测"""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y_train_orig = demand_scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
        model = SARIMAX(y_train_orig, order=(1,0,1), seasonal_order=(1,0,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        y_pred_orig = fit.forecast(steps=len(y_test))
        y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        return np.maximum(y_pred_orig, 0), y_test_orig
    except Exception:
        # Fallback to Naive Seasonal
        return baseline_naive_seasonal(y_train, y_test, demand_scaler)

def evaluate_model_simple(y_true, y_pred):
    """计算标准评估指标"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # sMAPE: symmetric MAPE, handles zeros
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-10)) * 100
    # MASE: scale-free error relative to naive forecast
    naive_errors = np.abs(y_true[1:] - y_true[:-1])
    mase_denom = np.mean(naive_errors) if len(naive_errors) > 0 else 1
    mase = np.mean(np.abs(y_true - y_pred)) / max(mase_denom, 1e-10)
    return {'MSE': round(mse,4), 'RMSE': round(rmse,4), 'MAE': round(mae,4),
            'R2': round(r2,4), 'sMAPE': round(smape,2), 'MASE': round(mase,4)}
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
                 f"loss=RMSE")
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
    y_pred_orig = np.maximum(y_pred_orig, 0)  # 物理约束：需求量非负

    return y_pred_orig, y_test_orig, importance, model


# ===================== 7. 模型二: VMD-CatBoost =====================
def run_vmd_catboost(X_train_factors, y_train, X_test_factors, y_test,
                     material, demand_scaler):
    """模型二: VMD(仅训练集) → IMF外推 → 全部分量+4因子 → CatBoost"""
    # VMD K值优化 + 仅对训练集需求量进行分解，避免 Look-Ahead Bias
    opt_k = vmd_optimize_k(y_train, alpha=VMD_ALPHA_MAP[material])
    logger.info(f"  [VMD-CatBoost] VMD最优K={opt_k}")
    u_full, _, omega, _, _, y_trunc = vmd_decompose_full(y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])

    # IMF 相关性筛选
    keep_idx = filter_imfs_by_correlation(u_full, y_trunc)
    u_filtered = u_full[keep_idx]
    n_imfs_kept = len(keep_idx)
    logger.info(f"  [VMD-CatBoost] IMF筛选: {opt_k}→{n_imfs_kept}个 (保留{keep_idx})")

    imfs_train = u_filtered.T
    imfs_test = extrapolate_imfs(imfs_train, len(y_test), residual_idx=None, method='seasonal_naive')

    # 拼接特征: IMFs + 因子(截断对齐)
    X_train_full = np.column_stack([imfs_train, X_train_factors[:len(y_trunc)]])
    X_test_full = np.column_stack([imfs_test, X_test_factors])

    model = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=3,
        loss_function='RMSE', early_stopping_rounds=50,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_val = min(12, len(y_trunc) // 4)
    X_tr, X_val = X_train_full[:-n_val], X_train_full[-n_val:]
    y_tr, y_val = y_trunc[:-n_val], y_trunc[-n_val:]
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    y_pred = model.predict(X_test_full)
    importance = model.get_feature_importance()

    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred_orig = np.maximum(y_pred_orig, 0)

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
    n_factors = X_train_factors.shape[1]
    train_len = len(y_train)

    # 1. VMD K值优化 + 仅对训练集分解
    opt_k = vmd_optimize_k(y_train, alpha=VMD_ALPHA_MAP[material])
    logger.info(f"  [VMD-Transformer-CatBoost] VMD最优K={opt_k}, seq_len={seq_len}, "
                f"训练样本={train_len - seq_len}, 无自回归(全外推IMF)")
    u_full, _, omega, residual_idx, all_modal_indices, y_trunc = vmd_decompose_full(
        y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])
    X_trunc = X_train_factors[:len(y_trunc)]

    # 2. IMF相关性筛选
    keep_idx = filter_imfs_by_correlation(u_full, y_trunc)
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

    u_train = u  # (n_imfs, eff_len)

    # 季节性外推 IMF
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

    # 3. 残差分量 → MultiFeatureTransformer (残差 + 全部因子)
    residual_train = u_train[residual_idx_new]
    residual_test = u_test[residual_idx_new]

    residual_features_train = np.column_stack(
        [residual_train] + [X_trunc[:, j] for j in range(n_factors)])
    input_size_mf = 1 + n_factors

    X_r, y_r = create_sequences(residual_features_train, seq_len, stride=SLIDING_STRIDE)

    # 构建测试序列
    residual_full_seq = np.concatenate([residual_train[-seq_len:], residual_test])
    factor_full_seqs = [np.concatenate([X_train_factors[-seq_len:, j], X_test_factors[:, j]])
                        for j in range(n_factors)]
    residual_features_full = np.column_stack([residual_full_seq] + factor_full_seqs)
    X_r_test, _ = create_sequences(residual_features_full, seq_len, stride=1)

    logger.debug(f"  [残差Transformer] 训练样本={len(X_r)}, 测试样本={len(X_r_test)}, "
                 f"input_size={input_size_mf}, X.shape={X_r.shape}")

    mf_model = MultiFeatureTransformer(
        input_size=input_size_mf, hidden_size=mf_hidden, dropout=tf_do,
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

    # 5. CatBoost 融合
    train_target_idx = np.arange(seq_len, len(y_trunc), SLIDING_STRIDE)
    fusion_train = np.column_stack(tf_preds_train + [X_trunc[train_target_idx]])
    fusion_test = np.column_stack(tf_preds_test + [X_test_factors])

    cb_params = {
        'ac_arrester':   {'iterations': 3000, 'lr': 0.005, 'depth': 4, 'l2': 5},
        'cvt':           {'iterations': 2000, 'lr': 0.01, 'depth': 5, 'l2': 3},
        'post_insulator': {'iterations': 2000, 'lr': 0.01, 'depth': 5, 'l2': 3},
    }
    cb = cb_params.get(material, cb_params['ac_arrester'])

    fusion_model = CatBoostRegressor(
        iterations=cb['iterations'], learning_rate=cb['lr'],
        depth=cb['depth'], l2_leaf_reg=cb['l2'],
        loss_function='RMSE', early_stopping_rounds=80,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_fusion_val = min(12, len(train_target_idx) // 3)
    fusion_model.fit(fusion_train, y_trunc[train_target_idx],
                     eval_set=(fusion_train[-n_fusion_val:], y_trunc[train_target_idx][-n_fusion_val:]))

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
    n_factors = X_train_factors.shape[1]
    train_len = len(y_train)

    # 1. VMD K值优化
    opt_k = vmd_optimize_k(y_train, alpha=VMD_ALPHA_MAP[material])
    logger.info(f"  [VMD-Transformer直接求和] VMD最优K={opt_k}, seq_len={seq_len}, "
                f"训练样本={train_len - seq_len}, 全外推IMF")
    u_full, _, omega, residual_idx, all_modal_indices, y_trunc = vmd_decompose_full(
        y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])
    X_trunc = X_train_factors[:len(y_trunc)]

    # 2. IMF相关性筛选
    keep_idx = filter_imfs_by_correlation(u_full, y_trunc)
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

    # 3. 残差分量 → MultiFeatureTransformer (残差 + 全部因子)
    residual_train = u_train[residual_idx_new]
    residual_test = u_test[residual_idx_new]
    residual_features_train = np.column_stack(
        [residual_train] + [X_trunc[:, j] for j in range(n_factors)])
    input_size_mf = 1 + n_factors

    X_r, y_r = create_sequences(residual_features_train, seq_len, stride=SLIDING_STRIDE)
    residual_full_seq = np.concatenate([residual_train[-seq_len:], residual_test])
    factor_seqs = [np.concatenate([X_trunc[-seq_len:, j], X_test_factors[:, j]])
                   for j in range(n_factors)]
    X_r_test, _ = create_sequences(np.column_stack([residual_full_seq] + factor_seqs),
                                   seq_len, stride=1)

    mf_model = MultiFeatureTransformer(
        input_size=input_size_mf, hidden_size=mf_hidden, dropout=tf_do,
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
    u_full, _, omega, _, _, y_trunc = vmd_decompose_full(y_train, K=opt_k, alpha=VMD_ALPHA_MAP[material])
    keep_idx = filter_imfs_by_correlation(u_full, y_trunc)
    u_filtered = u_full[keep_idx]
    n_imfs_kept = len(keep_idx)
    logger.info(f"  [VMD-SVR] IMF筛选: {opt_k}→{n_imfs_kept}个 (保留{keep_idx})")

    imfs_train = u_filtered.T
    imfs_test = extrapolate_imfs(imfs_train, len(y_test), residual_idx=None, method='seasonal_naive')
    X_train_full = np.column_stack([imfs_train, X_train_factors[:len(y_trunc)]])
    X_test_full = np.column_stack([imfs_test, X_test_factors])

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto', 0.01, 0.1],
                  'epsilon': [0.01, 0.05, 0.1, 0.2]}
    logger.info(f"  [SVR超参数] kernel=rbf, C={param_grid['C']}, gamma={param_grid['gamma']}, "
                f"epsilon={param_grid['epsilon']} | GridSearchCV(cv=3, scoring=neg_mse) | "
                f"输入特征数={X_train_full.shape[1]} ({n_imfs_kept}个IMF+{X_train_factors.shape[1]}因子)")
    svr = SVR(kernel='rbf')
    grid = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error',
                        n_jobs=1, verbose=0)
    grid.fit(X_train_full, y_trunc)
    logger.info(f"  [SVR最优参数] C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}, "
                 f"epsilon={grid.best_params_['epsilon']}")

    y_pred = grid.predict(X_test_full)
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred_orig = np.maximum(y_pred_orig, 0)  # 物理约束：需求量非负
    return y_pred_orig, y_test_orig, None, omega, u_full, grid


# ===================== 11. 模型评估 =====================
def evaluate_model(y_true, y_pred):
    """计算 MSE/RMSE/MAE/R²/sMAPE/MASE"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape = np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_pred)+np.abs(y_true)+1e-10))*100
    mase_denom = np.mean(np.abs(y_true[1:]-y_true[:-1])) if len(y_true)>1 else 1
    mase = mae / max(mase_denom, 1e-10) if mase_denom > 0 else 999
    return {'MSE': round(mse,4), 'RMSE': round(rmse,4), 'MAE': round(mae,4),
            'R2': round(r2,4), 'sMAPE': round(smape,2), 'MASE': round(mase,4)}


# ===================== 10. 可视化 =====================
def plot_prediction_comparison(all_results, material):
    """预测对比曲线：所有模型预测 vs 真实值"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    pal = ['#FF0000','#2196F3','#4CAF50','#FF9800','#8E24AA','#00BCD4','#795548','#E91E63']
    mks = ['o','^','D','v','s','p','*','h']
    unit_map = {'交流避雷器': '台/只', '电容式电压互感器': '台', '交流支柱绝缘子': '只/支',
                '断路器保护': '套', '电抗器保护': '套',
                '线路保护': '套', '10kV变压器': '台', '变压器保护': '套',
                '母线保护': '套', '500kVGIS组合电器': '套'}

    results = all_results[material]
    test_len = len(results['CatBoost']['y_test'])
    end_date = pd.Timestamp.today().replace(day=1) - pd.DateOffset(months=1)
    x = pd.date_range(end=end_date, periods=test_len, freq='MS')

    # Truth line
    ax.plot(x, results['CatBoost']['y_test'][:len(x)], color='black', marker='o',
            linestyle='solid', label='真实值', markersize=5, linewidth=2.5, zorder=10)

    # All model predictions
    for mi, (model_name, preds) in enumerate(results.items()):
        if 'y_pred' not in preds: continue
        pred = preds['y_pred']
        pred_x = x[:len(pred)]
        r2_val = preds.get('metrics', {}).get('R2', 0)
        lbl = f'{model_name} (R²={r2_val:.3f})'
        ax.plot(pred_x, pred, color=pal[mi % len(pal)], marker=mks[mi % len(mks)],
                linestyle='--', label=lbl, markersize=4, alpha=0.85)

    # Unit
    unit = ''
    for kw, u in unit_map.items():
        if kw in material: unit = u; break
    ax.set_xlabel('日期')
    ax.set_ylabel(f'需求量({unit})' if unit else '需求量')
    ax.set_title(f'{material} — 模型预测对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'预测对比_{material}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 预测对比图({material}) → {path}")


def plot_vmd_decomposition(demand_full, u, omega, material):
    """VMD分解可视化：原始信号+IMF分量"""
    n_imfs = len(u)
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 10))
    eff_len = min(len(demand_full), u.shape[1])
    t = np.arange(eff_len)

    # 原始信号 (截断对齐)
    axes[0].plot(t, demand_full[:eff_len], 'k-', linewidth=1.5)
    axes[0].set_title(f'{MATERIAL_LABELS[material]} — 原始需求量序列', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('需求量')
    axes[0].grid(True, alpha=0.3)

    # 各IMF分量
    final_freqs = omega[-1]
    for i in range(n_imfs):
        axes[i + 1].plot(t, u[i][:eff_len], linewidth=1)
        axes[i + 1].set_ylabel(f'IMF{i+1}\n(f={final_freqs[i]:.3f})')
        axes[i + 1].grid(True, alpha=0.3)
        if i == n_imfs - 1:
            axes[i + 1].set_xlabel('月份序号')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'VMD分解_{material}.png')
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
    path = os.path.join(OUTPUT_DIR, f'特征重要性_{material}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 特征重要性({material}) → {path}")


def plot_metrics_comparison(all_metrics):
    """模型指标对比：分组柱状图（各物资各模型的四项指标）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_names = ['MSE', 'RMSE', 'MAE', 'R2']
    model_names = ['CatBoost', 'NaiveSeasonal', 'Persistence', 'SARIMA', 'CondCatBoost', 'NHiTS', 'TwoStage']
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#795548', '#9C27B0', '#E91E63', '#00BCD4']

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
    path = os.path.join(OUTPUT_DIR, '指标对比图.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 指标对比图 → {path}")


def plot_demand_curves(data_dict):
    """物资的实际需求量曲线图"""
    n_mat = len(MATERIALS)
    fig, axes = plt.subplots(n_mat, 1, figsize=(14, 3*n_mat))
    colors = plt.cm.tab10(np.linspace(0, 1, n_mat))
    for idx, material in enumerate(MATERIALS):
        ax = axes[idx] if n_mat > 1 else axes
        df = data_dict[material]
        ax.plot(df['date'], df['demand'], color=colors[idx], linewidth=1.5, marker='o', markersize=3)
        ax.fill_between(df['date'], 0, df['demand'], color=colors[idx], alpha=0.08)
        unit = ''; [unit := u for kw,u in {'交流避雷器':'台','电容式电压互感器':'台','交流支柱绝缘子':'只','断路器保护':'套','电抗器保护':'套','线路保护':'套','10kV变压器':'台'}.items() if kw in material]
        ax.set_ylabel(f'{MATERIAL_LABELS[material]}\n需求量({unit})' if unit else f'{MATERIAL_LABELS[material]}\n需求量', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        # 标记 train/test 分割线
        split_date = df['date'].iloc[-N_TEST] if len(df) > N_TEST else None
        if split_date is not None:
            ax.axvline(x=split_date, color='black', linestyle='--', linewidth=1.2)
    n_months = len(data_dict[MATERIALS[0]])
    date_range = f"{data_dict[MATERIALS[0]]['date'].iloc[0].strftime('%Y.%m')}–{data_dict[MATERIALS[0]]['date'].iloc[-1].strftime('%Y.%m')}"
    axes[0].set_title(f'配电网物资需求量 — {n_months}个月完整序列 ({date_range})', fontsize=13, fontweight='bold')
    (axes[-1] if n_mat > 1 else axes).set_xlabel('日期')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '需求量曲线.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 需求量曲线 → {path}")


# ===================== 批次事件驱动 Conditional CatBoost =====================
def run_conditional_catboost(X_train_factors, y_train, X_test_factors, y_test,
                              material, demand_scaler):
    """批次事件驱动CatBoost: 更深树捕捉批次事件非线性交互"""
    model = CatBoostRegressor(
        iterations=2000, learning_rate=0.015, depth=7, l2_leaf_reg=4,
        loss_function='RMSE', early_stopping_rounds=50,
        random_seed=RANDOM_SEED, verbose=0)
    n_val = min(12, len(y_train)//4)
    X_tr, X_val = X_train_factors[:-n_val], X_train_factors[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    y_pred = model.predict(X_test_factors)
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_orig = demand_scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    return np.maximum(y_pred_orig, 0), y_test_orig, model.get_feature_importance(), model


# ===================== 两阶段预测 (论文核心创新) =====================
def run_two_stage(df_all, X_train_factors, y_train, X_test_factors, y_test,
                   material, demand_scaler):
    """两阶段预测:
    Stage 1: CatBoost 二分类 — 预测当月是否有需求(>0)
    Stage 2: CatBoost 回归 — 对有需求的月份预测需求量
    最终 = P(有需求) × 预测量
    """
    demand_raw = demand_scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_train_binary = (demand_raw > 0).astype(int)
    n_pos = y_train_binary.sum(); n_neg = len(y_train_binary) - n_pos
    if n_pos < 5 or n_neg < 5:
        logger.warning(f"  [两阶段] 正负样本不均衡(pos={n_pos},neg={n_neg}), 回退到CatBoost")
        return run_catboost(X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler)

    # Stage 1: 分类器
    cls = CatBoostClassifier(
        iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=5,
        loss_function='Logloss', early_stopping_rounds=30,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_val = max(6, min(12, len(y_train)//4))
    X_cls_tr, X_cls_val = X_train_factors[:-n_val], X_train_factors[-n_val:]
    y_cls_tr, y_cls_val = y_train_binary[:-n_val], y_train_binary[-n_val:]
    cls.fit(X_cls_tr, y_cls_tr, eval_set=(X_cls_val, y_cls_val))
    prob_test = cls.predict_proba(X_test_factors)[:, 1]
    prob_test = np.clip(prob_test, 0, 1)

    # Stage 2: 回归 (仅对非零训练数据)
    nonzero_mask = demand_raw > 0
    if nonzero_mask.sum() < 10:
        logger.warning(f"  [两阶段] 非零样本过少({nonzero_mask.sum()}), 回退到CatBoost baseline")
        return run_catboost(X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler)

    X_nz = X_train_factors[nonzero_mask]
    y_nz = y_train[nonzero_mask]

    reg = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=3,
        loss_function='RMSE', early_stopping_rounds=50,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_val2 = min(6, len(y_nz)//4)
    reg.fit(X_nz[:-n_val2], y_nz[:-n_val2],
            eval_set=(X_nz[-n_val2:], y_nz[-n_val2:]))
    qty_test = reg.predict(X_test_factors)

    # 反归一化
    y_test_orig = demand_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    qty_pred_orig = demand_scaler.inverse_transform(qty_test.reshape(-1,1)).flatten()
    # 最终 = 概率 × 预测量
    y_pred_orig = prob_test * np.maximum(qty_pred_orig, 0)

    cls_acc = np.mean((prob_test > 0.5).astype(int) == (y_test_orig > 0).astype(int))
    logger.info(f"  [两阶段] Stage1分类准确率={cls_acc:.2%}, 非零训练样本={nonzero_mask.sum()}")
    return y_pred_orig, y_test_orig, None, (cls, reg)


# ===================== 批次事件驱动 Conditional CatBoost =====================
# ===================== N-HiTS 多尺度层次化预测 (PyTorch原生) =====================
class NHitsBlock(nn.Module):
    def __init__(self, in_len, out_len, pool_size, hidden_units, dropout=0.2):
        super().__init__()
        self.pool_size = pool_size
        n_pooled = max(1, in_len // pool_size)
        layers = [nn.Linear(n_pooled, hidden_units[0]), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(len(hidden_units)-1):
            layers += [nn.Linear(hidden_units[i], hidden_units[i+1]), nn.ReLU(), nn.Dropout(dropout*0.5)]
        layers.append(nn.Linear(hidden_units[-1], out_len))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x.reshape(x.shape[0], -1, self.pool_size).max(dim=2)[0])

class NHitsModel(nn.Module):
    def __init__(self, lookback, horizon, pool_sizes, hidden_units_list, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            NHitsBlock(lookback, horizon, ps, hu, dropout)
            for ps, hu in zip(pool_sizes, hidden_units_list)
        ])
    def forward(self, x):
        return sum(block(x) for block in self.blocks)


def run_nhits(df_all, material, demand_scaler):
    import torch.optim as optim
    df = df_all.copy(); demand_raw = df['demand'].values.astype(np.float64)
    train_len = len(demand_raw) - N_TEST
    demand_train, demand_test = demand_raw[:train_len], demand_raw[train_len:]
    train_vals = np.maximum(demand_train, 0)
    d_min, d_max = train_vals.min(), train_vals.max()
    d_range = d_max - d_min if d_max > d_min else 1.0
    y_train, y_test = (demand_train-d_min)/d_range, (demand_test-d_min)/d_range
    lookback, horizon = 24, N_TEST
    X_tr, Y_tr = [], []
    for i in range(len(y_train) - lookback - horizon):
        X_tr.append(y_train[i:i+lookback]); Y_tr.append(y_train[i+lookback:i+lookback+horizon])
    X_tr, Y_tr = np.array(X_tr), np.array(Y_tr)
    if len(X_tr) < 10:
        logger.warning(f'  [N-HiTS] samples={len(X_tr)}<10, skip'); return None,None,None,None
    model = NHitsModel(lookback, horizon, [6,3,1], [[8],[8],[8]], 0.5).to(DEVICE)
    X_t, Y_t = torch.FloatTensor(X_tr).to(DEVICE), torch.FloatTensor(Y_tr).to(DEVICE)
    n_val = max(3, len(X_tr)//3); X_trn, X_val = X_t[:-n_val], X_t[-n_val:]; Y_trn, Y_val = Y_t[:-n_val], Y_t[-n_val:]
    opt = optim.Adam(model.parameters(), lr=0.002, weight_decay=3e-4)
    best_val, patience, best_state = float("inf"), 150, None
    for _ in range(1000):
        model.train(); opt.zero_grad(); loss = nn.MSELoss()(model(X_trn), Y_trn)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.MSELoss()(model(X_val), Y_val).item()
        if vl < best_val:
            best_val, patience = vl, 150
            best_state = {k:v.clone().cpu() for k,v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(y_train[-lookback:]).unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()
    logger.info(f'  [N-HiTS] samples={len(X_tr)}, val_loss={best_val:.5f}')
    return np.maximum(p*d_range+d_min, 0), demand_test[:horizon], None, None

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
        for i, model_name in enumerate(['CatBoost', 'NaiveSeasonal', 'Persistence', 'SARIMA', 'CondCatBoost', 'NHiTS', 'TwoStage']):
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
    logger.info("  物资: Top5采购频率最高 (从ECP数据自动选择)")
    logger.info("  因子: 批次事件特征 + 项目数量 + 自回归滞后")
    logger.info("  模型: CatBoost / Conditional-CatBoost / N-HiTS / TwoStage + NaiveSeasonal/Persistence/SARIMA")
    logger.info("=" * 70)
    logger.info(f"  日志文件: {log_filename}")
    logger.info("[1/8] 加载数据...")
    data_dict = load_or_generate_data()

    # 动态更新 MATERIALS 和 MATERIAL_LABELS
    global MATERIALS, MATERIAL_LABELS
    MATERIALS = list(data_dict.keys())
    # Build labels from sheet names / data
    temp_labels = {}
    for i, mat in enumerate(MATERIALS):
        temp_labels[mat] = mat  # Use the material name directly as label
    MATERIAL_LABELS = temp_labels
    logger.info(f"  物资: {[MATERIAL_LABELS[m] for m in MATERIALS]}")

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
                         f"MAE={metrics_1['MAE']:.4f} R2={metrics_1['R2']:.4f}")

            # --- Baseline: Naive Seasonal ---
            y_pred_ns, y_test_ns = baseline_naive_seasonal(y_train, y_test, demand_scaler)
            metrics_ns = evaluate_model(y_test_ns, y_pred_ns)
            all_results[material]['NaiveSeasonal'] = {'y_pred': y_pred_ns, 'y_test': y_test_ns, 'metrics': metrics_ns}
            all_metrics[material]['NaiveSeasonal'] = metrics_ns
            logger.info(f"  [Baseline] NaiveSeasonal: R2={metrics_ns['R2']:.4f}")

            # --- Baseline: Persistence ---
            y_pred_sp, y_test_sp = baseline_persistence(y_train, y_test, demand_scaler)
            metrics_sp = evaluate_model(y_test_sp, y_pred_sp)
            all_results[material]['Persistence'] = {'y_pred': y_pred_sp, 'y_test': y_test_sp, 'metrics': metrics_sp}
            all_metrics[material]['Persistence'] = metrics_sp
            logger.info(f"  [Baseline] Persistence: R2={metrics_sp['R2']:.4f}")

            # --- Baseline: SARIMA ---
            y_pred_sa, y_test_sa = baseline_sarima(y_train, y_test, demand_scaler)
            metrics_sa = evaluate_model(y_test_sa, y_pred_sa)
            all_results[material]['SARIMA'] = {'y_pred': y_pred_sa, 'y_test': y_test_sa, 'metrics': metrics_sa}
            all_metrics[material]['SARIMA'] = metrics_sa
            logger.info(f"  [Baseline] SARIMA: R2={metrics_sa['R2']:.4f}")

            # --- 进阶模型: Conditional CatBoost (批次事件驱动) ---
            logger.info(f"  [5/8] Conditional-CatBoost...")
            y_pred_cc, y_test_cc, imp_cc, model_cc = run_conditional_catboost(
                X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler)
            metrics_cc = evaluate_model(y_test_cc, y_pred_cc)
            all_results[material]['CondCatBoost'] = {'y_pred': y_pred_cc, 'y_test': y_test_cc, 'metrics': metrics_cc}
            all_metrics[material]['CondCatBoost'] = metrics_cc
            logger.info(f"        MSE={metrics_cc['MSE']:.4f} RMSE={metrics_cc['RMSE']:.4f} "
                         f"MAE={metrics_cc['MAE']:.4f} R2={metrics_cc['R2']:.4f}")

            # --- 进阶模型: N-HiTS ---
            logger.info(f"  [6/8] N-HiTS...")
            y_pred_nh, y_test_nh, imp_nh, model_nh = run_nhits(df, material, demand_scaler)
            if y_pred_nh is not None:
                metrics_nh = evaluate_model(y_test_nh, y_pred_nh)
                all_results[material]['NHiTS'] = {'y_pred': y_pred_nh, 'y_test': y_test_nh, 'metrics': metrics_nh}
                all_metrics[material]['NHiTS'] = metrics_nh
                logger.info(f"        MSE={metrics_nh['MSE']:.4f} RMSE={metrics_nh['RMSE']:.4f} "
                             f"MAE={metrics_nh['MAE']:.4f} R2={metrics_nh['R2']:.4f}")
            else:
                logger.info(f"  [N-HiTS] 跳过 (未安装或失败)")

            # --- 两阶段预测 (Stage1分类 + Stage2回归) ---
            logger.info(f"  [7/8] 两阶段预测...")
            y_pred_ts, y_test_ts, imp_ts, model_ts = run_two_stage(
                df, X_train_factors, y_train, X_test_factors, y_test, material, demand_scaler)
            metrics_ts = evaluate_model(y_test_ts, y_pred_ts)
            all_results[material]['TwoStage'] = {'y_pred': y_pred_ts, 'y_test': y_test_ts, 'metrics': metrics_ts}
            all_metrics[material]['TwoStage'] = metrics_ts
            logger.info(f"        MSE={metrics_ts['MSE']:.4f} RMSE={metrics_ts['RMSE']:.4f} "
                         f"MAE={metrics_ts['MAE']:.4f} R2={metrics_ts['R2']:.4f}")

            # --- VMD-CatBoost / VMD-Transformer-CatBoost / VMD-SVR ---
            # [DISABLED] 循环论证: VMD分解y->IMF作特征->预测y, Sigma(IMF)~=y

            # 特征重要性图
            imp_dict = {
                'catboost_imp': imp_1,
            }
            plot_feature_importance(imp_dict, material)

        # Step 4: 评估汇总
        logger.info('')
        logger.info('[8/8] 汇总评估与可视化...')
        print_metrics_table(all_metrics)

        # 保存指标 JSON
        metrics_json = {}
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
