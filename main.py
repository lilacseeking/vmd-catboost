"""
main.py -- 电力物资需求量预测 (期刊论文)
模型: CatBoost / CatBoost-2S / CondCatBoost / TwoStage / Ridge-2S / ElasticNet-2S / LightGBM / N-HiTS
基线: NaiveSeasonal / NaiveMean / Persistence / SARIMA / Chronos / Croston-SBA
指标: MSE / RMSE / MAE / R^2 / sMAPE / MASE / WRMSSE / sCRPS(Chronos)
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# === 【Dashboard插入点1】安全导入 ===
try:
    from dashboard import DashboardBuilder, ResultCollector
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    DASHBOARD_AVAILABLE = False
    print(f"[Warning] Dashboard模块不可用: {e}")

# ===================== 全局配置 =====================
MATERIALS = []  # 动态从 data.xlsx sheet 名加载
MATERIAL_LABELS = {}
USE_EXTERNAL_FACTORS = False # D-01: 外部宏观因子(铝价/工业用电) — 趋势伪相关, ΔR²=-0.021, 默认关闭
# 内部ECP因子(始终启用) + 外部宏观因子(由USE_EXTERNAL_FACTORS控制)
# D-03: has_batch/digital_bids加入候选池后160kN R²暴跌-0.32(Spearman竞争替换), 已回退
_INTERNAL_FACTORS = ['project_count', 'transformer_bids', 'monthly_bid_count', 'uhv_bids']
_EXTERNAL_FACTORS = ['aluminum_price', 'industrial_elec', 'industrial_elec_yoy']
FACTOR_NAMES = _INTERNAL_FACTORS + (_EXTERNAL_FACTORS if USE_EXTERNAL_FACTORS else [])
FACTOR_LABELS = {'project_count': '项目数量(同源)', 'transformer_bids': '输变电批次数',
                 'monthly_bid_count': '当月公告总数', 'uhv_bids': '特高压批次数',
                 'has_batch': '批次采购标记', 'digital_bids': '数字化批次数',
                 'aluminum_price': '铝价(元/吨)', 'industrial_elec': '工业用电量(亿kWh)',
                 'industrial_elec_yoy': '工业用电同比(%)'}
RANDOM_SEED = 42
DATA_LOCKED = True  # 严格模式 — 数据由外部手动生成，禁止自动回退
USE_QUARTER_DUMMIES = False  # 方案A: 行政季度末哑变量。实验确认无显著收益(ΔR²=-0.0046), 已关闭

# ===================== 可取消的实验开关 (设为True启用/False回退) =====================
USE_EVENT_FEATURES = True    # EVT事件特征 — 表征采购批次结构, 论文第3组特征
USE_POOLED_TRAINING = False  # 实验2: 相似性聚类池化
USE_LOG1P_TARGET = False     # M-02: log1p变换 — R²-0.259(0.724→0.465), 间歇性零值+反变换放大误差, 已关闭
USE_QUANTILE_REGR = False    # 实验4: 分位数回归
SKIP_REACTOR_PROTECT = True  # 开关: 跳过电抗器保护 (R²<0.05, 样本噪声主导, 无优化空间)
USE_MAG_FEATURES = False     # 实验6: 量级特征(lag-1) — 泄漏版R²虚高+0.28, 无泄漏版Δ=-0.02, 备选关闭
USE_GREY_FEATURES = True     # GINN创新点: 灰色系统先验特征。仅对NZ<20的稀疏物资生效
USE_ENHANCED_LAG_FEATURES = True  # 论文特征组1: 多分辨率自回归信号(8D, +0.23 R²贡献)
USE_PAPER_FEATURES = True    # 论文精简: NZ>=15→MAS+PES(22D) | NZ<15→MAS+GDEP(17D)
#   全局移除: sin_m/cos_m/sin_q/cos_q/ewm (4组Spearman均<0.13)
USE_BATCH_FEATURES = False   # D-04: 批次采购附加特征 — R²-0.027, 300kN暴跌-0.42, 已关闭
USE_SEASONAL_PROFILE = False # D-05: 月度季节剖面 — R²-0.018/MAPE+22pp, 间歇型物资月均值被零主导, 已关闭
USE_BLEND_ENSEMBLE = True    # M-01: Top-3模型加权融合(R²权重, 排除朴素基线)

# ===================== 0. 每物资超参数配置 =====================
# 两层字典: HP_DEFAULTS[model_key] = 默认参数; HP_OVERRIDES[model_key][material_substr] = 覆盖值
# get_hp(model_key, material_name) → 合并后的参数字典。
# 物资匹配: 在material_name中做子串搜索(find), 第一个命中的覆盖生效。
# 新增模型: 只需在 HP_DEFAULTS 加一行 + 在 HP_OVERRIDES 加物料特化(可选)
# 新增物资: 只需在 HP_OVERRIDES 各模型下加该物资的覆盖条目(如需要)

# 注意: HP键名使用 CatBoost/LightGBM 原生参数名, 通过 **hp 直接解包传入
HP_DEFAULTS = {
    # -- 共享Stage1分类器 (被 _two_stage_fit_predict 使用) --
    'stage1_cls':       {'iterations': 600, 'learning_rate': 0.03, 'depth': 5, 'l2_leaf_reg': 5},
    # -- CatBoost单阶段直接回归 (消融基线) --
    'catboost_direct':  {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
    # -- CatBoost单阶段×Tweedie损失 (波动数据专项,  复合Poisson-Gamma分布) --
    'catboost_tweedie': {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
    # -- CatBoost-2S (两阶段×RMSE) --
    'catboost_2s':      {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
    # -- CondCatBoost (两阶段×更强CatBoost) --
    'cond_catboost':    {'iterations': 2000, 'learning_rate': 0.015, 'depth': 7, 'l2_leaf_reg': 4},
    # -- TwoStage独立实现: Stage1分类器 --
    'twostage_cls':     {'iterations': 800, 'learning_rate': 0.03, 'depth': 5, 'l2_leaf_reg': 5},
    # -- TwoStage独立实现: Stage2回归器 --
    'twostage_reg':     {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
    # -- Ridge-2S --
    'ridge_2s':         {'alphas': [0.01, 0.1, 1.0, 10.0, 100.0]},
    # -- ElasticNet-2S --
    'elasticnet_2s':    {'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]},
    # -- LightGBM两阶段 --
    'lightgbm':         {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5,
                         'num_leaves': 31, 'reg_alpha': 1, 'reg_lambda': 3},
    # -- 分位数回归 (实验4: 三alpha训练, 输出中位数+预测区间) --
    'quantile':         {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
}

HP_OVERRIDES = {
    'stage1_cls': {
        # 按物资数据特征调参: 非零月多的物资可用更强分类器, 非零月少的需强正则
        '交流避雷器':      {'depth': 6, 'iterations': 800},           # ~42非零月, 高区分度
        '电容式电压互感器':  {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 6},  # ~38非零月
        '交流支柱绝缘子':    {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 6},  # ~36非零月, CV高
        '断路器保护':       {'depth': 3, 'iterations': 400, 'l2_leaf_reg': 8},  # ~36非零月, 极高CV
        '电抗器保护':       {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 6},  # ~40非零月
    },
    'catboost_2s': {
        '交流避雷器':      {'depth': 7, 'iterations': 2000},           # 高密度, 可加大容量
        '电容式电压互感器':  {'depth': 5, 'iterations': 1200, 'l2_leaf_reg': 5},  # 中密度
        '交流支柱绝缘子':    {'depth': 4, 'iterations': 800, 'l2_leaf_reg': 8},   # 低密度+高CV→浅树+强正则
        '断路器保护':       {'depth': 4, 'iterations': 800, 'l2_leaf_reg': 8},   # 同上
        '电抗器保护':       {'depth': 5, 'iterations': 1000, 'l2_leaf_reg': 5},  # 中密度
    },
    'cond_catboost': {
        # CondCatBoost 默认已较强(depth=7, iters=2000), 仅对过拟合高风险物资降级
        '断路器保护':       {'depth': 4, 'iterations': 1000, 'l2_leaf_reg': 8},
        '交流支柱绝缘子':    {'depth': 5, 'iterations': 1200, 'l2_leaf_reg': 6},
    },
    'twostage_cls': {
        '断路器保护':       {'depth': 3, 'iterations': 500, 'l2_leaf_reg': 8},
        '交流支柱绝缘子':    {'depth': 4, 'iterations': 600, 'l2_leaf_reg': 6},
    },
    'twostage_reg': {
        '交流避雷器':      {'depth': 7, 'iterations': 2000},
        '断路器保护':       {'depth': 4, 'iterations': 800, 'l2_leaf_reg': 8},
        '电抗器保护':       {'depth': 5, 'iterations': 1000, 'l2_leaf_reg': 5},
    },
    'catboost_direct': {
        # 直接回归在波动数据上(Ridge/CV>1)优于两阶段, 按CV调参防过拟合
        '断路器保护':       {'depth': 4, 'iterations': 3000, 'l2_leaf_reg': 5},  # CV=1.26→浅树+更多迭代+强正则
        '交流支柱绝缘子':    {'depth': 4, 'iterations': 2000, 'l2_leaf_reg': 6},  # CV=0.78+极端值→防过拟合
        '电抗器保护':       {'depth': 5, 'iterations': 2000, 'l2_leaf_reg': 4},  # CV=0.92→中等复杂度
    },
    'catboost_tweedie': {
        # Tweedie单阶段:  复合Poisson-Gamma分布天然处理零膨胀, 波动数据上预期优于RMSE
        # p值(CatBoost的tweedie_variance_power)默认=1.5, 波动大的物资可加大(方差∝均值^p)
        '断路器保护':       {'depth': 4, 'iterations': 3000, 'l2_leaf_reg': 5},  # CV=1.26: 同直接回归配置
        '交流支柱绝缘子':    {'depth': 5, 'iterations': 2000, 'l2_leaf_reg': 4},  # CV=0.78: 中配置
        '电抗器保护':       {'depth': 5, 'iterations': 2000, 'l2_leaf_reg': 3},  # CV=0.92: 中配置
    },
}


def _match_hp(material, overrides):
    """在 overrides 字典中查找匹配 material 的键, 返回合并参数(或空dict)"""
    for key, params in overrides.items():
        if key in material:
            return params
    return {}


def get_hp(model_key, material):
    """获取指定模型+物资的超参数: HP_DEFAULTS[model_key] + HP_OVERRIDES[model_key][material_match]

    用法: params = get_hp('catboost_2s', material)
          reg = CatBoostRegressor(**params, loss_function='RMSE', ...)
    """
    defaults = HP_DEFAULTS.get(model_key, {})
    overrides = HP_OVERRIDES.get(model_key, {})
    material_overrides = _match_hp(material, overrides)
    return {**defaults, **material_overrides}  # 覆盖合并 (material takes priority)
N_TEST = 12  # 测试集月数
OUTPUT_DIR = 'outputs/figures'
LOG_DIR = 'outputs/logs'
DATA_DIR = 'inputs'
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
# --data flag support: python main.py --data data_sgcc.xlsx
if '--data' in sys.argv:
    idx = sys.argv.index('--data')
    DATA_FILE = os.path.join(DATA_DIR, sys.argv[idx+1])

# === 【Dashboard插入点2】命令行参数 ===
NO_DASHBOARD = '--no-dashboard' in sys.argv
DASHBOARD_CONFIG = None
DASHBOARD_OUTPUT = None
DASHBOARD_DEBUG = '--dashboard-debug' in sys.argv
if '--dashboard-config' in sys.argv:
    _idx = sys.argv.index('--dashboard-config')
    DASHBOARD_CONFIG = sys.argv[_idx + 1]
if '--dashboard-output' in sys.argv:
    _idx = sys.argv.index('--dashboard-output')
    DASHBOARD_OUTPUT = sys.argv[_idx + 1]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- 日志系统 ----
# 运行时间戳：日志文件与看板输出文件共享同一编号，便于关联追溯
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOG_DIR, f'main_{RUN_TIMESTAMP}.log')
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


# ===================== 1.6 量级特征 (lag-1, 备选) =====================
_mag_cache = {}

def build_magnitude_features(df, material):
    """从bid_items表提取采购事件量级特征(lag-1, 无数据泄漏)。

    特征(6维): n_packages, n_sub_bids, n_orgs, n_items, avg_item_qty, max_item_qty
    关键设计: lag-1 — 当前月使用的是上个月的实际值, 避免同源泄漏。
    工业逻辑: '上次采购规模大→这次可能是间歇期'(lag-1 rho为负)。
    """
    if not USE_MAG_FEATURES:
        return np.zeros((len(df), 0))
    if material in _mag_cache:
        return _mag_cache[material]

    import sqlite3
    # Try both relative and absolute paths for portability
    db_path = r'D:\Users\dell\PycharmProjects\bidding-ecp-data\data\ecp_data.db'
    if not os.path.exists(db_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'bidding-ecp-data', 'data', 'ecp_data.db')
        db_path = os.path.normpath(alt)
    if not os.path.exists(db_path):
        return np.zeros((len(df), 0))

    mat_short = material
    if ',' in mat_short: mat_short = mat_short.split(',')[0]
    pattern = f'%{mat_short}%'
    conn = sqlite3.connect(db_path); c = conn.cursor()
    c.execute('''SELECT demand_month, demand_quantity, package_no, sub_bid_name, project_org_name
        FROM bid_items WHERE material_name LIKE ? AND demand_quantity > 0 AND demand_month >= ? AND demand_month <= ?''',
        (pattern, '201911', '202607'))
    rows = c.fetchall(); conn.close()

    from collections import defaultdict
    monthly = defaultdict(lambda: {'qty':0, 'pkgs':set(), 'subs':set(), 'orgs':set(), 'items':0, 'max_q':0})
    for dm, q, pkg, sub, org in rows:
        qf = float(q) if q else 0
        monthly[dm]['qty'] += qf; monthly[dm]['pkgs'].add(pkg or '')
        monthly[dm]['subs'].add(sub or ''); monthly[dm]['orgs'].add(org or '')
        monthly[dm]['items'] += 1; monthly[dm]['max_q'] = max(monthly[dm]['max_q'], qf)

    # Align with df dates (81 months from 2019-11)
    n_df = len(df)
    dates = df['date'].values
    arr = np.zeros((n_df, 6))
    for i in range(n_df):
        d = pd.Timestamp(dates[i])
        dm_key = f'{d.year}{d.month:02d}'
        entry = monthly.get(dm_key, {'qty':0, 'pkgs':set(), 'subs':set(), 'orgs':set(), 'items':0, 'max_q':0})
        arr[i,0] = len(entry['pkgs']); arr[i,1] = len(entry['subs'])
        arr[i,2] = len(entry['orgs']); arr[i,3] = entry['items']
        arr[i,4] = entry['qty'] / max(entry['items'], 1)
        arr[i,5] = entry['max_q']
    # LAG-1: row[t] uses row[t-1] value (only past info, no future leak)
    arr_lagged = np.vstack([np.zeros((1, 6)), arr[:-1]])
    _mag_cache[material] = arr_lagged
    return arr_lagged


# ===================== 1.7 GINN灰色先验特征 (SCI创新点) =====================
_grey_cache = {}
from scipy.special import gamma as _gamma_fn

def _gm11_fit(x0):
    """GM(1,1) OLS估计: dX⁽¹⁾/dt + a·X⁽¹⁾ = b → 返回(a,b)"""
    n = len(x0)
    if n < 4: return -0.01, np.mean(x0) if n > 0 else 0
    x1 = np.cumsum(np.maximum(x0, 0))
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([-z1, np.ones(n-1)]); Y = x0[1:]
    try:
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        return params[0], params[1]
    except: return -0.01, np.mean(x0)

def _gm11_fitted(x0, a, b):
    """GM(1,1)拟合值序列(与输入等长)"""
    n = len(x0)
    if abs(a) < 1e-10: return np.full(n, np.mean(np.maximum(x0, 0)))
    x1_hat = np.zeros(n); x1_hat[0] = max(x0[0], 0)
    for k in range(1, n):
        x1_hat[k] = (x1_hat[0] - b/a) * np.exp(-a*k) + b/a
    fitted = np.zeros(n); fitted[0] = x1_hat[0]
    for k in range(1, n): fitted[k] = max(x1_hat[k]-x1_hat[k-1], 0)
    return fitted

def _gm11_predict_next(last_n, a, b, n_pred):
    """GM(1,1)从last_n外推n_pred步"""
    if abs(a) < 1e-10: return np.full(n_pred, last_n)
    preds = np.zeros(n_pred); x1_prev = last_n
    for k in range(1, n_pred+1):
        x1_k = (last_n - b/a)*np.exp(-a*k) + b/a
        preds[k-1] = max(x1_k - x1_prev, 0); x1_prev = x1_k
    return preds

def _fractional_ago(x0, r):
    """分数阶r-AGO序列: Xr[k] = Σ C(k-j+r-1, k-j)·x0[j]"""
    n = len(x0); xr = np.zeros(n)
    for k in range(n):
        for j in range(k+1):
            coeff = _gamma_fn(k-j+r) / (_gamma_fn(k-j+1)*_gamma_fn(r)) if r > 0 else 1.0
            xr[k] += coeff * x0[j]
    return xr

def build_grey_features(demand_raw, train_len):
    """构建灰色先验特征 (GM(1,1)+分数阶AGO, 无数据泄漏)。

    训练集: 在每个训练位置t, 用[0:t]的序列拟合GM(1,1)→ gm_fitted[t] = t位置拟合值
    测试集: 滚动一步预测, 只用训练数据+已预测值, 不触碰测试集真实值
    """
    if not USE_GREY_FEATURES: return np.zeros((len(demand_raw), 0)), np.zeros((len(demand_raw), 0))

    n = len(demand_raw)
    tl = train_len
    d = np.maximum(demand_raw, 0)
    # 灰色系统理论处理稀疏序列: NZ>=15时普通特征足够, 灰色特征不必要
    if int(np.sum(demand_raw[:tl] > 0)) >= 15:
        return np.zeros((tl, 0)), np.zeros((n-tl, 0))

    # --- 训练集: 滑动window=12的GM(1,1)拟合 ---
    gm_fitted = np.zeros(n); gm_res = np.zeros(n); grey_a = np.zeros(n)
    for t in range(tl):
        start = max(0, t-11)
        seg = d[start:t+1]
        if len(seg) >= 4 and seg.sum() > 0:
            a, b = _gm11_fit(seg); fitted = _gm11_fitted(seg, a, b)
            gm_fitted[t] = fitted[-1]; gm_res[t] = d[t] - fitted[-1]; grey_a[t] = a
        else:
            gm_fitted[t] = np.mean(seg) if len(seg) > 0 else 0

    # --- 测试集: GM(1,1)外推预测 (无泄漏, 只用训练集拟合) ---
    # 用训练集最后12点拟合GM(1,1), 然后外推test_len步
    train_end_seg = d[max(0, tl-12):tl]
    test_len = n - tl
    if len(train_end_seg) >= 4 and train_end_seg.sum() > 0:
        a_hat, b_hat = _gm11_fit(train_end_seg)
        preds = _gm11_predict_next(train_end_seg[-1], a_hat, b_hat, test_len)
        gm_fitted[tl:] = preds
        grey_a[tl:] = a_hat
    else:
        gm_fitted[tl:] = np.mean(train_end_seg) if len(train_end_seg) > 0 else 0

    # --- 分数阶AGO (只对训练集+测试集做constant外推, 无泄漏) ---
    best_r = 0.7
    y_pos = np.maximum(d[:tl], 1e-6)
    if tl >= 5:
        best_smooth = float('inf')
        for r in [0.3, 0.5, 0.7, 0.9, 1.0]:
            try:
                fago_sub = _fractional_ago(y_pos[:min(20,tl)], r)
                smooth = np.std(np.diff(fago_sub))
                if smooth < best_smooth: best_smooth = smooth; best_r = r
            except: continue
    fago = np.zeros(n)
    try:
        fago_tr = _fractional_ago(y_pos, best_r)
        scale = np.mean(y_pos[y_pos>0]) / max(np.mean(fago_tr[fago_tr>0]), 1e-6) if (y_pos>0).any() else 1
        fago[:tl] = fago_tr * scale
        fago[tl:] = fago[tl-1]  # 测试集AGO固定为训练最后值 (保守外推)
    except: pass

    # gm_res(残差)在测试集为0: 测试时没有真实值可对比
    grey_tr = np.column_stack([gm_fitted[:tl], gm_res[:tl], grey_a[:tl], fago[:tl]])
    grey_te = np.column_stack([gm_fitted[tl:], gm_res[tl:], grey_a[tl:], fago[tl:]])
    return grey_tr, grey_te


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
    """时序安全特征工程: 树模型不需要y归一化, 仅对特征做MinMaxScaler."""
    top4 = get_top_factors(material, df.iloc[:len(df)-N_TEST])
    cols = ['demand'] + top4
    sub = df[cols].copy().ffill().bfill().fillna(0)
    data = sub.values.astype(np.float64)
    demand_raw = data[:, 0].copy()

    # Step 1: 分割
    train_len = len(data) - N_TEST
    data_train, data_test = data[:train_len], data[train_len:]
    demand_train, demand_test = demand_raw[:train_len], demand_raw[train_len:]

    # Step 1.5: 论文精简方案——按物资数据特征选择特征组
    _use_evt = USE_EVENT_FEATURES
    _use_grey = USE_GREY_FEATURES
    _use_sincos = True   # sin/cos 周期编码 (论文精简时移除, |r|<0.13)
    _use_ewm = True      # EWM指数平滑 (论文精简时移除, |r|<0.13)
    if USE_PAPER_FEATURES:
        nz_check = int(np.sum(demand_train > 0))
        if nz_check >= 15:
            _use_evt, _use_grey, _use_sincos, _use_ewm = True, False, False, False  # MAS+PES
        else:
            _use_evt, _use_grey, _use_sincos, _use_ewm = True, True, False, False # MAS+PES+GDEP

    # Step 2: 训练集lag/rolling (无未来泄露)
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
    m_train = (np.arange(train_len)+1) % 12; m_train[m_train==0]=12
    q_train = ((np.arange(train_len)+1) // 3) % 4; q_train[q_train==0]=4

    # D-02: 增强滞后/滚动统计特征 (纯自回归, 无外部依赖, 无未来泄露)
    enh_lag_feats_tr = []
    if USE_ENHANCED_LAG_FEATURES:
        seq = demand_train
        n = train_len
        # 多步滞后
        lag2 = np.zeros(n); lag2[2:] = seq[:-2]
        lag3 = np.zeros(n); lag3[3:] = seq[:-3]
        lag6 = np.zeros(n); lag6[6:] = seq[:-6]
        # 长窗口滚动均值
        roll6_mean = np.array([np.mean(seq[max(0,i-6):i]) if i > 0 else 0 for i in range(n)])
        roll12_mean = np.array([np.mean(seq[max(0,i-12):i]) if i > 0 else 0 for i in range(n)])
        # 滚动标准差(波动性)
        roll3_std = np.array([np.std(seq[max(0,i-3):i]) if i > 1 else 0 for i in range(n)])
        # 指数加权均值 (α=0.3, 近期权重更高)
        ewm = np.zeros(n)
        alpha = 0.3
        for i in range(1, n):
            ewm[i] = alpha * seq[i-1] + (1 - alpha) * ewm[i-1]
        # 同比差分: lag1 - lag13 (捕捉年度趋势变化)
        yoy_diff = np.zeros(n)
        for i in range(13, n):
            yoy_diff[i] = seq[i-1] - seq[i-13]
        enh_lag_feats_tr = [lag2, lag3, lag6, roll6_mean, roll12_mean, roll3_std, yoy_diff]
        if _use_ewm:
            enh_lag_feats_tr.append(ewm)  # 论文精简时跳过(|r|<0.13)

    # D-04: 批次采购附加特征 (附加模式, 不参与Spearman竞争, lag-1避免泄露)
    batch_feats_tr = []
    if USE_BATCH_FEATURES:
        for col in ['has_batch', 'digital_bids']:
            if col in df.columns:
                vals = df[col].values.astype(np.float64)
                # lag-1: 用上月批次信号预测当月需求
                lagged = np.zeros(train_len)
                lagged[1:] = vals[:train_len-1]
                batch_feats_tr.append(lagged)

    # D-05: 月度季节剖面 (同月历史扩展均值, 训练集计算, 无泄露)
    seasonal_profile_tr = []
    if USE_SEASONAL_PROFILE and 'date' in df.columns:
        cal_m = df['date'].dt.month.values  # 实际日历月 1-12
        sp = np.zeros(train_len)
        for i in range(train_len):
            same_month_past = [j for j in range(i) if cal_m[j] == cal_m[i]]
            if same_month_past:
                sp[i] = np.mean(demand_train[same_month_past])
            else:
                sp[i] = 0  # 首次出现该月, 无历史
        seasonal_profile_tr = [sp]

    # 实验1: 事件保持特征 (SHOS-based)
    evt_feats_tr = []
    if _use_evt:
        d = demand_train
        n = train_len
        # Stop类: 距上次事件的间隔
        gap_since_last = np.zeros(n)
        last_event = -999
        for i in range(n):
            if d[i] > 0: last_event = i
            gap_since_last[i] = i - last_event if last_event >= 0 else n
        # Hoover类: 事件密度
        evt_cnt_6m = np.zeros(n); evt_cnt_12m = np.zeros(n)
        cumul_12m = np.zeros(n)
        for i in range(n):
            evt_cnt_6m[i] = np.sum(d[max(0,i-5):i+1] > 0)
            evt_cnt_12m[i] = np.sum(d[max(0,i-11):i+1] > 0)
            cumul_12m[i] = np.sum(d[max(0,i-11):i+1])
        # Occurrence类: 月份事件频率(平滑)
        month_freq = np.zeros(n)
        cal_month = np.array([(4+i)%12+1 for i in range(n)])
        for i in range(12, n):
            past_same_month = [j for j in range(i) if cal_month[j]==cal_month[i]]
            if past_same_month:
                month_freq[i] = np.mean(d[past_same_month] > 0)
        # 注: last_evt_mag/evt_trend/evt_cv 在测试集为常数(仅复制训练集最后值), 已移除
        evt_feats_tr = [gap_since_last, evt_cnt_6m, evt_cnt_12m, cumul_12m, month_freq]

    # 方案A: 季度末哑变量
    qtr_end_cols = []
    if USE_QUARTER_DUMMIES:
        cal_month_tr = np.array([(4 + i) % 12 + 1 for i in range(train_len)])
        is_q1_end = (cal_month_tr == 3).astype(float)
        is_q2_end = (cal_month_tr == 6).astype(float)
        is_q3_q4_end = ((cal_month_tr == 9) | (cal_month_tr == 12)).astype(float)
        qtr_end_cols = [is_q1_end, is_q2_end, is_q3_q4_end]

    # 周期编码 (论文精简时移除: |r|<0.13)
    _sincos_cols_tr = []
    if _use_sincos:
        _sincos_cols_tr = [np.sin(2*np.pi*m_train/12), np.cos(2*np.pi*m_train/12),
                           np.sin(2*np.pi*q_train/4), np.cos(2*np.pi*q_train/4)]
    X_train_raw = np.column_stack([
        data_train[:,1:], lag1_tr, lag12_tr, roll3_tr,
        is_zero_lag1_tr, is_zero_lag12_tr,
    ] + _sincos_cols_tr + evt_feats_tr + qtr_end_cols + enh_lag_feats_tr + batch_feats_tr + seasonal_profile_tr)

    # 实验6: 量级特征(lag-1) → 追加到特征矩阵末尾
    mag_all = build_magnitude_features(df, material)
    if USE_MAG_FEATURES and mag_all.shape[1] > 0:
        X_train_raw = np.column_stack([X_train_raw, mag_all[:train_len]])

    # GINN: 灰色先验特征 (GM(1,1)+分数阶AGO, SCI创新点, 无泄漏)
    # ElasticNet L1正则自动筛选: 灰色特征有用→保留系数, 无用→归零
    grey_tr, grey_te = build_grey_features(demand_raw, train_len)
    if _use_grey and grey_tr.shape[1] > 0:
        X_train_raw = np.column_stack([X_train_raw, grey_tr])

    # Step 3: 防御性NaN清洗 + 特征Scaler仅对训练集fit
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(X_train_raw)
    y_train = demand_train.copy()

    # Step 4: 测试集特征
    lag1_te = np.zeros(N_TEST); lag12_te = np.zeros(N_TEST); roll3_te = np.zeros(N_TEST)
    for i in range(N_TEST):
        idx = train_len + i
        lag1_te[i] = demand_raw[idx-1] if idx>0 else 0
        lag12_te[i] = demand_raw[idx-12] if idx>=12 else 0
        roll3_te[i] = np.mean(demand_raw[max(0,idx-3):idx])
    is_zero_lag1_te = (lag1_te == 0).astype(float)
    is_zero_lag12_te = (lag12_te == 0).astype(float)
    m_test = (np.arange(train_len+1, train_len+N_TEST+1)) % 12; m_test[m_test==0]=12
    q_test = ((np.arange(train_len+1, train_len+N_TEST+1)) // 3) % 4; q_test[q_test==0]=4

    # D-02: 测试集增强滞后/滚动特征 (使用demand_raw的历史真实值, 无未来泄露)
    enh_lag_feats_te = []
    if USE_ENHANCED_LAG_FEATURES:
        nt = N_TEST; tl = train_len
        lag2_te = np.zeros(nt); lag3_te = np.zeros(nt); lag6_te = np.zeros(nt)
        roll6_te = np.zeros(nt); roll12_te = np.zeros(nt); roll3std_te = np.zeros(nt)
        ewm_te = np.zeros(nt); yoy_te = np.zeros(nt)
        # EWM需要训练集末尾状态作为初始值
        alpha = 0.3
        ewm_state = 0.0
        for i in range(1, tl):
            ewm_state = alpha * demand_raw[i-1] + (1 - alpha) * ewm_state
        for i in range(nt):
            idx = tl + i
            lag2_te[i] = demand_raw[idx-2] if idx >= 2 else 0
            lag3_te[i] = demand_raw[idx-3] if idx >= 3 else 0
            lag6_te[i] = demand_raw[idx-6] if idx >= 6 else 0
            roll6_te[i] = np.mean(demand_raw[max(0,idx-6):idx]) if idx > 0 else 0
            roll12_te[i] = np.mean(demand_raw[max(0,idx-12):idx]) if idx > 0 else 0
            roll3std_te[i] = np.std(demand_raw[max(0,idx-3):idx]) if idx > 1 else 0
            ewm_state = alpha * demand_raw[idx-1] + (1 - alpha) * ewm_state
            ewm_te[i] = ewm_state
            yoy_te[i] = (demand_raw[idx-1] - demand_raw[idx-13]) if idx >= 13 else 0
        enh_lag_feats_te = [lag2_te, lag3_te, lag6_te, roll6_te, roll12_te, roll3std_te, yoy_te]
        if _use_ewm:
            enh_lag_feats_te.append(ewm_te)  # 论文精简时跳过(|r|<0.13)

    # D-04: 测试集批次采购附加特征 (lag-1)
    batch_feats_te = []
    if USE_BATCH_FEATURES:
        for col in ['has_batch', 'digital_bids']:
            if col in df.columns:
                vals = df[col].values.astype(np.float64)
                lagged_te = np.zeros(N_TEST)
                for i in range(N_TEST):
                    idx = train_len + i
                    lagged_te[i] = vals[idx-1] if idx >= 1 else 0
                batch_feats_te.append(lagged_te)

    # D-05: 测试集月度季节剖面 (用训练集同月均值)
    seasonal_profile_te = []
    if USE_SEASONAL_PROFILE and 'date' in df.columns:
        cal_m_all = df['date'].dt.month.values
        # 训练集各月均值
        month_means = {}
        for mo in range(1, 13):
            idxs = [j for j in range(train_len) if cal_m_all[j] == mo]
            month_means[mo] = np.mean(demand_train[idxs]) if idxs else 0
        sp_te = np.array([month_means.get(cal_m_all[train_len + i], 0) for i in range(N_TEST)])
        seasonal_profile_te = [sp_te]

    # 实验1: 测试集事件特征
    evt_feats_te = []
    if _use_evt:
        d_all = demand_raw; tl = train_len; nt = N_TEST
        gap_sl_te = np.zeros(nt); last_ev = -999
        for i in range(tl):
            if d_all[i] > 0: last_ev = i
        for i in range(nt):
            idx = tl + i
            if d_all[idx] > 0: last_ev = idx
            gap_sl_te[i] = idx - last_ev if last_ev >= 0 else tl
        ec6_te = np.zeros(nt); ec12_te = np.zeros(nt); cu12_te = np.zeros(nt)
        for i in range(nt):
            seq = d_all[max(0,tl+i-5):tl+i+1]
            seq12 = d_all[max(0,tl+i-11):tl+i+1]
            ec6_te[i] = np.sum(np.array(seq) > 0)
            ec12_te[i] = np.sum(np.array(seq12) > 0)
            cu12_te[i] = np.sum(seq12)
        mf_te = np.zeros(nt); cal_m_all = np.array([(4+i)%12+1 for i in range(tl+nt)])
        for i in range(nt):
            past_same = [j for j in range(tl+i) if cal_m_all[j]==cal_m_all[tl+i]]
            if past_same: mf_te[i] = np.mean(d_all[past_same] > 0)
        evt_feats_te = [gap_sl_te, ec6_te, ec12_te, cu12_te, mf_te]

    qtr_end_cols_te = []
    if USE_QUARTER_DUMMIES:
        cal_month_te = np.array([(4 + (train_len + i)) % 12 + 1 for i in range(N_TEST)])
        is_q1_end_te = (cal_month_te == 3).astype(float)
        is_q2_end_te = (cal_month_te == 6).astype(float)
        is_q3_q4_end_te = ((cal_month_te == 9) | (cal_month_te == 12)).astype(float)
        qtr_end_cols_te = [is_q1_end_te, is_q2_end_te, is_q3_q4_end_te]

    _sincos_cols_te = []
    if _use_sincos:
        _sincos_cols_te = [np.sin(2*np.pi*m_test/12), np.cos(2*np.pi*m_test/12),
                           np.sin(2*np.pi*q_test/4), np.cos(2*np.pi*q_test/4)]
    X_test_raw = np.column_stack([
        data_test[:,1:], lag1_te, lag12_te, roll3_te,
        is_zero_lag1_te, is_zero_lag12_te,
    ] + _sincos_cols_te + evt_feats_te + qtr_end_cols_te + enh_lag_feats_te + batch_feats_te + seasonal_profile_te)
    if USE_MAG_FEATURES and mag_all.shape[1] > 0:
        X_test_raw = np.column_stack([X_test_raw, mag_all[train_len:train_len+N_TEST]])
    if USE_GREY_FEATURES and grey_te.shape[1] > 0:
        X_test_raw = np.column_stack([X_test_raw, grey_te])
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = feature_scaler.transform(X_test_raw)
    y_test = demand_test.copy()

    logger.info(f"  [特征工程] top4={top4}, n_feat={X_train.shape[1]}维, 训练={train_len}月"
                + (" [论文精简ON]" if USE_PAPER_FEATURES else "")
                + (" [事件特征ON]" if _use_evt else "")
                + (" [量级特征ON]" if USE_MAG_FEATURES else "")
                + (" [增强滞后ON]" if USE_ENHANCED_LAG_FEATURES else "")
                + (" [灰色先验ON]" if _use_grey else "")
                + (" [log1p]" if USE_LOG1P_TARGET else ""))
    return X_train, y_train, X_test, y_test, feature_scaler

def baseline_naive_seasonal(y_train, y_test, period=12):
    """季节性朴素预测: yhat_t = y_{t-period} (抄去年同期)"""
    preds = np.array([y_train[-period + (i % period)] for i in range(len(y_test))])
    y_test_orig = y_test
    y_pred_orig = preds
    return y_pred_orig, y_test_orig

def baseline_persistence(y_train, y_test):
    """持久性预测: yhat_{t+1} = y_t (抄上月)"""
    preds = np.full(len(y_test), y_train[-1])
    y_test_orig = y_test
    y_pred_orig = preds
    return y_pred_orig, y_test_orig

def baseline_sarima(y_train, y_test):
    """SARIMA(1,0,1)(1,0,1,12) 基线预测"""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y_train_orig = y_train
        model = SARIMAX(y_train_orig, order=(1,0,1), seasonal_order=(1,0,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        y_pred_orig = fit.forecast(steps=len(y_test))
        y_test_orig = y_test
        return np.maximum(y_pred_orig, 0), y_test_orig
    except Exception:
        # Fallback to Naive Seasonal
        return baseline_naive_seasonal(y_train, y_test)


def baseline_naive_mean(y_train, y_test):
    """历史均值预测: 所有测试月预测为训练集均值"""
    preds = np.full(len(y_test), np.mean(y_train))
    return preds, y_test

def evaluate_model(y_true, y_pred, y_train=None):
    """计算 MSE/RMSE/MAE/R²/sMAPE/MASE/WRMSSE

    WRMSSE (M5竞赛标准): 分母为训练集季节性naive误差，对零值不敏感。
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape = np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_pred)+np.abs(y_true)+1e-10))*100
    mase_denom = np.mean(np.abs(y_true[1:]-y_true[:-1])) if len(y_true)>1 else 1
    mase = mae / max(mase_denom, 1e-10) if mase_denom > 0 else 999
    # WRMSSE: 分母是训练集季节性naive的RMSE (M5标准)
    wmsse = None
    if y_train is not None and len(y_train) >= 12:
        seas_naive_pred = np.array([y_train[-12 + (i % 12)] for i in range(len(y_true))])
        seas_rmse = np.sqrt(np.mean((seas_naive_pred - y_true)**2))
        seas_naive_train_err = np.sqrt(np.mean((y_train[12:] - y_train[:-12])**2))
        wmsse = round(seas_rmse / max(seas_naive_train_err, 1e-10), 4)
    return {'MSE': round(mse,4), 'RMSE': round(rmse,4), 'MAE': round(mae,4),
            'R2': round(r2,4), 'sMAPE': round(smape,2), 'MASE': round(mase,4),
            'WRMSSE': wmsse,
            'zero_acc': round(np.mean((y_true==0)==(y_pred==0)),4)}


def crps_score(y_true, pred_quantiles):
    """sCRPS: 缩放连续排位概率分数。评估概率预测分布质量。

    Args:
        y_true: shape (n,) 真实值
        pred_quantiles: shape (n, n_quantiles) 每个时间步的各分位数预测
    Returns:
        sCRPS 值 (数值积分近似)
    """
    n = len(y_true)
    yt = np.array(y_true).reshape(-1)
    pq = np.array(pred_quantiles)
    if pq.ndim != 2 or pq.shape[0] != n:
        return None
    q_levels = np.linspace(0, 1, pq.shape[1])  # [0, 1/(k-1), 2/(k-1), ..., 1]
    crps_vals = np.zeros(n)
    for i in range(n):
        fi = pq[i]
        indicator = np.array(fi <= yt[i], dtype=float)
        integrand = (q_levels - indicator)**2
        # 梯形积分: np.trapz removed in numpy 2.0
        crps_vals[i] = np.sum((integrand[:-1] + integrand[1:]) / 2 * np.diff(q_levels))
    crps = np.mean(crps_vals)
    # 缩放因子: 真实值绝对均值
    scale = np.mean(np.abs(yt)) if np.mean(np.abs(yt)) > 0 else 1
    return round(crps / scale, 4)


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


def plot_feature_importance(importance_dict, material):
    """特征重要性条形图（CatBoost 特征重要性）"""
    top4 = get_top_factors(material)
    imp = importance_dict.get('catboost_imp')
    if imp is None or len(imp) == 0:
        return

    n_features = len(imp)
    feat_names = list(top4)
    if len(feat_names) > n_features:
        feat_names = feat_names[:n_features]
    elif len(feat_names) < n_features:
        feat_names = feat_names + [f'F{i+1}' for i in range(len(feat_names), n_features)]

    fig, ax = plt.subplots(figsize=(10, max(5, n_features * 0.4)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_features))
    ax.barh(range(n_features), imp, color=colors, edgecolor='navy', alpha=0.85)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feat_names)
    ax.set_xlabel('Importance')
    ax.set_title(f'CatBoost 特征重要性 — {MATERIAL_LABELS[material]}', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'特征重要性_{material}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [图表] 特征重要性({material}) → {path}")


def plot_metrics_comparison(all_metrics):
    """模型指标对比：分组柱状图（各物资各模型的四项指标）"""
    # 仅使用实际有预测结果的物资（跳过被SKIP_REACTOR_PROTECT等开关排除的）
    active_materials = [m for m in MATERIALS if m in all_metrics]
    if not active_materials:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_names = ['MSE', 'RMSE', 'MAE', 'R2']
    # Dynamically collect all model names from results
    model_names = []
    for material in active_materials:
        for m in all_metrics[material]:
            if m not in model_names:
                model_names.append(m)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(model_names), 1)))

    for ax_idx, metric in enumerate(metric_names):
        ax = axes[ax_idx // 2, ax_idx % 2]
        x = np.arange(len(active_materials))
        width = 0.8 / max(len(model_names), 1)

        for i, model_name in enumerate(model_names):
            values = []
            for material in active_materials:
                vals = all_metrics[material].get(model_name, {})
                values.append(vals.get(metric, 0))
            bars = ax.bar(x + i * width, values, width, label=model_name,
                          color=colors[i], alpha=0.85, edgecolor='white')

            # 数值标注
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels([MATERIAL_LABELS[m] for m in active_materials])
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


# ===================== log1p逆变换辅助 =====================
def _inverse_log1p(y_pred):
    """如果USE_LOG1P_TARGET启用, 将预测值从log空间逆变换回原始空间"""
    return np.expm1(y_pred) if USE_LOG1P_TARGET else y_pred


# ===================== 4. 两阶段预测框架 =====================
def _two_stage_fit_predict(X_tr, y_tr, X_te, stage2_regressor, material):
    """所有模型共享的两阶段预测: Stage1分类×Stage2回归 = P×Q"""
    y_bin = (y_tr > 0).astype(int)
    n_pos, n_neg = y_bin.sum(), len(y_bin) - y_bin.sum()
    if n_pos < 5 or n_neg < 5 or (y_tr > 0).sum() < 10:
        return None, None  # 回退信号

    hp = get_hp('stage1_cls', material)
    cls = CatBoostClassifier(**hp,
        loss_function='Logloss', early_stopping_rounds=30, random_seed=RANDOM_SEED, verbose=0)
    nv = max(6, len(y_tr)//4)
    cls.fit(X_tr[:-nv], y_bin[:-nv], eval_set=(X_tr[-nv:], y_bin[-nv:]))
    prob = np.clip(cls.predict_proba(X_te)[:,1], 0, 1)

    nz = y_tr > 0
    nv2 = min(6, nz.sum()//4)
    is_cb = hasattr(stage2_regressor, 'get_params')
    try:
        if np.std(y_tr[nz]) < 1e-10:
            return np.full(len(X_te), np.mean(y_tr[nz])), None
        stage2_regressor.fit(X_tr[nz][:-nv2], y_tr[nz][:-nv2],
            eval_set=(X_tr[nz][-nv2:], y_tr[nz][-nv2:]))
    except (TypeError, ValueError):
        try:
            stage2_regressor.fit(X_tr[nz], y_tr[nz])
        except:
            return prob * np.mean(y_tr[nz]), cls
    return prob * np.maximum(stage2_regressor.predict(X_te), 0), cls


def run_catboost(X_train_factors, y_train, X_test_factors, y_test, material):
    """CatBoost(朴素): 直接回归, 不使用两阶段框架(消融基线)"""
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('catboost_direct', material)
    reg = CatBoostRegressor(**hp,
        loss_function='RMSE', early_stopping_rounds=50, random_seed=RANDOM_SEED, verbose=0)
    nv = min(12, len(y_tr_log)//4)
    reg.fit(X_train_factors[:-nv], y_tr_log[:-nv], eval_set=(X_train_factors[-nv:], y_tr_log[-nv:]))
    yp = np.maximum(reg.predict(X_test_factors), 0)
    if USE_LOG1P_TARGET: yp = np.expm1(yp)
    return yp, y_test, reg.get_feature_importance(), reg

def run_catboost_tweedie(X_train_factors, y_train, X_test_factors, y_test, material):
    """CatBoost-Tweedie(单阶段):  复合Poisson-Gamma损失函数直接回归。

    关键差异: 不同于两阶段Tweedie(Stage1分类+Stage2的Tweedie loss), 这是单阶段——
    用全部69个月(含零值)训练, Tweedie损失天然建模"零=未发标+正=采购量"的生成机制。
    与RMSE直接回归的对比: Tweedie在零点有离散概率质量, 不会强迫模型预测非零。

    预期: 波动数据(CV>1)上 Tweedie > RMSE(直接) >> 两阶段
    """
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('catboost_tweedie', material)
    reg = CatBoostRegressor(**hp,
        loss_function='Tweedie:variance_power=1.5',
        early_stopping_rounds=50, random_seed=RANDOM_SEED, verbose=0)
    nv = min(12, len(y_tr_log)//4)
    reg.fit(X_train_factors[:-nv], y_tr_log[:-nv], eval_set=(X_train_factors[-nv:], y_tr_log[-nv:]))
    yp = np.maximum(reg.predict(X_test_factors), 0)
    if USE_LOG1P_TARGET: yp = np.expm1(yp)
    return yp, y_test, reg.get_feature_importance(), reg


# ===================== 实验4: 分位数回归 (可取消) =====================
def run_catboost_quantile(X_train_factors, y_train, X_test_factors, y_test, material):
    """分位数回归: 训练 alpha=0.1/0.5/0.9 三个分位数模型。

    中位数(0.5)作为点预测, (0.1, 0.9)作为80%预测区间。
    优势: 不做硬判决"有/无需求", 而是输出"不确定区间"。
    当下限>0 → 高置信度有需求; 上限≈0 → 高置信度没有。
    """
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('quantile', material)
    preds = {}
    for alpha in [0.1, 0.5, 0.9]:
        reg = CatBoostRegressor(**hp,
            loss_function=f'Quantile:alpha={alpha}',
            early_stopping_rounds=50, random_seed=RANDOM_SEED, verbose=0)
        nv = min(12, len(y_tr_log)//4)
        reg.fit(X_train_factors[:-nv], y_tr_log[:-nv], eval_set=(X_train_factors[-nv:], y_tr_log[-nv:]))
        p = np.maximum(reg.predict(X_test_factors), 0)
        if USE_LOG1P_TARGET: p = np.expm1(p)
        preds[alpha] = p
    yp = preds[0.5]  # 中位数点预测
    logger.info(f'  [分位数] 80%区间宽度均值={np.mean(preds[0.9]-preds[0.1]):.2f}')
    return yp, y_test, None, (preds[0.1], preds[0.9])


def run_catboost_2s(X_train_factors, y_train, X_test_factors, y_test, material):
    """TwoStage-CatBoost: 两阶段=分类×CatBoost回归"""
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('catboost_2s', material)
    reg = CatBoostRegressor(**hp,
        loss_function='RMSE', early_stopping_rounds=50, random_seed=RANDOM_SEED, verbose=0)
    yp, _ = _two_stage_fit_predict(X_train_factors, y_tr_log, X_test_factors, reg, material)
    if yp is None:
        logger.warning(f'  [CatBoost-2S] 回退到直接回归')
        nv = min(12, len(y_tr_log)//4)
        reg.fit(X_train_factors[:-nv], y_tr_log[:-nv], eval_set=(X_train_factors[-nv:], y_tr_log[-nv:]))
        yp = np.maximum(reg.predict(X_test_factors), 0)
    if USE_LOG1P_TARGET and yp is not None: yp = np.expm1(yp)
    return yp, y_test, reg.get_feature_importance(), reg


def run_conditional_catboost(X_train_factors, y_train, X_test_factors, y_test, material):
    """TwoStage-CondCatBoost: 两阶段=分类×更深CatBoost回归"""
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('cond_catboost', material)
    reg = CatBoostRegressor(**hp,
        loss_function='RMSE', early_stopping_rounds=50, random_seed=RANDOM_SEED, verbose=0)
    yp, _ = _two_stage_fit_predict(X_train_factors, y_tr_log, X_test_factors, reg, material)
    if yp is None:
        logger.warning(f'  [CondCatBoost-2S] 回退到直接回归')
        nv = min(12, len(y_tr_log)//4)
        reg.fit(X_train_factors[:-nv], y_tr_log[:-nv], eval_set=(X_train_factors[-nv:], y_tr_log[-nv:]))
        yp = np.maximum(reg.predict(X_test_factors), 0)
    if USE_LOG1P_TARGET and yp is not None: yp = np.expm1(yp)
    return yp, y_test, reg.get_feature_importance(), reg

# 原 TwoStage 保留不变


# ===================== 两阶段预测 (论文核心创新) =====================
def run_two_stage(df_all, X_train_factors, y_train, X_test_factors, y_test,
                   material):
    """两阶段预测:
    Stage 1: CatBoost 二分类 — 预测当月是否有需求(>0)
    Stage 2: CatBoost 回归 — 对有需求的月份预测需求量
    最终 = P(有需求) × 预测量
    """
    demand_raw = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    y_train_binary = (demand_raw > 0).astype(int)
    n_pos = y_train_binary.sum(); n_neg = len(y_train_binary) - n_pos
    if n_pos < 5 or n_neg < 5:
        logger.warning(f"  [两阶段] 正负样本不均衡(pos={n_pos},neg={n_neg}), 回退到CatBoost")
        return run_catboost(X_train_factors, y_train, X_test_factors, y_test, material)

    # Stage 1: 分类器
    hp_cls = get_hp('twostage_cls', material)
    cls = CatBoostClassifier(**hp_cls,
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
        return run_catboost(X_train_factors, y_train, X_test_factors, y_test, material)

    X_nz = X_train_factors[nonzero_mask]
    y_nz = demand_raw[nonzero_mask]  # demand_raw已log1p, y_train未变换

    reg = CatBoostRegressor(**get_hp('twostage_reg', material),
        loss_function='RMSE', early_stopping_rounds=50,
        random_seed=RANDOM_SEED, verbose=0
    )
    n_val2 = min(6, len(y_nz)//4)
    reg.fit(X_nz[:-n_val2], y_nz[:-n_val2],
            eval_set=(X_nz[-n_val2:], y_nz[-n_val2:]))
    qty_test = reg.predict(X_test_factors)

    # 反归一化
    y_test_orig = y_test
    qty_pred_orig = qty_test
    # 最终 = 概率 × 预测量
    y_pred_orig = prob_test * np.maximum(qty_pred_orig, 0)
    if USE_LOG1P_TARGET: y_pred_orig = np.expm1(np.maximum(y_pred_orig, 0))
    cls_acc = np.mean((prob_test > 0.5).astype(int) == (y_test_orig > 0).astype(int))
    logger.info(f"  [两阶段] Stage1分类准确率={cls_acc:.2%}, 非零训练样本={nonzero_mask.sum()}")
    return y_pred_orig, y_test_orig, None, (cls, reg)


# ===================== 线性两阶段模型 =====================
def run_ridge_2s(X_train_factors, y_train, X_test_factors, y_test, material):
    """TwoStage-Ridge: 两阶段=分类×Ridge回归 (小样本更稳定)"""
    from sklearn.linear_model import RidgeCV
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('ridge_2s', material)
    reg = RidgeCV(**hp)
    nz = y_tr_log > 0
    yp, _ = _two_stage_fit_predict(X_train_factors, y_tr_log, X_test_factors, reg, material)
    if yp is None:
        reg.fit(X_train_factors, y_tr_log)
        yp = np.maximum(reg.predict(X_test_factors), 0)
    if USE_LOG1P_TARGET and yp is not None: yp = np.expm1(yp)
    return yp, y_test, None, reg


def run_elasticnet_2s(X_train_factors, y_train, X_test_factors, y_test, material):
    """TwoStage-ElasticNet: 两阶段=分类×ElasticNet(L1+L2, 小样本强正则)"""
    from sklearn.linear_model import ElasticNetCV
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('elasticnet_2s', material)
    nz = y_tr_log > 0
    reg = ElasticNetCV(**hp, cv=min(5, nz.sum()), max_iter=5000, random_state=42)
    yp, _ = _two_stage_fit_predict(X_train_factors, y_tr_log, X_test_factors, reg, material)
    if yp is None:
        reg.fit(X_train_factors, y_tr_log)
        yp = np.maximum(reg.predict(X_test_factors), 0)
    if USE_LOG1P_TARGET and yp is not None: yp = np.expm1(yp)
    return yp, y_test, None, reg


def run_croston_sba(y_train, y_test):
    """Croston-SBA: 间歇性需求专用预测。Decompose into demand interval + size.
    SBA变体对Croston偏差做修正: yhat = (1-alpha/2) * size_ema / interval_ema"""
    d_raw = y_train  # y不再缩放, 直接是原始值
    alpha = 0.1
    # 初始化
    nonzero_idx = np.where(d_raw > 0)[0]
    if len(nonzero_idx) < 2:
        return np.full(len(y_test), np.mean(d_raw)), y_test
    size_ema = d_raw[nonzero_idx[0]]
    interval_ema = nonzero_idx[0] + 1 if nonzero_idx[0] > 0 else 1
    last_nz = nonzero_idx[0]
    size_log, intv_log = [], []
    for i in range(1, len(nonzero_idx)):
        interval = nonzero_idx[i] - last_nz
        demand = d_raw[nonzero_idx[i]]
        size_ema = alpha * demand + (1-alpha) * size_ema
        interval_ema = alpha * interval + (1-alpha) * interval_ema
        last_nz = nonzero_idx[i]
        size_log.append(size_ema); intv_log.append(interval_ema)
    if not size_log:
        return np.full(len(y_test), np.mean(d_raw)), y_test
    s, iv = size_log[-1], max(intv_log[-1], 1)
    sba = s / iv * (1 - alpha/2)  # SBA修正
    preds = np.array([sba if (i % max(1, int(round(iv)))) == 0 else 0 for i in range(len(y_test))])
    return np.maximum(preds, 0), y_test


# ===================== Chronos 零样本基线 (Phase 1: TSFM通用性边界) =====================
# chronos-t5-tiny (8M参数, ~50MB): 最小TSFM基线, 内存友好。
# 首次运行需联网下载。国内网络需代理 (设置HTTPS_PROXY=http://127.0.0.1:7890)
# 如需更强基线，安装 chronos-2 (pip install "chronos-forecasting>=2.0" + 120M模型)
CHRONOS_MODEL = os.environ.get('CHRONOS_MODEL', 'amazon/chronos-t5-tiny')
CHRONOS_LOCAL_PATH = os.environ.get('CHRONOS_LOCAL_PATH', '')

def baseline_chronos(y_train, y_test, material):
    """Chronos 零样本预测基线。

    Amazon预训练时序基础模型，在ECP数据上做零样本推断。默认为 chronos-t5-tiny (8M)。
    定位: 量化"通用预训练模型 vs 领域专项模型"的泛化差距。
    """
    try:
        from chronos import BaseChronosPipeline
    except ImportError:
        logger.warning("  [Chronos] chronos-forecasting 未安装, 跳过基线")
        return None, y_test, None, None

    import torch as _torch

    def _load(pretrained_path, **kw):
        logger.info(f"  [Chronos] 加载: {pretrained_path}")
        return BaseChronosPipeline.from_pretrained(pretrained_path, device_map="cpu", **kw)

    # 按优先级: 本地路径 → 环境变量指定 → 默认 tiny
    sources = []
    if CHRONOS_LOCAL_PATH and os.path.isdir(CHRONOS_LOCAL_PATH):
        sources.append(CHRONOS_LOCAL_PATH)
    sources.append(CHRONOS_MODEL)

    pipeline = None
    last_err = ""
    for src in sources:
        try:
            pipeline = _load(src, torch_dtype=_torch.float32)
            break
        except Exception as e:
            last_err = str(e)[:200]
            logger.info(f"  [Chronos] {src} 不可用: {last_err}")
            continue

    if pipeline is None:
        logger.warning(f"  [Chronos] 所有来源均失败, 跳过基线 (最后错误: {last_err})")
        return None, y_test, None, None

    test_len = len(y_test)
    # 检查模型能力: 某些chronos变体model_prediction_length可能低于test_len
    max_pred_len = getattr(pipeline, 'model_prediction_length', test_len)
    logger.info(f"  [Chronos] model_prediction_length={max_pred_len}, request={test_len}")

    # Chronos 输入: 1D float tensor (可含零值, 不需要特殊预处理)
    context = _torch.tensor(y_train.astype(np.float32), dtype=_torch.float32)
    try:
        # 如果模型单次预测长度不足，使用滚动预测
        if max_pred_len < test_len:
            logger.info(f"  [Chronos] 单次预测长度不足, 使用滚动预测")
            _preds = []
            _ctx = context.clone()
            for _step in range(test_len):
                step_quantiles = pipeline.predict_quantiles(
                    _ctx, prediction_length=1, quantile_levels=[0.1, 0.5, 0.9])
                if isinstance(step_quantiles, tuple):
                    step_quantiles = step_quantiles[0]
                step_vals = step_quantiles.numpy()[0, 0, :]  # shape (3,)
                _preds.append(step_vals)
                # 将本次预测的中位数追加到context用于下一步
                _ctx = _torch.cat([_ctx, _torch.tensor([step_vals[1]], dtype=_torch.float32)])
            quantile_preds = np.array(_preds)  # shape (test_len, 3)
        else:
            quantile_preds = pipeline.predict_quantiles(
                context, prediction_length=test_len, quantile_levels=[0.1, 0.5, 0.9])
            if isinstance(quantile_preds, tuple):
                quantile_preds = quantile_preds[0]  # chronos 返回 (predictions, metadata) 元组
            if hasattr(quantile_preds, 'numpy'):
                quantile_preds = quantile_preds.numpy()
            quantile_preds = quantile_preds[0]  # shape (test_len, n_quantiles)
        y_pred_chronos = quantile_preds[:, 1]  # 中位数作为点预测
        scrps = crps_score(y_test, quantile_preds)
    except Exception as e:
        logger.warning(f"  [Chronos] 预测失败: {e}, 回退到均值预测")
        y_pred_chronos = np.full(test_len, np.mean(y_train))
        scrps = None

    y_pred_chronos = np.maximum(y_pred_chronos, 0)  # 物理约束: 需求量非负
    logger.info(f"  [Chronos] 零样本完成, sCRPS={scrps}")
    return y_pred_chronos, y_test, scrps, None

def run_lightgbm(X_train_factors, y_train, X_test_factors, y_test, material):
    """LightGBM: 两阶段=分类×LGBM回归, 对比CatBoost"""
    try:
        import lightgbm as lgb
    except ImportError:
        return None, y_test
    y_tr_log = np.log1p(y_train) if USE_LOG1P_TARGET else y_train
    hp = get_hp('lightgbm', material)
    yp, _ = _two_stage_fit_predict(X_train_factors, y_tr_log, X_test_factors,
        lgb.LGBMRegressor(**hp, random_state=RANDOM_SEED, verbose=-1), material)
    if yp is None:
        nv = min(12, len(y_tr_log)//4)
        reg = lgb.LGBMRegressor(**hp, random_state=RANDOM_SEED, verbose=-1)
        reg.fit(X_train_factors[:-nv], y_tr_log[:-nv], eval_set=[(X_train_factors[-nv:], y_tr_log[-nv:])])
        yp = np.maximum(reg.predict(X_test_factors), 0)
    if USE_LOG1P_TARGET and yp is not None: yp = np.expm1(yp)
    return yp, y_test


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


def run_nhits(df_all, material):
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
    if len(X_tr) < 50:
        logger.warning(f'  [N-HiTS] samples={len(X_tr)}<50, skip (小样本不稳定)'); return None,None,None,None
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
    header = f"{'物资':<16s} {'模型':<18s} {'MSE':>10s} {'RMSE':>10s} {'sMAPE':>8s} {'MASE':>8s} {'R2':>8s}"
    lines.append(header)
    lines.append("-" * 130)

    MODEL_ORDER = ['CatBoost', 'CatBoost-Tweedie', 'NaiveSeasonal', 'NaiveMean', 'Persistence', 'SARIMA',
                    'Chronos', 'Croston-SBA', 'CatBoost-2S', 'Ridge-2S', 'ElasticNet-2S',
                    'CondCatBoost', 'LightGBM', 'TwoStage', 'NHiTS']
    for material in MATERIALS:
        if material not in all_metrics:
            continue  # 被SKIP_REACTOR_PROTECT等开关跳过的物资
        for i, model_name in enumerate(MODEL_ORDER):
            metrics = all_metrics[material].get(model_name, {})
            if metrics:
                if i == 0:
                    lines.append(f"{MATERIAL_LABELS[material]:<16s} {model_name:<18s} "
                                 f"{metrics['MSE']:>10.2f} {metrics['RMSE']:>10.2f} "
                                 f"{metrics['sMAPE']:>7.1f}% {metrics['MASE']:>7.3f} "
                                 f"{metrics['R2']:>7.3f}")
                else:
                    lines.append(f"{'':<16s} {model_name:<18s} "
                                 f"{metrics['MSE']:>10.2f} {metrics['RMSE']:>10.2f} "
                                 f"{metrics['sMAPE']:>7.1f}% {metrics['MASE']:>7.3f} "
                                 f"{metrics['R2']:>7.3f}")
        lines.append("-" * 130)
    lines.append("")

    for line in lines:
        logger.info(line)


# ===================== 11. 主函数 =====================
def main():
    # Step 1: 加载数据
    logger.info("=" * 70)
    logger.info("  配电网物资需求预测 —— 两阶段预测模型对比实验")
    logger.info("  物资: Top5采购频率最高 (从ECP数据自动选择)")
    logger.info("  因子: 批次事件特征 + 项目数量 + 自回归滞后")
    logger.info("  模型: 15个 (树模型单/两阶段 + Tweedie + 线性 + 统计基线 + Chronos)")
    logger.info("=" * 70)
    logger.info(f"  日志文件: {log_filename}")
    logger.info("[加载] 加载数据...")
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

    # === 【Dashboard插入点3】初始化数据收集器 ===
    collector = ResultCollector() if DASHBOARD_AVAILABLE else None
    if collector:
        collector.set_meta(
            time_labels=["1月", "2月", "3月", "4月", "5月", "6月",
                         "7月", "8月", "9月", "10月", "11月", "12月"],
            forecast_horizon=12
        )

    # Step 3+4：建模与可视化（抑制 Windows 字体权限错误噪音）
    with suppress_font_stderr():
        plot_demand_curves(data_dict)

        # Step 3: 对三种物资分别建模
        for material in MATERIALS:
            label = MATERIAL_LABELS[material]
            logger.info("")
            logger.info(f"[处理] {label} ({material})...")
            # Skip materials in blacklist (e.g. 电抗器保护 R^2 always < 0.05)
            if SKIP_REACTOR_PROTECT and '电抗器保护' in material:
                logger.info("  [跳过] 电抗器保护已禁用 (R2<0.05, 噪声主导)")
                continue
            df = data_dict[material]
            # 注意: 不要在 preprocess_data 之前调用 get_top_factors(material)
            # 缓存机制会在首次调用时填充 fallback 值
            # preprocess_data 内部会传 df → 触发真正的 Spearman 计算

            X_train_factors, y_train, X_test_factors, y_test, feature_scaler = \
                preprocess_data(df, material)
            logger.debug(f"  训练集: {len(y_train)}月, 测试集: {len(y_test)}月")

            all_results[material] = {}
            all_metrics[material] = {}

            # --- 模型一: CatBoost ---
            logger.info("  [模型] CatBoost (两阶段)...")
            y_pred_1, y_test_1, imp_1, model_1 = run_catboost(
                X_train_factors, y_train, X_test_factors, y_test, material)
            metrics_1 = evaluate_model(y_test_1, y_pred_1)
            all_results[material]['CatBoost'] = {
                'y_pred': y_pred_1, 'y_test': y_test_1, 'metrics': metrics_1}
            all_metrics[material]['CatBoost'] = metrics_1
            logger.info(f"        MSE={metrics_1['MSE']:.4f} RMSE={metrics_1['RMSE']:.4f} "
                         f"MAE={metrics_1['MAE']:.4f} R2={metrics_1['R2']:.4f}")

            # --- CatBoost-Tweedie 单阶段 (波动数据专项:   复合Poisson-Gamma) ---
            y_pred_ctw, y_test_ctw, imp_ctw, model_ctw = run_catboost_tweedie(
                X_train_factors, y_train, X_test_factors, y_test, material)
            metrics_ctw = evaluate_model(y_test_ctw, y_pred_ctw)
            all_results[material]['CatBoost-Tweedie'] = {'y_pred': y_pred_ctw, 'y_test': y_test_ctw, 'metrics': metrics_ctw}
            all_metrics[material]['CatBoost-Tweedie'] = metrics_ctw
            logger.info(f"  [模型] CatBoost-Tweedie: R2={metrics_ctw['R2']:.4f} "
                         f"(vs RMSE Δ={metrics_ctw['R2']-metrics_1['R2']:+.4f})")

            # --- Baseline: Naive Seasonal ---
            y_pred_ns, y_test_ns = baseline_naive_seasonal(y_train, y_test)
            metrics_ns = evaluate_model(y_test_ns, y_pred_ns)
            all_results[material]['NaiveSeasonal'] = {'y_pred': y_pred_ns, 'y_test': y_test_ns, 'metrics': metrics_ns}
            all_metrics[material]['NaiveSeasonal'] = metrics_ns
            logger.info(f"  [Baseline] NaiveSeasonal: R2={metrics_ns['R2']:.4f}")

            # --- Baseline: Naive-Mean ---
            y_pred_nm, y_test_nm = baseline_naive_mean(y_train, y_test)
            metrics_nm = evaluate_model(y_test_nm, y_pred_nm)
            all_results[material]['NaiveMean'] = {'y_pred': y_pred_nm, 'y_test': y_test_nm, 'metrics': metrics_nm}
            all_metrics[material]['NaiveMean'] = metrics_nm
            logger.info(f"  [Baseline] NaiveMean: R2={metrics_nm['R2']:.4f}")

            # --- Baseline: Persistence ---
            y_pred_sp, y_test_sp = baseline_persistence(y_train, y_test)
            metrics_sp = evaluate_model(y_test_sp, y_pred_sp)
            all_results[material]['Persistence'] = {'y_pred': y_pred_sp, 'y_test': y_test_sp, 'metrics': metrics_sp}
            all_metrics[material]['Persistence'] = metrics_sp
            logger.info(f"  [Baseline] Persistence: R2={metrics_sp['R2']:.4f}")

            # --- Baseline: SARIMA ---
            y_pred_sa, y_test_sa = baseline_sarima(y_train, y_test)
            metrics_sa = evaluate_model(y_test_sa, y_pred_sa)
            all_results[material]['SARIMA'] = {'y_pred': y_pred_sa, 'y_test': y_test_sa, 'metrics': metrics_sa}
            all_metrics[material]['SARIMA'] = metrics_sa
            logger.info(f"  [Baseline] SARIMA: R2={metrics_sa['R2']:.4f}")

            # --- CatBoost-2S (两阶段) ---
            logger.info(f"  [两阶段] CatBoost-2S...")
            y_pred_cb2, y_test_cb2, imp_cb2, model_cb2 = run_catboost_2s(
                X_train_factors, y_train, X_test_factors, y_test, material)
            metrics_cb2 = evaluate_model(y_test_cb2, y_pred_cb2)
            all_results[material]['CatBoost-2S'] = {'y_pred': y_pred_cb2, 'y_test': y_test_cb2, 'metrics': metrics_cb2}
            all_metrics[material]['CatBoost-2S'] = metrics_cb2
            logger.info(f"  [模型] CatBoost-2S: R2={metrics_cb2['R2']:.4f}")

            # --- CondCatBoost ---
            logger.info("  [模型] CondCatBoost...")
            y_pred_cc, y_test_cc, imp_cc, model_cc = run_conditional_catboost(
                X_train_factors, y_train, X_test_factors, y_test, material)
            metrics_cc = evaluate_model(y_test_cc, y_pred_cc, y_train=y_train)
            all_results[material]['CondCatBoost'] = {'y_pred': y_pred_cc, 'y_test': y_test_cc, 'metrics': metrics_cc}
            all_metrics[material]['CondCatBoost'] = metrics_cc
            logger.info(f"        MSE={metrics_cc['MSE']:.4f} RMSE={metrics_cc['RMSE']:.4f} "
                         f"MAE={metrics_cc['MAE']:.4f} R2={metrics_cc['R2']:.4f}")

            # --- N-HiTS ---
            logger.info("  [模型] N-HiTS...")
            y_pred_nh, y_test_nh, imp_nh, model_nh = run_nhits(df, material)
            if y_pred_nh is not None:
                metrics_nh = evaluate_model(y_test_nh, y_pred_nh)
                all_results[material]['NHiTS'] = {'y_pred': y_pred_nh, 'y_test': y_test_nh, 'metrics': metrics_nh}
                all_metrics[material]['NHiTS'] = metrics_nh
                logger.info(f"        MSE={metrics_nh['MSE']:.4f} RMSE={metrics_nh['RMSE']:.4f} "
                             f"MAE={metrics_nh['MAE']:.4f} R2={metrics_nh['R2']:.4f}")
            else:
                logger.info(f"  [N-HiTS] 跳过 (未安装或失败)")

            # --- Chronos 零样本基线 (TSFM通用性边界) ---
            y_pred_ch, y_test_ch, scrps_ch, _ = baseline_chronos(y_train, y_test, material)
            if y_pred_ch is not None:
                metrics_ch = evaluate_model(y_test_ch, y_pred_ch, y_train=y_train)
                if scrps_ch is not None:
                    metrics_ch['sCRPS'] = scrps_ch
                all_results[material]['Chronos'] = {'y_pred': y_pred_ch, 'y_test': y_test_ch, 'metrics': metrics_ch}
                all_metrics[material]['Chronos'] = metrics_ch
                logger.info(f"  [基线] Chronos: R2={metrics_ch['R2']:.4f}, sCRPS={scrps_ch}")
            else:
                logger.info(f"  [基线] Chronos: 跳过 (未安装)")

            # --- Croston-SBA 间歇性需求基线 ---
            y_pred_cr, y_test_cr = run_croston_sba(y_train, y_test)
            metrics_cr = evaluate_model(y_test_cr, y_pred_cr)
            all_results[material]['Croston-SBA'] = {'y_pred': y_pred_cr, 'y_test': y_test_cr, 'metrics': metrics_cr}
            all_metrics[material]['Croston-SBA'] = metrics_cr
            logger.info(f"  [基线] Croston-SBA: R2={metrics_cr['R2']:.4f}")

            # --- LightGBM 对比 ---
            # --- TwoStage-Ridge ---
            y_pred_rd, y_test_rd, _, _ = run_ridge_2s(X_train_factors, y_train, X_test_factors, y_test, material)
            metrics_rd = evaluate_model(y_test_rd, y_pred_rd)
            all_results[material]['Ridge-2S'] = {'y_pred': y_pred_rd, 'y_test': y_test_rd, 'metrics': metrics_rd}
            all_metrics[material]['Ridge-2S'] = metrics_rd

            # --- LightGBM 对比 ---
            y_pred_lgb, y_test_lgb = run_lightgbm(X_train_factors, y_train, X_test_factors, y_test, material)
            if y_pred_lgb is not None:
                metrics_lgb = evaluate_model(y_test_lgb, y_pred_lgb)
                all_results[material]['LightGBM'] = {'y_pred': y_pred_lgb, 'y_test': y_test_lgb, 'metrics': metrics_lgb}
                all_metrics[material]['LightGBM'] = metrics_lgb
                logger.info(f"  [对比] LightGBM: R2={metrics_lgb['R2']:.4f}")
            else:
                logger.info(f"  [LightGBM] lightgbm未安装, 跳过")

            # --- 两阶段预测 (Stage1分类 + Stage2回归) ---
            logger.info("  [模型] TwoStage (独立实现)...")
            y_pred_ts, y_test_ts, imp_ts, model_ts = run_two_stage(
                df, X_train_factors, y_train, X_test_factors, y_test, material)
            metrics_ts = evaluate_model(y_test_ts, y_pred_ts)
            all_results[material]['TwoStage'] = {'y_pred': y_pred_ts, 'y_test': y_test_ts, 'metrics': metrics_ts}
            all_metrics[material]['TwoStage'] = metrics_ts
            logger.info(f"        MSE={metrics_ts['MSE']:.4f} RMSE={metrics_ts['RMSE']:.4f} "
                         f"MAE={metrics_ts['MAE']:.4f} R2={metrics_ts['R2']:.4f}")

            # --- ElasticNet-2S ---
            y_pred_en, y_test_en, _, _ = run_elasticnet_2s(X_train_factors, y_train, X_test_factors, y_test, material)
            metrics_en = evaluate_model(y_test_en, y_pred_en)
            all_results[material]['ElasticNet-2S'] = {'y_pred': y_pred_en, 'y_test': y_test_en, 'metrics': metrics_en}
            all_metrics[material]['ElasticNet-2S'] = metrics_en

            # 特征重要性图
            imp_dict = {
                'catboost_imp': imp_1,
            }
            plot_feature_importance(imp_dict, material)

        # Step 4: 评估汇总
        logger.info('')
        logger.info('[汇总] 评估与可视化...')
        print_metrics_table(all_metrics)

        # 保存指标 JSON
        metrics_json = {}
        for material in MATERIALS:
            if material not in all_results:
                continue
            plot_prediction_comparison(all_results, material)

        # 指标对比柱状图
        plot_metrics_comparison(all_metrics)

    # M-01: Top-3模型加权融合 (R²权重, 排除朴素基线)
    _BASELINE_MODELS = {'NaiveSeasonal', 'NaiveMean', 'Persistence', 'SARIMA', 'Chronos', 'Croston-SBA'}
    if USE_BLEND_ENSEMBLE:
        for material in list(all_results.keys()):
            if material not in all_metrics:
                continue
            # 筛选非基线模型, 按R²排序取top-3
            candidates = [(k, v['R2']) for k, v in all_metrics[material].items()
                          if k not in _BASELINE_MODELS and v['R2'] > 0]
            if len(candidates) < 2:
                continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            top3 = candidates[:3]
            # R²权重 (clip到正, 归一化)
            weights = np.array([max(r2, 0.01) for _, r2 in top3])
            weights = weights / weights.sum()
            # 加权融合预测
            y_preds = [all_results[material][name]['y_pred'] for name, _ in top3]
            y_test_ref = all_results[material][top3[0][0]]['y_test']
            y_blend = sum(w * yp for w, yp in zip(weights, y_preds))
            y_blend = np.maximum(y_blend, 0)  # 非负约束
            blend_metrics = evaluate_model(y_test_ref, y_blend)
            all_results[material]['Blend-Top3'] = {
                'y_pred': y_blend, 'y_test': y_test_ref, 'metrics': blend_metrics}
            all_metrics[material]['Blend-Top3'] = blend_metrics

    # === 【Dashboard插入点4+5】批量收集结果并生成看板 ===
    if collector:
        for material in all_results:
            for model_name, result_dict in all_results[material].items():
                collector.collect(material, model_name, result_dict)
    if collector and not NO_DASHBOARD:
        # 看板文件名与日志文件共享同一时间戳编号，便于关联追溯
        dashboard_output = DASHBOARD_OUTPUT or os.path.join(
            'output', f'dashboard_{RUN_TIMESTAMP}.html')
        try:
            DashboardBuilder(
                collector=collector,
                config_path=DASHBOARD_CONFIG,
                output_path=dashboard_output,
                debug=DASHBOARD_DEBUG
            ).build()
        except Exception as e:
            print(f"[Warning] 看板生成失败（不影响预测结果）: {e}")
            import traceback
            traceback.print_exc()

    # 最佳模型识别（纯日志输出，无需抑制）
    logger.info("")
    logger.info("=" * 70)
    logger.info("  模型性能排序 (按R^2)")
    logger.info("=" * 70)
    for material in MATERIALS:
        if material not in all_metrics:
            continue
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
