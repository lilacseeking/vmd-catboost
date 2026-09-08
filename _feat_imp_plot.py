"""图5-6 特征重要性图: 2物资 x 2阶段(发生/规模), 竖排, 风格统一."""
import sys, io, warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for f in fm.findSystemFonts():
    try:
        if any(n in f.lower() for n in ['simhei','msyh','yahei','simsun']): fm.fontManager.addfont(f)
    except: pass
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

TRAIN = 69
N_TEST = 12
MATS = [
    ('电抗器保护', 'inputs/data.xlsx'),
    ('视频监视系统', 'inputs/data_jibei_20.xlsx'),
]
FEAT_CN = {
    'F:project_count':'项目数量', 'F:transformer_bids':'输变电批次', 'F:monthly_bid_count':'公告数', 'F:uhv_bids':'特高压批次',
    'lag1':'上月需求', 'lag12':'去年同期', 'roll3':'近3月均值', 'is_zero_lag1':'上期零值',
    'lag2':'前2月', 'lag3':'前3月', 'lag6':'前6月', 'roll6':'近6月均值', 'roll12':'近12月均值',
    'roll3_std':'近3月波动', 'yoy_diff':'同比差', 'gap':'距上次采购', 'evt6':'近6月采购次数',
    'evt12':'近12月采购次数', 'cum12':'近12月采购量', 'month_freq':'月频率',
}


def build_features(df, material):
    FACTORS = ['project_count','transformer_bids','monthly_bid_count','uhv_bids']
    y = df['demand'].astype(float).values
    scores = {}
    for col in FACTORS:
        if col in df.columns:
            v = df[col].astype(float).values
            rho,_ = spearmanr(v, y)
            scores[col] = abs(rho)
    top4 = sorted(scores, key=scores.get, reverse=True)[:4]
    cols = ['demand'] + top4
    sub = df[cols].copy().ffill().bfill().fillna(0)
    data = sub.values.astype(float)
    demand_raw = data[:,0].copy()
    train_len = len(data) - N_TEST
    dtr = demand_raw[:train_len]
    n = train_len
    seq = dtr
    lag1 = np.zeros(n); lag1[1:] = seq[:-1]
    lag12 = np.zeros(n); lag12[12:] = seq[:-12]
    roll3 = np.array([np.mean(seq[max(0,i-3):i]) for i in range(n)])
    is_z1 = (seq==0).astype(float)
    lag2 = np.zeros(n); lag2[2:] = seq[:-2]
    lag3 = np.zeros(n); lag3[3:] = seq[:-3]
    lag6 = np.zeros(n); lag6[6:] = seq[:-6]
    roll6 = np.array([np.mean(seq[max(0,i-6):i]) if i>0 else 0 for i in range(n)])
    roll12 = np.array([np.mean(seq[max(0,i-12):i]) if i>0 else 0 for i in range(n)])
    roll3s = np.array([np.std(seq[max(0,i-3):i]) if i>1 else 0 for i in range(n)])
    yoy = np.zeros(n)
    for i in range(13,n): yoy[i] = seq[i-1]-seq[i-13]
    gap = np.zeros(n); le=-999
    for i in range(n):
        if seq[i]>0: le=i
        gap[i] = i-le if le>=0 else n
    evt6 = np.array([np.sum(seq[max(0,i-5):i+1]>0) for i in range(n)])
    evt12 = np.array([np.sum(seq[max(0,i-11):i+1]>0) for i in range(n)])
    cum12 = np.array([np.sum(seq[max(0,i-11):i+1]) for i in range(n)])
    mf = np.zeros(n)
    cm = np.array([(4+i)%12+1 for i in range(n)])
    for i in range(12,n):
        past=[j for j in range(i) if cm[j]==cm[i]]
        if past: mf[i]=np.mean(seq[past]>0)
    feats = [lag1, lag12, roll3, is_z1, lag2, lag3, lag6, roll6, roll12, roll3s, yoy,
             gap, evt6, evt12, cum12, mf]
    fnames = ['lag1','lag12','roll3','is_zero_lag1','lag2','lag3','lag6','roll6','roll12','roll3_std','yoy_diff',
              'gap','evt6','evt12','cum12','month_freq']
    for col in top4:
        if col in df.columns:
            feats.append(df[col].astype(float).values[:train_len])
            fnames.append(f'F:{col}')
    X = np.column_stack(feats)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    sc = MinMaxScaler()
    X_tr = sc.fit_transform(X)
    return X_tr, dtr, fnames


def get_imps(mat, fpath):
    df = pd.read_excel(fpath, sheet_name=mat)
    rn = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if 'demand' in cl or '需求' in str(c): rn[c]='demand'
        elif 'project' in cl or '项目' in str(c): rn[c]='project_count'
        elif 'transformer' in cl: rn[c]='transformer_bids'
        elif 'monthly' in cl or '公告' in str(c): rn[c]='monthly_bid_count'
        elif 'uhv' in cl or '特高压' in str(c): rn[c]='uhv_bids'
        elif 'date' in cl or '日期' in str(c): rn[c]='date'
    df = df.rename(columns=rn)
    X, y, fnames = build_features(df, mat)
    y_bin = (y>0).astype(int)
    clf = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                             loss_function='Logloss', random_seed=42, verbose=0)
    clf.fit(X, y_bin)
    imp1 = clf.get_feature_importance()
    nz = y>0
    reg = CatBoostRegressor(iterations=300, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                            loss_function='RMSE', random_seed=42, verbose=0)
    reg.fit(X[nz], y[nz])
    imp2 = reg.get_feature_importance()
    return fnames, imp1, imp2


def plot():
    # 4个子图: 2物资 x 2阶段
    fig, axes = plt.subplots(4, 1, figsize=(8, 13))
    colors = {'occ':'#5B7F9A', 'size':'#B86B4D'}
    for row, (mat, fpath) in enumerate(MATS):
        fnames, imp1, imp2 = get_imps(mat, fpath)
        # 中文名
        cn = [FEAT_CN.get(fn, fn) for fn in fnames]
        # Stage1
        ax = axes[row*2]
        top_idx = np.argsort(imp1)[::-1][:8]
        ypos = np.arange(len(top_idx))
        ax.barh(ypos, imp1[top_idx], color=colors['occ'], edgecolor='#2C3E50', linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([cn[i] for i in top_idx][::-1], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(f'{mat} · 发生过程（该月是否采购）', fontsize=13, pad=8)
        ax.grid(axis='x', alpha=0.3)
        # 数值
        for i,(idx,v) in enumerate(zip(top_idx, imp1[top_idx])):
            ax.text(v+0.5, i, f'{v:.1f}', va='center', fontsize=10)
        # Stage2
        ax = axes[row*2+1]
        top_idx = np.argsort(imp2)[::-1][:8]
        ypos = np.arange(len(top_idx))
        ax.barh(ypos, imp2[top_idx], color=colors['size'], edgecolor='#2C3E50', linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([cn[i] for i in top_idx][::-1], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(f'{mat} · 规模过程（采购多少）', fontsize=13, pad=8)
        ax.grid(axis='x', alpha=0.3)
        for i,(idx,v) in enumerate(zip(top_idx, imp2[top_idx])):
            ax.text(v+0.5, i, f'{v:.1f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/figures/fig5-6_feature_importance.svg', bbox_inches='tight')
    plt.savefig('outputs/figures/fig5-6_feature_importance.png', dpi=400, bbox_inches='tight')
    print('图5-6 已保存')
    import shutil, os
    dstdir = r'D:\Users\dell\Desktop\研究生文献收集'
    for src_n, dst_n in [('fig5-6_feature_importance.svg','图5-6_特征重要性.svg'),
                         ('fig5-6_feature_importance.png','图5-6_特征重要性.png')]:
        shutil.copy2(os.path.join(r'D:\Users\dell\PycharmProjects\vmd-catboost\outputs\figures', src_n),
                     os.path.join(dstdir, dst_n))
    print('已复制到桌面')


if __name__ == '__main__':
    plot()
