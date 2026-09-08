"""图5-6 特征重要性: 两阶段(Stage1发生/Stage2规模)的特征重要性.

选2个代表物资(电抗器保护-高频, 视频监视系统-中频), 分别提取:
  - Stage1 分类器(CatBoostClassifier)特征重要性 -> 哪些特征决定"该月是否采购"
  - Stage2 回归器(CatBoostRegressor)特征重要性 -> 哪些特征决定"采购多少"
"""
import sys, io, warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
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
FEAT_NAMES = ['项目数量', '输变电批次', '公告数', '特高压批次',
              'lag1', 'lag12', 'roll3', '上期是否零值',
              'lag2', 'lag3', 'lag6', 'roll6', 'roll12', 'roll3_std', '同比差',
              '距上次采购', '近6月采购次数', '近12月采购次数', '近12月采购量', '月频率']


def build_features(df, material):
    """构建特征(与main.py两阶段一致), 返回 X_train/y_train/X_test/y_test"""
    from scipy.stats import spearmanr
    # 因子选择: Spearman top4
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
    dtr, dte = demand_raw[:train_len], demand_raw[train_len:]

    # 特征
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
    # 加 top4 因子
    for col in top4:
        if col in df.columns:
            feats.append(df[col].astype(float).values[:train_len])
    feat_names = ['lag1','lag12','roll3','is_zero_lag1','lag2','lag3','lag6','roll6','roll12','roll3_std','yoy_diff',
                  'gap','evt6','evt12','cum12','month_freq'] + [f'F:{col}' for col in top4]
    X = np.column_stack(feats)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    sc = MinMaxScaler()
    X_tr = sc.fit_transform(X)
    y_tr = dtr
    return X_tr, y_tr, feat_names


def main():
    for mat, fpath in MATS:
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
        X, y, feat_names = build_features(df, mat)
        # Stage1: 分类
        y_bin = (y>0).astype(int)
        clf = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                                 loss_function='Logloss', random_seed=42, verbose=0)
        clf.fit(X, y_bin)
        imp1 = clf.get_feature_importance()
        # Stage2: 回归(仅非零)
        nz = y>0
        X_nz, y_nz = X[nz], y[nz]
        reg = CatBoostRegressor(iterations=300, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                                loss_function='RMSE', random_seed=42, verbose=0)
        reg.fit(X_nz, y_nz)
        imp2 = reg.get_feature_importance()

        # 输出
        print(f'=== {mat} ===')
        print('Stage1(发生) 特征重要性:')
        idx1 = np.argsort(imp1)[::-1][:8]
        for i in idx1:
            print(f'  {feat_names[i]:<15}{imp1[i]:.3f}')
        print('Stage2(规模) 特征重要性:')
        idx2 = np.argsort(imp2)[::-1][:8]
        for i in idx2:
            print(f'  {feat_names[i]:<15}{imp2[i]:.3f}')
        print()


if __name__ == '__main__':
    main()
