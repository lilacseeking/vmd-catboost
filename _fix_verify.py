"""验证修复后的点预测是否都在区间内, 并生成图3-1和图5-4."""
import sys, io, warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
import lightgbm as lgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for f in fm.findSystemFonts():
    try:
        if any(n in f.lower() for n in ['simhei','msyh','yahei','simsun']): fm.fontManager.addfont(f)
    except: pass
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

TRAIN, TEST = 62, 12
CALIB = 12
MATS = [
    ('电抗器保护', 'inputs/data.xlsx'),
    ('火灾报警系统', 'inputs/data_jibei_20.xlsx'),
    ('视频监视系统', 'inputs/data_jibei_20.xlsx'),
    ('330kV油浸电磁CT', 'inputs/data.xlsx'),
]
SHORT = {'电抗器保护':'电抗器保护','火灾报警系统':'火灾报警系统','视频监视系统':'视频监视系统','330kV油浸电磁CT':'330kV电磁CT'}


def build_self(y_tr, y_te):
    y_all = np.concatenate([y_tr, y_te])
    def extr(N, offset):
        F = np.zeros((N, 8))
        for i in range(N):
            t = offset + i
            F[i,0]=y_all[t-1] if t>=1 else 0
            F[i,1]=y_all[t-2] if t>=2 else 0
            F[i,2]=y_all[t-3] if t>=3 else 0
            F[i,3]=y_all[t-6] if t>=6 else 0
            F[i,4]=y_all[t-12] if t>=12 else 0
            F[i,5]=np.mean(y_all[max(0,t-2):t+1]) if t>=1 else y_all[0]
            F[i,6]=np.mean(y_all[max(0,t-5):t+1]) if t>=1 else y_all[0]
            last=-1
            for j in range(t-1,-1,-1):
                if y_all[j]>0: last=j; break
            F[i,7]=float(t-last) if last>=0 else 99.0
        return F
    return extr(TRAIN,0), extr(TEST,TRAIN)


def zi_fixed(y_all):
    y_tr, y_te = y_all[:TRAIN], y_all[TRAIN:]
    L_tr, L_te = build_self(y_tr, y_te)
    proper = TRAIN - CALIB
    y_bin = (y_tr>0).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    clf.fit(L_tr[:proper], y_bin[:proper])
    P = np.clip(clf.predict_proba(L_te)[:,1], 0.05, 0.95)
    nz = y_tr>0
    L_nz, y_nz = L_tr[nz], np.log1p(y_tr[nz])
    nz_idx = np.where(nz)[0]
    calib_mask = nz_idx>=proper
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
    m.fit(L_nz[~calib_mask], y_nz[~calib_mask])
    cp = m.predict(L_nz[calib_mask])
    q = np.quantile(np.abs(y_nz[calib_mask]-cp), 0.9)
    pred_log = m.predict(L_te)
    q_high = np.maximum(np.expm1(pred_log+q), 0)
    q_hat = np.maximum(np.expm1(pred_log), 0)
    # 修复: 点预测和上界都用P缩放
    pred = np.where(P>0.5, q_hat, P*q_hat)
    upper = np.where(P>0.5, q_high, P*q_high)
    return pred, upper, P, y_te


def main():
    # 1. 验证修复
    print('=== 修复验证: 点预测是否都在区间内 ===')
    results = {}
    for m, f in MATS:
        df = pd.read_excel(f, sheet_name=m)
        rn={}
        for c in df.columns:
            if '需求' in str(c) or 'demand' in str(c).lower(): rn[c]='demand'
        df=df.rename(columns=rn)
        y=df['demand'].astype(float).values
        y_all=y[-81:][:TRAIN+TEST]
        if len(y_all)<TRAIN+TEST: y_all=np.pad(y_all,(0,TRAIN+TEST-len(y_all)))
        pred, upper, P, y_te = zi_fixed(y_all)
        results[m] = (pred, upper, P, y_te)
        nz_te = y_te>0
        cov = np.mean((y_te>=0)&(y_te<=upper))
        covp = np.mean((y_te[nz_te]>=0)&(y_te[nz_te]<=upper[nz_te])) if nz_te.sum()>0 else np.nan
        exceed = [i+1 for i in range(12) if pred[i]>upper[i]]
        print(f'  {m}: 点预测>上界={exceed if exceed else "无"} 边际={cov:.2f} 非零={covp:.2f}')

    # 2. 画图3-1 需求量曲线 (新4物资)
    fig, axes = plt.subplots(4, 1, figsize=(9, 10))
    for i, (m, f) in enumerate(MATS):
        df = pd.read_excel(f, sheet_name=m)
        rn={}
        for c in df.columns:
            if '需求' in str(c) or 'demand' in str(c).lower(): rn[c]='demand'
            elif '日期' in str(c) or 'date' in str(c).lower(): rn[c]='date'
        df=df.rename(columns=rn); df['date']=pd.to_datetime(df['date'])
        y=df['demand'].astype(float).values
        nz = int((y[:69]>0).sum())
        ax = axes[i]
        ax.plot(df['date'], y, color='#2C3E50', lw=1.8, marker='o', ms=3)
        ax.fill_between(df['date'], 0, y, color='#5B7F9A', alpha=0.15)
        ax.set_ylabel(f'{SHORT[m]}\n需求量', fontsize=12)
        ax.set_title(f'{SHORT[m]}（非零月 {nz}/81）', fontsize=13, pad=8)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/fig3-1_demand_curves.svg', bbox_inches='tight')
    plt.savefig('outputs/figures/fig3-1_demand_curves.png', dpi=400, bbox_inches='tight')
    print('\n图3-1 已保存')

    # 3. 画图5-4 区间图 (修复后)
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    x = np.arange(1, 13)
    for i, (m, f) in enumerate(MATS):
        pred, upper, P, y_te = results[m]
        nz_te = y_te>0
        cov = np.mean((y_te>=0)&(y_te<=upper))
        covp = np.mean((y_te[nz_te]>=0)&(y_te[nz_te]<=upper[nz_te])) if nz_te.sum()>0 else np.nan
        fp = np.mean((y_te==0)&(pred>0))
        ax = axes[i]
        ax.plot(x, y_te, 'o-', color='#2C3E50', lw=2, ms=6, label='真实需求', zorder=5)
        ax.fill_between(x, 0, upper, color='#5B7F9A', alpha=0.25, label='预测区间')
        ax.plot(x, pred, '^--', color='#B86B4D', lw=1.5, ms=5, label='点预测', zorder=4)
        ax.set_title(f'{SHORT[m]}（边际覆盖{cov:.2f}·非零覆盖{covp:.2f}·假阳性{fp:.2f}）', fontsize=13, pad=8)
        ax.set_ylabel('需求量', fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
        if i==0: ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('outputs/figures/fig5-4_intervals.svg', bbox_inches='tight')
    plt.savefig('outputs/figures/fig5-4_intervals.png', dpi=400, bbox_inches='tight')
    print('图5-4 已保存')

    # 复制到桌面
    import shutil, os
    dstdir = r'D:\Users\dell\Desktop\研究生文献收集'
    pairs = [('fig3-1_demand_curves.svg','图3-1_需求量曲线.svg'),
             ('fig3-1_demand_curves.png','图3-1_需求量曲线.png'),
             ('fig5-4_intervals.svg','图5-4_代表性案例区间图.svg'),
             ('fig5-4_intervals.png','图5-4_代表性案例区间图.png')]
    for src_n, dst_n in pairs:
        shutil.copy2(os.path.join(r'D:\Users\dell\PycharmProjects\vmd-catboost\outputs\figures', src_n),
                     os.path.join(dstdir, dst_n))
    print('已复制到桌面')


if __name__ == '__main__':
    main()
