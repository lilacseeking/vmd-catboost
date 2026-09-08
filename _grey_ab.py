"""Grey feature AB test on JiBei sparse materials (NZ=4)."""
import sys, io, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.special import gamma as _gamma_fn

# === Grey feature functions (from main.py) ===
def _gm11_fit(x0):
    n = len(x0)
    if n < 4: return -0.01, np.mean(x0) if n > 0 else 0
    x1 = np.cumsum(np.maximum(x0, 0))
    z1 = 0.5 * (x1[1:] + x1[:-1]); B = np.column_stack([-z1, np.ones(n-1)]); Y = x0[1:]
    try: params = np.linalg.lstsq(B, Y, rcond=None)[0]; return params[0], params[1]
    except: return -0.01, np.mean(x0)

def _gm11_fitted(x0, a, b):
    n = len(x0)
    if abs(a) < 1e-10: return np.full(n, np.mean(np.maximum(x0, 0)))
    x1_hat = np.zeros(n); x1_hat[0] = max(x0[0], 0)
    for k in range(1,n): x1_hat[k] = (x1_hat[0]-b/a)*np.exp(-a*k)+b/a
    fitted = np.zeros(n); fitted[0]=x1_hat[0]
    for k in range(1,n): fitted[k]=max(x1_hat[k]-x1_hat[k-1],0)
    return fitted

def _gm11_predict_next(last_n, a, b, n_pred):
    if abs(a) < 1e-10: return np.full(n_pred, last_n)
    preds=np.zeros(n_pred); x1_prev=last_n
    for k in range(1,n_pred+1):
        x1_k=(last_n-b/a)*np.exp(-a*k)+b/a; preds[k-1]=max(x1_k-x1_prev,0); x1_prev=x1_k
    return preds

def _fractional_ago(x0, r):
    n=len(x0); xr=np.zeros(n)
    for k in range(n):
        for j in range(k+1):
            coeff=_gamma_fn(k-j+r)/(_gamma_fn(k-j+1)*_gamma_fn(r)) if r>0 else 1.0
            xr[k]+=coeff*x0[j]
    return xr

def build_grey(d, tl):
    n=len(d); gm_f=np.zeros(n); gm_r=np.zeros(n); ga=np.zeros(n)
    for t in range(tl):
        start=max(0,t-11); seg=d[start:t+1]
        if len(seg)>=4 and seg.sum()>0:
            a,b=_gm11_fit(seg); f=_gm11_fitted(seg,a,b)
            gm_f[t]=f[-1]; gm_r[t]=d[t]-f[-1]; ga[t]=a
        else: gm_f[t]=np.mean(seg) if len(seg)>0 else 0
    train_end=d[tl-12:tl]
    if len(train_end)>=4 and train_end.sum()>0:
        ah,bh=_gm11_fit(train_end); preds=_gm11_predict_next(train_end[-1],ah,bh,n-tl)
        gm_f[tl:]=preds; ga[tl:]=ah
    br=0.7; yp=np.maximum(d[:tl],1e-6)
    if tl>=5:
        bs=float('inf')
        for r in [0.3,0.5,0.7,0.9,1.0]:
            try:
                fs=_fractional_ago(yp[:min(20,tl)],r); sm=np.std(np.diff(fs))
                if sm<bs: bs=sm; br=r
            except: continue
    fa=np.zeros(n)
    try:
        ft=_fractional_ago(yp,br)
        sc=np.mean(yp[yp>0])/max(np.mean(ft[ft>0]),1e-6) if (yp>0).any() else 1
        fa[:tl]=ft*sc; fa[tl:]=fa[tl-1]
    except: pass
    gr_tr=np.column_stack([gm_f[:tl],gm_r[:tl],ga[:tl],fa[:tl]])
    gr_te=np.column_stack([gm_f[tl:],gm_r[tl:],ga[tl:],fa[tl:]])
    return gr_tr, gr_te

def make_features(df, grey_on):
    demand_raw = df['demand'].values[-81:].astype(np.float64)
    tl = 69; N_TEST = 12
    ext_cols = []
    for c in ['project_count','transformer_bids','monthly_bid_count','uhv_bids']:
        ext_cols.append(df[c].values[-81:].astype(np.float64) if c in df.columns else np.zeros(81))
    ext = np.column_stack(ext_cols)
    seq = demand_raw[:tl]
    lag1 = np.zeros(tl); lag1[1:]=seq[:-1]
    lag12 = np.zeros(tl); lag12[12:]=seq[:-12]
    roll3 = np.array([np.mean(seq[max(0,i-3):i]) for i in range(tl)])
    is_z1 = (seq==0).astype(float)
    is_z12 = np.zeros(tl)
    for i in range(12,tl): is_z12[i]=(seq[i-12]==0)
    lag2=np.zeros(tl); lag2[2:]=seq[:-2]
    lag3=np.zeros(tl); lag3[3:]=seq[:-3]
    lag6=np.zeros(tl); lag6[6:]=seq[:-6]
    roll6=np.array([np.mean(seq[max(0,i-6):i]) if i>0 else 0 for i in range(tl)])
    roll12=np.array([np.mean(seq[max(0,i-12):i]) if i>0 else 0 for i in range(tl)])
    roll3s=np.array([np.std(seq[max(0,i-3):i]) if i>1 else 0 for i in range(tl)])
    yoy=np.zeros(tl)
    for i in range(13,tl): yoy[i]=seq[i-1]-seq[i-13]
    gap=np.zeros(tl); le=-999
    for i in range(tl):
        if seq[i]>0: le=i
        gap[i]=i-le if le>=0 else tl
    evt6=np.array([np.sum(seq[max(0,i-5):i+1]>0) for i in range(tl)])
    evt12=np.array([np.sum(seq[max(0,i-11):i+1]>0) for i in range(tl)])
    cum12=np.array([np.sum(seq[max(0,i-11):i+1]) for i in range(tl)])
    mf=np.zeros(tl); cm=np.array([(4+i)%12+1 for i in range(tl)])
    for i in range(12,tl):
        past=[j for j in range(i) if cm[j]==cm[i]]
        if past: mf[i]=np.mean(seq[past]>0)
    X_tr = np.column_stack([ext[:tl], lag1,lag12,roll3,is_z1,is_z12,
        lag2,lag3,lag6,roll6,roll12,roll3s,yoy, gap,evt6,evt12,cum12,mf])
    # test features
    full_d = demand_raw
    lag1_te=np.zeros(N_TEST); lag1_te[0]=seq[-1]
    for i in range(1,N_TEST): lag1_te[i]=full_d[tl+i-1]
    lag12_te=np.zeros(N_TEST)
    for i in range(N_TEST):
        src=tl+i-12; lag12_te[i]=full_d[src] if 0<=src<tl else (full_d[src] if src>=tl else 0)
    roll3_te=np.array([np.mean(full_d[max(0,tl+i-3):tl+i]) for i in range(N_TEST)])
    isz1_te=np.zeros(N_TEST); isz1_te[0]=float(seq[-1]==0)
    for i in range(1,N_TEST): isz1_te[i]=float(full_d[tl+i-1]==0)
    isz12_te=np.zeros(N_TEST)
    for i in range(N_TEST):
        src=tl+i-12; isz12_te[i]=float(full_d[src]==0) if 0<=src else 0
    lag2_te=np.zeros(N_TEST); lag3_te=np.zeros(N_TEST); lag6_te=np.zeros(N_TEST)
    for i in range(N_TEST):
        for lv, arr in [(2,lag2_te),(3,lag3_te),(6,lag6_te)]:
            src=tl+i-lv; arr[i]=full_d[src] if src>=0 and src<tl else (full_d[src] if src>=tl else 0)
    roll6_te=np.array([np.mean(full_d[max(0,tl+i-6):tl+i]) for i in range(N_TEST)])
    roll12_te=np.array([np.mean(full_d[max(0,tl+i-12):tl+i]) for i in range(N_TEST)])
    roll3s_te=np.array([np.std(full_d[max(0,tl+i-3):tl+i]) for i in range(N_TEST)])
    yoy_te=np.zeros(N_TEST)
    for i in range(N_TEST):
        s1=tl+i-1; s13=tl+i-13
        yoy_te[i]=full_d[s1] - (full_d[s13] if 0<=s13<tl else 0)
    gap_te=np.zeros(N_TEST); le2=-999
    for i in range(tl):
        if full_d[i]>0: le2=i
    for i in range(N_TEST):
        if full_d[tl+i]>0: le2=tl+i
        gap_te[i]=(tl+i)-le2 if le2>=0 else tl+N_TEST
    evt6_te=np.array([np.sum(full_d[max(0,tl+i-5):tl+i+1]>0) for i in range(N_TEST)])
    evt12_te=np.array([np.sum(full_d[max(0,tl+i-11):tl+i+1]>0) for i in range(N_TEST)])
    cum12_te=np.array([np.sum(full_d[max(0,tl+i-11):tl+i+1]) for i in range(N_TEST)])
    mf_te=np.zeros(N_TEST); cm_all=np.array([(4+i)%12+1 for i in range(tl+N_TEST)])
    for i in range(N_TEST):
        past=[j for j in range(tl+i) if cm_all[j]==cm_all[tl+i]]
        if past: mf_te[i]=np.mean(full_d[past]>0)
    X_te = np.column_stack([ext[tl:tl+N_TEST], lag1_te,lag12_te,roll3_te,isz1_te,isz12_te,
        lag2_te,lag3_te,lag6_te,roll6_te,roll12_te,roll3s_te,yoy_te,
        gap_te,evt6_te,evt12_te,cum12_te,mf_te])
    if grey_on:
        gr_tr, gr_te = build_grey(demand_raw, tl)
        X_tr = np.column_stack([X_tr, gr_tr])
        X_te = np.column_stack([X_te, gr_te])
    X_tr=np.nan_to_num(X_tr,nan=0,posinf=0,neginf=0)
    X_te=np.nan_to_num(X_te,nan=0,posinf=0,neginf=0)
    sc=MinMaxScaler(); X_tr=sc.fit_transform(X_tr); X_te=sc.transform(X_te)
    return X_tr, demand_raw[:tl], X_te, demand_raw[tl:tl+N_TEST]

def predict(X_tr, y_tr, X_te):
    nz = y_tr > 0
    nnz = nz.sum()
    if nnz < 6:
        # Extreme sparse: Ridge(alpha=10) - strong L2, always stable
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=10.0)
        reg.fit(X_tr[nz], y_tr[nz])
        return np.maximum(reg.predict(X_te), 0)
    en = ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,0.95,1.0], cv=min(3,nnz-1), max_iter=5000, random_state=42)
    en.fit(X_tr[nz], y_tr[nz])
    return np.maximum(en.predict(X_te), 0)

def ev(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred)}

data = pd.read_excel('inputs/data_jibei_grey_test.xlsx', sheet_name=None)

print('冀北极端稀疏物资 — 灰色特征必要性验证')
print('5种物资 NZ=4/81, 训练69月/测试12月')
print()
for mat_name in data:
    df=data[mat_name]
    col_map = {}
    for c in df.columns:
        if '需求' in c: col_map[c]='demand'
        elif '项目' in c: col_map[c]='project_count'
        elif 'transformer' in c: col_map[c]='transformer_bids'
        elif 'monthly' in c or '公告' in c: col_map[c]='monthly_bid_count'
        elif 'uhv' in c or '特高压' in c: col_map[c]='uhv_bids'
    df = df.rename(columns=col_map)
    if 'demand' not in df.columns: continue
    d_all = df['demand'].values[-81:].astype(np.float64)
    nz = int(sum(d_all[:69]>0))
    nzv = d_all[:69][d_all[:69]>0]
    cv = np.std(nzv)/np.mean(nzv) if len(nzv)>0 and np.mean(nzv)>0 else 0
    Xtr,ytr,Xte,yte = make_features(df, False); yp_off=predict(Xtr,ytr,Xte); r_off=ev(yte,yp_off)['R2']
    Xtr2,ytr2,Xte2,yte2 = make_features(df, True); yp_on=predict(Xtr2,ytr2,Xte2); r_on=ev(yte2,yp_on)['R2']
    d=r_on-r_off
    v='GREY SAVED (R2 revives from death)' if d>0.2 else ('GREY HELPS' if d>0.05 else 'NO EFFECT')
    print(f'{mat_name[:35]:35s} NZ={nz} CV={cv:.2f}  OFF={r_off:+.4f}  ON={r_on:+.4f}  D={d:+.4f}  [{v}]')
