"""JiBei grey feature validation - direct DB query + AB test."""
import sqlite3, os, numpy as np, warnings; warnings.filterwarnings('ignore')
import pandas as pd, openpyxl
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import r2_score
from scipy.special import gamma as _gamma_fn

DB = r'D:\Users\dell\PycharmProjects\bidding-ecp-data\data_jibei\ecp_data.db'

def all_months(s='201911', e='202607'):
    m = []; y, mo = int(s[:4]), int(s[4:])
    ye, me = int(e[:4]), int(e[4:])
    while y<ye or (y==ye and mo<=me):
        m.append(f'{y}{mo:02d}'); mo+=1
        if mo>12: mo=1; y+=1
    return m
ALL = all_months()

# === Step 1: Get 5 sparse materials directly from DB ===
conn = sqlite3.connect(DB); c = conn.cursor()
# Pick 5 diverse materials from NZ=6-12 directly via LIKE patterns (DB names may differ)
PATTERNS = [
    '%电缆终端%', '%柱式复合绝缘子%', '%设备线夹-螺栓%', '%针式%绝缘子%', '%智能变电站%'
]
placeholders = ','.join(['?']*len(PATTERNS))
c.execute(f'''SELECT material_name, demand_month, demand_quantity, notice_count
    FROM material_demand_total WHERE ({' OR '.join(['material_name LIKE ?']*len(PATTERNS))})
    ORDER BY material_name, demand_month''', PATTERNS)
rows = c.fetchall()
# Get unique material names
mat_names = sorted(set(r[0] for r in rows))
print(f'Selected {len(mat_names)} sparse materials:')
for i, n in enumerate(mat_names, 1): print(f'  {i}. {n}')

# Build per-material monthly data
materials = {}
for name in mat_names:
    materials[name] = {'qty': {}, 'nc': {}}
for name, dm, q, nc in rows:
    materials[name]['qty'][dm] = (q or 0)
    materials[name]['nc'][dm] = (nc or 0)

# Batch features
c.execute('''SELECT notice_publish_time, title FROM bid_notices
    WHERE category=\"material\" AND doctype=\"doci-bid\" AND title NOT LIKE \"%变更%\"''')
bids = c.fetchall(); conn.close()
mb = defaultdict(lambda: {'transformer':0, 'total':0, 'uhv':0})
for pt, title in bids:
    ym = pt[:4]+pt[5:7]; mb[ym]['total'] += 1
    if any(k in title for k in ['输变电','变电设备','变压器']): mb[ym]['transformer'] += 1
    if '特高压' in title: mb[ym]['uhv'] += 1

# === Step 2: Build Excel (for reuse) ===
wb = openpyxl.Workbook(); wb.remove(wb.active)
HEADERS = ['日期', '需求量', '项目数量', 'transformer_bids', 'monthly_bid_count', 'uhv_bids']
for mat_name in mat_names:
    qd = materials[mat_name]['qty']; nd = materials[mat_name]['nc']
    ws = wb.create_sheet(mat_name[:31])
    for ci, h in enumerate(HEADERS, 1): ws.cell(row=1, column=ci, value=h)
    for ri, dm in enumerate(ALL, 2):
        bf = mb.get(dm, {}); y, m = int(dm[:4]), int(dm[4:])
        ws.cell(row=ri, column=1, value=f'{y}-{m:02d}-01')
        ws.cell(row=ri, column=2, value=qd.get(dm, 0))
        ws.cell(row=ri, column=3, value=nd.get(dm, 0))
        ws.cell(row=ri, column=4, value=bf.get('transformer', 0))
        ws.cell(row=ri, column=5, value=bf.get('total', 0))
        ws.cell(row=ri, column=6, value=bf.get('uhv', 0))
os.makedirs('inputs', exist_ok=True)
out = 'inputs/_jibei_grey_5.xlsx'; wb.save(out)

# === Step 3: Grey feature functions ===
def _gm11_fit(x0):
    n=len(x0)
    if n<4: return -0.01, np.mean(x0) if n>0 else 0
    x1=np.cumsum(np.maximum(x0,0))
    z1=0.5*(x1[1:]+x1[:-1]); B=np.column_stack([-z1,np.ones(n-1)]); Y=x0[1:]
    try: params=np.linalg.lstsq(B,Y,rcond=None)[0]; return params[0],params[1]
    except: return -0.01, np.mean(x0)

def _gm11_fitted(x0,a,b):
    n=len(x0)
    if abs(a)<1e-10: return np.full(n,np.mean(np.maximum(x0,0)))
    x1h=np.zeros(n); x1h[0]=max(x0[0],0)
    for k in range(1,n): x1h[k]=(x1h[0]-b/a)*np.exp(-a*k)+b/a
    f=np.zeros(n); f[0]=x1h[0]
    for k in range(1,n): f[k]=max(x1h[k]-x1h[k-1],0)
    return f

def _gm11_predict_next(ln,a,b,np_):
    if abs(a)<1e-10: return np.full(np_,ln)
    p=np.zeros(np_); xp=ln
    for k in range(1,np_+1):
        xk=(ln-b/a)*np.exp(-a*k)+b/a; p[k-1]=max(xk-xp,0); xp=xk
    return p

def _fractional_ago(x0,r):
    n=len(x0); xr=np.zeros(n)
    for k in range(n):
        for j in range(k+1):
            c=_gamma_fn(k-j+r)/(_gamma_fn(k-j+1)*_gamma_fn(r)) if r>0 else 1.0
            xr[k]+=c*x0[j]
    return xr

def build_grey(d,tl):
    n=len(d); gf=np.zeros(n); gr=np.zeros(n); ga=np.zeros(n)
    for t in range(tl):
        s=max(0,t-11); sg=d[s:t+1]
        if len(sg)>=4 and sg.sum()>0:
            a,b=_gm11_fit(sg); f=_gm11_fitted(sg,a,b)
            gf[t]=f[-1]; gr[t]=d[t]-f[-1]; ga[t]=a
        else: gf[t]=np.mean(sg) if len(sg)>0 else 0
    te=d[tl-12:tl]
    if len(te)>=4 and te.sum()>0:
        ah,bh=_gm11_fit(te); p=_gm11_predict_next(te[-1],ah,bh,n-tl)
        gf[tl:]=p; ga[tl:]=ah
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
    return np.column_stack([gf[:tl],gr[:tl],ga[:tl],fa[:tl]]), np.column_stack([gf[tl:],gr[tl:],ga[tl:],fa[tl:]])

# === Step 4: Feature engineering ===
def make_features(df, grey_on):
    dr = df['需求量'].values[-81:].astype(np.float64)
    tl=69; N_TEST=12
    ext = np.column_stack([df[c].values[-81:].astype(np.float64) if c in df.columns else np.zeros(81)
        for c in ['项目数量','transformer_bids','monthly_bid_count','uhv_bids']])
    s=dr[:tl]
    l1=np.zeros(tl); l1[1:]=s[:-1]
    l12=np.zeros(tl); l12[12:]=s[:-12]
    r3=np.array([np.mean(s[max(0,i-3):i]) for i in range(tl)])
    iz1=(s==0).astype(float)
    iz12=np.zeros(tl)
    for i in range(12,tl): iz12[i]=(s[i-12]==0)
    l2=np.zeros(tl); l2[2:]=s[:-2]; l3=np.zeros(tl); l3[3:]=s[:-3]; l6=np.zeros(tl); l6[6:]=s[:-6]
    r6=np.array([np.mean(s[max(0,i-6):i]) if i>0 else 0 for i in range(tl)])
    r12=np.array([np.mean(s[max(0,i-12):i]) if i>0 else 0 for i in range(tl)])
    r3s=np.array([np.std(s[max(0,i-3):i]) if i>1 else 0 for i in range(tl)])
    yy=np.zeros(tl)
    for i in range(13,tl): yy[i]=s[i-1]-s[i-13]
    gp=np.zeros(tl); le=-999
    for i in range(tl):
        if s[i]>0: le=i
        gp[i]=i-le if le>=0 else tl
    e6=np.array([np.sum(s[max(0,i-5):i+1]>0) for i in range(tl)])
    e12=np.array([np.sum(s[max(0,i-11):i+1]>0) for i in range(tl)])
    cu12=np.array([np.sum(s[max(0,i-11):i+1]) for i in range(tl)])
    mf=np.zeros(tl); cm=np.array([(4+i)%12+1 for i in range(tl)])
    for i in range(12,tl):
        past=[j for j in range(i) if cm[j]==cm[i]]
        if past: mf[i]=np.mean(s[past]>0)
    Xtr=np.column_stack([ext[:tl],l1,l12,r3,iz1,iz12,l2,l3,l6,r6,r12,r3s,yy,gp,e6,e12,cu12,mf])
    # test
    fd=dr
    l1t=np.zeros(N_TEST); l1t[0]=s[-1]
    for i in range(1,N_TEST): l1t[i]=fd[tl+i-1]
    l12t=np.zeros(N_TEST)
    for i in range(N_TEST):
        src=tl+i-12; l12t[i]=fd[src] if 0<=src<tl else (fd[src] if src>=tl else 0)
    r3t=np.array([np.mean(fd[max(0,tl+i-3):tl+i]) for i in range(N_TEST)])
    iz1t=np.zeros(N_TEST); iz1t[0]=float(s[-1]==0)
    for i in range(1,N_TEST): iz1t[i]=float(fd[tl+i-1]==0)
    iz12t=np.zeros(N_TEST)
    for i in range(N_TEST):
        src=tl+i-12; iz12t[i]=float(fd[src]==0) if 0<=src else 0
    l2t=np.zeros(N_TEST);l3t=np.zeros(N_TEST);l6t=np.zeros(N_TEST)
    for i in range(N_TEST):
        for lv,arr in [(2,l2t),(3,l3t),(6,l6t)]:
            src=tl+i-lv; arr[i]=fd[src] if 0<=src<tl else (fd[src] if src>=tl else 0)
    r6t=np.array([np.mean(fd[max(0,tl+i-6):tl+i]) for i in range(N_TEST)])
    r12t=np.array([np.mean(fd[max(0,tl+i-12):tl+i]) for i in range(N_TEST)])
    r3st=np.array([np.std(fd[max(0,tl+i-3):tl+i]) for i in range(N_TEST)])
    yyt=np.zeros(N_TEST)
    for i in range(N_TEST):
        s1=tl+i-1; s13=tl+i-13
        yyt[i]=fd[s1]-(fd[s13] if 0<=s13<tl else 0)
    gpt=np.zeros(N_TEST); le2=-999
    for i in range(tl):
        if fd[i]>0: le2=i
    for i in range(N_TEST):
        if fd[tl+i]>0: le2=tl+i
        gpt[i]=(tl+i)-le2 if le2>=0 else tl+N_TEST
    e6t=np.array([np.sum(fd[max(0,tl+i-5):tl+i+1]>0) for i in range(N_TEST)])
    e12t=np.array([np.sum(fd[max(0,tl+i-11):tl+i+1]>0) for i in range(N_TEST)])
    cu12t=np.array([np.sum(fd[max(0,tl+i-11):tl+i+1]) for i in range(N_TEST)])
    mft=np.zeros(N_TEST); cma=np.array([(4+i)%12+1 for i in range(tl+N_TEST)])
    for i in range(N_TEST):
        past=[j for j in range(tl+i) if cma[j]==cma[tl+i]]
        if past: mft[i]=np.mean(fd[past]>0)
    Xte=np.column_stack([ext[tl:tl+N_TEST],l1t,l12t,r3t,iz1t,iz12t,l2t,l3t,l6t,r6t,r12t,r3st,yyt,gpt,e6t,e12t,cu12t,mft])
    if grey_on:
        gtr,gte=build_grey(dr,tl); Xtr=np.column_stack([Xtr,gtr]); Xte=np.column_stack([Xte,gte])
    Xtr=np.nan_to_num(Xtr,nan=0,posinf=0,neginf=0); Xte=np.nan_to_num(Xte,nan=0,posinf=0,neginf=0)
    sc=MinMaxScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    return Xtr, dr[:tl], Xte, dr[tl:tl+N_TEST]

# === Step 5: Predict (Ridge for ultra-sparse) ===
def predict(X_tr, y_tr, X_te):
    nz = y_tr > 0; nnz = nz.sum()
    if nnz < 6:
        reg = Ridge(alpha=10.0); reg.fit(X_tr[nz], y_tr[nz])
    else:
        reg = ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,0.95,1.0], cv=min(3,nnz-1), max_iter=5000, random_state=42)
        reg.fit(X_tr[nz], y_tr[nz])
    return np.maximum(reg.predict(X_te), 0)

def ev(yt, yp): return {'R2': r2_score(yt, yp)}

# === Step 6: Run AB ===
data = pd.read_excel(out, sheet_name=None)
print()
print('=== 冀北极端稀疏物资 — 灰色特征 AB 对比 ===')
print()
for mat_name in data:
    df=data[mat_name]; d=df['需求量'].values[-81:].astype(float)
    nz_train = int(sum(d[:69]>0)); nz_total = int(sum(d>0))
    if nz_train < 5: continue  # too few samples for meaningful evaluation
    nzv = d[:69][d[:69]>0]
    cv = np.std(nzv)/np.mean(nzv) if len(nzv)>1 and np.mean(nzv)>0 else 0
    try:
        Xtr,ytr,Xte,yte = make_features(df, False); yp_off=predict(Xtr,ytr,Xte); r_off=ev(yte,yp_off)['R2']
        Xtr2,ytr2,Xte2,yte2 = make_features(df, True); yp_on=predict(Xtr2,ytr2,Xte2); r_on=ev(yte2,yp_on)['R2']
    except: continue
    d_r = r_on - r_off
    v = 'GREY SAVED' if d_r > 0.2 else ('HELPS' if d_r > 0.05 else 'NEUTRAL')
    print(f'{mat_name[:30]:30s} NZ_tr={nz_train:2d}(tot={nz_total}) CV={cv:.2f}  OFF={r_off:+.4f}  ON={r_on:+.4f}  D={d_r:+.4f}  [{v}]')
