"""
Top 10 高密度物资可视化展示
选取非零月最多的10种物资，生成六张综合分析图
"""
import sqlite3, numpy as np, pandas as pd, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

# ---- 中文字体 ----
for d in [matplotlib.get_cachedir(), os.path.expanduser('~/.matplotlib')]:
    try:
        for f in os.listdir(d):
            if f.startswith('fontlist'):
                os.remove(os.path.join(d, f))
    except: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

db = sqlite3.connect(r"D:\Users\dell\PycharmProjects\bidding-ecp-data\data\ecp_data.db")
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]
MONTH_DATES = pd.date_range('2020-05-01', periods=74, freq='MS')

# ==============================================================================
# 1. 选取 TOP10 物资 (非零月最多, 且覆盖不同品类)
# ==============================================================================
cur = db.execute("""
    SELECT material_name, COUNT(DISTINCT demand_month) as nz,
           SUM(demand_quantity) as tot,
           AVG(CASE WHEN demand_quantity > 0 THEN demand_quantity END) as avg_nz
    FROM material_demand_item
    WHERE demand_month >= '202005' AND demand_month <= '202606'
    GROUP BY material_name
    HAVING nz >= 35
    ORDER BY nz DESC, tot DESC
""")
all_top = cur.fetchall()

# Manual curation: pick 10 from different categories per 研究纪要 CATEGORY_MAP
CATEGORIES = {
    '变压器': ['变压器', '变压器台成套', '调压器'],
    '开关柜/环网柜': ['开关柜', '环网柜', '环网箱', 'GIS', '隔离开关'],
    '电缆/导线': ['电缆', '导线', '光缆', '线', '绞线'],
    '避雷器/绝缘子': ['避雷器', '绝缘子', '穿墙套管'],
    '断路器/组合电器': ['断路器', '组合电器', '负荷开关', '熔断器'],
    '保护/监控': ['保护', '故障录波', '监控', '自动化', '时间同步', '测控', '安全'],
    '通信设备': ['通信', '光端机', '交换机', 'SDH', '接入'],
    '电源/蓄电池': ['电源', '蓄电池', 'UPS', '充电', '直流'],
    '杆塔/铁件': ['水泥杆', '钢管杆', '铁塔', '金具', '铁附件'],
}

def find_category(name):
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in name:
                return cat
    return '其他'

# Pick top materials from each category
MATS = []
used_cats = set()
for name, nz, tot, avg in all_top:
    cat = find_category(name)
    mats_in_cat = sum(1 for m in MATS if m['cat'] == cat)
    if mats_in_cat < 3:
        # Get full monthly data
        cur2 = db.execute("""
            SELECT demand_month, SUM(demand_quantity)
            FROM material_demand_item
            WHERE material_name = ? AND demand_month >= '202005' AND demand_month <= '202606'
            GROUP BY demand_month ORDER BY demand_month
        """, (name,))
        dmap = {r[0]: r[1] for r in cur2.fetchall()}
        vals = np.array([dmap.get(m, 0) for m in MONTHS])
        nzv = vals[vals > 0]
        cv = np.std(nzv) / np.mean(nzv) if nzv.size > 0 else -1
        MATS.append({
            'name': name, 'cn_short': name.split(',')[0],
            'nz': nz, 'tot': tot, 'cv': cv, 'cat': cat,
            'vals': vals, 'train': vals[:62], 'test': vals[62:],
        })
        used_cats.add(cat)
    if len(MATS) >= 10:
        break

print(f"Selected {len(MATS)} materials from {len(used_cats)} categories:")
for i, m in enumerate(MATS):
    train_nz = (m['train'] > 0).sum()
    test_nz = (m['test'] > 0).sum()
    print(f"  {i+1:>2d}. {m['cn_short'][:30]:<30s} [{m['cat']:<12s}] "
          f"nz={m['nz']}/74  train={train_nz}/62  test={test_nz}/12  CV={m['cv']:.2f}")

# ==============================================================================
# 2. 生成六张图
# ==============================================================================
OUT = r"D:\Users\dell\PycharmProjects\vmd-catboost\outputs\figures"
os.makedirs(OUT, exist_ok=True)

N = len(MATS)
colors = plt.cm.tab10(np.linspace(0, 1, N))

# ---- Fig 1: 10种物资完整74月需求曲线 (5行×2列) ----
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
fig.suptitle('TOP10 高密度物资月度需求量 (2020-05 ~ 2026-06)', fontsize=16, fontweight='bold')

for idx, m in enumerate(MATS):
    ax = axes[idx // 2, idx % 2]
    color = colors[idx]
    ax.plot(MONTH_DATES, m['vals'], color=color, linewidth=1.2, marker='o', markersize=2, alpha=0.8)
    ax.fill_between(MONTH_DATES, 0, m['vals'], color=color, alpha=0.08)
    # Mark train/test boundary
    ax.axvline(x=pd.Timestamp('2025-07-01'), color='black', linestyle='--', linewidth=1, alpha=0.5)

    title = f"{idx+1}. {m['cn_short'][:25]}\n[{m['cat']}] 非零={m['nz']}/74月 | CV={m['cv']:.2f} | 总量={m['tot']/1e4:.1f}万"
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylabel('需求量', fontsize=8)
    ax.tick_params(labelsize=7)

axes[-1, 0].set_xlabel('日期', fontsize=9)
axes[-1, 1].set_xlabel('日期', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'top10_material_demand.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [1/6] 需求曲线: top10_material_demand.png")

# ---- Fig 2: Non-zero density per month (heatmap) ----
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 12))

# 2a: Monthly non-zero count for EACH material (heatmap)
ax = axes2[0]
n_mat = len(MATS)
heat_data = np.zeros((n_mat, 74))
for i, m in enumerate(MATS):
    heat_data[i] = (m['vals'] > 0).astype(float)

im = ax.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
               interpolation='nearest')
ax.set_yticks(range(n_mat))
ax.set_yticklabels([m['cn_short'][:25] for m in MATS], fontsize=8)
# X ticks yearly
yr_mid = [0, 8, 20, 32, 44, 56, 68]
yr_lbl = ['2020','2021','2022','2023','2024','2025','2026']
ax.set_xticks(yr_mid)
ax.set_xticklabels(yr_lbl, fontsize=7)
ax.set_title('月度需求矩阵 (绿色=有需求, 红色=无需求)', fontsize=12, fontweight='bold')

# 2b: Aggregate monthly non-zero count
ax2 = axes2[1]
monthly_nz = np.sum(heat_data, axis=0)
ax2.fill_between(range(74), monthly_nz, color='steelblue', alpha=0.6)
ax2.plot(range(74), monthly_nz, 'o-', color='darkblue', linewidth=1.5, markersize=4)
ax2.axvline(x=62, color='red', linestyle='--', linewidth=1.5, label='Train/Test分界')
ax2.set_title(f'每月有需求的物资种数 (共{N}种)', fontsize=12, fontweight='bold')
ax2.set_ylabel('活跃物资数', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 2c: Monthly total demand (all 10 combined)
ax3 = axes2[2]
monthly_total = np.sum([m['vals'] for m in MATS], axis=0)
ax3.fill_between(range(74), monthly_total, color='darkorange', alpha=0.5)
ax3.plot(range(74), monthly_total, 'o-', color='darkred', linewidth=1.5, markersize=4)
ax3.axvline(x=62, color='red', linestyle='--', linewidth=1.5)
ax3.set_title(f'{N}种物资月度总需求量', fontsize=12, fontweight='bold')
ax3.set_ylabel('总需求量', fontsize=10)
ax3.set_xlabel('月份序号 (0=2020-05, 73=2026-06)', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(OUT, 'top10_density_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [2/6] 密度热力图: top10_density_heatmap.png")

# ---- Fig 3: Seasonal pattern per material ----
fig3, axes3 = plt.subplots(n_mat, 1, figsize=(14, 2.5 * n_mat))
for i, m in enumerate(MATS):
    ax = axes3[i] if n_mat > 1 else axes3
    month_data = defaultdict(list)
    for j, v in enumerate(m['vals']):
        if v > 0:
            month_idx = (j + 4) % 12 + 1
            month_data[month_idx].append(v)

    # Box plot per month
    positions = list(range(1, 13))
    box_data = [month_data.get(mth, [0]) for mth in range(1, 13)]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})

    for patch in bp['boxes']:
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.3)

    nz_months = sorted(month_data.keys())
    ax.set_title(f"{i+1}. {m['cn_short'][:30]} [{m['cat']}]  非零={m['nz']}/74月, CV={m['cv']:.2f}", fontsize=10, fontweight='bold')
    ax.set_ylabel('需求量', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=8)

axes3[-1].set_xlabel('月份', fontsize=9)
plt.tight_layout()
fig3.savefig(os.path.join(OUT, 'top10_seasonal_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [3/6] 季节箱线图: top10_seasonal_boxplot.png")

# ---- Fig 4: Density & CV scatter plot ----
fig4, ax4 = plt.subplots(figsize=(12, 8))

nz_list = [m['nz'] / 74 for m in MATS]
cv_list = [m['cv'] for m in MATS]
tot_list = [m['tot'] for m in MATS]
sizes = np.clip(np.log10(tot_list) * 60, 50, 400)

scatter = ax4.scatter(nz_list, cv_list, c=range(N), cmap='tab10', s=sizes,
                      alpha=0.7, edgecolors='black', linewidth=0.5)
for i, m in enumerate(MATS):
    ax4.annotate(f"{m['cn_short'][:20]}", (nz_list[i], cv_list[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax4.set_xlabel('非零月密度', fontsize=12)
ax4.set_ylabel('变异系数 (CV)', fontsize=12)
ax4.set_title('TOP10物资 密度-CV 散点图 (气泡大小=总需求量)', fontsize=14, fontweight='bold')
ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='CV=1.0 分界线')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig(os.path.join(OUT, 'top10_density_cv_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [4/6] 密度-CV散点: top10_density_cv_scatter.png")

# ---- Fig 5: Train/Test comparison ----
fig5, axes5 = plt.subplots(n_mat, 1, figsize=(14, 2.5 * n_mat))
for i, m in enumerate(MATS):
    ax = axes5[i]
    test_months = list(range(62, 74))
    actual = m['test']
    ax.plot(test_months, actual, 'ko-', linewidth=2, markersize=6, label='实际值')

    # Simple baselines
    naive_mean = np.full(12, np.mean(m['train'][m['train'] > 0]))
    naive_seas = m['train'][-12:]

    ax.plot(test_months, naive_seas, 'g^--', linewidth=1.5, markersize=4, alpha=0.6, label='Naive-Seasonal')
    ax.plot(test_months, naive_mean, 'rx--', linewidth=1.5, markersize=4, alpha=0.5, label='Naive-Mean')

    from sklearn.metrics import r2_score
    r2_seas = r2_score(actual, naive_seas) if np.std(actual) > 1e-10 else np.nan

    ax.set_title(f"{i+1}. {m['cn_short'][:30]}  test R²_seas={r2_seas:.3f}", fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

axes5[-1].set_xlabel('测试月 (0=2025-07, 11=2026-06)', fontsize=9)
plt.tight_layout()
fig5.savefig(os.path.join(OUT, 'top10_test_baselines.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [5/6] 测试集基线: top10_test_baselines.png")

# ---- Fig 6: Year-over-year pattern for top 3 ----
fig6, ax6 = plt.subplots(3, 1, figsize=(16, 10))
top3 = sorted(MATS, key=lambda x: -(x['nz'] / (x['cv'] + 0.1)))[:3]
for i, m in enumerate(top3):
    ax = ax6[i]
    for yr_idx, yr in enumerate(range(2020, 2027)):
        start = max(0, (yr - 2020) * 12 + (5 - 1))
        end = start + 12
        if end > 74:
            end = 74
        yr_vals = m['vals'][start:end]
        yr_months_s = [f"{mth}月" for mth in range(1, 13)][:len(yr_vals)]
        alpha = 0.3 + yr_idx * 0.12
        ax.plot(range(len(yr_vals)), yr_vals, 'o-', linewidth=1.5, markersize=4,
                alpha=min(1, alpha), label=str(yr))

    ax.set_title(f"{i+1}. {m['cn_short'][:30]} — 年度对比", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('需求量', fontsize=9)

ax6[-1].set_xlabel('月份序号 (1-12)', fontsize=10)
plt.tight_layout()
fig6.savefig(os.path.join(OUT, 'top10_yoy_pattern.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [6/6] 年度对比: top10_yoy_pattern.png")

# ---- Metrics summary table save to text ----
summary_path = os.path.join(OUT, 'top10_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"TOP10 高密度物资汇总 (2020-05 ~ 2026-06)\n")
    f.write(f"{'='*80}\n")
    f.write(f"{'物资':<35s} {'品类':<12s} {'非零月':>6s} {'训练':>5s} {'测试':>5s} {'CV':>6s} {'总量(万)':>10s}\n")
    f.write(f"{'-'*80}\n")
    for m in MATS:
        train_nz = (m['train'] > 0).sum()
        test_nz = (m['test'] > 0).sum()
        f.write(f"{m['cn_short'][:33]:<35s} {m['cat']:<12s} {m['nz']:>5d}/74 {train_nz:>3d}/62 {test_nz:>3d}/12 "
                f"{m['cv']:>5.2f} {m['tot']/1e4:>10.1f}\n")
    f.write(f"\n平均密度: {np.mean([m['nz']/74 for m in MATS]):.1%}\n")
    f.write(f"平均CV: {np.mean([m['cv'] for m in MATS]):.2f}\n")
    f.write(f"覆盖品类数: {len(set(m['cat'] for m in MATS))} 个\n")

print(f"\n  Summary: {summary_path}")
print(f"\nDone! 6 charts in {OUT}")

db.close()
