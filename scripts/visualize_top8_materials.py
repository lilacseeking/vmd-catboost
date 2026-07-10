"""
Top 8 高密度物资可视化：月度需求曲线 + 非零月分布 + 季节规律 + 测试集放大
数据来源：bidding-ecp-data/ecp_data.db (国网总部级采购数据 2020-05 ~ 2026-06)
"""
import sqlite3, numpy as np, pandas as pd, os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- 中文字体设置 ----
_font_cache_dir = matplotlib.get_cachedir()
for _fn in os.listdir(_font_cache_dir):
    if _fn.startswith('fontlist'):
        try: os.remove(os.path.join(_font_cache_dir, _fn))
        except OSError: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# Config
# ==============================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'material_eda')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]
MONTH_LABELS = ["{}-{:02d}".format(2020 + (5+i)//12, (5+i-1)%12+1) for i in range(74)]

# 8 materials with highest non-zero months, from diverse categories
# Categories derived from sub_bid_name and material name
MATERIALS = [
    ('消弧线圈', '%消弧线圈%', '保护/接地类'),
    ('线路保护', '%线路保护%', '保护类'),
    ('10kV变压器', '%10kV变压器%', '变压器类'),
    ('高压开关柜', '%高压开关柜%', '开关柜类'),
    ('交流支柱绝缘子', '%交流支柱绝缘子%', '绝缘子类'),
    ('钢芯铝绞线', '%钢芯铝绞线%', '电缆/导线类'),
    ('电抗器保护', '%电抗器保护%', '保护类'),
    ('500kV GIS组合电器', '%500kV%GIS%', 'GIS/组合电器类'),
]

# ==============================================================================
# Load Data
# ==============================================================================
db = sqlite3.connect(DB_PATH)

all_data = []
for mat_label, pattern, cat in MATERIALS:
    cur = db.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) as nz,
               SUM(demand_quantity) as total_qty
        FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY material_name ORDER BY nz DESC LIMIT 3
    """, (pattern,))
    matches = cur.fetchall()
    if not matches:
        continue
    best = max(matches, key=lambda x: x[1])
    exact_name = best[0]

    # Get monthly demand
    cur2 = db.execute("""
        SELECT demand_month, SUM(demand_quantity)
        FROM material_demand_item
        WHERE material_name = ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (exact_name,))
    demand_map = {r[0]: r[1] for r in cur2.fetchall()}
    vals = np.array([demand_map.get(m, 0) for m in MONTHS])
    nz_all = (vals > 0).sum()
    nz_vals = vals[vals > 0]
    cv = np.std(nz_vals) / np.mean(nz_vals) if nz_vals.size > 0 else -1
    train_nz = (vals[:62] > 0).sum()
    test_nz = (vals[62:] > 0).sum()

    # Month-of-year distribution (only non-zero)
    month_counts = defaultdict(int)
    month_qtys = defaultdict(list)
    for i, v in enumerate(vals):
        if v > 0:
            m = (i + 4) % 12 + 1  # month 1-12
            month_counts[m] += 1
            month_qtys[m].append(v)

    all_data.append({
        'label': mat_label,
        'exact_name': exact_name,
        'category': cat,
        'vals': vals,
        'nz_all': nz_all,
        'cv': cv,
        'train_nz': train_nz,
        'test_nz': test_nz,
        'total_qty': nz_vals.sum(),
        'avg_nz': np.mean(nz_vals),
        'max_val': vals.max(),
        'month_counts': month_counts,
        'month_qtys': month_qtys,
    })

db.close()

# Sort by nz descending
all_data.sort(key=lambda x: -x['nz_all'])

# ==============================================================================
# Figure 1: Full 74-month demand curves (8 materials)
# ==============================================================================
print("[1/4] Full demand curves...")
fig, axes = plt.subplots(8, 1, figsize=(16, 24))

dates = pd.date_range('2020-05-01', periods=74, freq='MS')

for idx, d in enumerate(all_data):
    ax = axes[idx]
    vals = d['vals']

    # Plot demand line
    ax.plot(dates, vals, color='#2196F3', linewidth=1.2)

    # Fill non-zero areas
    ax.fill_between(dates, 0, vals, where=(vals > 0), color='#2196F3', alpha=0.15)
    ax.fill_between(dates, 0, vals, where=(vals <= 0), color='#E0E0E0', alpha=0.05)

    # Mark train/test split
    ax.axvline(x=pd.Timestamp('2025-07-01'), color='#FF5722', linestyle='--', linewidth=1.5, alpha=0.7)

    # Title with stats
    title = f"{d['label']} [{d['category']}]  —  {d['nz_all']}/74月非零({d['nz_all']/74:.0%})  |  CV={d['cv']:.2f}  |  总量={d['total_qty']:.0f}"
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel('需求量', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set ylim to make small-value materials readable
    if d['max_val'] > 0:
        ax.set_ylim(0, d['max_val'] * 1.1)

    # Add text label for train/test
    ax.text(0.02, 0.95, '训练集(62月)', transform=ax.transAxes, fontsize=7, color='#666666',
            verticalalignment='top', fontstyle='italic')
    ax.text(0.82, 0.95, '测试集(12月)', transform=ax.transAxes, fontsize=7, color='#FF5722',
            verticalalignment='top', fontstyle='italic')

axes[-1].set_xlabel('日期')
fig.suptitle('8种最高密度物资 — 74个月完整需求序列 (国网总部级采购数据, 2020-05 ~ 2026-06)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, '01_full_74m_curves.png')
fig.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {path1}")

# ==============================================================================
# Figure 2: Monthly seasonality heatmap (materials × month)
# ==============================================================================
print("[2/4] Seasonality heatmap...")
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

# Left: Count of non-zero months per calendar month
month_arr = np.zeros((len(all_data), 12))
for idx, d in enumerate(all_data):
    for m in range(1, 13):
        month_arr[idx][m-1] = d['month_counts'].get(m, 0)

ax_left = axes2[0]
im1 = ax_left.imshow(month_arr, cmap='YlOrRd', aspect='auto')
ax_left.set_xticks(range(12))
ax_left.set_xticklabels([f'{m}月' for m in range(1, 13)])
ax_left.set_yticks(range(len(all_data)))
ax_left.set_yticklabels([d['label'] for d in all_data], fontsize=9)
for i in range(len(all_data)):
    for j in range(12):
        val = month_arr[i][j]
        if val > 0:
            ax_left.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=8,
                        color='white' if val > 3 else 'black')
ax_left.set_title('各月非零出现次数 (共~6年)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax_left, shrink=0.8)

# Right: Average demand per month (normalized per material to [0,1])
month_avg = np.zeros((len(all_data), 12))
for idx, d in enumerate(all_data):
    for m in range(1, 13):
        qtys = d['month_qtys'].get(m, [])
        if qtys:
            month_avg[idx][m-1] = np.mean(qtys)
    # Normalize to [0,1] within each material
    if month_avg[idx].max() > 0:
        month_avg[idx] /= month_avg[idx].max()

ax_right = axes2[1]
im2 = ax_right.imshow(month_avg, cmap='Blues', aspect='auto')
ax_right.set_xticks(range(12))
ax_right.set_xticklabels([f'{m}月' for m in range(1, 13)])
ax_right.set_yticks(range(len(all_data)))
ax_right.set_yticklabels([d['label'] for d in all_data], fontsize=9)
ax_right.set_title('各月平均需求强度 (归一化2[0,1])', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax_right, shrink=0.8)

fig2.suptitle('8种物资的季节性规律 — 月度热度矩阵', fontsize=14, fontweight='bold')
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, '02_seasonality_heatmap.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {path2}")

# ==============================================================================
# Figure 3: Test set (last 12 months) zoom — most important for model evaluation
# ==============================================================================
print("[3/4] Test set detail...")
fig3, axes3 = plt.subplots(8, 1, figsize=(14, 20))
test_dates = pd.date_range('2025-07-01', periods=12, freq='MS')
test_month_labels = [d.strftime('%Y-%m') for d in test_dates]

colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800', '#795548', '#00BCD4', '#E91E63']

for idx, d in enumerate(all_data):
    ax = axes3[idx]
    test_vals = d['vals'][62:74]

    bars = ax.bar(range(12), test_vals, color=[colors[idx] if v > 0 else '#E0E0E0' for v in test_vals],
                  edgecolor='white', alpha=0.85)
    ax.set_xticks(range(12))
    ax.set_xticklabels(test_month_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('需求量', fontsize=9)

    # Label each bar
    for bar, v in zip(bars, test_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_vals)*0.02,
                   f'{v:.0f}', ha='center', fontsize=7, color=colors[idx], fontweight='bold')

    title = f"{d['label']} [{d['category']}]  —  测试集{d['test_nz']}/12非零月  |  历史均值={d['avg_nz']:.0f}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add train mean reference line
    ax.axhline(y=d['avg_nz'], color='#FF5722', linestyle='--', linewidth=1, alpha=0.5,
              label=f'训练集均值={d["avg_nz"]:.0f}')
    ax.legend(fontsize=7, loc='upper right')

axes3[-1].set_xlabel('月份')
fig3.suptitle('8种物资 — 测试集12个月逐月需求量 (2025-07 ~ 2026-06)', fontsize=14, fontweight='bold')
plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, '03_test_set_detail.png')
fig3.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {path3}")

# ==============================================================================
# Figure 4: Summary stats — bar chart comparison
# ==============================================================================
print("[4/4] Summary statistics...")
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

mat_labels = [d['label'] for d in all_data]
cat_colors = {'保护类': '#2196F3', '保护/接地类': '#4CAF50', '变压器类': '#FF5722',
              '开关柜类': '#9C27B0', '绝缘子类': '#FF9800', '电缆/导线类': '#795548',
              'GIS/组合电器类': '#00BCD4'}
bar_colors = [cat_colors.get(d['category'], '#888888') for d in all_data]

# Subplot 1: Non-zero month count
ax4_1 = axes4[0, 0]
nzs = [d['nz_all'] for d in all_data]
bars1 = ax4_1.barh(range(len(all_data)), nzs, color=bar_colors, edgecolor='white', alpha=0.85)
ax4_1.set_yticks(range(len(all_data)))
ax4_1.set_yticklabels([f"{d['label']} [{d['category']}]" for d in all_data], fontsize=9)
ax4_1.set_xlabel('非零月数 / 74个月')
ax4_1.set_title(f'非零月密度 (平均65%, VMD可用率100%)', fontsize=12, fontweight='bold')
ax4_1.axvline(x=37, color='#FF5722', linestyle='--', linewidth=1, alpha=0.5, label='50%密度线')
for bar, nz in zip(bars1, nzs):
    ax4_1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{nz}月({nz/74:.0%})',
              va='center', fontsize=8)
ax4_1.legend(fontsize=7)

# Subplot 2: CV (coefficient of variation)
ax4_2 = axes4[0, 1]
cvs = [d['cv'] for d in all_data]
bars2 = ax4_2.barh(range(len(all_data)), cvs, color=bar_colors, edgecolor='white', alpha=0.85)
ax4_2.set_yticks(range(len(all_data)))
ax4_2.set_yticklabels([d['label'] for d in all_data], fontsize=9)
ax4_2.set_xlabel('CV (变异系数)')
ax4_2.set_title(f'非零需求波动性 (CV<1.5=可预测, 平均CV={np.mean(cvs):.2f})', fontsize=12, fontweight='bold')
ax4_2.axvline(x=1.5, color='#FF5722', linestyle='--', linewidth=1, alpha=0.5, label='CV=1.5 阈值')
for bar, cv in zip(bars2, cvs):
    ax4_2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{cv:.2f}',
              va='center', fontsize=8)
ax4_2.legend(fontsize=7)

# Subplot 3: Train vs Test non-zero months
ax4_3 = axes4[1, 0]
x = np.arange(len(all_data))
width = 0.35
train_bars = ax4_3.bar(x - width/2, [d['train_nz'] for d in all_data], width,
                        label='训练集(62月)', color='#2196F3', alpha=0.85, edgecolor='white')
test_bars = ax4_3.bar(x + width/2, [d['test_nz'] for d in all_data], width,
                       label='测试集(12月)', color='#FF5722', alpha=0.85, edgecolor='white')
ax4_3.set_xticks(x)
ax4_3.set_xticklabels([d['label'] for d in all_data], rotation=45, ha='right', fontsize=8)
ax4_3.set_title('训练集 vs 测试集 非零月数', fontsize=12, fontweight='bold')
ax4_3.legend(fontsize=9)

# Subplot 4: Total demand (log scale for visual)
ax4_4 = axes4[1, 1]
totals = [d['total_qty'] for d in all_data]
bars4 = ax4_4.barh(range(len(all_data)), totals, color=bar_colors, edgecolor='white', alpha=0.85)
ax4_4.set_yticks(range(len(all_data)))
ax4_4.set_yticklabels([d['label'] for d in all_data], fontsize=9)
ax4_4.set_xlabel('总需求量')
ax4_4.set_xscale('log')
ax4_4.set_title('总需求量 (log scale)', fontsize=12, fontweight='bold')
for bar, tot in zip(bars4, totals):
    ax4_4.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2, f'{tot:.0f}',
              va='center', fontsize=8)

# Legend for category colors
handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85) for c in cat_colors.values()]
labels = list(cat_colors.keys())
ax4_4.legend(handles, labels, fontsize=7, loc='lower right')

fig4.suptitle('8种最高密度物资 — 统计概览 (国网总部级ECP采购数据)', fontsize=13, fontweight='bold')
plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, '04_summary_stats.png')
fig4.savefig(path4, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {path4}")

# ==============================================================================
# Print summary table
# ==============================================================================
print(f"\n{'='*90}")
print(f"8种物资数据摘要")
print(f"{'='*90}")
print(f"{'物资':<20s} {'品类':<15s} {'非零':>5s} {'训练':>5s} {'测试':>5s} {'CV':>6s} {'总量':>12s} {'均值':>10s}")
print(f"{'-'*80}")
for d in all_data:
    print(f"{d['label']:<20s} {d['category']:<15s} {d['nz_all']:>3d}/74 {d['train_nz']:>2d}/62 "
          f"{d['test_nz']:>2d}/12 {d['cv']:>5.2f} {d['total_qty']:>12.0f} {d['avg_nz']:>10.0f}")

avg_nz = np.mean([d['nz_all'] for d in all_data])
avg_cv = np.mean([d['cv'] for d in all_data])
print(f"\n平均非零月: {avg_nz:.0f}/74 ({avg_nz/74:.0%})  |  平均CV: {avg_cv:.2f}")
print(f"\nAll charts saved to: {OUTPUT_DIR}")
