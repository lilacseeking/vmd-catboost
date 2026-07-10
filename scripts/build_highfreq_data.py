"""
生成8种高密度物资的预测数据文件。
每种物资 >=35 非零月 (训练集密度 >=47%), 来自不同品类。
"""
import sqlite3, numpy as np, pandas as pd, os

db = sqlite3.connect(r"D:\Users\dell\PycharmProjects\bidding-ecp-data\data\ecp_data.db")
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if f"{y}{m:02d}" >= "202005" and f"{y}{m:02d}" <= "202606"]

# 8 materials from different categories, all >=35 non-zero months
MATERIALS = [
    ('线路保护', '线路保护', '保护类'),
    ('电抗器保护', '电抗器保护', '保护类'),
    ('10kV变压器', '10kV变压器', '变压器类'),
    ('低压开关柜', '低压开关柜', '开关柜类'),
    ('钢芯铝绞线', '钢芯铝绞线', '电缆/导线类'),
    ('交流支柱绝缘子', '交流支柱绝缘子', '绝缘子类'),
    ('GPS定位仪', 'GPS定位仪', '通信/仪器类'),
    ('采集终端', '采集终端', '计量/采集类'),
]

output_path = r"D:\Users\dell\PycharmProjects\vmd-catboost\inputs\data_highfreq.xlsx"

all_rows = {}
for mat_name, query_name, cat in MATERIALS:
    cur = db.execute("""
        SELECT demand_month, SUM(demand_quantity)
        FROM material_demand_item
        WHERE material_name LIKE ? AND demand_month >= '202005' AND demand_month <= '202606'
        GROUP BY demand_month ORDER BY demand_month
    """, (f'%{query_name}%',))
    data = {r[0]: r[1] for r in cur.fetchall()}
    vals = np.array([data.get(m, 0) for m in MONTHS])
    nz = (vals > 0).sum()
    nz_vals = vals[vals > 0]
    cv = np.std(nz_vals) / np.mean(nz_vals) if nz_vals.size > 0 else -1

    rows = []
    for i, m in enumerate(MONTHS):
        rows.append({
            '日期': f"{m[:4]}-{m[4:]}-01",
            '需求量': vals[i],
        })
    all_rows[mat_name] = (rows, nz, cv, cat)

# Print summary
print("=" * 70)
print("8种高密度物资 — 数据概览")
print("=" * 70)
for mat_name, (rows, nz, cv, cat) in all_rows.items():
    train_nz = sum(1 for r in rows[:62] if r['需求量'] > 0)
    test_nz = sum(1 for r in rows[62:] if r['需求量'] > 0)
    print(f"  {mat_name:<15s} [{cat:<10s}]: total={nz}/74非零, train={train_nz}/62, test={test_nz}/12, CV={cv:.2f}")

# Write Excel
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for mat_name, (rows, _, _, _) in all_rows.items():
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=mat_name, index=False)

print(f"\n  Excel: {output_path}")

# Also print first 3 rows and last 3 rows for each material
for mat_name, (rows, _, _, _) in all_rows.items():
    demand_vals = [r['需求量'] for r in rows]
    print(f"\n  [{mat_name}] 测试集12月:")
    for i, r in enumerate(rows[62:]):
        marker = " <--" if r['需求量'] > 0 else ""
        print(f"    {r['日期'][:7]}: {r['需求量']:>10.0f}{marker}")

db.close()
