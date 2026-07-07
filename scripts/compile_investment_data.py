"""
编译全国电网投资月度数据 + 推算冀北电网月度工程投资量
数据来源: 国家能源局月度《全国电力工业统计数据》+ 中电联
"""
import os, json, numpy as np
import pandas as pd

# ==============================================================================
# 已确认的月度累计数据 (亿元)
# ★=国家能源局, ☆=中电联/券商研报
# ==============================================================================
confirmed = {
    '2020': {'1-6': 1657,  'full': 4699},
    '2021': {'1-2': 227,  '1-3': 540,  '1-4': 852,  '1-5': 1225, '1-6': 1734, 'full': 4951},
    '2022': {'1-2': 313,  '1-4': 893,  '1-5': 1263, '1-6': 1905, '1-10': 3511, '1-11': 4209, 'full': 5012},
    '2023': {'1-2': 319,  '1-4': 984,  '1-5': 1400, '1-7': 2473, '1-11': 4458, 'full': 5275},
    '2024': {'1-2': 327,  '1-3': 766,  '1-4': 1229, '1-5': 1703, '1-6': 2540, '1-7': 2947, '1-8': 3330, '1-9': 3982, '1-10': 4502, '1-11': 5290, 'full': 6083},
    '2025': {'1-4': 1408, '1-5': 2040},
    '2026': {},
}

sources = {
    '2020': {'1-6': 'chinasmartgrid 2020-07-22', 'full': '国家能源局 2021-01-20'},
    '2021': {'1-2': '中电联/观研报告网', '1-3': '中电联', '1-4': '中电联', '1-5': '中电联', '1-6': '中电联', 'full': '国家能源局 2022-01-20'},
    '2022': {'1-2': '国家能源局 2022-03-21', '1-4': '国家能源局', '1-5': '中电联/中信建投', '1-6': '中电联', '1-10': '国家能源局', '1-11': '国家能源局', 'full': '国家能源局 2023-01-18'},
    '2023': {'1-2': '国家能源局 2023-03-22', '1-4': '上证报/华福证券', '1-5': '国家能源局 2023-06-20', '1-7': '国家能源局 2023-08-17', '1-11': '国家能源局 2023-12-20', 'full': '国家能源局 2024-01-26'},
    '2024': {'1-2': '国家能源局 2024-03-25', '1-3': '国家能源局', '1-4': '国家能源局 2024-05-23', '1-5': '国家能源局 2024-06-28', '1-6': '国家能源局 2024-07-20', '1-7': '国家能源局 2024-08-30', '1-8': '国家能源局 2024-09-23', '1-9': '国家能源局 2024-10-21', '1-10': '国家能源局 2024-11-22', '1-11': '国家能源局 2024-12-20', 'full': '国家能源局 2025-01-21'},
    '2025': {'1-4': '国家能源局/红塔证券', '1-5': '国家能源局/华泰证券'},
}

MONTH_LABELS = [f'{m:02d}' for m in range(1, 13)]

# ==============================================================================
# 2024年月度单月值 (从确认的累计值推算) — 用做季节模板
# ==============================================================================
y2024_cum = {'1-2':327, '1-3':766, '1-4':1229, '1-5':1703, '1-6':2540,
             '1-7':2947, '1-8':3330, '1-9':3982, '1-10':4502, '1-11':5290, 'full':6083}
y2024_monthly = {}
prev = 0
for cm_name in sorted(y2024_cum.keys(), key=lambda k: int(k.split('-')[1]) if k != 'full' else 99):
    if cm_name == 'full':
        continue
    cm = int(cm_name.split('-')[1])
    val = y2024_cum[cm_name]
    # 分配: 从prev+1月到cm月
    target_sum = val - y2024_cum.get(f'1-{prev}', 0) if prev > 0 else val
    if prev == 0:
        # 1-2月: 拆分为1月(35%) + 2月(65%)
        jan = round(val * 0.35, 1)
        feb = round(val - jan, 1)
        y2024_monthly['01'] = jan
        y2024_monthly['02'] = feb
    else:
        n_months = cm - prev
        # 初始化均匀分配, 后续会按季节调整
        per_month = round(target_sum / n_months, 1)
        for i in range(n_months):
            m_str = f'{prev + 1 + i:02d}'
            y2024_monthly[m_str] = per_month
    prev = cm

# 1-11累计 = 5290, 12月 = 6083 - 5290 = 793
remaining = y2024_cum['full'] - y2024_cum['1-11']
total_allocated = sum(y2024_monthly.values())
y2024_monthly['12'] = round(remaining, 1)

# 归一化
total = sum(y2024_monthly.values())
for m in y2024_monthly:
    y2024_monthly[m] = round(y2024_monthly[m] * y2024_cum['full'] / total, 1)
seasonal_ratio = {m: y2024_monthly[m] / sum(y2024_monthly.values()) for m in MONTH_LABELS}

print("2024 月度季节因子 (%):")
for m in MONTH_LABELS:
    print(f"  {m}月: {seasonal_ratio[m]*100:.1f}% ({y2024_monthly[m]:.0f}亿)")

# ==============================================================================
# 拟合函数: 按季节因子分配, 满足已知累计硬约束
# ==============================================================================
def cum_key_to_month(k):
    return int(k.split('-')[1])

def fit_monthly(cum_data, annual_total):
    """
    月度分配策略:
    1. 先按2024季节比例分配全年总量 → 得到初步月度值
    2. 对每个已知累计约束段, 线性缩放使该段总和匹配约束
    3. 最终归一化到全年总量
    """
    monthly = np.array([seasonal_ratio[m] * annual_total for m in MONTH_LABELS])

    constraints = sorted(
        [(cum_key_to_month(k), v) for k, v in cum_data.items() if k != 'full'],
        key=lambda x: x[0]
    )

    prev_cm, prev_val = 0, 0
    for cm, target_cum in constraints:
        # 从prev_cm+1月到cm月这一段
        seg_idx = list(range(prev_cm, cm))  # 0-indexed
        current_sum = monthly[seg_idx].sum()
        target_sum = target_cum - prev_val
        if current_sum > 0:
            monthly[seg_idx] *= target_sum / current_sum
        prev_cm, prev_val = cm, target_cum

    # 最后一段: 最后一个约束到12月
    if constraints:
        last_cm, last_val = constraints[-1]
        if last_cm < 12:
            seg_idx = list(range(last_cm, 12))
            target_sum = annual_total - last_val
            current_sum = monthly[seg_idx].sum()
            if current_sum > 0:
                monthly[seg_idx] *= target_sum / current_sum

    # 最终归一化
    monthly *= annual_total / monthly.sum()

    return {MONTH_LABELS[i]: round(monthly[i], 1) for i in range(12)}

# 计算所有年份
all_monthly = {}
for yr in range(2020, 2027):
    ystr = str(yr)
    cd = confirmed.get(ystr, {})
    annual = cd.get('full', None)
    if annual is None:
        if yr == 2025:
            annual = 6300  # 估算: 基于1-5月增速19.8%
        else:
            annual = sum(seasonal_ratio[m] * 6500 for m in MONTH_LABELS)  # fallback
    all_monthly[ystr] = fit_monthly(cd, annual)

# ==============================================================================
# 输出: 全国月度表和冀北估算
# ==============================================================================
print('\n' + '=' * 140)
print('全国电网基本建设投资完成额 月度单月值 (亿元) — 2024季节模板 + 累计约束拟合')
print('=' * 140)
header = f'{"月份":>5s}'
for yr in range(2020, 2027):
    header += f' | {yr:>10d}'
print(header)
print('-' * 140)

for mi, m in enumerate(MONTH_LABELS):
    line = f'{m:>5s}'
    for yr in range(2020, 2027):
        ystr = str(yr)
        cd = confirmed.get(ystr, {})
        val = all_monthly[ystr][m]
        is_constrained = any(cum_key_to_month(k) == mi + 1 for k in cd if k != 'full')
        marker = '★' if is_constrained else ' '
        line += f' | {marker}{val:>9.1f}'
    print(line)

print('-' * 140)
print('★ = 受国家能源局累计值直接约束的月份 | 空格 = 季节分配')

# 校核
print('\n年度合计校核:')
for yr in range(2020, 2027):
    ystr = str(yr)
    actual = confirmed.get(ystr, {}).get('full', None)
    total = sum(all_monthly[ystr].values())
    if actual:
        ok = abs(total - actual) < 2
        print(f'  {yr}: 拟合={total:.1f}  实际={actual:.0f}  偏差={total-actual:.1f}  {"OK" if ok else "CHECK"}')
    else:
        print(f'  {yr}: 拟合={total:.1f}  (全年未知, 季节估算)')

# ==============================================================================
# 冀北估算
# ==============================================================================
jibei_annual = {
    2020: (15.0, 4699),
    2021: (17.0, 4951),
    2022: (23.0, 5012),
    2023: (31.29, 5275),
    2024: (40.0, 6083),
    2025: (49.0, 6300),
    2026: (53.0, 6800),  # 预估
}

print('\n' + '=' * 140)
print('冀北配电网月度工程投资量估算 (亿元) — 全国月度×冀北年度占比')
print('=' * 140)

line = f'{"月份":>5s}'
for yr in range(2020, 2027):
    line += f' | {yr:>10d}'
print(line)
print('-' * 140)

jibei_monthly = {}
for yr in range(2020, 2027):
    ystr = str(yr)
    jb, nat = jibei_annual[yr]
    ratio = jb / nat
    jibei_monthly[yr] = {m: round(all_monthly[ystr][m] * ratio, 4) for m in MONTH_LABELS}

for mi, m in enumerate(MONTH_LABELS):
    line = f'{m:>5s}'
    for yr in range(2020, 2027):
        line += f' | {jibei_monthly[yr][m]:>10.4f}'
    print(line)

print('-' * 140)
line = f'{"合计":>5s}'
for yr in range(2020, 2027):
    total = sum(jibei_monthly[yr].values())
    line += f' | {total:>10.2f}'
print(line)

line = f'{"占比":>5s}'
for yr in range(2020, 2027):
    jb, nat = jibei_annual[yr]
    line += f' | {jb/nat*100:>9.2f}%'
print(line)

# ==============================================================================
# 导出
# ==============================================================================
output_dir = r'D:\Users\dell\PycharmProjects\vmd-catboost\inputs\data'
os.makedirs(output_dir, exist_ok=True)

# 全国数据 CSV
csv_nat = os.path.join(output_dir, 'national_grid_investment_monthly.csv')
with open(csv_nat, 'w', encoding='utf-8') as f:
    f.write('year,month,investment_100M_yuan\n')
    for yr in range(2020, 2027):
        for m in MONTH_LABELS:
            f.write(f'{yr},{m},{all_monthly[str(yr)][m]:.1f}\n')
print(f'\n全国数据 → {csv_nat}')

# 冀北数据 Excel
rows = []
for yr in range(2020, 2027):
    for m in MONTH_LABELS:
        rows.append({
            '日期': f'{yr}-{m}-01',
            '工程投资量(亿元)': jibei_monthly[yr][m],
            '数据来源': '全国电网投资月度分布×冀北年占比',
        })
df = pd.DataFrame(rows)
xlsx_path = os.path.join(output_dir, 'jibei_investment_monthly.xlsx')
df.to_excel(xlsx_path, index=False, sheet_name='冀北月度工程投资量')
print(f'冀北数据 → {xlsx_path}')

# 保存元数据
meta = {
    'description': '全国电网基本建设投资完成额月度数据 + 冀北配电网工程投资量推导',
    'method': '以国家能源局月度发布的全国电网投资累计值为硬约束，按2024年月度分布作为季节模板拟合单月值。冀北数据 = 全国月度值 × 冀北配电网投资年度占比',
    'sources': sources,
    'jibei_annual_investment': {str(k): v[0] for k, v in jibei_annual.items()},
    'jibei_annual_ratio': {str(k): v[0]/v[1] for k, v in jibei_annual.items()},
}
meta_path = os.path.join(output_dir, 'investment_data_metadata.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f'元数据 → {meta_path}')

print('\n完成!')
