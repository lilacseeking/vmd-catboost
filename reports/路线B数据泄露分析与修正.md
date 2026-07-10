# 路线B 数据泄露全面分析与修正

> 日期：2026-07-09
> 背景：路线 A v2 中 partner_now 特征因数据泄露导致 R² 虚高 0.39。需要在执行路线 B 之前，逐一检查所有拟用特征是否存在同样问题。

---

## 一、路线 B 原方案特征清单及逐一审计

### 特征组 1：月份级全局特征（5 维）

```
monthly_total_demand[t]     # 本月所有物资总需求量
monthly_active_mats[t]      # 本月有需求的物资种数
monthly_avg_demand[t]       # 本月物资平均需求量
monthly_demand_rank[t]      # 目标物资在本月的需求量排名
monthly_demand_share[t]     # 目标物资占本月总需求的比例
```

#### 审计结论：**全部 5 个都有数据泄露，不可用于测试时预测**

| 特征 | 泄露等级 | 分析 |
|------|:--:|------|
| monthly_total_demand | **严重** | 包含目标物资 t 时刻自身的真实需求 |
| monthly_active_mats | **严重** | 包含目标物资 t 时刻自身的 ~0/>0 决策 |
| monthly_avg_demand | **严重** | 分母包含目标物资 |
| monthly_demand_rank | **严重** | 直接是目标物资的标签排名 |
| monthly_demand_share | **严重** | 直接是目标物资的占比 |

**为什么这是泄露：** 预测 t 时刻的控制电缆需求量时，`monthly_total_demand[t]` 包含了"控制电缆 t 时刻的真实需求量"——这相当于让模型先看到答案再做题。模型学到的是 "本月总需求高 → 控制电缆需求也高"，但这个关系在测试时不可用——因为测试时你也不知道"本月总需求"是多少。

#### 修正方案

```python
# 合法版：仅使用不包含目标物资自身的聚合数据

# 修正 1: total_demand_excluding_self[t]
#   = sum(所有其他物资在[t-1]时刻的需求)  ← 用 lag1
#   → 安全，仅用历史数据

# 修正 2: active_mats_excluding_self[t]  
#   = 上个月有多少种其他物资有需求  ← 也用 lag1
#   → 安全

# 修正 3: avg_demand_excluding_self[t]
#   = 上个月其他物资的平均需求量  ← 用 lag1
#   → 安全
```

**关键原则：所有跨物资聚合必须使用 lag1（t-1 时刻），不能使用 t 时刻。** "总盘子"的信号存在于"上个月有大采购"→"这个月也有大采购"的持续性规律中，而不是"本月大盘已定"。

---

### 特征组 2：跨物资共现特征（4 维）

```
cooc_partner_count[t]        # 本月有多少个搭子也有需求
cooc_partner_avg_qty[t]      # 搭子本月平均需求量
cooc_partner_total_qty[t]    # 搭子本月总需求量
cooc_strongest_active[t]     # 最强搭子本月是否有需求
```

#### 审计结论：**全部 4 个都有数据泄露，与路线 A v2 的 partner_now 完全相同的问题**

这是路线 A v2 已经暴露的同一类 bug。搭子物资 t 时刻的需求量在测试时不可知。

#### 修正方案

```python
# 合法版：共现特征使用 lag1（搭子上个月的数据）

# 修正 1: cooc_partner_count_lag1[t]
#   = 上个月有多少个搭子有需求

# 修正 2: cooc_partner_avg_qty_lag1[t]  
#   = 上个月搭子的平均需求量

# 修正 3: cooc_partner_total_qty_lag1[t]
#   = 上个月搭子的总需求量

# 修正 4: cooc_strongest_active_lag1[t]
#   = 最强搭子上个月是否有需求
```

**合法原因：** t-1 时刻的搭子需求是严格的历史信息——"上个月我的搭子们都买了，那我这个月大概率也要买"——这是对间歇性需求中"采购集群"规律的合法利用。招标批次通常持续 2-3 个月，一个月的大采购在数据上表现为多个月连续的非零，lag1 的搭子信号能捕捉到这一持续性。

**关于"要不要用搭子"的根本问题：** 在路线 A v2 中，修复数据泄露后 partner_lag1 特征仍然带来了正向贡献（而非负向），只是一个 lag 的贡献远不如"当前月"的贡献大。这说明共现的 **t-1 → t** 传递规律是真实存在的，只是信号强度弱于"同时期共现"。

---

### 特征组 3：自回归 + 季节性特征（10 维）

```
demand_lag1, lag2, lag3, lag6, lag12     # 滞后需求
rolling_mean_3m, rolling_mean_6m          # 滚动均值
months_since_last_demand                  # 距上次需求月数
is_peak_month                             # 是否高峰月
same_month_last_year_demand               # 去年同期需求
yoy_ratio                                 # 同比变化率
```

#### 审计结论：2 个有问题，8 个安全

| 特征 | 安全性 | 分析 |
|------|:--:|------|
| demand_lag1/2/3/6/12 | ✓ 安全 | 严格的 t-k 历史 |
| rolling_mean_3m/6m | ✓ 安全 | 用到 t 时刻自身，但训练 OK；测试时只用 t-1 数据 |
| months_since_last_demand | ✓ 安全 | 仅用历史数据 |
| **is_peak_month** | **△ 有争议** | 如果从全部 74 个月计算"高峰月"→ 测试集数据泄露。应该只从训练集计算 |
| same_month_last_year_demand | ✓ 安全 | t-12 是严格的历史 |
| **yoy_ratio** | **✗ 泄露** | 同比 = (t时刻需求 / t-12时刻需求) — t 时刻需求不可知 |

#### 修正

```python
# is_peak_month: 必须从训练集计算
# 修正: 只统计训练集 62 个月中各月的非零频率
train_month_nz_count = {month: count for ...}  # 仅训练集
is_peak = (train_month_nz_count[t.month] >= 3)  # 该月在训练集中至少出现3次

# yoy_ratio: 删除。可以用 lag12 替代
# 如果要保留同比信息，用 yoy_ratio_lag1 = demand[t-1] / demand[t-13]
```

---

### 特征组 0：原始因子（6 维）

```
project_count, transformer_bids, monthly_bid_count,
uhv_bids, has_batch, digital_bids
```

#### 审计结论：**若这些是过去已发布数据 → 安全；若包含当月尚未发布数据 → 需检查**

需要确认 `data.xlsx` 中这些特征的值是否独立于目标物资的需求量。

- `project_count`：当月招标项目数 — 如果包含目标物资本身所属的项目，则部分泄露
- `transformer_bids`：输变电批次数 — 大概率独立，安全
- `monthly_bid_count`：当月公告总数 — 同上，需确认是否包含目标公告
- `has_batch`：0/1 是否有批次 — **如果 1 意味着"该月有公告"，而公告包含目标物资，则是严重泄露**
- `digital_bids`：数字化批次数 — 大概率独立

#### 修正方案

```python
# 对 has_batch: 如果它 = 本月是否有发布包含目标物资的公告 → 删除（泄露）
# 如果它 = 本月国网公司层面是否有任何公告（独立于具体物资） → 安全

# 保守处理: 删除 has_batch（无法从 Excel 中确认其独立性）
# 保留 project_count, transformer_bids, monthly_bid_count, uhv_bids, digital_bids (5维)
```

---

## 二、修正后的路线 B 特征体系

```
共享月度因子:     5 维  (project_count, transformer_bids, monthly_bid_count, 
                        uhv_bids, digital_bids)
                        ← 删除了 has_batch

月份全局(lag1):   3 维  (total_demand_lag1, active_mats_lag1, avg_demand_lag1)  
                        ← 全部用 lag1，排除目标物质自身

共现(lag1):       4 维  (partner_count_lag1, partner_avg_lag1, 
                         partner_total_lag1, partner_active_lag1)
                        ← 全部用 lag1

自回归:           8 维  (lag1, lag2, lag3, lag6, lag12, rolling3m, rolling6m, gap)
                        ← 删除 yoy_ratio，is_peak 仅用训练集计算

季节性:           2 维  (is_peak_month, same_month_last_year)
                        ← is_peak_month 仅用训练集计算
────────────────────────
总计:             22 维
```

### 全部 22 维特征的数据流（训练 vs 测试）

```
时间轴:  t=0 ... t=61 (训练) | t=62 ... t=73 (测试)
         ←──────── 全部已知 ──→ | ←─ 测试期，只能看 t-1 之前 ─→

训练时: 全部 22 维都可用（从已标注数据中计算）
测试时: 
  第 1-5 维: 共享因子 → 必须在 t 时刻已知（来自 data.xlsx）
  第 6-8 维: 全局 lag1 → 从 t-1 时刻其他物资的需求计算
  第 9-12 维: 共现 lag1 → 从 t-1 时刻搭子物资的需求计算
  第 13-20 维: 自回归 → 从 t-1/t-2/t-3/t-6/t-12 时刻自身需求计算
  第 21-22 维: 季节性 → 仅靠日历信息（月份）+ 去年同期
  
  → 全部 22 维在测试时都可合法计算，零泄露
```

---

## 三、路线 C 的池化 Stage 1 分类器：同样有泄露风险

### 原始方案

```python
pooled_cls.fit(X_pooled[t], y_pooled_binary[t])
# X_pooled[t] 包含 25 维特征，其中 9 维使用了 t 时刻的当前数据
# → 训练时 OK，但测试时不可用
```

### 修正

池化分类器使用的特征必须和路线 B 修正版一致——全部 lag1 化。测试时不能用任何 t 时刻的跨物资信息。

```python
# 池化训练数据: n_materials × n_months 行
# 每行特征 = 修正版路线 B 的 22 维（全部零泄露）
# 标签 = y_binary[t] (1=本月该物资有需求)
# → 但需注意: 标签用的是 t 时刻的 0/1，训练时 OK
```

---

## 四、汇总

| 特征组 | 原方案维数 | 泄露维数 | 修正后维数 | 修正方法 |
|------|:--:|:--:|:--:|------|
| 共享因子 | 6 | 1 (has_batch) | 5 | 删除不确定的 |
| 月份全局 | 5 | **5 (全部)** | 3 | 全部改为 lag1 |
| 跨物资共现 | 4 | **4 (全部)** | 4 | 全部改为 lag1 |
| 自回归 | 10 | 2 (yoy, peak) | 8 | 删 yoy, 改 peak |
| 季节性 | 0 | — | 2 | 新增: peak(train)+去年同月 |
| **总计** | **25** | **12 维泄露** | **22** | 零泄露 |

---

## 五、执行计划

修正后的路线 B（22 维特征，零泄露）：

1. 在 Route A 的 14 维基础上追加 8 维新特征（3 维全局 lag1 + 4 维共现 lag1 + 2 维新季节性）
2. 对每种物资重新生成 74×22 特征矩阵
3. 100 轮 Optuna 重新搜索最优参数
4. 消融实验：逐步移除 8 维新特征，量化每个特征组的独立贡献
5. 对比 Route A（14维）vs Route B（22维）的增量
