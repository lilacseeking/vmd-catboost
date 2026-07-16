# EVT 采购计划增强方案——用确定性未来事件替换测试集估算

> 日期: 2026-07-15 | 状态: 技术方案 → 待审核

---

## 一、问题定位

### 1.1 当前 EVT 测试集的致命缺陷

`preprocess_data()` 中测试集的 EVT 特征计算（`main.py:419-442`）存在结构性缺陷：

```python
# 测试集 evt_cnt_6m 的计算 (line 431-435)
for i in range(nt):
    seq = d_all[max(0,tl+i-5):tl+i+1]
    seq12 = d_all[max(0,tl+i-11):tl+i+1]
    ec6_te[i] = np.sum(np.array(seq) > 0)   # ← 这里
    ec12_te[i] = np.sum(np.array(seq12) > 0) # ← 这里
```

这段代码对测试期每个月，数"过去6/12个月（含本月）有多少月有需求"。"含本月"意味着：测试期第 i 个月还没发生，但代码 `d_all[tl + i]` 已经读取了 `demand_raw` 对应位置的值。这**在训练时是未来数据、在真实场景下不可用**。

实际上 `demand_raw[tl+i]` 在 i≥0 时是测试集的真实标签，预测时不应被任何特征使用。但代码把它作为特征输入了——这是一处**数据泄露**。

### 1.2 泄露量分析

| 测试月 i | seq 中含测试月 i 吗？ | 泄露？ |
|:---:|:---:|:---:|
| 0 | `d_all[tl-5 : tl+1]` 含 `d_all[tl]` | ✅ 已泄露 |
| 1 | `d_all[tl-4 : tl+2]` 含 `d_all[tl+1]` | ✅ |
| ... | ... | 全部 12 个月均泄露 |

对于 `evt_cnt_6m`，12个月中前5个月的窗口完全在训练集内，后7个月的窗口跨入了自己月份。对 `evt_cnt_12m`，前11个月的窗口完全在训练集内，仅第12个月窗口跨入自己。`gap_since_last` 也有同样问题——它用 `d_all[idx] > 0` 来判断测试月是否发生了事件。

### 1.3 为什么不能"修"而需要"替"

即使修复了数据泄露（改成只用训练集信息），测试集的 EVT 特征仍然是"向后看"的：

> "过去6个月有3次事件" → 对测试期第1个月的预测有用  
> "过去6个月有2次事件" → 但对测试期第6个月，其中一半的窗口在测试期内，信息半盲

正确的做法：用**采购预安排信息**直接告知模型"接下来N个月计划有K次招标"，替代估算。

---

## 二、数据源：SGCC 年度采购预安排

### 2.1 数据是什么

国家电网有限公司每年 1-2 月在 ECP2.0 平台公布《国家电网有限公司XXXX年度总部集中采购批次预安排》。这份文件以表格形式列出全年各月计划的招标批次类型和数量。

### 2.2 数据格式（从 ECP 页面提取）

采购预安排是一个按月组织的计划表，典型结构：

```
月份   | 批次数量 | 批次类型
1月    | 2       | 输变电设备第一批、服务类第一批
2月    | 0       | 
3月    | 3       | 输变电设备第二批、特高压第一批、配农网第一批
...
```

### 2.3 数据获取方式

由于 ECP 的 `doci-purplan` 页面无法通过 API 获取内容（加密+SPA架构），采用：
- **手动提取**：从浏览器访问 ECP 采购预安排页面，提取月份-批次数对照表
- **存储格式**：JSON 文件，存放在 `inputs/procurement_plan.json`
- **维护方式**：每年2月更新一次（SGCC公布新计划后）

### 2.4 数据结构设计

```json
{
  "2025": {
    "source_url": "https://ecp.sgcc.com.cn/ecp2.0/portal/#/doc/doci-purplan/2502053460622845_2020052000175277",
    "source_title": "国家电网有限公司2025年度总部集中采购批次预安排",
    "publish_date": "2025-02-05",
    "monthly_schedule": {
      "1": {"batch_count": 2, "has_material_batch": true},
      "2": {"batch_count": 0, "has_material_batch": false},
      "3": {"batch_count": 3, "has_material_batch": true},
      "...": "..."
    }
  },
  "2026": {
    "source_url": "https://ecp.sgcc.com.cn/ecp2.0/portal/#/doc/doci-purplan/2602034384651452_2020052000175277",
    "source_title": "国家电网有限公司2026年度总部集中采购批次预安排",
    "publish_date": "2026-02-03",
    "monthly_schedule": {
      "1": {"batch_count": 2, "has_material_batch": true},
      ...
    }
  }
}
```

---

## 三、技术方案

### 3.1 方案概览

在 `preprocess_data()` 中新增一类特征：**采购计划特征 (Plan Features)**。

这类特征的核心不同：
- **训练集**：用实际历史数据计算（等价于知道"这个月实际发生了什么"）
- **测试集**：用采购预安排数据计算（等价于知道"这个月计划发生什么"）

因为采购预安排在预测时是**确定已知的未来信息**（年初公布全年计划），使用它不构成数据泄露。

### 3.2 新增采购计划特征（5个）

| 特征名 | 类型 | 训练集计算 | 测试集计算 |
|--------|------|-----------|-----------|
| `plan_evt_this_month` | 计划事件指示 | 实际该月 demand>0 | 采购计划中该月是否有物资批次 |
| `plan_evt_cnt_3m` | 未来3月计划事件数 | 实际未来3月事件数 | 采购计划未来3月批次数 |
| `plan_evt_cnt_6m` | 未来6月计划事件数 | 实际未来6月事件数 | 采购计划未来6月批次数 |
| `plan_gap_to_next` | 距下次计划事件月数 | 实际距下次事件月数 | 采购计划距下次批次的月数 |
| `plan_month_is_batch` | 该月是否为计划批次月 | 实际该月>0 | 采购计划中has_material_batch |

### 3.3 对现有 EVT 特征的替代关系

下表说明新旧特征的关系：

| 现有 EVT 特征 | 问题 | 对应的采购计划特征 | 改进 |
|--------------|------|------------------|------|
| `gap_since_last` | 向后看，不知道下次事件在哪 | `plan_gap_to_next` | 向前看，知道下次事件距离 |
| `evt_cnt_6m` | 测试期数据泄露 + 半年内一半窗口在测试集 | `plan_evt_cnt_6m` | 无泄露，直接从计划计算 |
| `evt_cnt_12m` | 测试期窗口跨测试集 | `plan_evt_cnt_6m` × 2 + `plan_evt_this_month` | 用计划+已发生组合计算 |
| `month_freq` | 只基于历史统计 | `plan_month_is_batch` | 确定性真值 |
| `cumul_12m` | 测试期窗口跨测试集 | 用 `plan_evt_cnt_*` + train最近值推算 | 减少泄露风险 |

### 3.4 特征工程改造点

#### 3.4.1 训练集

```python
def make_plan_features_train(demand_train, train_months, plan_data):
    """
    训练集: 用实际历史数据计算"计划"特征
    等价逻辑: 如果我在 t 月做预测，我知道 t+1..t+N 月的计划
    但训练时 t+1..t+N 已有实际数据 → 用实际数据训练模型理解"计划→实际"的映射
    """
    n = len(demand_train)
    # plan_evt_this_month: 实际该月是否有需求（训练时等价于知道计划是否兑现）
    plan_evt_this = (demand_train > 0).astype(float)
    
    # plan_evt_cnt_3m: 未来3个月（含本月）的需求事件数
    plan_cnt_3m = np.zeros(n)
    for i in range(n):
        end = min(n, i + 3)
        plan_cnt_3m[i] = np.sum(demand_train[i:end] > 0)
    
    # plan_evt_cnt_6m: 未来6个月的事件数
    plan_cnt_6m = np.zeros(n)
    for i in range(n):
        end = min(n, i + 6)
        plan_cnt_6m[i] = np.sum(demand_train[i:end] > 0)
    
    # plan_gap_to_next: 距下一次事件的月数
    plan_gap = np.zeros(n)
    for i in range(n):
        future_events = np.where(demand_train[i:] > 0)[0]
        plan_gap[i] = future_events[0] if len(future_events) > 0 else n - i
    
    # plan_month_is_batch: 该月是计划批次月
    cal_month = np.array([(4 + i) % 12 + 1 for i in range(n)])
    plan_is_batch = np.array([
        plan_data.get(str(m), {}).get('has_material_batch', False)
        for m in cal_month
    ]).astype(float)
    
    return plan_evt_this, plan_cnt_3m, plan_cnt_6m, plan_gap, plan_is_batch
```

#### 3.4.2 测试集

```python
def make_plan_features_test(demand_train, test_months, plan_data):
    """
    测试集: 用采购计划数据计算特征（无泄露）
    关键差异: 对测试月, plan_cnt_3m 使用采购计划中的批次数, 而非实际需求量
    """
    n_train = len(demand_train)
    n_test = len(test_months)
    
    # 组合: 训练期末尾实际值 + 测试期计划值
    # plan_evt_this_month: 测试期 = 计划中该月的 has_material_batch
    plan_evt_this = np.array([
        plan_data.get(str(m), {}).get('has_material_batch', False)
        for m in test_months
    ]).astype(float)
    
    # plan_evt_cnt_3m/6m: 向前看, 使用计划数据
    plan_cnt_3m = np.zeros(n_test)
    plan_cnt_6m = np.zeros(n_test)
    plan_gap = np.zeros(n_test)
    
    for i in range(n_test):
        # 从当前测试月开始, 向前看N个月
        # 若超出测试期范围, 按计划中的批次数估算
        cnt3 = 0; cnt6 = 0; gap = 999
        for j in range(6):
            idx = i + j
            if idx < n_test:
                m = test_months[idx]
                has_batch = plan_data.get(str(m), {}).get('has_material_batch', False)
                batch_cnt = plan_data.get(str(m), {}).get('batch_count', 0)
                if has_batch or batch_cnt > 0:
                    if j < 3: cnt3 += 1
                    cnt6 += 1
                    if gap == 999:
                        gap = j
        plan_cnt_3m[i] = cnt3
        plan_cnt_6m[i] = cnt6
        plan_gap[i] = min(gap, 12)  # cap at 12 months
    
    # plan_month_is_batch: 该月是计划批次月
    plan_is_batch = np.array([
        plan_data.get(str(m), {}).get('has_material_batch', False)
        for m in test_months
    ]).astype(float)
    
    return plan_evt_this, plan_cnt_3m, plan_cnt_6m, plan_gap, plan_is_batch
```

### 3.5 训练/测试的对称性设计

训练时特征使用"实际事件"、测试时特征使用"计划事件"——这种训练/测试的不对称性使得模型学习到：

> "计划说下月有批次 → 上个月实际有需求 ∴ 这个月需求量 ≈ 上个月量"

而非：

> "过去6个月有3次事件 → 这个月也有事件"  ← 这种纯历史自回归视角

---

## 四、与现有 EVT 的关系

### 4.1 不删除现有 EVT

现有 EVT 的 5 个特征（`gap_since_last`, `evt_cnt_6m`, `evt_cnt_12m`, `cumul_12m`, `month_freq`）在训练集上仍然有效——它们提供了"历史事件节奏"的信息。采购计划特征提供的是"未来事件安排"的信息——两者互补。

### 4.2 修复 EVT 的数据泄露

同时修复测试集中 EVT 特征的数据泄露问题（见 §1.1）：
- `evt_cnt_6m_te`: 不包含 `d_all[tl+i]`，只用 `d_all[max(0, tl+i-5):tl+i]`（不含当前月）
- `gap_since_last_te`: 不更新测试月的事件状态
- `cumul_12m_te`: 同样窗口不含当前测试月

### 4.3 最终特征配置

```
基线 (EVT=OFF):  4因子 + 6lag/rolling + 4sin/cos = 14维,  // 这是假的: 少了一个 N_TEST
基线 (EVT=ON):   + 5 EVT = 19维
增强 (PLAN=ON):  + 5 PLAN = 24维
```

可通过 `USE_PLAN_FEATURES` 开关控制。

---

## 五、实施计划

### 5.1 数据准备

1. 创建 `inputs/procurement_plan.json`
   - 从 ECP 手动提取 2025 和 2026 年度批次预安排
   - 格式见 §2.4

### 5.2 代码改动（main.py）

1. **新增全局开关** `USE_PLAN_FEATURES = True`（~line 116）
2. **新增数据加载函数** `load_procurement_plan()`（~line 170）
3. **新增特征函数** `make_plan_features_train()` 和 `make_plan_features_test()`（~line 350）
4. **修改 preprocess_data()**：
   - 训练集：追加 plan_feats_tr 到 X_train_raw
   - 测试集：追加 plan_feats_te 到 X_test_raw
   - 修复现有 EVT 的数据泄露
5. **更新 all_models 的 feature count 日志输出**

### 5.3 预期效果

| 物资 | 当前 EVT R² | 预期 PLAN R² | 改善来源 |
|------|:---:|:---:|------|
| 电容式电压互感器 | 0.634 | 0.68-0.72 | 已知下一个批次精确日期 |
| 交流支柱绝缘子 | 0.748 | 0.77-0.80 | 计划+EVT双重信号 |
| 断路器保护 | 0.727 | 0.75-0.78 | 低CV物资受益于计划确定性 |
| 电抗器保护 | 0.609 | 0.65-0.70 | 同上 |
| 交流避雷器 | 0.611 | 0.60-0.65 | 暴降模式受益有限 |

---

## 六、风险评估

| 风险 | 概率 | 缓解 |
|------|:---:|------|
| 采购计划数据难以从 ECP 提取 | 中 | 手动输入 JSON（每年1次，2小时工作量）|
| 计划≠实际（计划可能调整） | 低 | 计划特征训练时用实际事件，学习"计划偏差"模式 |
| 计划特征与 EVT 特征高度相关 | 中 | Spearman 检查 + 消融实验 |
| JSON 数据年份覆盖不足 | 低 | 优先录入 2025-2026，逐步补录历史年份 |

---

## 七、附录：采购预安排页面数据结构示例

从浏览器打开 https://ecp.sgcc.com.cn/ecp2.0/portal/#/doc/doci-purplan/2602034384651452_2020052000175277 后，页面展示的表格大致为：

| 序号 | 批次编号 | 批次名称 | 公告发布时间 | 开标时间 | 采购类型 |
|:---:|---------|---------|:---:|:---:|------|
| 1 | SG2601 | 总部实施设备、材料第一批 | 2026年1月 | 2026年2月 | 设备材料 |
| 2 | SG2602 | 总部实施服务类第一批 | 2026年1月 | 2026年2月 | 服务类 |
| 3 | SG2603 | 总部实施设备、材料第二批 | 2026年3月 | 2026年4月 | 设备材料 |
| 4 | SG2604 | 总部实施特高压工程第一批 | 2026年3月 | 2026年4月 | 特高压 |
| ... | ... | ... | ... | ... | ... |

从该表格可以提取出 `monthly_schedule`（每月批次数量 + 是否有物资类批次）。
