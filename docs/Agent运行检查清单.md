# Agent 运行效果检查清单

> 创建日期: 2026-07-13
> 用途: Agent（任何角色）在执行完代码修改或模型运行后，必须逐条核对本清单。防止Agent"失忆"——漏掉关键步骤、跳过必跑模型、声称完成了但其实没跑全。
> 
> **规则**: 本清单是强制性的。Agent 在声称"运行完成"或"任务完成"前，必须逐条标注 ✅/❌/⚠️。

---

## 一、数据源完整性检查

| # | 检查项 | 要求 | 检查方法 |
|---|--------|------|---------|
| 1.1 | **国家电网(SGCC)数据存在** | `bidding-ecp-data/data/ecp_data.db` 必须存在且 >100MB | `ls -la data/ecp_data.db` |
| 1.2 | **冀北公司数据存在** | `bidding-ecp-data/data_jibei/ecp_data.db` 必须存在 | `ls -la data_jibei/ecp_data.db` |
| 1.3 | **训练数据XLSX已生成** | `vmd-catboost/inputs/data.xlsx` 必须存在，包含5个Sheet | `python -c "import openpyxl; wb=openpyxl.load_workbook('inputs/data.xlsx'); print(wb.sheetnames)"` |
| 1.4 | **两家公司的5种物资各不缺失** | SGCC和冀北各5种物资，不允许某公司只有3种 | 在log中搜索"物资:"行，确认列出5个 |

---

## 二、模型运行完整性检查（缺一不可）

### 2.1 必须运行的模型清单

以下模型在每次实验运行中**必须全部被执行**，不允许跳过（除非环境不支持的 N-HiTS/Chronos）：

| # | 模型 | 组别 | 检查方法 |
|---|------|------|---------|
| 2.1.1 | CatBoost | 核心模型 | log中搜索"CatBoost (两阶段)" |
| 2.1.2 | CatBoost-2S | 消融 | log中搜索"CatBoost-2S:" |
| 2.1.3 | CondCatBoost | 消融 | log中搜索"CondCatBoost" |
| 2.1.4 | TwoStage | 对照 | log中搜索"TwoStage (独立实现)" |
| 2.1.5 | Ridge-2S | 线性 | log中搜索"Ridge-2S"(或确认Ridge-2S在指标汇总表中出现) |
| 2.1.6 | ElasticNet-2S | 线性 | log中搜索"ElasticNet-2S"(或确认在汇总表中出现) |
| 2.1.7 | LightGBM | 对比 | log中搜索"LightGBM:" |
| 2.1.8 | N-HiTS | 深度（允许跳过） | 允许"N-HiTS: 跳过"但必须出现在log中 |
| 2.1.9 | NaiveSeasonal | 基线 | log中搜索"NaiveSeasonal:" |
| 2.1.10 | NaiveMean | 基线 | log中搜索"NaiveMean:" |
| 2.1.11 | Persistence | 基线 | log中搜索"Persistence:" |
| 2.1.12 | SARIMA | 基线 | log中搜索"SARIMA:" |
| 2.1.13 | Croston-SBA | 基线 | log中搜索"Croston-SBA:" |
| 2.1.14 | Chronos-2 | TSFM基线（允许跳过） | 允许"Chronos-2: 跳过"但必须出现在log中 |

### 2.2 物资覆盖检查

| # | 检查项 | 要求 |
|---|--------|------|
| 2.2.1 | **SGCC 5种物资全部运行** | 指标汇总表中必须出现5个物资行的SGCC数据 |
| 2.2.2 | **冀北 5种物资全部运行** | 如果使用冀北数据源，同上 |
| 2.2.3 | 不允许"只跑了3种就声称完成" | 核对汇总表的物资行数 |

### 2.3 两家公司对比检查

| # | 检查项 | 要求 |
|---|--------|------|
| 2.3.1 | **SGCC vs 冀北指标对比** | 必须输出两家公司的R²对比。如果只跑了一家公司，必须明确声明另一家未跑 |
| 2.3.2 | --org参数正确使用 | `--org sgcc` 和 `--org jibei` 分别运行 |

---

## 三、输出产物完整性检查

| # | 检查项 | 要求 | 检查方法 |
|---|--------|------|---------|
| 3.1 | **评估指标汇总表** | log中列出所有物资×模型的R²/MSE/RMSE/sMAPE/MASE | 搜索"评估指标汇总表" |
| 3.2 | **模型性能排序** | log末尾列出每物资Top3模型 | 搜索"模型性能排序" |
| 3.3 | **图表全部生成** | `outputs/figures/` 下存在预测对比图、特征重要性图、指标对比图 | `ls outputs/figures/` |
| 3.4 | **日志文件已保存** | `outputs/logs/` 下存在时间戳命名的log | `ls outputs/logs/` |

---

## 四、代码质量检查

| # | 检查项 | 要求 |
|---|--------|------|
| 4.1 | **无语法错误** | `python -c "import py_compile; py_compile.compile('main.py', doraise=True)"` |
| 4.2 | **无import错误** | `python -c "from main import *"` 不报错 |
| 4.3 | **不引入新的硬编码路径** | 代码中不出现 `C:\Users\董文涛\` 或类似绝对路径 |
| 4.4 | **MODEL_ORDER列表与实际运行的模型一致** | `print_metrics_table` 中的 MODEL_ORDER 不能包含已删除/未运行的模型名 |

---

## 五、方法论约束检查

| # | 检查项 | 要求 |
|---|--------|------|
| 5.1 | **不引入VMD系列模型** | VMD系已因循环论证永久删除。任何Agent不得重新引入 |
| 5.2 | **不引入已删除的模型族** | Transformer/DLinear/ModernTCN/Theta/SES/TSB/GP-2S/LightGBM-pure已被删除。不得重新引入，除非有明确的审核通过的方案 |
| 5.3 | **损失函数选择** | Stage 2回归器默认使用RMSE。Tweedie/零膨胀损失仅在单阶段或有明确审核通过的场景下使用 |
| 5.4 | **不破坏 train/test split** | 绝对不能在全量数据上做scaler.fit或Spearman计算（出现就是数据泄露） |
| 5.5 | **每物资超参数已配置** | 见下方 §5.5 详细要求 ↓ |

### 5.5 每物资超参数配置强制检查

**背景**: HP_DEFAULTS + HP_OVERRIDES 两层字典为每种模型×每种物资提供独立超参数。Agent 在新增/修改模型时**必须**同步更新这些配置。

| # | 检查项 | 要求 |
|---|--------|------|
| 5.5.1 | **HP_DEFAULTS 包含所有模型的默认参数** | 每新增一个模型函数，必须在 HP_DEFAULTS 中增加对应 model_key |
| 5.5.2 | **HP_OVERRIDES 为每物资提供特化参数** | 对于低非零月（≤36月）/高CV（>1.2）的物资，必须降低 depth 和 iterations，增大 l2 防过拟合 |
| 5.5.3 | **新模型函数使用 get_hp(model_key, material)** | 禁止在函数体内硬编码超参数。必须从 HP 字典中读取 |
| 5.5.4 | **model_key 命名规范** | 使用 lowercase_with_underscore 格式。与函数名一致或一一对应 |
| 5.5.5 | **material 匹配使用子串查找** | HP_OVERRIDES 的 key 使用物资简称（如'交流避雷器'），通过 `in material` 匹配全名 |
| 5.5.6 | **SGCC 和冀北两套数据均需配置** | 如果两家公司使用不同物资集，HP_OVERRIDES 需覆盖两边的物资名 |

**超参数调优原则**（按物资数据特征）:
| 物资特征 | 非零月 | Stage1分类器 | Stage2回归器 |
|---------|:---:|------------|-------------|
| 高密度 (>40月) | 高 | depth=6~7, iters=800~2000 | depth=6~7, iters=1500~2000, l2=3~4 |
| 中密度 (36~40月) | 中 | depth=4~5, iters=500~1200 | depth=5, iters=1000~1200, l2=5~6 |
| 低密度+高CV (<36月, CV>1.2) | 低 | depth=3~4, iters=400~500, l2=6~8 | depth=4, iters=800, l2=8 |

---

## 六、文档同步检查

| # | 检查项 | 要求 |
|---|--------|------|
| 6.1 | `CLAUDE.md` 与代码一致 | 新增/删除模型后必须更新 CLAUDE.md 的模型清单 |
| 6.2 | `docs/index.md` 更新（如果bidding-ecp-data侧变动） | 新增分析文档后必须更新文档地图 |
| 6.3 | **关键决策记录到Memory** | 重要的方法论决策（引入/删除模型、数据源切换等）存入Memory |

---

## 七、声明模板（Agent完成任务时必须输出）

Agent在报告"任务完成"时，必须输出以下表格（不可跳过）：

```
| # | 检查项 | 状态 |
|---|--------|:---:|
| 1 | SGCC 5种物资全部运行 | ✅/❌ |
| 2 | 冀北 5种物资全部运行 | ✅/❌ |
| 3 | 14个模型全部执行 | ✅/❌ |
| 4 | 指标汇总表已输出 | ✅/❌ |
| 5 | 图表已生成 | ✅/❌ |
| 6 | 语法无错误 | ✅/❌ |
| 7 | CLAUDE.md已同步 | ✅/❌ |
| 8 | 无数据泄露 | ✅/❌ |
| 9 | 每物资超参数已配置(HP_DEFAULTS+HP_OVERRIDES) | ✅/❌ |

如有❌，必须说明原因和计划修复时间。
