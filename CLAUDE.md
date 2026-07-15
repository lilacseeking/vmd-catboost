# CLAUDE.md

## 项目目的

配电网物资需求预测模型对比实验。基于国网 ECP2.0 平台采购数据，对多种模型在 5 种物资的月度需求预测任务上进行对比评估，为期刊论文提供实验支撑。

## 命令

```bash
pip install -r requirements.txt
python main.py                    # 运行全量实验
python main.py --data data.xlsx   # 指定数据文件
```

## 架构

### 数据流

```
inputs/data/data.xlsx (5 sheets, 各代表一种物资)
  → load_or_generate_data()     → data_dict[material] = DataFrame
  → get_top_factors()           → 每物资 Spearman 动态选择 Top-4 因子
  → preprocess_data()           → 训练/测试分割 + lag/rolling特征 + 因子缩放
  → 13 个模型逐一训练评估       → metrics + 图表
```

### 模型清单（13 个，全部在 `main.py` 中）

| 类别 | 模型 | 函数 | 说明 |
|------|------|------|------|
| **核心** | CatBoost | `run_catboost` | TwoStage-CatBoost：两阶段框架 + CatBoost 回归 |
| **消融** | CatBoost-2S | `run_catboost_2s` | CatBoost 两阶段变体（不同超参数） |
| **消融** | CondCatBoost | `run_conditional_catboost` | 更强 CatBoost 参数配置 |
| **对照** | TwoStage | `run_two_stage` | 独立两阶段实现（不依赖 `_two_stage_fit_predict` 框架） |
| **线性** | Ridge-2S | `run_ridge_2s` | 两阶段 + Ridge 回归（小样本更稳定） |
| **线性** | ElasticNet-2S | `run_elasticnet_2s` | 两阶段 + ElasticNet（L1+L2 正则化） |
| **对比** | LightGBM | `run_lightgbm` | 两阶段 + LightGBM（树模型族对比） |
| **深度** | N-HiTS | `run_nhits` | 多尺度层次化预测（当前因样本 <50 被跳过） |
| **基线** | NaiveSeasonal | `baseline_naive_seasonal` | 去年同期值 |
| **基线** | NaiveMean | `baseline_naive_mean` | 历史均值 |
| **基线** | Persistence | `baseline_persistence` | 最后观测值 |
| **基线** | SARIMA | `baseline_sarima` | 经典统计模型 |
| **基线** | Croston-SBA | `run_croston_sba` | 间歇性需求标准方法 |

### 两阶段框架

核心函数 `_two_stage_fit_predict`：Stage 1 用 CatBoostClassifier 预测某月是否有需求（P），Stage 2 在非零样本上训练回归器预测需求量（Q）。最终预测 ŷ = P × Q。

### 评估指标

MSE、RMSE、MAE、R²、sMAPE、MASE、zero_acc（零值预测准确率）。

### 输出

`outputs/figures/`：预测对比图、特征重要性、指标对比柱状图、需求曲线
`outputs/logs/`：时间戳日志

## 已知关键结论

- **VMD 系列已移除（2026-07-11）**：循环论证 —— ΣIMF ≈ y，用 IMF 预测 y 是方法论错误
- **Transformer/DLinear/ModernTCN 已移除（2026-07-11）**：74 个月数据对深度学习时序模型差两个数量级
- **Theta/SES/TSB 已移除（2026-07-11）**：指数平滑族假设随机噪声，ECP 零值来自确定性的批次招标节奏
- **GP-2S 已移除（2026-07-11）**：36 个 Stage 2 非零样本不足以可靠学习核超参数
- **LightGBM-pure 已移除（2026-07-11）**：单阶段回归无法处理零膨胀数据，两阶段版已展示正确处理方式
- 最佳单模型 R² ≈ 0.35-0.66（SGCC 数据，5 种物资）
- 两阶段框架对大需求物资提升显著（+0.35~+0.47 R²），小需求物资退化
- Ridge-2S 在极小样本（36 非零月）上有时优于 CatBoost——线性模型的低方差优势
- ECP 零值不是随机间歇（Croston-SBA R² 全负证明）——是确定性的批次招标节奏
- 4 个外部因子中 3 个是全局变量（对所有物资相同），模型本质是 fancy 自回归
- 上游数据存在致命问题 demand_month=公告月（非交货月）和单位未归一化（bidding-ecp-data 项目）

## scripts/ 目录

| 文件 | 用途 |
|------|------|
| `batch_event_model.py` | 批次事件三层预测模型（后续方向） |
| `build_highfreq_data.py` | 高密度物资数据准备 |
| `compile_investment_data.py` | 投资数据编译 |
| `conformal_prediction.py` | 共形预测不确定性量化 |
| `jackknife_conformal.py` | Jackknife+ 共形预测 |
| `ensemble_analysis.py` | 集成分析 |
| `visualize_top10.py` | Top-10 物资 EDA 可视化 |
| `visualize_top8_materials.py` | Top-8 物资 EDA 可视化 |
| `logger_utils.py` | 日志工具（conformal 系列依赖） |

## 清理记录

**2026-07-11**：从 main.py 移除以下模型族（从 2096 行精简至 979 行）：
- VMD 全系列（循环论证）
- Transformer 全部基础设施（样本效率不足以支持 self-attention）
- DLinear/ModernTCN（参数/样本比失衡）
- Theta/SES/TSB（模型类与确定性批次数据不匹配）
- LightGBM-pure（单阶段在零膨胀数据上的结构性劣势）
- GP-2S（核超参数在 36 样本上不可辨识）
- 旧版 run_catboost（代码重复）

同时删除 scripts/ 下 7 个 Route ABC 实验脚本，其核心发现记录在 reports/ 中。
