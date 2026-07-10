Route C: 池化两阶段预测模型 — 设计文档
日期: 2026-07-10

================================================================
一、架构设计
================================================================

预测流程:
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 1: 池化分类器 (10种物资共享)                           │
  │   输入: 当前月份 + 物资标识 + 14维特征                       │
  │   输出: P(该物资本月有需求)                                   │
  │   训练数据: 10 materials × 62 months = 620行                │
  │   模型: LightGBMClassifier (class_weight='balanced')        │
  ├─────────────────────────────────────────────────────────────┤
  │ Guardrail: Stage 1 分类准确率检查                            │
  │   对每种物资: 在验证集(训练集最后12月)上计算F1              │
  │   基线: "全部预测为0"的F1-score(正样本比例)                   │
  │   通过条件: F1 > baseline_F1 × margin (margin=1.3)         │
  │   未通过: fallback到单阶段LightGBM + Optuna                  │
  ├─────────────────────────────────────────────────────────────┤
  │ Stage 2: 物资专属回归器 (每种物资独立训练)                   │
  │   输入: Stage 1判断"有需求"的月份 + 14维特征                │
  │   输出: 需求量Q                                              │
  │   模型: LightGBMRegressor + Optuna 100 trials               │
  │   训练数据: 仅非零样本                                       │
  ├─────────────────────────────────────────────────────────────┤
  │ 最终预测: ŷ = P(有需求|通过guardrail) × Q                   │
  │         或者 ŷ = P_guardrail_adjusted × Q                   │
  └─────────────────────────────────────────────────────────────┘

================================================================
二、Guardrail 设计 (用户要求的关键部分)
================================================================

2.1 Stage 1 误差来源分类:
  Stage 1 预测结果 vs 实际标签:
    ┌──────────┬────────────┬────────────┐
    │          │ Actual=1   │ Actual=0   │
    ├──────────┼────────────┼────────────┤
    │ Predict=1│ TRUE POS   │ FALSE POS  │
    │ Predict=0│ FALSE NEG  │ TRUE NEG   │
    └──────────┴────────────┴────────────┘

2.2 Stage 1 误差对最终预测的传播:
  True Positive (Predict=1, Actual=1):
    → ŷ = Q_stage2  → MSE贡献 = (Q_stage2 - actual)²
    → 纯Stage 2误差 (可接受)
  
  False Negative (Predict=0, Actual>0):
    → ŷ = 0  → MSE贡献 = actual²
    → Stage 1直接摧毁预测 — 是最致命的误差

  False Positive (Predict>0, Actual=0):
    → ŷ = Q_stage2  → MSE贡献 = Q_stage2²
    → 与True Negative (ŷ=0, Actual=0 → MSE=0)对比:
    → FP比TN多了 Q_stage2² 的误差


2.3 Guardrail 判定逻辑:
  def guardrail_pass(stage1_f1, baseline_f1, n_pos_test, material_name):
      # 条件1: F1 > baseline (即比"全预测为0"好)
      if stage1_f1 <= baseline_f1 * 1.3:
          return False, "F1 not better than baseline"
      
      # 条件2: 至少预测正确一次正类
      if n_pred_pos < 1:
          return False, "Classifier predicts all zeros"
      
      # 条件3: 假阴性率不能太高
      fnr = fn_count / max(actual_pos_count, 1)
      if fnr > 0.4:
          return False, f"FNR too high ({fnr:.1%}) — too many missed demands"
      
      return True, "PASS"

2.4 Guardrail 未通过时的改进方案:
  方案A: 提高分类阈值 (降低P阈值 → 更多预测为正 → 减少假阴性)
    代价: 增加假阳性 → 更多的"预测有需求但实际没有" → MSE增加
  
  方案B: 使用概率加权预测 (而非硬阈值)
    ŷ = P(stage1) × Q(stage2)  而非  ŷ = (P>0.5 ? Q : 0)
    优点: 假阴性的影响被减弱 (P=0.3 × Q < 0.5 × Q但>0)
    缺点: 假阳性的影响也被保留 (P=0.3 × Q 仍贡献误差)
  
  方案C: 使用两组超参数 — 分别优化分类和回归
    分类器: 更深的树 + 更高的min_child_samples (避免全部预测为0/1)
    回归器: 与Route A保持一致的超参数策略
  
  默认选择: 方案B + 方案C组合
    所有物资使用概率加权预测(不设硬阈值)
    分类器超参数专门搜索(F1最大化)
    回归器超参数同Route A(R2最大化)

================================================================
三、对比实验设计
================================================================

实验1: 主对比 — 池化两阶段 vs 单阶段基线
  模型1: Route C (LightGBMPooledClassifier + LightGBMStage2)
  模型2: Route A baseline (单阶段LightGBM + Optuna, already available)
  对比指标: 每种物资的R2差, 均值/中位数差

实验2: 消融 — 量化Stage1误差贡献
  计算每个物资的MSE分解:
    MSE_total = MSE_correct + MSE_false_neg + MSE_false_pos
    MSE_correct: 那些Stage1正确判定为"有需求"的月份 → 纯S2误差
    MSE_false_neg: Stage1判定"没有需求"但实际有 → 灾维性误差
    MSE_false_pos: Stage1判定"有需求"但实际没有 → S2预测值本身成为误差

实验3: 概率加权 vs 硬阈值对比
  概率加权: ŷ = P × Q  (所有月份都参与)
  硬阈值: ŷ = (P>0.5 ? Q : 0)  (仅Stage 1判定有需求的月份)
  对比: R2差异 → 量化"去硬阈值"的价值

实验4: "有Guardrail vs 无Guardrail"对比
  无Guardrail = 强制所有10种物资都用池化两阶段
  有Guardrail = 未通过的物资fallback到单阶段Optuna
  对比: 哪种策略整体R2更高？

================================================================
四、实现步骤 (main.py 的增量改版)
================================================================

Step A: 放入Route A已积累的改进
  - StepLogger结构化日志
  - 10种高密度物资从ECP数据库自动预选
  - Per-material Optuna超参数搜索框架
  - 14维零泄露特征工程

Step B: 实现池化两阶段模型
  - PooledStage1Classifier: 620行训练数据
  - Guardrail检查函数 (基于验证集)
  - Stage2 Regressor: 每种物资独立Optuna
  - 概率加权最终预测

Step C: 对比实验
  - Runtime自动输出对比表格
  - 消融分析: MSE分解
  - 图表: Guardrail通过/未通过的物资标记

Step D: 代码整理
  - 删除Route A/B独立脚本的导入引用
  - 删除已不再使用的模型函数
  - 保留CatBoost/TwoStage作为对比基线
