---
name: analyze-results
description: Analyze VMD-CatBoost experiment results — compare models, find best performers, generate paper-ready analysis. Use after "run-experiment" or when the user says "analyze results", "分析结果", "which model is best".
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
user-invocable: true
disable-model-invocation: false
---

# Analyze Results — 实验结果分析

Analyze the VMD-CatBoost experiment outputs and produce paper-ready findings.

## Step 1: Load metrics

Read `outputs/figures/metrics_summary.json` for structured evaluation metrics.

Or read the latest log file: `outputs/logs/main_*.log`

## Step 2: Identify best models

For each material, rank models by R² (primary metric) and RMSE (secondary).

Look for patterns:
- Does VMD decomposition consistently improve over pure CatBoost?
- Does LSTM fusion (Model 3) beat direct CatBoost fusion (Model 2)?
- How does VMD-SVR compare to VMD-CatBoost? (ablation: ML method choice)
- Does VMD-LSTM direct sum underperform VMD-LSTM-CatBoost? (ablation: fusion benefit)

## Step 3: Generate paper-ready tables

Format the results into a comparison table suitable for the paper:

```
| 模型 | 10KV电缆 R² | 柱上变压器 R² | 避雷器 R² | 平均R² |
|------|-----------|-------------|----------|-------|
| CatBoost | ... | ... | ... | ... |
| VMD-CatBoost | ... | ... | ... | ... |
| VMD-LSTM-CatBoost | ... | ... | ... | ... |
| VMD-LSTM | ... | ... | ... | ... |
| VMD-SVR | ... | ... | ... | ... |
```

## Step 4: Key findings for paper discussion

Summarize:
1. Which model is best overall and why
2. Does the best model vary by material category (infrastructure vs user-expansion vs emergency)?
3. What does the ablation study reveal about each component's contribution?
4. What are the practical implications for grid material procurement?

## Step 5: Check charts

Review generated figures in `outputs/figures/`:
- `prediction_comparison_{material}.png` — per-material prediction curves
- `vmd_decomposition_{material}.png` — VMD IMF waveforms
- `feature_importance_{material}.png` — which factors drive predictions
- `metrics_comparison.png` — grouped bar chart for all models
