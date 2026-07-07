---
name: run-experiment
description: Run the VMD-CatBoost electricity material demand forecasting experiment. Use when the user says "run experiment", "train models", "跑实验", or wants to execute model training and get results.
argument-hint: [--material cable|transformer|arrester|all]
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
user-invocable: true
disable-model-invocation: false
---

# Run Experiment — 配电网物资需求预测模型实验

Run the VMD-CatBoost experiment comparing 5 models across 3 materials.

## Quick start

```bash
cd D:\Users\dell\PycharmProjects\vmd-catboost
python main.py
```

If `.venv` is not activated:
```bash
D:\Users\dell\PycharmProjects\vmd-catboost\.venv\Scripts\python.exe main.py
```

## What the experiment does

1. Loads data from `inputs/data/data.xlsx` (3 sheets: cable/transformer/arrester)
2. Preprocesses: MinMax normalize, split train(0:24)/test(24:36)
3. Runs 5 models per material:
   - Model 1: CatBoost (baseline — 4 factors direct regression)
   - Model 2: VMD-CatBoost (VMD K=5 → 5 IMFs + 4 factors → CatBoost)
   - Model 3: VMD-LSTM-CatBoost (VMD → LSTM per IMF → CatBoost fusion)
   - Model 4: VMD-LSTM direct sum (ablation — no CatBoost fusion)
   - Model 5: VMD-SVR (ablation — SVR instead of CatBoost per IMF)
4. Outputs metrics (MSE, RMSE, MAE, R²) to console + `outputs/figures/metrics_summary.json`
5. Generates charts in `outputs/figures/`

## After execution

Read `outputs/figures/metrics_summary.json` for structured results. The console output and log file (`outputs/logs/main_*.log`) contain the full evaluation table with model rankings by R².

## Key methodological notes for the paper

- VMD is applied ONLY on training set (24 months) — test IMFs via persistence extrapolation
- Spearman correlation used for factor selection (not Pearson — handles nonlinear monotonic relationships)
- Sequence length = 6 months for LSTM window construction
- LSTM uses small-sample optimization: hidden_size=12/8, dropout=0.5, full-batch training
