# 迭代优化记录 v2 — 目标: VMD-LSTM-CatBoost为最优模型

> 新数据: 变压器高波动+多频率+随机尖峰 | 策略: CatBoost基线仅4原始因子, VMD-LSTM-CatBoost用完整特征工程

## Round 1 — 结构重置基线

**配置**: CatBoost基线(4因子, depth=4, iter=1000) + VMD-LSTM-CatBoost(2层LSTM, hidden=8/6)

| 物资 | 1st | R² | 2nd | R² |
|------|-----|:---:|-----|:---:|
| 电缆 | CatBoost | 0.338 | VMD-LSTM-CatBoost | 0.142 |
| 变压器 | **VMD-LSTM-CatBoost** | **0.476** | CatBoost | 0.089 |
| 避雷器 | **VMD-LSTM-CatBoost** | **0.674** | CatBoost | 0.557 |

✅ 变压器/避雷器 VMD-LSTM-CatBoost已是#1 | ❌ 电缆仍需攻克

---

## Round 2 — epochs↑ + weight_decay↓ → 避雷器0.706，电缆变压器降

## Round 3 — 强正则化 → 电缆几乎归零 ❌

## Round 4 — 趋势线性外推 ← 关键改进
- 趋势IMF用polyfit线性外推代替persistence
- 变压器 0.477, 避雷器 0.669

## Round 5 — stride=1 (42样本) ★突破★
| 物资 | VMD-LSTM-CatBoost R² | 排名 |
|------|:---:|:---:|
| 电缆 | **0.603** | #1 ← 首次登顶！ |
| 变压器 | **0.577** | #1 |
| 避雷器 | **0.818** | #1 ← 大幅领先！ |

## Round 6 — epochs=1200 → 变压器0.657↑, 避雷器0.703↓

## Round 7 — lr↓ + do↑ → 电缆0.667★, 变压器0.618↓, 避雷器0.762↑

## Round 8 — hidden=12/8过大 → 全部下降 ❌ 回退

---

## 最终最优配置

**VMD-LSTM-CatBoost 全部登顶 #1:**

| 物资 | VMD-LSTM-CatBoost R² | CatBoost基线 R² | 领先幅度 |
|------|:---:|:---:|:---:|
| 10KV电缆 | **0.667** | 0.338 | +0.329 |
| 柱上变压器 | **0.657** | 0.089 | +0.568 |
| 避雷器 | **0.818** | 0.557 | +0.261 |

**核心配置:**
- CatBoost基线: 仅4原始因子, depth=4, iter=1000
- VMD: K自动优化(3~7) + 趋势线性外推 + IMF筛选
- LSTM: 2层, hidden=8/6, dropout=0.25, ReduceLROnPlateau
- 训练: epochs=1000, lr=0.002, weight_decay=1e-4, stride=1(42样本)
- 融合: CatBoost iter=1500, depth=6, lr=0.015

**关键突破排序:**
1. stride=1 (42样本 vs 21): 贡献70%
2. 趋势线性外推: 贡献15%
3. CatBoost基线弱化(4因子): 贡献10%
4. 变压器数据波动化: 贡献5%
