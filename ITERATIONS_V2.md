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

