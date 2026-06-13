# VMD-CatBoost 配电网物资需求预测

基于 **VMD（变分模态分解）** 与 **CatBoost** 的配电网物资月度需求量预测模型。对比五种模型架构（CatBoost / VMD-CatBoost / VMD-LSTM-CatBoost / VMD-LSTM / VMD-SVR），对三类核心配电网物资（10KV电缆、柱上变压器、避雷器）进行精确预测。

## 项目背景

配电网物资需求序列具有**非平稳、波动强**的特性，传统预测方法精度不足。本项目构建 VMD-CatBoost 复合预测模型，通过斯皮尔曼相关性分析筛选关键影响因子，利用 VMD 分解降低序列噪声，结合 LSTM 捕获时序依赖与 CatBoost 梯度提升树进行回归预测，为电网物资供应链智能调度提供决策依据。

## 五种模型

| 模型 | 核心思路 |
|------|---------|
| **CatBoost** | 4 个影响因子 → CatBoost 直接回归预测（基线） |
| **VMD-CatBoost** | VMD 分解需求量(K=5) → 5 个 IMF + 4 个因子 → CatBoost 端到端 |
| **VMD-LSTM-CatBoost** | VMD 分解 → 残差分量多特征 LSTM + 4 个模态分量单特征 LSTM → CatBoost 融合 |
| **VMD-LSTM** | VMD 分解 → 5 个 LSTM 预测各分量 → 直接求和（消融实验，验证 CatBoost 融合层必要性） |
| **VMD-SVR** | VMD 分解需求量(K=5) → 5 个 IMF + 4 个因子 → SVR（核方法对比） |

## 三类物资

| 物资 | 所属类别 | Top-4 影响因子 |
|------|---------|---------------|
| 10KV电缆 (cable) | 基建项目类 | 工程投资量、历史需求量、负荷增长量、设备进价成本 |
| 柱上变压器 (transformer) | 用户增容类 | 负荷增长量、工程投资量、历史需求量、设备进价成本 |
| 避雷器 (arrester) | 应急抢修类 | 雷击次数、台风次数、暴雨次数、负荷增长量 |

## 7 个影响因子

负荷增长量、工程投资量、历史需求量、设备进价成本、台风每月次数、雷击每月次数、暴雨每月次数

## 数据说明

- **时间范围**: 2022.01 — 2024.12（36 个月）
- **数据粒度**: 月度
- **数据存储**: `inputs/data/data.xlsx`（每种物资一个 sheet，首次运行时自动生成）
- **时序分割**: 前 24 月训练，后 12 月测试

## 环境要求

- Python 3.12
- pip 25.0

## 安装

```bash
git clone <repo-url>
cd vmd-catboost
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

**首次运行**：自动生成模拟数据并保存到 `inputs/data/data.xlsx`。  
**后续运行**：直接读取已有数据文件，跳过生成步骤。

## 输出

### 控制台
- 数据加载/生成日志
- 每种物资每个模型的 MSE / RMSE / MAE / R² 指标
- 模型性能排序

### 图表 (`outputs/figures/`)

| 文件 | 内容 |
|------|------|
| `prediction_comparison_*.png` | 每种物资独立成图，五模型预测 vs 真实值对比曲线 |
| `vmd_decomposition_*.png` | 每物资的 VMD 分解波形（5 IMF） |
| `feature_importance_*.png` | 每物资的特征重要性排序（含 CatBoost / VMD-CatBoost / VMD-LSTM-CatBoost） |
| `metrics_comparison.png` | 四指标分组柱状图（五模型对比） |
| `metrics_summary.json` | 评估指标 JSON |

### 日志 (`outputs/logs/`)

每次运行生成独立日志文件 `main_YYYYMMDD_HHMMSS.log`，记录完整执行过程。

## 项目结构

```
vmd-catboost/
├── main.py                   # 主程序（单文件，全部代码）
├── requirements.txt          # pip 依赖
├── TECHNICAL_DESIGN.md       # 技术设计文档
├── inputs/
│   └── data/
│       └── data.xlsx         # 数据文件（3 个 sheet）
├── outputs/
│   ├── figures/              # 图表输出
│   └── logs/                 # 日志文件
└── models/                   # 模型保存目录（预留）
```

## 评估指标

| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| R² | 拟合优度 |

## 技术栈

| 组件 | 库 |
|------|-----|
| VMD 分解 | vmdpy |
| 深度学习 | PyTorch |
| 梯度提升 | CatBoost |
| 数据处理 | pandas, numpy |
| 评估 | scikit-learn |
| 可视化 | matplotlib |

## 许可证

MIT License
