## 配电网物资需求预测系统 - 第一性原理审计报告（代码级）

审计日期: 2026-07-08 (首次) → 2026-07-08 (第二轮复审)
审计范围: main.py 全部 1759 行代码 + inputs/data.xlsx 5 个物资数据
审计方法: 逐函数代码走读 + 数值实验验证 + 数据质量分析

---

### 〇、修复状态总览

| 编号 | 原始问题 | 状态 | 说明 |
|------|---------|------|------|
| Bug #1 | log1p 反变换缺失 (expm1) | **已修复** | preprocess_data 不再使用 log1p/MinMaxScaler 变换 y |
| Bug #2 | roll3_te 滚动均值数据泄露 | **已修复** | `idx-3:idx` 仅使用历史值 |
| Bug #3 | SARIMA 在 log 空间拟合 | **已修复** | SARIMA 直接在原始需求空间拟合 |
| Bug #4 | N-HiTS 被阈值跳过 | **未修复** | `len(X_tr) < 50` 阈值不变，仍然跳过 |
| Bug #5 | NaiveSeasonal/Persistence log 空间反归一化 | **已修复** | 不再需要反归一化 |
| Issue #6 | 5/6 因子全局相同 | **部分修复** | 移除 has_batch/digital_bids，但 3/4 因子仍全局相同 |
| Issue #7 | 数据高度稀疏 | **未修复** | 35-38% 零值率 + CV > 1 不变 |
| Issue #8 | project_count 一枝独秀 | **未修复** | 仍然是唯一有区分度的因子 |
| Issue #9 | CatBoost 基线注释与实现不符 | **新形态** | run_catboost 被二次定义，语义混乱 |
| Issue #10 | N-HiTS 仅用单变量需求序列 | **未修复** | 本地实现仍只使用 demand |
| Issue #11 | 评估指标体系不完整 | **部分修复** | sMAPE/MASE 已在汇总表展示 |
| Issue #12 | 无交叉验证 | **未修复** | 仍为单一 train/test 分割 |
| 新增 | Croston-SBA 间歇性需求基线已加入 | **正面改进** | — |
| 新增 | LightGBM 对比模型已加入 | **正面改进** | — |
| 新增 | VMD 模型循环论证已识别并禁用 | **正面改进** | 但有潜伏 Bug |

---

### 一、已修复的致命 Bug（3 个致命 + 1 个严重）

#### Bug #1: log1p 反变换缺失 (expm1) — ✅ 已修复

**修复方式**: `preprocess_data()` (L236-296) 彻底重构。不再对 y 做 `log1p` + `MinMaxScaler` 变换，直接使用原始需求值。注释明确写道"树模型不需要 y 归一化"。

**影响**: 所有模型的预测值和评估指标现在在同一个尺度上（原始需求量），expm1 问题从架构层面消除。

#### Bug #2: roll3_te 滚动均值数据泄露 — ✅ 已修复

**修复方式**:
- 训练集 (L254): `np.mean(seq[max(0,i-3):i])` — 仅使用 i-3 到 i-1 的历史值
- 测试集 (L280): `np.mean(demand_raw[max(0,idx-3):idx])` — 仅使用 idx-3 到 idx-1 的历史值

**验证**: Python 切片 `idx-3:idx` 右端开，不包含 `idx` 本身。修复正确。

#### Bug #3: SARIMA 在 log 空间拟合 — ✅ 已修复

**修复方式**: y 不再经过 log1p 变换，`baseline_sarima()` (L737-750) 直接在原始需求空间拟合。量纲一致。

#### Bug #5: NaiveSeasonal/Persistence log 空间反归一化 — ✅ 已修复

**修复方式**: y 不再变换，基线模型直接返回原始尺度的预测值。`baseline_naive_seasonal()` (L723-728) 和 `baseline_persistence()` (L730-735) 直接返回原始值。

---

### 二、未修复的 Bug

#### Bug #4: N-HiTS 被阈值跳过 — ❌ 未修复

**位置**: `run_nhits()` L1527

**当前代码**:
```python
if len(X_tr) < 50:
    logger.warning(f'  [N-HiTS] samples={len(X_tr)}<50, skip (小样本不稳定)'); return None,None,None,None
```

**实际数据**: `train_len=69`, `lookback=24`, `horizon=12` → `n_samples = 69-24-12 = 33`。33 < 50 → N-HiTS 在所有 5 个物资上仍然被跳过。

**附加问题**: N-HiTS 仍使用本地 PyTorch 实现（`NHitsBlock`/`NHitsModel` at L1488-1510），`requirements.txt` 中已添加 `darts>=0.30.0` 但代码中无任何 import darts。

---

### 三、新发现的 Bug

#### Bug #6: run_catboost 函数双重定义 — 严重语义混乱 🔴

**位置**: L767 和 L1353 两次定义 `run_catboost`

**第一次定义** (L767):
```python
def run_catboost(...):
    """模型一: 仅使用原始4因子(无特征工程)，CatBoost基线回归预测"""
    # 实际使用了全部 13 维特征
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    y_pred = model.predict(X_te_raw)
```

**第二次定义** (L1353):
```python
def run_catboost(...):
    """TwoStage-CatBoost: 两阶段=分类×CatBoost回归"""
    yp, _ = _two_stage_fit_predict(X_train_factors, y_train, X_test_factors, reg)
```

Python 中后定义的函数覆盖先定义的。`main()` 在 L1635 调用 `run_catboost` 时执行的是 L1353 的 TwoStage 版本。

**影响**:
1. L767 的代码是死代码，永远不会执行
2. 汇总表中 "CatBoost" 标签实际对应 TwoStage-CatBoost
3. "CatBoost" 和 "TwoStage" (L1382 的 `run_two_stage`) 是两个几乎相同的 TwoStage 实现，失去了真正的朴素 CatBoost 基线对比点

**建议**: 删除 L767 死代码，或将 L1353 重命名为 `run_twostage_catboost()`，并恢复一个真正的朴素 CatBoost 基线。

#### Bug #7: VMD/Transformer 超参字典键名与物资标识不匹配 — 潜伏崩溃 🔴

**位置**: L109-117

```python
VMD_ALPHA_MAP = {'ac_arrester': 4000, 'cvt': 2000, 'post_insulator': 3000}
TF_MULTI_DIM = {'ac_arrester': 32, 'cvt': 24, 'post_insulator': 32}
TF_NLAYERS = {'ac_arrester': 2, 'cvt': 2, 'post_insulator': 2}
# ... 等等
```

实际物资标识（data.xlsx sheet 名）是中文：`交流避雷器`、`电容式电压互感器`、`交流支柱绝缘子`、`断路器保护`、`电抗器保护`。

当 VMD/Transformer 函数调用 `VMD_ALPHA_MAP[material]` 时，`material` 是中文字符串，会触发 `KeyError`。

**当前影响**: VMD 模型已被禁用（L1714-1715），Bug 不会触发。但如果未来重新启用将导致崩溃。Transformer 超参字典有 `.get(material, default)` 回退机制（如 L888 `TF_MULTI_DIM[material]` 无 fallback，会直接崩溃）。

**建议**: 统一键名（改为中文物资名或物资索引），或添加 `.get()` 回退默认值。

---

### 四、部分修复的设计问题

#### Issue #6: 因子全局相同 — 部分修复

**改进**: FACTOR_NAMES 从 6 个缩减为 4 个，移除了 `has_batch` 和 `digital_bids`（对所有物资完全相同且相关性 < 0.27 的因子）。

**残留问题**: 数据验证结果——当前 4 个因子中仍有 3 个对所有物资全局相同：
```
project_count:      all_identical=False  ← 唯一有区分度
transformer_bids:   all_identical=True
monthly_bid_count:  all_identical=True
uhv_bids:           all_identical=True
```

#### Issue #9: CatBoost 基线注释与实现不符 — 新形态

**原始问题**: 注释说"仅使用原始4因子"但实际用全部特征。
**当前状态**: 第一个 `run_catboost`（L767）注释说"仅4因子"但实际用 13 维特征；第二个定义（L1353）覆盖了第一个，变为 TwoStage。函数定义冲突成为新形态的问题。

#### Issue #11: 评估指标体系不完整 — 部分修复

**改进**: `print_metrics_table()` (L1551-1583) 展示 MSE、RMSE、sMAPE、MASE、R2 五项指标。

**残留问题**: 缺少零值预测准确率、方向准确率、统计显著性检验。

---

### 五、未修复的设计问题

#### Issue #7: 数据高度稀疏 — ❌ 未修复

| 物资 | 0值比例 | CV | 均值 |
|------|---------|-----|------|
| 交流避雷器 | 35.8% | 1.58 | 952.6 |
| 电容式电压互感器 | 35.8% | 1.07 | 112.6 |
| 交流支柱绝缘子 | 38.3% | 1.23 | 838.4 |
| 断路器保护 | 38.3% | 1.79 | 15.0 |
| 电抗器保护 | 38.3% | 1.39 | 9.6 |

**部分缓解**: 统一 TwoStage 框架更好地处理了零值。

#### Issue #8: project_count 一枝独秀 — ❌ 未修复

#### Issue #10: N-HiTS 仅用单变量需求序列 — ❌ 未修复

#### Issue #12: 无交叉验证 — ❌ 未修复

仍为单一 69/12 月 train/test 分割，12 个测试点中约 4 个为零值。

---

### 六、正面改进

#### 改进 #1: VMD 循环论证已识别并禁用

main() L1714-1715:
```python
# [DISABLED] 循环论证: VMD分解y->IMF作特征->预测y, Sigma(IMF)~=y
```
这是对 VMD 分解方法论层面的根本性质疑的正确回应。

#### 改进 #2: Croston-SBA 间歇性需求基线已加入

`run_croston_sba()` (L1441-1466) 实现 SBA 修正的 Croston 方法——学术界公认的间歇性需求预测标准方法。

#### 改进 #3: LightGBM 对比模型已加入

`run_lightgbm()` (L1470-1485) 使用 TwoStage 框架 + LightGBM，作为 CatBoost 对照。

#### 改进 #4: 统一 TwoStage 框架

所有主模型（CatBoost、CondCatBoost、TwoStage、LightGBM）统一使用两阶段预测框架。

#### 改进 #5: 特征工程重构

`preprocess_data()` 完全重写，消除了 log1p + MinMaxScaler 对 y 的变换。13 维特征结构清晰。

---

### 七、仍需关注的问题清单

#### 代码级

| 优先级 | 问题 | 位置 | 建议 |
|--------|------|------|------|
| 高 | run_catboost 双重定义 | L767 + L1353 | 删除死代码或重命名，恢复朴素基线 |
| 高 | VMD_ALPHA_MAP / TF_* 键名不匹配 | L109-117 | 更新键名或添加 .get() 回退 |
| 中 | N-HiTS 阈值 50 | L1527 | 降低到 10 或缩短 lookback |
| 中 | N-HiTS 未迁移到 darts | L1488-1549 | 用 darts NHiTSModel 替代 |
| 低 | darts 在 requirements.txt 但未使用 | requirements.txt L3 | 使用或移除 |
| 低 | get_top_factors 选择无意义 | L205-232 | 4 个因子选 top-4 无选择余地 |

#### 评估级

| 优先级 | 问题 | 建议 |
|--------|------|------|
| 高 | 无交叉验证 | 滚动窗口回测（至少 3 折） |
| 中 | 缺少零值预测准确率 | 添加 "正确预测需求=0" 的比例指标 |
| 中 | 缺少统计显著性检验 | 添加 Diebold-Mariano 或 Wilcoxon 检验 |
| 低 | 缺少方向准确率 | 添加需求变化方向准确率 |

---

### 八、结论

**5 个原始致命/严重 Bug 中 4 个已修复**（expm1 通过架构重构消除，roll3_te 泄露已修正，SARIMA 和基线模型已正常）。**N-HiTS 阈值问题仍未修复**。

**新增 2 个代码级 Bug**: run_catboost 双重定义（影响当前运行——"CatBoost"标签实际是 TwoStage），VMD/Transformer 超参字典键名不匹配（潜伏 Bug，当前被禁用不触发）。

**架构层面显著改善**: VMD 循环论证已禁用、Croston-SBA 和 LightGBM 已加入、统一 TwoStage 框架。但核心方法论问题（时间序列 vs 离散事件预测）仍存在——详见 DESIGN_AUDIT_REPORT.md。

---

## 九、2026-07-11 代码清理

从 main.py 移除以下模型族（2096 → 979 行，精简 53%）。每条删除基于"模型架构假设与项目数据特征的根本性不匹配"：

| 移除内容 | 理由 |
|---------|------|
| VMD 全部函数（含 run_vmd_catboost/transformer/SVR） | 循环论证：ΣIMF≈y，用 IMF 预测 y 等于用答案反推答案 |
| Transformer 基础设施（PositionalEncoding/ProbSparseAttention/MultiFeatureTransformer 等） | 自注意力在 45 训练窗口下无法学习有意义的注意力模式（576 pairwise 关系 vs 45 样本） |
| DLinear/DLinear-2S | lookback=24 时趋势/季节分量在确定性零值模式下退化，参数/样本比 0.5:1 |
| ModernTCN/ModernTCN-2S | 参数/样本比 >2:1，教科书级过拟合 |
| Theta/SES | 模型假设"信号+i.i.d.噪声"，ECP 零值是确定性批次节奏（"事件+空白"） |
| TSB | 与 Croston-SBA 同族，对确定性零值问题的回答必然一致，不增加信息 |
| LightGBM-pure | 单阶段回归在零膨胀数据上的结构性劣势；正确消融是 CatBoost vs CatBoost-2S |
| GP-2S | 核超参数在 36 个 Stage 2 样本上不可辨识（Rasmussen & Williams, 2006） |
| 旧版 run_catboost（L776 第一定义） | 被 L1505 第二定义覆盖，死代码 |

同时删除 `scripts/` 下 7 个独立实验脚本（route_a/b/c 系列 + run_new_models.py），其核心发现已记录在 reports/ 中。
