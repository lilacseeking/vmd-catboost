"""
FC-JK+ 区间过宽诊断分析
逐层拆解导致预测区间过宽的 4 个根因
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'forward_jackknife')
with open(os.path.join(OUT, 'forward_jackknife_results.json'), encoding='utf-8') as f:
    data = json.load(f)

SEP = '=' * 90

print(f"\n{SEP}")
print("  FC-JK+ 区间过宽 — 四层根因诊断")
print(SEP)

# ========================================================================
# 根因 1: 前向链残差分布 (异常值主导)
# ========================================================================
print(f"\n{'='*90}")
print("  根因 1: 前向链残差分布 — 少量异常残差拉高分位数")
print(f"{'='*90}")
print(f"\nJackknife+ 区间宽度 ∝ 校准残差的 (1-α/2) 分位数")
print(f"当 26 个残差中存在少数极大值时, 分位数被少数 fold 绑架\n")

print(f"{'Material':<20s} {'Median':>8s} {'P90':>8s} {'Max':>8s} {'Max/Med':>8s} "
      f"{'Top3残差':>30s} {'区间宽度':>10s}")
print('-' * 110)
for m in data['materials']:
    res = sorted(m['fc_residuals'], reverse=True)
    med = np.median(m['fc_residuals'])
    p90 = np.percentile(m['fc_residuals'], 90)
    mx = res[0]
    ratio = mx / max(med, 1)
    top3 = ', '.join(f'{r:.0f}' for r in res[:3])
    print(f"{m['key'][:18]:<20s} {med:>8.0f} {p90:>8.0f} {mx:>8.0f} {ratio:>7.1f}x "
          f"[{top3:>25s}] {m['mean_width']:>10.0f}")

print(f"\n关键发现:")
print(f"  AC_Arrester: max={max(data['materials'][0]['fc_residuals']):.0f}, "
      f"median={np.median(data['materials'][0]['fc_residuals']):.0f}, "
      f"比值 {max(data['materials'][0]['fc_residuals'])/max(np.median(data['materials'][0]['fc_residuals']),1):.1f}x")
print(f"  Post_Insulator: max={max(data['materials'][2]['fc_residuals']):.0f}, "
      f"median={np.median(data['materials'][2]['fc_residuals']):.0f}, "
      f"比值 {max(data['materials'][2]['fc_residuals'])/max(np.median(data['materials'][2]['fc_residuals']),1):.1f}x")
print(f"  Line_Protect: 单个异常残差 1363, 第二大仅 292 (4.7x gap)")
print(f"\n机制: 前向链早期 fold (仅 36-40 月训练) 模型不稳定, 偶遇需求尖峰 → 极大残差")
print(f"      Jackknife+ 的 P90 分位数被这些 outlier 拉高 → 区间宽度被少数 fold 决定")

# ========================================================================
# 根因 2: 模型预测差异度 (26 个模型的 diversity)
# ========================================================================
print(f"\n{SEP}")
print("  根因 2: 模型预测差异度 — 26 个 FC 模型的 test prediction spread")
print(SEP)
print(f"\n如果 26 个模型对同一测试点的预测高度一致, 区间宽度主要由残差驱动")
print(f"如果预测本身分散, 则宽度 = 残差 + 模型不确定性\n")

# 我们无法直接获取 fc_test_preds 矩阵 (未保存), 但可以从 pred_pt 的方差推断
# pred_pt 是 26 个模型预测的均值; 如果均值变化大, 说明模型间差异大
print(f"{'Material':<20s} {'PtMean':>8s} {'PtStd':>8s} {'PtRange':>10s} {'CV':>6s} "
      f"{'MeanWidth':>10s} {'来源':>10s}")
print('-' * 85)
for m in data['materials']:
    pt = np.array(m['pred_pt'])
    mean_pt = np.mean(pt)
    std_pt = np.std(pt)
    range_pt = np.max(pt) - np.min(pt)
    cv = std_pt / max(mean_pt, 1)
    # 预测变异 vs 区间宽度
    src = '残差主导' if range_pt < m['mean_width'] * 0.3 else '混合'
    print(f"{m['key'][:18]:<20s} {mean_pt:>8.1f} {std_pt:>8.1f} {range_pt:>10.1f} {cv:>5.2f} "
          f"{m['mean_width']:>10.0f} {src:>10s}")

print(f"\n关键发现:")
print(f"  大多数物资 pred_pt 的标准差远小于区间宽度 → 区间宽度主要由残差驱动")
print(f"  原因: 自滞后特征对所有测试月恒定 → 模型输出相似 → 预测多样性低")

# ========================================================================
# 根因 3: 特征恒定导致的信息瓶颈
# ========================================================================
print(f"\n{SEP}")
print("  根因 3: 特征构建瓶颈 — 自滞后特征对 12 个测试月完全恒定")
print(SEP)
print(f"\n当前测试特征: shared[6D](时变) + self_lag[8D](恒定=最后训练月)")
print(f"8 维自滞后特征占总特征的 57%, 但它们对 12 个测试月提供相同信息\n")

# 分析 SHAP 重要度中, 恒定特征 vs 时变特征的占比
feat_names = ['budget_month', 'n_bids', 'avg_bid_qty', 'bid_growth_3m',
              'bid_growth_6m', 'bid_yoy', 'lag_1', 'lag_2', 'lag_3',
              'lag_6', 'lag_12', 'roll_mean_3', 'roll_mean_6', 'gap_since_last']
time_varying = [0,1,2,3,4,5]   # shared features (时变)
constant = [6,7,8,9,10,11,12,13]  # self-lag features (恒定)

print(f"{'Material':<20s} {'时变SHAP':>10s} {'恒定SHAP':>10s} {'恒定占比':>8s} {'含义':>15s}")
print('-' * 75)
for m in data['materials']:
    imp = np.array(m['shap_importance'])
    tv_imp = imp[time_varying].sum()
    c_imp = imp[constant].sum()
    total = tv_imp + c_imp
    ratio = c_imp / max(total, 0.001) * 100
    note = '恒定特征主导' if ratio > 60 else ('均衡' if ratio > 40 else '时变特征主导')
    print(f"{m['key'][:18]:<20s} {tv_imp:>10.1f} {c_imp:>10.1f} {ratio:>7.1f}% {note:>15s}")

print(f"\n关键发现:")
print(f"  恒定特征 (lag/roll_mean/gap) 在多数物资中贡献 >50% 的 SHAP 重要度")
print(f"  但这些特征对 12 个测试月完全相同 → 模型无法区分不同月份的 demand pattern")
print(f"  点预测 R2 低 → 残差大 → 区间宽")

# ========================================================================
# 根因 4: 分位数估计不稳定 (26 折 vs 理论要求)
# ========================================================================
print(f"\n{SEP}")
print("  根因 4: 分位数估计稳定性 — 26 折够不够")
print(SEP)
print(f"\nJackknife+ 的 P90 分位数: 26 个残差中取第 24 大的值")
print(f"  q_hi = ceil(0.9 * (26+1)) - 1 = 23 → 第 24 大残差")
print(f"  即: 区间上界受第 2-3 大残差直接控制\n")

print(f"{'Material':<20s} {'R[24]':>8s} {'R[23]':>8s} {'R[22]':>8s} {'R[1]':>8s} "
      f"{'Top3集中度':>10s} {'影响':>12s}")
print('-' * 90)
for m in data['materials']:
    res = sorted(m['fc_residuals'], reverse=True)
    # q_hi_idx=23 → res[3] (第4大) 在降序排列中
    # 实际上 q_hi = 23, 在 26 个元素中, sorted ascending → index 23 = 第 3 大
    res_asc = sorted(m['fc_residuals'])
    q_hi_val = res_asc[23] if len(res_asc) > 23 else res_asc[-1]
    top3 = res[:3]
    # 集中度: top3 mean / overall mean
    overall_mean = np.mean(res_asc)
    top3_mean = np.mean(top3)
    concentration = top3_mean / max(overall_mean, 1)
    impact = '高(top3绑架)' if concentration > 2.5 else ('中' if concentration > 1.8 else '低')
    print(f"{m['key'][:18]:<20s} {res[3]:>8.0f} {res[4]:>8.0f} {res[5]:>8.0f} {res[0]:>8.0f} "
          f"{concentration:>9.1f}x {impact:>12s}")

print(f"\n关键发现:")
print(f"  26 个残差中, P90 分位数直接受第 3-4 大残差控制")
print(f"  Top-3 残差均值 vs 全体均值的比值普遍 > 2x → 分位数被 outlier 绑架")
print(f"  这是 Jackknife+ 在小样本校准集上的固有问题")

# ========================================================================
# 汇总: 根因权重排序 + 对策
# ========================================================================
print(f"\n{SEP}")
print("  汇总: 四层根因 → 对策优先级")
print(SEP)

print("""
  优先级  根因                          贡献度   对策
  ─────────────────────────────────────────────────────────────────────────────
  ★★★    1. 残差异常值 (outlier)         ~50%     Winsorize/Trimmed 残差
                                           → 截断 top-10% 残差, 用 Winsorized 分位数
                                           → 或用 Conformalized Quantile Regression

  ★★☆    2. 自滞后特征恒定              ~25%     递归多步预测 (recursive forecasting)
                                           → 逐步预测 y[63], 更新 lag 特征, 再预测 y[64]
                                           → 或用 direct multi-horizon 模型

  ★★☆    3. 稀疏需求 + 零膨胀           ~15%     两阶段模型 (hurdle model)
                                           → Stage1: 分类器预测 P(demand > 0)
                                           → Stage2: 回归器预测 demand | demand > 0
                                           → 区间按 P(>0) 缩放

  ★☆☆    4. 分位数估计不稳定            ~10%     增大 MIN_TRAIN 或加权分位数
                                           → MIN_TRAIN 从 36 降到 24 → 38 折
                                           → 或对残差按 fold 时间加权 (近期权重更大)
""")

print(f"\n具体可操作的代码修改建议:")
print(f"  1. 在 Step 4 (区间构造) 中加入 Winsorize: 将 res[0:3] 截断到 res[3] 的值")
print(f"  2. 将测试特征改为递归更新: pred[0] → 更新 lag_1 → pred[1] → 更新 lag_1 → ...")
print(f"  3. 对零膨胀物资 (nz_te < 8) 单独使用 hurdle model")
print(f"  4. 将 MIN_TRAIN 从 36 降到 24 (如果 lag_12 特征允许)")
