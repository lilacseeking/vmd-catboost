"""诊断: 检查R²≈1的物资是否存在过拟合/数据问题"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs')
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
N_TEST = 12

targets = ['交流避雷器', '电抗器保护']

xl = pd.ExcelFile(DATA_FILE)
for mat in targets:
    if mat not in xl.sheet_names:
        print(f"[SKIP] {mat} not found")
        continue
    df = xl.parse(mat)
    # find demand column
    demand_col = None
    for c in df.columns:
        if '需求' in c or 'demand' in c.lower():
            demand_col = c
            break
    if demand_col is None:
        print(f"[SKIP] {mat}: no demand column")
        continue

    seq = df[demand_col].fillna(0).values.astype(float)
    train_seq = seq[:-N_TEST]
    test_seq = seq[-N_TEST:]

    print(f"\n{'='*60}")
    print(f"  物资: {mat}")
    print(f"{'='*60}")
    print(f"  总长度: {len(seq)}, 训练: {len(train_seq)}, 测试: {N_TEST}")
    print(f"  训练集: mean={np.mean(train_seq):.2f}, std={np.std(train_seq):.2f}, "
          f"min={np.min(train_seq):.1f}, max={np.max(train_seq):.1f}")
    print(f"  测试集: mean={np.mean(test_seq):.2f}, std={np.std(test_seq):.2f}, "
          f"min={np.min(test_seq):.1f}, max={np.max(test_seq):.1f}")
    print(f"  训练集非零月数: {np.sum(train_seq > 0)}/{len(train_seq)}")
    print(f"  测试集非零月数: {np.sum(test_seq > 0)}/{N_TEST}")
    print(f"  测试集原始值: {test_seq.tolist()}")
    print(f"  训练集最后12月: {train_seq[-12:].tolist()}")

    # 检查测试集方差是否极小 (R²高的常见原因)
    ss_tot = np.sum((test_seq - np.mean(test_seq))**2)
    print(f"  测试集SS_tot: {ss_tot:.4f}")
    if ss_tot < 1e-6:
        print(f"  ⚠️ 测试集方差≈0, R²无意义!")

    # 检查是否存在明显趋势/周期
    print(f"  训练集前12月: {train_seq[:12].tolist()}")
    print(f"  训练集中间12月(30-42): {train_seq[30:42].tolist()}")
