"""灰色特征 AB 对比：NZ∈[5,15) 的稀疏物资上，无灰色 vs 有灰色 的预测效果与收益。

唯一变量是灰色先验特征（GM(1,1)拟合值+灰色残差+发展系数a+分数阶AGO）。
模型: ElasticNet-2S 与 CatBoost-2S（论文核心两阶段框架）。
复用 main.py 的特征管线，避免重复实现。
"""
import sys, io, warnings, json
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import main

MATERIALS = ['330kV油浸电磁CT', '160kN单联双线夹地线悬垂串', '70kN双联地线耐张串',
             '300kN双联双挂点V型预绞单线夹750kV导线悬垂串', 'OPGW']
DATA_FILE = 'inputs/data.xlsx'


def smape(y, p):
    return np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100


def mase(y_tr, y_te, p):
    naive = np.mean(np.abs(np.diff(y_tr)))
    return np.mean(np.abs(y_te - p)) / max(naive, 1e-9)


def run_one(df, material, grey_on, model_key):
    """跑一次：返回 (metrics, y_test, y_pred)。grey_on 控制灰色特征开关。"""
    main.USE_GREY_FEATURES = grey_on
    main.USE_PAPER_FEATURES = False  # 保证唯一变量是 grey 特征
    X_tr, y_tr, X_te, y_te, _ = main.preprocess_data(df, material)
    if model_key == 'elasticnet_2s':
        yp, _, _, _ = main.run_elasticnet_2s(X_tr, y_tr, X_te, y_te, material)
    elif model_key == 'catboost_2s':
        yp, _, _, _ = main.run_catboost_2s(X_tr, y_tr, X_te, y_te, material)
    metrics = {
        'R2': float(r2_score(y_te, yp)),
        'sMAPE': float(smape(y_te, yp)),
        'MASE': float(mase(y_tr, y_te, yp)),
        'MAE': float(mean_absolute_error(y_te, yp)),
    }
    return metrics, np.asarray(y_te), np.asarray(yp)


def main_run():
    xl = pd.ExcelFile(DATA_FILE)
    all_results = {'materials': {}}
    for mat in MATERIALS:
        df = xl.parse(mat)
        df.rename(columns=main.COLUMN_EN, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        tl = len(df) - main.N_TEST
        nz = int((df['demand'].astype(float).values[:tl] > 0).sum())

        row = {'NZ': nz, 'models': {}}
        for model_key in ['elasticnet_2s', 'catboost_2s']:
            m_off, yt, poff = run_one(df, mat, grey_on=False, model_key=model_key)
            m_on, _, pon = run_one(df, mat, grey_on=True, model_key=model_key)
            row['models'][model_key] = {
                'off': m_off, 'on': m_on,
                'delta_R2': round(m_on['R2'] - m_off['R2'], 4),
                'y_test': [float(x) for x in yt],
                'pred_off': [float(x) for x in poff],
                'pred_on': [float(x) for x in pon],
            }
        all_results['materials'][mat] = row

    # ---- 输出汇总表 ----
    print('=' * 104)
    print('灰色系统先验特征 AB 对比  —  NZ∈[5,15) 稀疏物资  (训练69月/测试12月)')
    print('=' * 104)
    for model_key, mlabel in [('elasticnet_2s', 'ElasticNet-2S'), ('catboost_2s', 'CatBoost-2S')]:
        print(f'\n[{mlabel}]')
        print(f"{'物资':<36}{'NZ':>4}  {'无灰色R²':>9} {'有灰色R²':>9} {'ΔR²':>8}  {'收益':>6}")
        print('-' * 104)
        gains = []
        for mat, r in all_results['materials'].items():
            m = r['models'][model_key]
            d = m['delta_R2']
            gains.append((mat, d, m['off']['R2'], m['on']['R2']))
        gains.sort(key=lambda x: -x[1])
        for mat, d, roff, ron in gains:
            tag = '✓✓' if d > 0.10 else ('✓' if d > 0.02 else ('—' if d >= -0.02 else '✗'))
            print(f"{mat:<36}{all_results['materials'][mat]['NZ']:>4}  {roff:>9.4f} {ron:>9.4f} {d:>+8.4f}  {tag:>6}")
        print('-' * 104)
        gpos = sum(1 for _, d, _, _ in gains if d > 0)
        gavg = np.mean([d for _, d, _, _ in gains])
        gbest = max(gains, key=lambda x: x[1])
        print(f'灰色增益为正的物资: {gpos}/{len(gains)} | 平均ΔR² = {gavg:+.4f} | 最佳: {gbest[0]} ΔR²={gbest[1]:+.4f}')

    # ---- 保存用于画图 ----
    with open('experiments/grey_top5_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print('\n结果已保存: experiments/grey_top5_results.json')


if __name__ == '__main__':
    main_run()
