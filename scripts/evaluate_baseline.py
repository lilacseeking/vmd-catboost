#!/usr/bin/env python
"""Baseline 评测脚本：跑现有模型，产出全指标基线报告。

用法: python scripts/evaluate_baseline.py
产出: experiments/baseline_report.json + 更新 experiments/experiment_log.json

设计原则:
  - 复用 main.py 的数据加载/预处理/模型函数，不重复造轮子
  - 补充 main.py 未计算的 MAPE 指标
  - 按物资分组报告: 均值/最差/达标比例
  - 固定验证协议: 时间序列切分(前69月训练/后12月测试), 禁止shuffle
"""
import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 确保从项目根目录运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 main.py 的核心函数 (不会触发 main() 因为 __name__ != '__main__')
import main as m


def mape(y_true, y_pred):
    """MAPE: 跳过 y_true=0 的月份(间歇性需求中零值无意义)"""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_full(y_true, y_pred):
    """计算完整指标集: R²/MAPE/RMSE/MAE"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_baseline():
    """运行现有全部模型，收集基线指标。"""
    print("=" * 70)
    print("  Baseline 评测: 运行现有模型管道")
    print("=" * 70)

    # 1. 加载数据
    data_dict = m.load_or_generate_data()
    materials = list(data_dict.keys())

    # 设置 main.py 的全局变量 (函数内部依赖)
    m.MATERIALS = materials
    m.MATERIAL_LABELS = {mat: mat for mat in materials}

    print(f"  物资: {materials}")
    print(f"  数据: 每物资 {len(data_dict[materials[0]])} 月")
    print(f"  切分: 训练 {len(data_dict[materials[0]]) - m.N_TEST} 月 / 测试 {m.N_TEST} 月")
    print()

    # 2. 对每个物资跑全部已启用模型
    # 模型注册表: (名称, 调用函数, 是否需要df参数)
    all_results = {}  # {material: {model_name: {r2, mape, rmse, mae}}}

    for material in materials:
        # 跳过被禁用的物资
        if m.SKIP_REACTOR_PROTECT and '电抗器保护' in material:
            print(f"  [跳过] {material} (SKIP_REACTOR_PROTECT=True)")
            continue

        print(f"  [评测] {material}...")
        df = data_dict[material]
        X_train, y_train, X_test, y_test, scaler = m.preprocess_data(df, material)

        material_results = {}

        # --- CatBoost 直接回归 ---
        try:
            yp, yt, _, _ = m.run_catboost(X_train, y_train, X_test, y_test, material)
            material_results['CatBoost'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] CatBoost: {e}")

        # --- CatBoost-Tweedie ---
        try:
            yp, yt, _, _ = m.run_catboost_tweedie(X_train, y_train, X_test, y_test, material)
            material_results['CatBoost-Tweedie'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] CatBoost-Tweedie: {e}")

        # --- CatBoost-2S ---
        try:
            yp, yt, _, _ = m.run_catboost_2s(X_train, y_train, X_test, y_test, material)
            material_results['CatBoost-2S'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] CatBoost-2S: {e}")

        # --- CondCatBoost ---
        try:
            yp, yt, _, _ = m.run_conditional_catboost(X_train, y_train, X_test, y_test, material)
            material_results['CondCatBoost'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] CondCatBoost: {e}")

        # --- TwoStage ---
        try:
            yp, yt, _, _ = m.run_two_stage(df, X_train, y_train, X_test, y_test, material)
            material_results['TwoStage'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] TwoStage: {e}")

        # --- Ridge-2S ---
        try:
            yp, yt, _, _ = m.run_ridge_2s(X_train, y_train, X_test, y_test, material)
            material_results['Ridge-2S'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] Ridge-2S: {e}")

        # --- ElasticNet-2S ---
        try:
            yp, yt, _, _ = m.run_elasticnet_2s(X_train, y_train, X_test, y_test, material)
            material_results['ElasticNet-2S'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] ElasticNet-2S: {e}")

        # --- 基线: NaiveSeasonal ---
        try:
            yp, yt = m.baseline_naive_seasonal(y_train, y_test)
            material_results['NaiveSeasonal'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] NaiveSeasonal: {e}")

        # --- 基线: SARIMA ---
        try:
            yp, yt = m.baseline_sarima(y_train, y_test)
            material_results['SARIMA'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] SARIMA: {e}")

        # --- 基线: Croston-SBA ---
        try:
            yp, yt = m.run_croston_sba(y_train, y_test)
            material_results['Croston-SBA'] = evaluate_full(yt, yp)
        except Exception as e:
            print(f"    [ERROR] Croston-SBA: {e}")

        # --- LightGBM (如已安装) ---
        try:
            yp, yt = m.run_lightgbm(X_train, y_train, X_test, y_test, material)
            if yp is not None:
                material_results['LightGBM'] = evaluate_full(yt, yp)
        except Exception as e:
            pass  # 未安装则静默跳过

        all_results[material] = material_results

        # 打印当前物资最优
        if material_results:
            best_model = max(material_results, key=lambda k: material_results[k]['r2'])
            best_r2 = material_results[best_model]['r2']
            print(f"    最优: {best_model} R²={best_r2:.4f}")

    # 3. 汇总统计
    print()
    print("=" * 70)
    print("  Baseline 汇总报告")
    print("=" * 70)

    # 每物资最优模型的指标
    per_material_best = {}
    for material, results in all_results.items():
        if not results:
            continue
        best_model = max(results, key=lambda k: results[k]['r2'])
        per_material_best[material] = {
            "best_model": best_model,
            **results[best_model]
        }

    # 全局统计 (跨物资均值)
    r2_values = [v['r2'] for v in per_material_best.values()]
    mape_values = [v['mape'] for v in per_material_best.values() if not np.isnan(v['mape'])]
    rmse_values = [v['rmse'] for v in per_material_best.values()]
    mae_values = [v['mae'] for v in per_material_best.values()]

    overall = {
        "r2_mean": float(np.mean(r2_values)) if r2_values else None,
        "r2_min": float(np.min(r2_values)) if r2_values else None,
        "r2_max": float(np.max(r2_values)) if r2_values else None,
        "mape_mean": float(np.mean(mape_values)) if mape_values else None,
        "rmse_mean": float(np.mean(rmse_values)) if rmse_values else None,
        "mae_mean": float(np.mean(mae_values)) if mae_values else None,
    }

    # 达标比例 (R² >= 0.85 为合格线)
    n_pass = sum(1 for v in r2_values if v >= 0.85)
    n_total = len(r2_values)
    pass_ratio = n_pass / n_total if n_total > 0 else 0

    print(f"  物资数: {n_total} (跳过电抗器保护)")
    print(f"  R² 均值: {overall['r2_mean']:.4f}")
    print(f"  R² 最差: {overall['r2_min']:.4f}")
    print(f"  R² 最优: {overall['r2_max']:.4f}")
    print(f"  MAPE 均值: {overall['mape_mean']:.2f}%")
    print(f"  达标比例 (R²≥0.85): {n_pass}/{n_total} = {pass_ratio:.0%}")
    print()

    for material, info in per_material_best.items():
        print(f"  {material}: {info['best_model']} "
              f"R²={info['r2']:.4f} MAPE={info['mape']:.2f}% "
              f"RMSE={info['rmse']:.4f} MAE={info['mae']:.4f}")

    # 4. 写入报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_version": "v0",
        "model_version": "baseline",
        "validation_protocol": {
            "split_method": "time_series",
            "train_months": int(len(data_dict[materials[0]]) - m.N_TEST),
            "test_months": int(m.N_TEST),
            "shuffle": False,
        },
        "overall": overall,
        "pass_ratio": pass_ratio,
        "pass_threshold": 0.85,
        "per_material": per_material_best,
        "all_models": {mat: results for mat, results in all_results.items()},
        "config": {
            "USE_EVENT_FEATURES": m.USE_EVENT_FEATURES,
            "USE_LOG1P_TARGET": m.USE_LOG1P_TARGET,
            "USE_MAG_FEATURES": m.USE_MAG_FEATURES,
            "USE_QUARTER_DUMMIES": m.USE_QUARTER_DUMMIES,
            "SKIP_REACTOR_PROTECT": m.SKIP_REACTOR_PROTECT,
            "RANDOM_SEED": m.RANDOM_SEED,
        }
    }

    report_path = Path("experiments/baseline_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  报告已写入: {report_path.resolve()}")

    # 5. 更新 experiment_log.json 的 baseline 字段
    log_path = Path("experiments/experiment_log.json")
    if log_path.exists():
        log_data = json.loads(log_path.read_text(encoding='utf-8'))
    else:
        log_data = {"baseline": {}, "current_best": {}, "rounds": []}

    log_data["baseline"] = {
        "r2": overall["r2_mean"],
        "mape": overall["mape_mean"],
        "rmse": overall["rmse_mean"],
        "mae": overall["mae_mean"],
        "pass_ratio": pass_ratio,
        "data_version": "v0",
        "model_version": "baseline",
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    log_data["current_best"] = log_data["baseline"].copy()
    log_data["current_best"]["source_round"] = "baseline"

    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  台账已更新: {log_path.resolve()}")

    print("\n  Baseline 评测完成!")
    return report


if __name__ == "__main__":
    run_baseline()
