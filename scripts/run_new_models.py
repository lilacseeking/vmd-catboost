"""
Croston-TSB-LightGBM + LightGBM Direct Multi-Step
对比现有 CatBoost TwoStage, 在冀北ECP真实数据上评估

执行: python scripts/run_new_models.py
"""
import os, sys, warnings, json, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# Configuration
# ==============================================================================
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data', 'real_material_data.xlsx')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'new_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHEET_NAMES = ['控制电缆', '电缆保护管', '通信单元', '电缆接线端子', '蝶式绝缘子']
MAT_KEYS = ['control_cable', 'cable_protect_pipe', 'comm_unit', 'cable_terminal', 'butterfly_insulator']

TRAIN_LEN = 62  # 2020-05 ~ 2025-06
TEST_LEN = 12   # 2025-07 ~ 2026-06

# ==============================================================================
# Data loading
# ==============================================================================
def load_all_data():
    data = {}
    for sheet, key in zip(SHEET_NAMES, MAT_KEYS):
        df = pd.read_excel(DATA_FILE, sheet_name=sheet)
        df['日期'] = pd.to_datetime(df['日期'])
        # Forward fill NaN in factors
        factor_cols = [c for c in df.columns if c not in ['日期', '需求量', '单位']]
        for c in factor_cols:
            if df[c].dtype in [np.float64, np.int64]:
                df[c] = df[c].ffill().bfill()
        data[key] = {
            'df': df,
            'label': sheet,
            'demand': df['需求量'].values.astype(np.float64),
        }
    return data

# ==============================================================================
# Naive baselines
# ==============================================================================
def naive_seasonal_forecast(train_demand):
    """Repeat last 12 months (naive seasonal)"""
    if len(train_demand) < 12:
        mean_nz = np.mean(train_demand[train_demand > 0]) if (train_demand > 0).sum() > 0 else 0
        return np.full(TEST_LEN, mean_nz)
    return train_demand[-12:].copy()

def naive_mean_forecast(train_demand):
    """Mean of non-zero values"""
    nz = train_demand[train_demand > 0]
    mean_val = np.mean(nz) if len(nz) > 0 else 0
    return np.full(TEST_LEN, mean_val)

# ==============================================================================
# Model 1: Croston-TSB-LightGBM
# ==============================================================================
def extract_croston_sequence(demand_series):
    """Extract interval and size sequences from intermittent demand."""
    intervals, sizes = [], []
    last_nz_idx = -1
    for i, val in enumerate(demand_series):
        if val > 0:
            if last_nz_idx >= 0:
                intervals.append(i - last_nz_idx)
            sizes.append(val)
            last_nz_idx = i
    return np.array(intervals), np.array(sizes), last_nz_idx

def build_interval_features(months_data, intervals, sizes, n_events):
    """Build feature matrix for interval prediction."""
    features = []
    targets = []
    for i in range(2, n_events - 1):
        m = months_data[i]
        feat = [
            np.sin(2 * np.pi * m / 12),
            np.cos(2 * np.pi * m / 12),
            m / 12.0,  # normalized month index
            intervals[i - 1],  # prev interval
            np.mean(intervals[max(0, i-3):i]),  # avg interval (last 3)
            np.mean(intervals[:i]),  # avg interval (all)
            np.sum(intervals[:i]),  # days since first event
        ]
        features.append(feat)
        targets.append(intervals[i])
    return np.array(features), np.array(targets)

def build_size_features(months_data, sizes, intervals, n_events):
    """Build feature matrix for size prediction."""
    features = []
    targets = []
    for i in range(2, n_events - 1):
        m = months_data[i]
        feat = [
            np.sin(2 * np.pi * m / 12),
            np.cos(2 * np.pi * m / 12),
            m / 12.0,
            sizes[i - 1],  # prev size
            np.mean(sizes[max(0, i-3):i]),  # avg size (last 3)
            np.mean(sizes[:i]),  # avg size (all)
            intervals[i - 1],  # prev interval (link to size)
        ]
        features.append(feat)
        targets.append(sizes[i])
    return np.array(features), np.array(targets)

def croston_tsb_lgb_forecast(train_demand, test_len):
    """Croston-TSB-LightGBM forecast.

    Returns: (forecast_12m, model_info_dict)
    """
    intervals, sizes, last_nz_idx = extract_croston_sequence(train_demand)
    n_events = len(sizes)
    n_intervals = len(intervals)

    result = {
        'n_events': n_events,
        'n_intervals': n_intervals,
        'avg_interval': float(np.mean(intervals)) if n_intervals > 0 else 99,
        'avg_size': float(np.mean(sizes)) if n_events > 0 else 0,
        'model_type': 'Croston-TSB-LGB',
    }

    # Fallback: too few events
    if n_events < 4:
        result['fallback'] = 'too_few_events'
        return naive_seasonal_forecast(train_demand), result

    # Build event-month indices
    event_months = []
    month_count = 0
    for v in train_demand:
        if v > 0:
            event_months.append(month_count)
        month_count += 1

    # Interval model
    interval_features, interval_targets = build_interval_features(
        event_months, intervals, sizes, n_events)
    size_features, size_targets = build_size_features(
        event_months, sizes, intervals, n_events)

    pred_interval = np.mean(intervals)
    pred_size = np.mean(sizes)

    if len(interval_targets) >= 3:
        try:
            model_int = lgb.LGBMRegressor(
                n_estimators=30, max_depth=2, min_child_samples=2,
                learning_rate=0.05, reg_alpha=1.0, reg_lambda=2.0,
                verbosity=-1, random_state=42)
            model_int.fit(interval_features, interval_targets)

            last_m = event_months[-1]
            last_feat = np.array([[
                np.sin(2*np.pi*last_m/12), np.cos(2*np.pi*last_m/12),
                last_m/12.0, intervals[-1],
                np.mean(intervals[-3:]), np.mean(intervals),
                np.sum(intervals)
            ]])
            pred_interval = max(1.0, model_int.predict(last_feat)[0])
        except Exception:
            pass

    if len(size_targets) >= 3:
        try:
            model_size = lgb.LGBMRegressor(
                n_estimators=30, max_depth=2, min_child_samples=2,
                learning_rate=0.05, reg_alpha=1.0, reg_lambda=2.0,
                verbosity=-1, random_state=42)
            model_size.fit(size_features, size_targets)

            last_m2 = event_months[-1]
            last_feat2 = np.array([[
                np.sin(2*np.pi*last_m2/12), np.cos(2*np.pi*last_m2/12),
                last_m2/12.0, sizes[-1],
                np.mean(sizes[-3:]), np.mean(sizes),
                intervals[-1] if n_intervals > 0 else pred_interval
            ]])
            pred_size = max(0, model_size.predict(last_feat2)[0])
        except Exception:
            pass

    result['pred_interval'] = float(pred_interval)
    result['pred_size'] = float(pred_size)

    # Build monthly forecast
    months_since_last = len(train_demand) - 1 - last_nz_idx if last_nz_idx >= 0 else 999
    forecast = np.zeros(test_len)

    for h in range(test_len):
        current_gap = months_since_last + h + 1

        # TSB probability decay
        base_prob = 1.0 / max(pred_interval, 1.0)
        if current_gap > 2 * result['avg_interval']:
            decay_factor = 0.5 ** (current_gap / result['avg_interval'])
            prob = base_prob * decay_factor
        else:
            prob = base_prob

        prob = min(prob, 1.0)
        forecast[h] = prob * pred_size

    return forecast, result

# ==============================================================================
# Model 2: LightGBM Direct Multi-Step
# ==============================================================================
def build_lgb_direct_features(demand_series, factor_df):
    """Build 26-dim feature matrix for direct multi-step LightGBM."""
    n = len(demand_series)
    features = np.zeros((n, 23))
    eps = 1e-8

    # Time features (3 dim)
    for i in range(n):
        m = (i % 12) + 1  # month 1-12
        features[i, 0] = np.sin(2 * np.pi * m / 12)
        features[i, 1] = np.cos(2 * np.pi * m / 12)
        features[i, 2] = i / 12.0  # year index (0, 0.083, ..., ~6)

    # Lag features (7 dim) — direct, no fill
    for i in range(n):
        for j, lag in enumerate([1, 2, 3, 6, 9, 12, 18]):
            src = i - lag
            features[i, 3 + j] = demand_series[src] if src >= 0 else 0.0

    # Rolling statistics (6 dim)
    for i in range(n):
        for j, w in enumerate([3, 6, 12]):
            start = max(0, i - w + 1)
            window = demand_series[start:i+1]
            features[i, 10 + j*2] = np.mean(window)
            features[i, 10 + j*2 + 1] = np.max(window)

    # Demand interval features (4 dim, Croston-inspired)
    nz_indices = [ii for ii, v in enumerate(demand_series) if v > 0]
    for i in range(n):
        # months since last non-zero
        prev_nz = [idx for idx in nz_indices if idx < i]
        features[i, 16] = i - prev_nz[-1] if prev_nz else i + 1  # months_since_last_nz

        # zero streak
        if demand_series[i] > 0:
            features[i, 17] = 0
        else:
            zs = 0
            j = i
            while j >= 0 and demand_series[j] <= 0:
                zs += 1
                j -= 1
            features[i, 17] = zs  # zero_streak

        # avg interval & nz frequency
        if len(nz_indices) >= 2:
            gaps = np.diff([idx for idx in nz_indices if idx <= i])
            features[i, 18] = np.mean(gaps) if len(gaps) > 0 else 12.0
        else:
            features[i, 18] = 12.0

        # nz frequency in last 12 months
        recent = demand_series[max(0,i-11):i+1]
        features[i, 19] = (recent > 0).sum()

    # Batch rhythm features (3 dim)
    for i in range(n):
        m = (i % 12) + 1
        features[i, 20] = 1.0 if m == 3 else 0.0   # march
        features[i, 21] = 1.0 if m == 5 else 0.0   # may
        features[i, 22] = 1.0 if m == 9 else 0.0   # september

    # NaN guard
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    return features

def lgb_direct_multistep_forecast(train_demand, factor_df, test_len):
    """LightGBM Direct Multi-Step forecast."""
    result = {'model_type': 'LGB-Direct-MultiStep'}

    # Build features for full 74-month period
    features_all = build_lgb_direct_features(train_demand, factor_df)

    n_train = len(train_demand)
    if n_train < 24:
        result['fallback'] = 'too_short'
        return naive_seasonal_forecast(train_demand), result

    # For each forecast horizon h=1..12, train an independent model
    models = {}
    forecast = np.zeros(test_len)

    for h in range(1, min(test_len + 1, 13)):
        # Training: (X[0 : n_train-h], y[h : n_train])
        X_train = features_all[:n_train - h]
        y_train = train_demand[h:n_train]

        if len(X_train) < 5:
            forecast[h-1] = np.mean(train_demand[train_demand > 0]) if (train_demand > 0).sum() > 0 else 0
            continue

        try:
            model_h = lgb.LGBMRegressor(
                n_estimators=100, max_depth=3, min_child_samples=5,
                learning_rate=0.03, reg_alpha=0.5, reg_lambda=2.0,
                subsample=0.7, colsample_bytree=0.7,
                verbosity=-1, random_state=42 + h)
            model_h.fit(X_train, y_train)

            # Predict using last available feature row
            X_last = features_all[n_train - 1:n_train]  # shape (1, 26)
            forecast[h-1] = max(0.0, model_h.predict(X_last)[0])
            models[h] = model_h
        except Exception:
            forecast[h-1] = 0.0

    result['n_models'] = len(models)
    return forecast, result

# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate(actual, predicted, label):
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) < 3:
        return {'MSE': float('nan'), 'RMSE': float('nan'), 'MAE': float('nan'),
                'R2': float('nan'), 'MAPE_nz': float('nan'), 'label': label}
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    try:
        r2 = r2_score(actual, predicted)
    except:
        r2 = float('nan')

    # MAPE on non-zero actual only
    nz_mask = actual > 0
    if nz_mask.sum() > 0:
        mape_nz = np.mean(np.abs((actual[nz_mask] - predicted[nz_mask]) / actual[nz_mask])) * 100
    else:
        mape_nz = float('nan')

    return {'MSE': round(mse, 2), 'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
            'R2': round(r2, 4), 'MAPE_nz': round(mape_nz, 1) if not np.isnan(mape_nz) else float('nan'),
            'label': label}

# ==============================================================================
# Model 3: CatBoost TwoStage (现有的,从 TwoStage doc 引用)
# ==============================================================================
# TwoStage 在个体物资上的结果从前次实验获取
KNOWN_TWOSTAGE_R2 = {
    'control_cable': 0.10,   # VMD-LSTM-CatBoost best
    'cable_protect_pipe': 0.16,
    'comm_unit': 0.17,
    'cable_terminal': -0.03,
    'butterfly_insulator': 0.49,
}

KNOWN_CATBOOST_R2 = {
    'control_cable': -0.15,
    'cable_protect_pipe': 0.08,
    'comm_unit': -0.02,
    'cable_terminal': -0.24,
    'butterfly_insulator': -0.13,
}

# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 80)
    print("  Croston-TSB-LightGBM vs LightGBM Direct Multi-Step vs CatBoost TwoStage")
    print("  Data: ECP 国网冀北电力物资采购 (2020-05 ~ 2026-06, 74 months)")
    print("  Train: 2020-05 ~ 2025-06 (62 months) | Test: 2025-07 ~ 2026-06 (12 months)")
    print("=" * 80)

    all_data = load_all_data()

    all_results = {}
    all_details = {}

    for mat_key in MAT_KEYS:
        mat_info = all_data[mat_key]
        demand = mat_info['demand']
        label = mat_info['label']
        df = mat_info['df']
        factor_df = df.drop(columns=['日期', '需求量', '单位'], errors='ignore')

        train = demand[:TRAIN_LEN]
        test = demand[TRAIN_LEN:TRAIN_LEN + TEST_LEN]

        nz_train = (train > 0).sum()
        nz_test = (test > 0).sum()

        print(f"\n{'='*60}")
        print(f"  [{label}]  训练: {nz_train}/{TRAIN_LEN}非零月 | 测试: {nz_test}/{TEST_LEN}非零月")
        print(f"  训练集: mean={train.mean():.0f}, max={train.max():.0f}")
        print(f"  测试集: mean={test.mean():.0f}, max={test.max():.0f}")
        print(f"{'='*60}")

        results_for_mat = {}
        details_for_mat = {}

        # ---- Naive baselines ----
        fcst_seasonal = naive_seasonal_forecast(train)
        fcst_mean = naive_mean_forecast(train)

        eval_seasonal = evaluate(test, fcst_seasonal, 'Naive-Seasonal')
        eval_mean = evaluate(test, fcst_mean, 'Naive-Mean')

        # ---- Croston-TSB-LGB ----
        print(f"\n  [1] Croston-TSB-LGB...")
        fcst_croston, detail_croston = croston_tsb_lgb_forecast(train, TEST_LEN)
        eval_croston = evaluate(test, fcst_croston, 'Croston-TSB-LGB')
        details_for_mat['Croston-TSB-LGB'] = detail_croston

        # ---- LGB Direct Multi-Step ----
        print(f"  [2] LGB Direct Multi-Step...")
        fcst_lgb, detail_lgb = lgb_direct_multistep_forecast(train, factor_df, TEST_LEN)
        eval_lgb = evaluate(test, fcst_lgb, 'LGB-Direct')
        details_for_mat['LGB-Direct'] = detail_lgb

        results_for_mat = {
            'Naive-Seasonal': eval_seasonal,
            'Naive-Mean': eval_mean,
            'Croston-TSB-LGB': eval_croston,
            'LGB-Direct': eval_lgb,
            'CatBoost (known)': {'R2': KNOWN_CATBOOST_R2.get(mat_key, float('nan')), 'label': 'CatBoost'},
            'TwoStage (known)': {'R2': KNOWN_TWOSTAGE_R2.get(mat_key, float('nan')), 'label': 'TwoStage'},
        }

        # Print comparison
        print(f"\n  {'Model':<25s} {'R2':>8s} {'MAE':>12s} {'RMSE':>12s} {'MAPE_nz':>10s}")
        print(f"  {'-'*70}")
        for model_name, metrics in results_for_mat.items():
            r2_str = f"{metrics.get('R2', float('nan')):.4f}" if not np.isnan(metrics.get('R2', float('nan'))) else 'N/A'
            mae_str = f"{metrics.get('MAE', float('nan')):.0f}" if not np.isnan(metrics.get('MAE', float('nan'))) else 'N/A'
            rmse_str = f"{metrics.get('RMSE', float('nan')):.0f}" if not np.isnan(metrics.get('RMSE', float('nan'))) else 'N/A'
            mape_str = f"{metrics.get('MAPE_nz', float('nan')):.1f}%" if not np.isnan(metrics.get('MAPE_nz', float('nan'))) else 'N/A'
            print(f"  {model_name:<25s} {r2_str:>8s} {mae_str:>12s} {rmse_str:>12s} {mape_str:>10s}")

        # Print forecast details
        print(f"\n  测试集真实值: {[f'{v:.0f}' for v in test]}")
        print(f"  Croston-TSB-LGB:   {[f'{v:.0f}' for v in fcst_croston]}")
        print(f"  LGB-Direct:        {[f'{v:.0f}' for v in fcst_lgb]}")
        print(f"  Naive-Seasonal:    {[f'{v:.0f}' for v in fcst_seasonal]}")

        all_results[mat_key] = results_for_mat
        all_details[mat_key] = details_for_mat

    # ---- Summary ----
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: R² Comparison")
    print(f"{'='*80}")
    header = f"{'Material':<20s} {'Naive-Seas':>10s} {'Croston-LGB':>10s} {'LGB-Direct':>10s} {'CatBoost':>10s} {'TwoStage':>10s}"
    print(header)
    print(f"  {'-'*70}")

    summary_table = []
    for mat_key, mat_label in zip(MAT_KEYS, SHEET_NAMES):
        res = all_results[mat_key]
        line = f"  {mat_label:<18s}"
        for model in ['Naive-Seasonal', 'Croston-TSB-LGB', 'LGB-Direct', 'CatBoost (known)', 'TwoStage (known)']:
            r2 = res[model].get('R2', float('nan'))
            r2_str = f"{r2:>10.4f}" if not np.isnan(r2) else f"{'N/A':>10s}"
            line += f" {r2_str}"
        print(line)

        # Find best
        best = max(
            [(m, res[m].get('R2', -999)) for m in ['Croston-TSB-LGB', 'LGB-Direct']],
            key=lambda x: x[1]
        )
        summary_table.append({
            'material': mat_label,
            'croston_r2': res['Croston-TSB-LGB'].get('R2', float('nan')),
            'lgb_r2': res['LGB-Direct'].get('R2', float('nan')),
            'catboost_r2': res['CatBoost (known)'].get('R2', float('nan')),
            'twostage_r2': res['TwoStage (known)'].get('R2', float('nan')),
            'best_new': best[0],
            'best_new_r2': best[1],
        })

    # Save
    output = {
        'summary': summary_table,
        'detailed_results': {
            mat_key: {
                model: {k: v for k, v in metrics.items() if k != 'label'}
                for model, metrics in all_results[mat_key].items()
            }
            for mat_key in MAT_KEYS
        },
        'model_details': {
            mat_key: {
                model_name: {k: v for k, v in detail.items()}
                for model_name, detail in all_details[mat_key].items()
            }
            for mat_key in MAT_KEYS
        },
        'run_time': datetime.now().isoformat(),
    }

    json_path = os.path.join(OUTPUT_DIR, 'new_models_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {json_path}")

    # Best model per material
    print(f"\n  Best New Model per Material:")
    for row in summary_table:
        print(f"    {row['material']}: {row['best_new']} (R2={row['best_new_r2']:.4f})")

    return output

if __name__ == '__main__':
    results = main()
