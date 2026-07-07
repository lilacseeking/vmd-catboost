"""
批次事件三层预测模型 (Batch Event Three-Layer Prediction)

Layer 1 — When:   季节性批次时间预测 (Seasonal probability + interval constraint)
Layer 2 — What:   批次物资共现预测 (Conditional co-occurrence probability)
Layer 3 — How Much: 物资需求量预测 (Seasonal mean + year-over-year trend)

原则: 零训练参数, 纯统计分析。数据生成过程是离散批次事件, 不是连续时间序列。

执行: python scripts/batch_event_model.py
"""
import os, sys, json, sqlite3, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- 中文字体 ----
_font_cache_dir = matplotlib.get_cachedir()
for _fn in os.listdir(_font_cache_dir):
    if _fn.startswith('fontlist'):
        try: os.remove(os.path.join(_font_cache_dir, _fn))
        except OSError: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# Configuration
# ==============================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                       'bidding-ecp-data', 'data', 'ecp_data.db')
EXCEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'data',
                          'real_material_data.xlsx')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'batch_event_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_MONTH = "202005"
END_MONTH = "202606"
TRAIN_LEN = 62
TEST_LEN = 12

MONTHS_ALL = [f"{y}{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
              if f"{y}{m:02d}" >= START_MONTH and f"{y}{m:02d}" <= END_MONTH]

TARGET_MATERIALS = ['控制电缆', '电缆保护管', '通信单元', '电缆接线端子', '蝶式绝缘子']

# ==============================================================================
# Load Data
# ==============================================================================

def load_target_materials():
    """Load the 5 target materials from Excel (already prepared)."""
    data = {}
    for sheet in TARGET_MATERIALS:
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
        data[sheet] = df['需求量'].values.astype(float)
    return data

def load_all_materials_from_db(min_nz=8):
    """Load all materials with >= min_nz non-zero months from database."""
    if not os.path.exists(DB_PATH):
        return {}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Find materials with sufficient history
    cur.execute("""
        SELECT material_name, COUNT(DISTINCT demand_month) as nz,
               SUM(demand_quantity) as total_qty
        FROM material_demand_item
        WHERE demand_month >= ? AND demand_month <= ?
        GROUP BY material_name
        HAVING nz >= ?
        ORDER BY nz DESC
    """, (START_MONTH, END_MONTH, min_nz))
    rows = cur.fetchall()

    data = {}
    for mat_name, nz, total in rows:
        cur.execute("""
            SELECT demand_month, SUM(demand_quantity)
            FROM material_demand_item
            WHERE material_name = ? AND demand_month >= ? AND demand_month <= ?
            GROUP BY demand_month ORDER BY demand_month
        """, (mat_name, START_MONTH, END_MONTH))
        demand_map = {r[0]: r[1] for r in cur.fetchall()}
        vals = np.array([demand_map.get(m, 0) for m in MONTHS_ALL])
        data[mat_name] = vals

    conn.close()
    return data

# ==============================================================================
# Data: Extract batch events
# ==============================================================================

def extract_batches(material_data, n_months):
    """Extract batch events from material demand data.

    A batch = any month where at least 1 material has demand > 0.

    Returns:
        batches: list of {idx, month, year, materials: {name: qty}, n_mats, total_qty}
    """
    mat_names = list(material_data.keys())
    batches = []

    for i in range(n_months):
        mats_in_batch = {}
        for name in mat_names:
            qty = material_data[name][i]
            if qty > 0:
                mats_in_batch[name] = qty

        if mats_in_batch:
            y, m = (2020 + (5 + i) // 12, (5 + i - 1) % 12 + 1)
            batches.append({
                'idx': i,
                'month': m,
                'year': y,
                'label': f"{y}-{m:02d}",
                'materials': mats_in_batch,
                'n_mats': len(mats_in_batch),
                'total_qty': sum(mats_in_batch.values()),
            })

    return batches

# ==============================================================================
# Layer 1: When — batch timing prediction
# ==============================================================================

def build_layer1_when(train_batches):
    """Build seasonal batch probability table from training batches.

    Returns:
        month_prob: dict {month (1-12): probability}
        interval_stats: (mean, std, max) of batch intervals
    """
    n_years = max(1, (len(set(b['year'] for b in train_batches))) + 1)

    # Count batches per month over training years
    train_years = set(b['year'] for b in train_batches)
    n_train_years = max(1, len(train_years))

    month_count = defaultdict(int)
    for b in train_batches:
        month_count[b['month']] += 1

    month_prob = {}
    for m in range(1, 13):
        month_prob[m] = month_count.get(m, 0) / n_train_years

    # Batch interval statistics
    batch_indices = sorted([b['idx'] for b in train_batches])
    if len(batch_indices) >= 2:
        intervals = np.diff(batch_indices)
        interval_stats = {
            'mean': np.mean(intervals), 'std': np.std(intervals),
            'median': np.median(intervals), 'max': np.max(intervals),
            'min': np.min(intervals),
        }
    else:
        interval_stats = {'mean': 3, 'std': 2, 'median': 3, 'max': 8, 'min': 1}

    return month_prob, interval_stats

def predict_batch_months(month_prob, interval_stats, test_months_start, n_test=12):
    """Predict which test months will have batch events.

    Strategy: For each test month, compute probability as:
        P(batch) = seasonal_freq × interval_boost_factor

    Where interval_boost_factor > 1 if it's been long since last predicted batch.
    Returns list of (month_idx, prob) sorted by prob.
    """
    probs = []
    months_since_last_batch = 0

    for h in range(n_test):
        m = (5 + TRAIN_LEN + h - 1) % 12 + 1
        base_prob = month_prob.get(m, 0.15)

        # Interval boost: if >2 std beyond mean, increase probability
        if months_since_last_batch > interval_stats['mean'] + interval_stats['std']:
            base_prob *= 1.5
        if months_since_last_batch > interval_stats['max']:
            base_prob *= 2.0

        probs.append((h, m, base_prob))

        if base_prob >= 0.4:
            months_since_last_batch = 0
        else:
            months_since_last_batch += 1

    return probs

# ==============================================================================
# Layer 2: What — material co-occurrence
# ==============================================================================

def build_layer2_what(train_batches, mat_names):
    """Build conditional co-occurrence probability matrix.

    Returns:
        cooc_matrix: n_mats × n_mats — count of co-occurrences
        cond_prob: P(mat_i | mat_j) — conditional probability
        month_mat_prob: P(mat | month) — per-month material probability
    """
    n_mats = len(mat_names)
    mat_idx = {name: i for i, name in enumerate(mat_names)}
    cooc = np.zeros((n_mats, n_mats))

    # Also track: per month, which materials appear
    month_mat_count = defaultdict(lambda: defaultdict(int))
    month_total = defaultdict(int)

    for b in train_batches:
        present = list(b['materials'].keys())
        month_total[b['month']] += 1
        for m_name in present:
            if m_name in mat_idx:
                month_mat_count[b['month']][m_name] += 1
                for m2_name in present:
                    if m2_name in mat_idx:
                        cooc[mat_idx[m_name]][mat_idx[m2_name]] += 1

    # Conditional probability P(i | j)
    cond_prob = np.zeros((n_mats, n_mats))
    for i in range(n_mats):
        for j in range(n_mats):
            denom = cooc[j][j]
            cond_prob[i][j] = cooc[i][j] / max(denom, 1)

    # Per-month probability P(mat | month)
    month_mat_prob = {}
    for month in range(1, 13):
        n_batches = max(month_total.get(month, 1), 1)
        month_mat_prob[month] = {}
        for m_name in mat_names:
            month_mat_prob[month][m_name] = month_mat_count[month].get(m_name, 0) / n_batches

    return cooc, cond_prob, month_mat_prob

def predict_materials_for_batch(cond_prob, mat_names, month, month_mat_prob, n_expected=4):
    """Predict which materials appear in a batch.

    Uses month-specific probability + conditional probability from companion materials.
    """
    n_mats = len(mat_names)
    mat_idx = {name: i for i, name in enumerate(mat_names)}
    mat_probs = {}

    for m_name in mat_names:
        # Base: historical frequency in this month
        base = month_mat_prob.get(month, {}).get(m_name, 0.1)
        mat_probs[m_name] = base

    # Find the "core cluster" — materials that tend to appear together
    # Find material with highest probability
    sorted_by_prob = sorted(mat_probs.items(), key=lambda x: -x[1])

    # Boost companions
    included = set()
    for m_name, prob in sorted_by_prob:
        if prob >= 0.4:
            included.add(m_name)
            # Bootstrap companions via conditional probability
            idx = mat_idx[m_name]
            for m2_name in mat_names:
                idx2 = mat_idx[m2_name]
                if m2_name not in included and cond_prob[idx2][idx] >= 0.6:
                    included.add(m2_name)

    return list(included)

# ==============================================================================
# Layer 3: How Much — quantity prediction
# ==============================================================================

def build_layer3_quantity(train_batches, mat_names):
    """Build seasonal quantity statistics.

    Returns:
        mat_month_stats: {mat: {month: {'mean', 'std', 'n', 'trend'}}}
        mat_year_stats: {mat: {year: {'mean', 'n'}}} — year-over-year for trend
    """
    mat_month_stats = {}
    mat_year_stats = {}

    for m_name in mat_names:
        month_data = defaultdict(list)
        year_data = defaultdict(list)

        for b in train_batches:
            if m_name in b['materials']:
                qty = b['materials'][m_name]
                month_data[b['month']].append(qty)
                year_data[b['year']].append(qty)

        # Monthly statistics
        month_stats = {}
        for mth in range(1, 13):
            vals = month_data.get(mth, [])
            if vals:
                vals_arr = np.array(vals)
                month_stats[mth] = {
                    'mean': float(np.mean(vals_arr)),
                    'median': float(np.median(vals_arr)),
                    'std': float(np.std(vals_arr)) if len(vals_arr) > 1 else 0,
                    'n': len(vals_arr),
                    'trend': float(np.polyfit(range(len(vals_arr)), vals_arr, 1)[0]) if len(vals_arr) >= 3 else 0,
                }
            else:
                month_stats[mth] = {'mean': 0, 'median': 0, 'std': 0, 'n': 0, 'trend': 0}

        mat_month_stats[m_name] = month_stats

        # Yearly statistics (for trend)
        year_stats = {}
        for yr in sorted(year_data.keys()):
            vals = year_data[yr]
            year_stats[yr] = {'mean': float(np.mean(vals)), 'n': len(vals)}
        mat_year_stats[m_name] = year_stats

    return mat_month_stats, mat_year_stats

def predict_quantity(m_name, month, year, mat_month_stats, mat_year_stats):
    """Predict quantity for a material in a given month."""
    monthly = mat_month_stats.get(m_name, {}).get(month, {})
    if not monthly or monthly.get('n', 0) == 0:
        return 0

    base = monthly['mean']

    # Apply year-over-year trend (if we have 3+ data points)
    trend = monthly.get('trend', 0)
    if trend != 0 and monthly['n'] >= 3:
        years_since = year - 2025
        base += trend * max(0, years_since)

    return max(0, base)

# ==============================================================================
# Full Predict Pipeline
# ==============================================================================

TRAIN_START_YEAR = 2020

def run_batch_event_model(material_data, mat_names, train_len=TRAIN_LEN, test_len=TEST_LEN):
    """Run the full three-layer batch event prediction model.

    Returns:
        predictions: {mat_name: np.array(test_len)} — monthly predictions
        metrics: {mat_name: {'R2', 'MAE', 'RMSE', 'MAPE_nz'}}
        details: dict with batch prediction details
    """
    total_months = train_len + test_len

    # ---- Extract batches from full training data ----
    train_batches = extract_batches(
        {n: material_data[n][:train_len] for n in mat_names}, train_len)
    full_batches = extract_batches(material_data, total_months)

    # ---- Layer 1: When ----
    month_prob, interval_stats = build_layer1_when(train_batches)
    batch_probs = predict_batch_months(month_prob, interval_stats, train_len, test_len)

    # ---- Layer 2: What ----
    cooc, cond_prob, month_mat_prob = build_layer2_what(train_batches, mat_names)

    # ---- Layer 3: How Much ----
    mat_month_stats, mat_year_stats = build_layer3_quantity(train_batches, mat_names)

    # ---- Predict ----
    predictions = {n: np.zeros(test_len) for n in mat_names}
    actuals = {n: material_data[n][train_len:train_len+test_len] for n in mat_names}

    batch_predictions = []

    for h, month, prob in batch_probs:
        y = TRAIN_START_YEAR + (5 + train_len + h) // 12
        included = predict_materials_for_batch(cond_prob, mat_names, month, month_mat_prob)

        # Expected value prediction: E[demand] = P(batch|month) * E[qty|batch,month]
        # This eliminates false positives -- months with low prob get small predictions
        # instead of zero-or-full dichotomy
        for m_name in mat_names:
            if m_name in included:
                qty_full = predict_quantity(m_name, month, y, mat_month_stats, mat_year_stats)
                predictions[m_name][h] = prob * qty_full

    # ---- Evaluate ----
    metrics = {}
    for m_name in mat_names:
        act = actuals[m_name]
        pred = predictions[m_name]

        mask = np.isfinite(act) & np.isfinite(pred)
        act_clean, pred_clean = act[mask], pred[mask]

        if len(act_clean) < 2:
            metrics[m_name] = {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE_nz': np.nan}
            continue

        mse = mean_squared_error(act_clean, pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(act_clean, pred_clean)
        try:
            r2 = r2_score(act_clean, pred_clean)
        except ValueError:
            r2 = np.nan

        nz_mask = act_clean > 0
        if nz_mask.sum() > 0:
            mape_nz = np.mean(np.abs((act_clean[nz_mask] - pred_clean[nz_mask]) / act_clean[nz_mask])) * 100
        else:
            mape_nz = np.nan

        metrics[m_name] = {'R2': round(r2, 4), 'MAE': round(mae, 2),
                          'RMSE': round(rmse, 2),
                          'MAPE_nz': round(mape_nz, 1) if not np.isnan(mape_nz) else np.nan}

    # ---- Details ----
    details = {
        'n_train_batches': len(train_batches),
        'n_total_batches': len(full_batches),
        'train_intervals': np.diff([b['idx'] for b in train_batches]).tolist() if len(train_batches) >= 2 else [],
        'month_prob': {str(k): round(v, 3) for k, v in month_prob.items()},
        'interval_stats': {k: round(v, 2) if isinstance(v, float) else v for k, v in interval_stats.items()},
        'predicted_batches': [{'month': h+1, 'prob': round(p, 3), 'is_batch': p >= 0.35}
                             for h, m, p in batch_probs],
        'batch_predictions': batch_predictions,
    }

    return predictions, actuals, metrics, details

# ==============================================================================
# Baselines for comparison
# ==============================================================================

def naive_seasonal_forecast(train_vals, test_len=12):
    """Repeat last 12 months."""
    if len(train_vals) < 12:
        return np.zeros(test_len)
    return train_vals[-12:].copy()

def naive_mean_nz_forecast(train_vals, test_len=12):
    """Mean of non-zero values."""
    nz = train_vals[train_vals > 0]
    val = np.mean(nz) if len(nz) > 0 else 0
    return np.full(test_len, val)

def evaluate_model(actual, predicted):
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]
    if len(a) < 2:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
    return {
        'R2': round(r2_score(a, p), 4),
        'MAE': round(mean_absolute_error(a, p), 2),
        'RMSE': round(np.sqrt(mean_squared_error(a, p)), 2),
    }

# ==============================================================================
# Visualization
# ==============================================================================

def plot_results(target_data, batch_preds, batch_metrics, output_dir=OUTPUT_DIR):
    """Generate comparison charts for the 5 target materials."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    colors = {
        '实际值': 'black', '批次事件模型': '#2196F3',
        'Naive-Seasonal': '#4CAF50', 'Naive-Mean': '#FF9800',
    }

    MATS = ['控制电缆', '电缆保护管', '通信单元', '电缆接线端子', '蝶式绝缘子']

    for idx, mat_name in enumerate(MATS):
        ax = axes[idx]
        vals = target_data[mat_name]
        actual = vals[62:74]
        pred = batch_preds[mat_name]

        # Baselines
        train_vals = vals[:62]
        naive_s = naive_seasonal_forecast(train_vals)
        naive_m = naive_mean_nz_forecast(train_vals)

        months = pd.date_range('2025-07-01', periods=12, freq='MS')

        ax.plot(months, actual, 'ko-', linewidth=2, markersize=6, label='实际值', zorder=3)
        ax.plot(months, pred, 's-', color='#2196F3', linewidth=2, markersize=5,
                label='批次事件模型', zorder=2)
        ax.plot(months, naive_s, '^--', color='#4CAF50', linewidth=1.5, markersize=4,
                alpha=0.7, label='Naive-Seasonal')
        ax.plot(months, naive_m, 'x--', color='#FF9800', linewidth=1.5, markersize=4,
                alpha=0.5, label='Naive-Mean')

        r2_batch = batch_metrics[mat_name].get('R2', np.nan)
        r2_s = evaluate_model(actual, naive_s)['R2']
        title = f'{mat_name} (批次R²={r2_batch:.3f}, SeasR²={r2_s:.3f})'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('需求量')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)

        # Mark predicted batch months
        for h in range(12):
            if pred[h] > 0:
                ax.axvline(x=months[h], color='#2196F3', alpha=0.1, linewidth=20)

    axes[-1].set_xlabel('日期')
    fig.suptitle('批次事件三层预测模型 vs 基线方法\n(国网冀北电力真实采购数据, 2025-07~2026-06)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'batch_event_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {path}")

    # ---- Metrics comparison bar chart ----
    fig2, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MATS))
    width = 0.2

    models = ['批次事件模型', 'Naive-Seasonal', 'Naive-Mean']
    all_r2 = {m: [] for m in models}

    for mat_name in MATS:
        r2_b = batch_metrics[mat_name].get('R2', np.nan)
        r2_s = evaluate_model(target_data[mat_name][62:74],
                              naive_seasonal_forecast(target_data[mat_name][:62]))['R2']
        r2_m = evaluate_model(target_data[mat_name][62:74],
                              naive_mean_nz_forecast(target_data[mat_name][:62]))['R2']
        all_r2['批次事件模型'].append(r2_b if not np.isnan(r2_b) else -5)
        all_r2['Naive-Seasonal'].append(r2_s if not np.isnan(r2_s) else -5)
        all_r2['Naive-Mean'].append(r2_m if not np.isnan(r2_m) else -5)

    colors_bar = ['#2196F3', '#4CAF50', '#FF9800']
    for i, (model, r2_vals) in enumerate(all_r2.items()):
        bars = ax.bar(x + i * width, r2_vals, width, label=model, color=colors_bar[i], alpha=0.85)
        for bar, val in zip(bars, r2_vals):
            if val > -4:
                ax.text(bar.get_x() + bar.get_width()/2, max(0, bar.get_height()) + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_xticks(x + width)
    ax.set_xticklabels(MATS, fontsize=10)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('批次事件模型 vs 朴素基线 — R² 对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'batch_event_metrics.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {path2}")

    # ---- Layer 1 visualization: seasonal batch probability ----
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    month_prob, interval_stats = build_layer1_when(
        extract_batches({n: target_data[n][:62] for n in MATS}, 62))

    months_label = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
    probs = [month_prob.get(m, 0) for m in range(1, 13)]
    bars = ax3.bar(range(12), probs, color=['#4CAF50' if p >= 0.3 else '#E0E0E0' for p in probs],
                   alpha=0.85, edgecolor='white')
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(months_label)
    ax3.set_ylabel('批次发生概率', fontsize=12)
    ax3.set_title('Layer 1 — 季节性批次概率 (基于62个月训练集)', fontsize=13, fontweight='bold')
    ax3.axhline(y=0.35, color='#FF5722', linestyle='--', linewidth=1.5, label='决策阈值=0.35')
    for bar, prob in zip(bars, probs):
        if prob > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.2f}', ha='center', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'layer1_seasonal_prob.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {path3}")

    # ---- Layer 2 visualization: co-occurrence heatmap ----
    fig4, ax4 = plt.subplots(figsize=(8, 7))
    cooc, cond_prob, _ = build_layer2_what(
        extract_batches({n: target_data[n][:62] for n in MATS}, 62), MATS)
    im = ax4.imshow(cond_prob, cmap='YlOrRd', vmin=0, vmax=1)
    ax4.set_xticks(range(len(MATS)))
    ax4.set_xticklabels([m[:6] for m in MATS], fontsize=9, rotation=45, ha='right')
    ax4.set_yticks(range(len(MATS)))
    ax4.set_yticklabels([m[:6] for m in MATS], fontsize=9)
    for i in range(len(MATS)):
        for j in range(len(MATS)):
            ax4.text(j, i, f'{cond_prob[i][j]:.2f}', ha='center', va='center',
                    fontsize=10, color='white' if cond_prob[i][j] > 0.7 else 'black')
    ax4.set_title('Layer 2 — 物资共现条件概率 P(row|col)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8)
    plt.tight_layout()
    path4 = os.path.join(output_dir, 'layer2_coccurrence.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {path4}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("  批次事件三层预测模型 (Batch Event Three-Layer)")
    print("  Layer 1: When | Layer 2: What | Layer 3: How Much")
    print("  数据: 国网冀北电力 ECP (2020-05 ~ 2026-06)")
    print("  原则: 零训练参数 — 时序预测 → 离散事件预测范式转换")
    print("=" * 80)

    # ---- Load 5 target materials ----
    target_data = load_target_materials()

    # ---- Also check expanded materials from DB ----
    all_mats = load_all_materials_from_db(min_nz=8)
    if all_mats:
        n_mats = len(all_mats)
        # Filter to materials with >=10 non-zero months
        high_freq = {k: v for k, v in all_mats.items()
                     if (v[:TRAIN_LEN] > 0).sum() >= 10}
        print(f"\n数据库中共 {n_mats} 种物资 (>=8非零月), {len(high_freq)} 种 (>=10非零月)")

    # ---- Run Batch Event Model on 5 target materials ----
    print(f"\n{'='*60}")
    print(f"  对比实验: 批次事件模型 vs Naive-Seasonal vs Naive-Mean")
    print(f"{'='*60}")

    batch_preds, batch_actuals, batch_metrics, batch_details = \
        run_batch_event_model(target_data, TARGET_MATERIALS)

    all_metrics = {}

    for mat_name in TARGET_MATERIALS:
        actual = target_data[mat_name][TRAIN_LEN:TRAIN_LEN+TEST_LEN]
        train = target_data[mat_name][:TRAIN_LEN]

        pred_batch = batch_preds[mat_name]
        pred_seasonal = naive_seasonal_forecast(train)
        pred_mean = naive_mean_nz_forecast(train)

        eval_batch = evaluate_model(actual, pred_batch)
        eval_seasonal = evaluate_model(actual, pred_seasonal)
        eval_mean = evaluate_model(actual, pred_mean)

        all_metrics[mat_name] = {
            'BatchEvent': eval_batch,
            'NaiveSeasonal': eval_seasonal,
            'NaiveMean': eval_mean,
        }

        nz_train = (train > 0).sum()
        nz_test = (actual > 0).sum()
        print(f"\n  [{mat_name}] train={nz_train}/62 nz, test={nz_test}/12 nz")
        print(f"    {'Model':<20s} {'R2':>10s} {'MAE':>12s} {'RMSE':>12s}")
        print(f"    {'-'*48}")
        for model, mets in [('BatchEvent', eval_batch), ('NaiveSeasonal', eval_seasonal), ('NaiveMean', eval_mean)]:
            r2_s = f"{mets['R2']:.4f}" if not np.isnan(mets['R2']) else 'N/A'
            print(f"    {model:<20s} {r2_s:>10s} {mets['MAE']:>12.0f} {mets['RMSE']:>12.0f}")

        # Show actual vs predicted
        print(f"    实际值: {[f'{v:.0f}' for v in actual]}")
        print(f"    预测值: {[f'{v:.0f}' for v in pred_batch]}")

    # ---- Summary ----
    print("\n" + "="*60)
    print("  R2 Summary")
    print("="*60)
    header = f"{'物资':<16s} {'BatchEvent':>12s} {'NaiveSeas':>12s} {'Win/Lose':>10s}"
    print(header)
    print(f"  {'-'*54}")
    wins = 0
    for mat_name in TARGET_MATERIALS:
        r2_b = all_metrics[mat_name]['BatchEvent']['R2']
        r2_s = all_metrics[mat_name]['NaiveSeasonal']['R2']
        winner = 'WIN' if not np.isnan(r2_b) and (np.isnan(r2_s) or r2_b > r2_s) else 'lose'
        if winner == 'WIN':
            wins += 1
        r2b_s = f"{r2_b:>12.4f}" if not np.isnan(r2_b) else f"{'N/A':>12s}"
        r2s_s = f"{r2_s:>12.4f}" if not np.isnan(r2_s) else f"{'N/A':>12s}"
        print(f"  {mat_name:<16s} {r2b_s} {r2s_s} {winner:>10s}")

    print(f"\n  胜出: {wins}/{len(TARGET_MATERIALS)} 种物资")

    # ---- Generate charts ----
    print(f"\n生成图表...")
    plot_results(target_data, batch_preds, batch_metrics)

    # ---- Save JSON ----
    json_output = {
        'metrics': {
            mat: {m: v for m, v in mods.items()}
            for mat, mods in all_metrics.items()
        },
        'batch_details': {k: v for k, v in batch_details.items()
                         if k not in ['month_prob'] or True},
        'run_time': datetime.now().isoformat(),
    }
    json_path = os.path.join(OUTPUT_DIR, 'batch_event_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    print(f"\n{'='*80}")
    print(f"  DONE. 核心创新: 从 '时间序列预测' 到 '批次事件预测' 的范式转换")
    print(f"{'='*80}")

    return all_metrics, batch_details

if __name__ == '__main__':
    results = main()
