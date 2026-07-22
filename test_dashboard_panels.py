"""
测试脚本：验证dashboard改造后面板渲染正确性。
使用现有output/dashboard_data.json数据，无需重跑预测流程。
"""
import sys
import json
from pathlib import Path

# 确保项目根目录在path中
project_root = Path(r"C:\Users\董文涛\PycharmProjects\vmd-catboost")
sys.path.insert(0, str(project_root))

from dashboard.collector import ResultCollector
from dashboard.builder import DashboardBuilder


def load_collector_from_json(json_path: str) -> ResultCollector:
    """从dashboard_data.json反向构建ResultCollector"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    collector = ResultCollector()
    meta = data["meta"]
    collector.set_meta(
        forecast_horizon=meta.get("forecast_horizon", 12),
        time_labels=meta.get("time_labels", []),
    )

    results = data["results"]
    for material, material_data in results.items():
        y_test = material_data.get("__y_test__", [])
        for model_name, model_data in material_data.items():
            if model_name.startswith("__"):
                continue
            raw_result = {
                "y_pred": model_data["y_pred"],
                "y_test": y_test,
                "metrics": model_data["metrics"],
            }
            collector.collect(material, model_name, raw_result)

    return collector


def main():
    data_path = project_root / "output" / "dashboard_data.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} 不存在")
        return 1

    print(f"[Test] 加载数据: {data_path}")
    collector = load_collector_from_json(str(data_path))

    materials = collector.materials
    models = collector.models
    print(f"[Test] 物资数: {len(materials)}, 模型数: {len(models)}")
    print(f"[Test] 物资: {materials}")
    print(f"[Test] 模型: {models}")

    # 构建看板
    output_path = project_root / "output" / "dashboard_test.html"
    builder = DashboardBuilder(
        collector=collector,
        output_path=str(output_path),
        debug=True
    )
    result_path = builder.build()

    if not result_path:
        print("[Test] FAIL: 看板生成失败")
        return 1

    print(f"\n[Test] 看板已生成: {result_path}")

    # 验证面板数量
    # 预期: 原有固定面板(gauge_r2, gauge_mape, summary, heatmap, rank_table, bar_material, bar_model)
    #       + 5个line_chart(物资数)
    #       + 5个material_model_rank(物资数)
    #       + N个model_material_rank(模型数)
    from dashboard.config_loader import ConfigLoader
    from dashboard.data_binding import DataBindingResolver
    import dashboard.panels  # noqa: trigger registration

    config = ConfigLoader(str(DashboardBuilder.DEFAULT_CONFIG)).load()
    data = collector.to_dashboard_data()

    # 统计repeat展开后的面板数
    builder2 = DashboardBuilder(collector=collector)
    panels_expanded = builder2._expand_repeats(config["panels"], data, config["layout"])

    panel_types = {}
    for p in panels_expanded:
        t = p["type"]
        panel_types[t] = panel_types.get(t, 0) + 1

    print(f"\n[Test] 展开后面板总数: {len(panels_expanded)}")
    print(f"[Test] 面板类型分布: {panel_types}")

    # 验证预期
    n_materials = len(materials)
    n_models = len(models)
    expected_metrics_tables = n_materials + n_models  # A + C
    actual_metrics_tables = panel_types.get("metrics_rank_table", 0)

    print(f"\n[Test] 验证:")
    print(f"  metrics_rank_table面板数: {actual_metrics_tables} (预期: {expected_metrics_tables})")
    print(f"  line_chart面板数: {panel_types.get('line_chart', 0)} (预期: {n_materials})")

    # 检查line_chart的top_n配置
    for p in panels_expanded:
        if p["type"] == "line_chart":
            top_n = p.get("data_binding", {}).get("top_n")
            if top_n != 7:
                print(f"  [WARN] line_chart '{p['id']}' top_n={top_n}, 预期7")
            break

    # 验证HTML文件内容
    with open(result_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 检查新面板标题是否出现在HTML中
    for material in materials:
        expected_title = f"{material} - 模型效果排名 Top-7"
        if expected_title in html_content:
            print(f"  [PASS] 物资面板标题存在: {expected_title}")
        else:
            print(f"  [FAIL] 物资面板标题缺失: {expected_title}")

    for model in models:
        expected_title = f"{model} - 物资效果排名 Top-7"
        if expected_title in html_content:
            print(f"  [PASS] 模型面板标题存在: {expected_title}")
        else:
            print(f"  [FAIL] 模型面板标题缺失: {expected_title}")

    # 检查表格中是否有rank-table类
    rank_table_count = html_content.count('class="rank-table"')
    print(f"\n[Test] HTML中rank-table数量: {rank_table_count}")

    if actual_metrics_tables == expected_metrics_tables:
        print("\n[Test] === ALL CHECKS PASSED ===")
        return 0
    else:
        print("\n[Test] === SOME CHECKS FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
