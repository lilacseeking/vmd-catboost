"""
dashboard/panels/heatmap.py - 物资×模型热力图
"""

from dashboard.registry import register_panel, PanelContext, PanelRenderResult, wrap_echarts_div


@register_panel("heatmap")
def render_heatmap(ctx: PanelContext) -> PanelRenderResult:
    """
    生成ECharts热力图。

    data_binding字段说明：
    - x_axis: 路径表达式，解析为模型名称列表
    - y_axis: 路径表达式，解析为物资名称列表
    - metric: 控制参数（非路径表达式），指定从metrics中取哪个指标

    逻辑：
    1. x_axis = models, y_axis = materials
    2. 遍历 results[material][model].metrics[metric] 构建 [x_idx, y_idx, value] 数组
    3. visualMap 映射颜色
    """
    resolved = ctx.resolved_data
    options = ctx.panel_config.get("options", {})
    data = ctx.data

    models = resolved.get("x_axis", [])
    materials = resolved.get("y_axis", [])
    metric = resolved.get("metric", "R2")  # 控制参数，直接透传的字符串

    color_range = options.get("color_range", ["#f5222d", "#faad14", "#52c41a"])
    min_val = options.get("min", 0)
    max_val = options.get("max", 1)
    show_value = options.get("show_value", True)

    # 构建热力数据点
    data_points = []
    actual_values = []
    for y_idx, material in enumerate(materials):
        material_results = data.get("results", {}).get(material, {})
        for x_idx, model in enumerate(models):
            model_data = material_results.get(model, {})
            val = model_data.get("metrics", {}).get(metric)
            if val is not None:
                rounded = round(val, 3)
                data_points.append([x_idx, y_idx, rounded])
                actual_values.append(rounded)
            else:
                data_points.append([x_idx, y_idx, "-"])

    # 动态计算min/max（如果配置未指定或数据超出范围）
    if actual_values:
        data_min = min(actual_values)
        data_max = max(actual_values)
        # 使用配置值和数据值的合理范围
        vis_min = min(min_val, data_min)
        vis_max = max(max_val, data_max)
    else:
        vis_min = min_val
        vis_max = max_val

    option = {
        "tooltip": {
            "position": "top",
            "formatter": None,  # 使用默认
        },
        "grid": {
            "left": "12%",
            "right": "8%",
            "bottom": "15%",
            "top": "3%",
        },
        "xAxis": {
            "type": "category",
            "data": models,
            "axisLabel": {"rotate": 35, "color": "#ccc", "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#444"}},
            "splitArea": {"show": False},
        },
        "yAxis": {
            "type": "category",
            "data": materials,
            "axisLabel": {"color": "#ccc", "fontSize": 11},
            "axisLine": {"lineStyle": {"color": "#444"}},
            "splitArea": {"show": False},
        },
        "visualMap": {
            "min": vis_min,
            "max": vis_max,
            "calculable": True,
            "orient": "vertical",
            "right": "2%",
            "top": "center",
            "inRange": {"color": color_range},
            "textStyle": {"color": "#ccc"},
        },
        "series": [{
            "type": "heatmap",
            "data": data_points,
            "label": {
                "show": show_value,
                "fontSize": 10,
                "color": "#fff",
            },
            "emphasis": {
                "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
            },
        }],
    }

    return wrap_echarts_div(ctx.panel_id, option)
