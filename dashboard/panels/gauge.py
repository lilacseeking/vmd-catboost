"""
dashboard/panels/gauge.py - 仪表盘
"""

from dashboard.registry import register_panel, PanelContext, PanelRenderResult, wrap_echarts_div


@register_panel("gauge")
def render_gauge(ctx: PanelContext) -> PanelRenderResult:
    """
    生成ECharts仪表盘。

    逻辑：
    1. 取resolved_data["value"]作为当前值
    2. 根据thresholds设置颜色分段
    3. inverse=true时（如sMAPE，越低越好），颜色反转
    """
    resolved = ctx.resolved_data
    options = ctx.panel_config.get("options", {})

    value = resolved.get("value", 0)
    label = resolved.get("label", "")
    min_val = options.get("min", 0)
    max_val = options.get("max", 1)
    thresholds = options.get("thresholds", [])
    inverse = options.get("inverse", False)

    # 确保value是数值
    if not isinstance(value, (int, float)):
        value = 0

    # 构建颜色分段
    if thresholds:
        if inverse:
            # 越低越好：颜色从绿到红
            color_stops = []
            for t in thresholds:
                ratio = (t["value"] - min_val) / (max_val - min_val) if max_val > min_val else 0
                color_stops.append([ratio, t["color"]])
            color_stops.append([1, thresholds[-1]["color"] if thresholds else "#52c41a"])
        else:
            # 越高越好：颜色从红到绿
            color_stops = []
            for t in thresholds:
                ratio = (t["value"] - min_val) / (max_val - min_val) if max_val > min_val else 0
                color_stops.append([ratio, t["color"]])
            color_stops.append([1, thresholds[-1]["color"] if thresholds else "#52c41a"])
    else:
        color_stops = [[0.6, "#f5222d"], [0.8, "#faad14"], [1, "#52c41a"]]

    option = {
        "series": [{
            "type": "gauge",
            "min": min_val,
            "max": max_val,
            "startAngle": 210,
            "endAngle": -30,
            "progress": {"show": True, "width": 12},
            "pointer": {"show": True, "length": "60%", "width": 4},
            "axisLine": {
                "lineStyle": {
                    "width": 12,
                    "color": color_stops,
                }
            },
            "axisTick": {"show": False},
            "splitLine": {"show": False},
            "axisLabel": {"show": False},
            "detail": {
                "formatter": f"{{value}}",
                "fontSize": 22,
                "fontWeight": "bold",
                "color": "#d8d9da",
                "offsetCenter": [0, "70%"],
            },
            "title": {
                "show": True,
                "color": "#8e8e8e",
                "fontSize": 12,
                "offsetCenter": [0, "90%"],
            },
            "data": [{"value": round(value, 3), "name": label}],
        }]
    }

    return wrap_echarts_div(ctx.panel_id, option)
