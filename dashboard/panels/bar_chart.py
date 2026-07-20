"""
dashboard/panels/bar_chart.py - 指标对比柱状图
"""

from dashboard.registry import register_panel, PanelContext, PanelRenderResult, wrap_echarts_div


@register_panel("bar_chart")
def render_bar_chart(ctx: PanelContext) -> PanelRenderResult:
    """
    生成ECharts柱状图。

    支持：
    - horizontal: 横向/纵向
    - sort: 排序方向
    - show_value: 柱顶/柱侧显示数值
    """
    resolved = ctx.resolved_data
    options = ctx.panel_config.get("options", {})

    categories = resolved.get("categories", [])
    values = resolved.get("values", [])
    horizontal = options.get("horizontal", False)
    show_value = options.get("show_value", True)
    color = options.get("color", "#1890ff")
    sort_dir = options.get("sort", None)

    # 处理values可能是dict的情况
    if isinstance(values, dict):
        values = list(values.values())
    if isinstance(categories, dict):
        categories = list(categories.keys())

    # 排序
    if sort_dir and categories and values:
        paired = list(zip(categories, values))
        reverse = sort_dir == "desc"
        paired.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=reverse)
        categories = [p[0] for p in paired]
        values = [p[1] for p in paired]

    if horizontal:
        option = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": "3%", "right": "10%", "bottom": "3%", "top": "3%", "containLabel": True},
            "xAxis": {
                "type": "value",
                "axisLabel": {"color": "#999"},
                "splitLine": {"lineStyle": {"color": "#333"}},
            },
            "yAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {"color": "#ccc", "fontSize": 11},
                "axisLine": {"lineStyle": {"color": "#444"}},
            },
            "series": [{
                "type": "bar",
                "data": values,
                "itemStyle": {"color": color, "borderRadius": [0, 3, 3, 0]},
                "label": {
                    "show": show_value,
                    "position": "right",
                    "color": "#ccc",
                    "fontSize": 11,
                    "formatter": "{c}" if not isinstance(values[0] if values else 0, float) else None,
                },
            }],
        }
    else:
        option = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "top": "8%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {"color": "#ccc", "fontSize": 11, "rotate": 15 if len(categories) > 5 else 0},
                "axisLine": {"lineStyle": {"color": "#444"}},
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"color": "#999"},
                "splitLine": {"lineStyle": {"color": "#333"}},
            },
            "series": [{
                "type": "bar",
                "data": values,
                "itemStyle": {"color": color, "borderRadius": [3, 3, 0, 0]},
                "label": {
                    "show": show_value,
                    "position": "top",
                    "color": "#ccc",
                    "fontSize": 11,
                },
            }],
        }

    return wrap_echarts_div(ctx.panel_id, option)
