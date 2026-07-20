"""
dashboard/panels/line_chart.py - 多模型预测曲线对比
"""

from dashboard.registry import register_panel, PanelContext, PanelRenderResult, wrap_echarts_div


@register_panel("line_chart")
def render_line_chart(ctx: PanelContext) -> PanelRenderResult:
    """
    生成ECharts折线图配置。

    逻辑：
    1. 从resolved_data获取所有模型的y_pred
    2. 按rank_by指标排序，取top_n个模型
    3. y_test从resolved_data["y_test"]获取
    4. 生成ECharts option JSON
    """
    resolved = ctx.resolved_data
    options = ctx.panel_config.get("options", {})

    # 获取series数据（dict: model_name -> y_pred list）
    series_data = resolved.get("series", {})
    y_test = resolved.get("y_test", [])
    x_axis = resolved.get("x_axis", [])
    top_n = resolved.get("top_n", 5)
    rank_by = resolved.get("rank_by", "R2")

    # 按rank_by指标排序取top_n
    if isinstance(series_data, dict) and series_data:
        # 获取每个模型的指标用于排序
        data = ctx.data
        # 从panel title推断物资名（repeat展开后title含物资名）
        # 更可靠：从data_binding的series路径推断
        material = _infer_material(ctx)

        model_scores = {}
        if material and material in data.get("results", {}):
            for model_name in series_data.keys():
                model_data = data["results"][material].get(model_name, {})
                score = model_data.get("metrics", {}).get(rank_by, 0)
                if score is not None:
                    model_scores[model_name] = score

        # 排序取top_n
        if model_scores:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            selected_models = [m[0] for m in sorted_models[:top_n]]
        else:
            selected_models = list(series_data.keys())[:top_n]
    else:
        selected_models = []

    # 构建ECharts series
    model_names = selected_models
    echarts_series = []
    for model in model_names:
        y_pred = series_data.get(model, [])
        echarts_series.append({
            "name": model,
            "type": "line",
            "data": y_pred,
            "smooth": options.get("smooth", True),
            "lineWidth": 2,
        })

    # 添加实际值曲线
    if y_test:
        y_test_style = options.get("y_test_style", "dashed_scatter")
        echarts_series.append({
            "name": "实际值",
            "type": "line",
            "data": y_test,
            "lineStyle": {"type": "dashed", "width": 2},
            "symbol": "circle",
            "symbolSize": 6,
            "itemStyle": {"color": "#fff"},
        })

    legend_data = model_names + (["实际值"] if y_test else [])

    # dataZoom缩放支持：滑块 + 内部滚轮缩放
    use_data_zoom = options.get("data_zoom", False)
    data_zoom_cfg = []
    if use_data_zoom:
        data_zoom_cfg = [
            {"type": "inside", "start": 0, "end": 100},
            {
                "type": "slider",
                "start": 0,
                "end": 100,
                "height": 18,
                "bottom": 4,
                "borderColor": "#444",
                "fillerColor": "rgba(50, 116, 217, 0.25)",
                "handleStyle": {"color": "#3274d9"},
                "textStyle": {"color": "#999", "fontSize": 10},
            },
        ]

    option = {
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": legend_data,
            "textStyle": {"color": "#ccc", "fontSize": 11},
            "top": 0,
            "type": "scroll",
        },
        "grid": {
            "left": "3%",
            "right": "4%",
            "bottom": "12%" if use_data_zoom else "3%",
            "top": "15%",
            "containLabel": True,
        },
        "xAxis": {
            "type": "category",
            "data": x_axis,
            "axisLabel": {"color": "#999"},
            "axisLine": {"lineStyle": {"color": "#444"}},
            "boundaryGap": False,
        },
        "yAxis": {
            "type": "value",
            "name": "需求量",
            "nameTextStyle": {"color": "#999"},
            "axisLabel": {"color": "#999"},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "dataZoom": data_zoom_cfg,
        "series": echarts_series,
    }

    return wrap_echarts_div(ctx.panel_id, option)


def _infer_material(ctx: PanelContext) -> str | None:
    """从panel配置推断物资名"""
    # 方法1：从data_binding的series路径中提取
    binding = ctx.panel_config.get("data_binding", {})
    series_expr = binding.get("series", "")
    # 格式: $.results.交流避雷器.*.y_pred
    if "$.results." in series_expr:
        parts = series_expr.replace("$.results.", "").split(".")
        if parts and parts[0] != "*":
            return parts[0]
    # 方法2：从title中提取（repeat展开后title为 "物资名 - 多模型预测对比"）
    title = ctx.panel_config.get("title", "")
    if " - " in title:
        return title.split(" - ")[0]
    return None
