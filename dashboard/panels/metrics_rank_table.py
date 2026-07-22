"""
dashboard/panels/metrics_rank_table.py - 物资/模型维度效果参数排名表格

支持两种维度：
- dimension="material": 给定物资，展示该物资下所有模型按R²降序Top-N，列含模型名/R²/sMAPE/RMSE/MAE
- dimension="model": 给定模型，展示该模型预测各物资按R²降序Top-N，列含物资名/R²/sMAPE/RMSE/MAE
"""

from dashboard.registry import register_panel, PanelContext, PanelRenderResult


@register_panel("metrics_rank_table")
def render_metrics_rank_table(ctx: PanelContext) -> PanelRenderResult:
    """
    生成HTML表格（非ECharts）。

    通过resolved_data中的dimension和target参数决定展示维度：
    - dimension="material", target="交流避雷器" → 该物资下各模型排名
    - dimension="model", target="CatBoost" → 该模型预测各物资排名
    """
    resolved = ctx.resolved_data
    options = ctx.panel_config.get("options", {})
    data = ctx.data

    dimension = resolved.get("dimension", "material")
    target = resolved.get("target", "")
    top_n = resolved.get("top_n", 7)
    rank_by = resolved.get("rank_by", "R2")
    highlight_top = options.get("highlight_top", 3)

    # 展示指标列
    display_metrics = options.get("metrics", ["R2", "sMAPE", "RMSE", "MAE"])
    metric_labels = options.get("metric_labels", {
        "R2": "R²", "sMAPE": "sMAPE", "RMSE": "RMSE", "MAE": "MAE"
    })

    results = data.get("results", {})
    materials = data.get("meta", {}).get("materials", [])
    models = data.get("meta", {}).get("models", [])

    rows = []

    if dimension == "material":
        # 给定物资，排名各模型
        entity_label = "模型"
        material_data = results.get(target, {})
        for model_name in models:
            model_entry = material_data.get(model_name, {})
            metrics = model_entry.get("metrics", {})
            r2_val = metrics.get(rank_by)
            if r2_val is None:
                continue
            rows.append({
                "name": model_name,
                "metrics": metrics,
                "score": r2_val,
            })
    elif dimension == "model":
        # 给定模型，排名各物资
        entity_label = "物资"
        for material in materials:
            material_data = results.get(material, {})
            model_entry = material_data.get(target, {})
            metrics = model_entry.get("metrics", {})
            r2_val = metrics.get(rank_by)
            if r2_val is None:
                continue
            rows.append({
                "name": material,
                "metrics": metrics,
                "score": r2_val,
            })
    else:
        return PanelRenderResult(
            html_content=f'<div class="panel-error">未知dimension: {dimension}</div>',
            echarts_option=None
        )

    # 按rank_by降序排序，取top_n
    rows.sort(key=lambda x: x["score"], reverse=True)
    rows = rows[:top_n]

    # 生成HTML表格
    header_cols = ["排名", entity_label] + [metric_labels.get(m, m) for m in display_metrics]
    header_html = "".join(f"<th>{col}</th>" for col in header_cols)

    rows_html = []
    for idx, row in enumerate(rows):
        rank = idx + 1
        highlight = ' class="highlight"' if rank <= highlight_top else ''
        cells = [f"<td>{rank}</td>", f"<td>{row['name']}</td>"]
        for m in display_metrics:
            val = row["metrics"].get(m)
            if val is None:
                cells.append("<td>-</td>")
            elif m == "R2":
                cells.append(f"<td>{val:.4f}</td>")
            elif m == "sMAPE":
                cells.append(f"<td>{val:.2f}%</td>")
            else:
                cells.append(f"<td>{val:.4f}</td>")
        rows_html.append(f"<tr{highlight}>{''.join(cells)}</tr>")

    table_html = f'''
    <div style="overflow-y:auto;max-height:100%;">
    <table class="rank-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
    </table>
    </div>
    '''

    return PanelRenderResult(html_content=table_html, echarts_option=None)
