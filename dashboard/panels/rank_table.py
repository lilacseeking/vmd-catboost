"""
dashboard/panels/rank_table.py - 指标排名表格
"""

from dashboard.registry import register_panel, PanelContext, PanelRenderResult


@register_panel("rank_table")
def render_rank_table(ctx: PanelContext) -> PanelRenderResult:
    """
    生成HTML表格（非ECharts）。

    逻辑：
    1. 按group_by聚合（model维度：对每个模型取所有物资的指标均值）
    2. 按metric排序（desc/asc）
    3. 取top_n行
    4. 生成带样式的HTML table，前3名高亮
    """
    resolved = ctx.resolved_data
    options = ctx.panel_config.get("options", {})
    data = ctx.data

    metric = resolved.get("metric", "R2")
    sort_dir = resolved.get("sort", "desc")
    top_n = resolved.get("top_n", 10)
    group_by = resolved.get("group_by", "model")

    materials = data.get("meta", {}).get("materials", [])
    models = data.get("meta", {}).get("models", [])
    columns = options.get("columns", ["排名", "模型", "R²均值", "最佳物资", "最差物资"])
    highlight_top = options.get("highlight_top", 3)

    # 对每个模型计算跨物资的指标均值
    model_stats = []
    for model in models:
        scores = []
        material_scores = {}
        for m in materials:
            model_data = data.get("results", {}).get(m, {}).get(model, {})
            val = model_data.get("metrics", {}).get(metric)
            if val is not None:
                scores.append(val)
                material_scores[m] = val

        if not scores:
            continue

        avg_score = sum(scores) / len(scores)
        best_material = max(material_scores, key=material_scores.get) if material_scores else ""
        worst_material = min(material_scores, key=material_scores.get) if material_scores else ""

        model_stats.append({
            "model": model,
            "avg": avg_score,
            "best_material": best_material,
            "worst_material": worst_material,
        })

    # 排序
    reverse = sort_dir == "desc"
    model_stats.sort(key=lambda x: x["avg"], reverse=reverse)

    # 取top_n
    model_stats = model_stats[:top_n]

    # 生成HTML表格
    rows_html = []
    for idx, stat in enumerate(model_stats):
        rank = idx + 1
        highlight = ' class="highlight"' if rank <= highlight_top else ''
        rows_html.append(
            f'<tr{highlight}>'
            f'<td>{rank}</td>'
            f'<td>{stat["model"]}</td>'
            f'<td>{stat["avg"]:.4f}</td>'
            f'<td>{stat["best_material"]}</td>'
            f'<td>{stat["worst_material"]}</td>'
            f'</tr>'
        )

    header_html = "".join(f"<th>{col}</th>" for col in columns)

    table_html = f'''
    <div style="overflow-y:auto;max-height:100%;">
    <table class="rank-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
    </table>
    </div>
    '''

    return PanelRenderResult(html_content=table_html, echarts_option=None)
