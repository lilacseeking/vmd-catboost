"""
dashboard/panels/markdown_panel.py - Markdown文本/AI总结面板
"""

import re
from dashboard.registry import register_panel, PanelContext, PanelRenderResult


@register_panel("markdown_panel")
def render_markdown_panel(ctx: PanelContext) -> PanelRenderResult:
    """
    渲染Markdown文本为HTML。

    支持：标题、加粗、列表、换行（轻量内置解析，不引入额外依赖）。
    """
    resolved = ctx.resolved_data
    options = ctx.panel_config.get("options", {})

    content = resolved.get("content", "")
    font_size = options.get("font_size", 14)
    max_height = options.get("max_height", 200)

    # 轻量Markdown -> HTML转换
    html = _markdown_to_html(content)

    html_div = (
        f'<div class="markdown-content" style="font-size:{font_size}px;'
        f'max-height:{max_height}px;overflow-y:auto;line-height:1.7;">'
        f'{html}</div>'
    )

    return PanelRenderResult(html_content=html_div, echarts_option=None)


def _markdown_to_html(text: str) -> str:
    """轻量Markdown转HTML（支持加粗、标题、列表、换行）"""
    if not text:
        return "<p>暂无数据</p>"

    lines = text.split("\n")
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        # 标题
        if stripped.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<h4 style="margin:8px 0 4px;color:#d8d9da;">{stripped[4:]}</h4>')
        elif stripped.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<h3 style="margin:8px 0 4px;color:#d8d9da;">{stripped[3:]}</h3>')
        elif stripped.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<h2 style="margin:8px 0 4px;color:#d8d9da;">{stripped[2:]}</h2>')
        # 列表
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_parts.append('<ul style="margin:4px 0;padding-left:20px;">')
                in_list = True
            item = stripped[2:]
            item = _inline_format(item)
            html_parts.append(f"<li>{item}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            formatted = _inline_format(stripped)
            html_parts.append(f"<p style='margin:4px 0;'>{formatted}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def _inline_format(text: str) -> str:
    """处理行内格式：加粗、斜体"""
    # 加粗 **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#58a6ff;">\1</strong>', text)
    # 斜体 *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    return text
