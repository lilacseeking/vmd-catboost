"""
dashboard/registry.py - 窗口类型注册机制

提供 @register_panel 装饰器、PanelContext、PanelRenderResult 定义。
"""

from typing import Callable, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class PanelContext:
    """渲染上下文，传递给每个panel渲染函数"""
    panel_config: dict          # 该panel的配置（含data_binding, options）
    data: dict                  # 完整的dashboard_data
    resolved_data: dict         # data_binding解析后的数据
    panel_id: str              # panel唯一ID


@dataclass
class PanelRenderResult:
    """渲染结果"""
    html_content: str              # panel body内的HTML（div容器或table）
    echarts_option: dict | None   # ECharts配置（非图表类型为None）


@runtime_checkable
class PanelRenderer(Protocol):
    """窗口渲染器协议"""
    def __call__(self, ctx: PanelContext) -> PanelRenderResult:
        ...


# 全局注册表
_PANEL_REGISTRY: dict[str, PanelRenderer] = {}


def register_panel(panel_type: str):
    """
    装饰器：注册一个窗口类型渲染器。

    用法：
        @register_panel("line_chart")
        def render_line_chart(ctx: PanelContext) -> PanelRenderResult:
            ...
    """
    def decorator(func: PanelRenderer) -> PanelRenderer:
        if panel_type in _PANEL_REGISTRY:
            raise ValueError(f"Panel type '{panel_type}' already registered")
        _PANEL_REGISTRY[panel_type] = func
        return func
    return decorator


def get_renderer(panel_type: str) -> PanelRenderer:
    """获取已注册的渲染器，未注册则抛出明确错误"""
    if panel_type not in _PANEL_REGISTRY:
        available = ", ".join(sorted(_PANEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown panel type '{panel_type}'. Available: {available}"
        )
    return _PANEL_REGISTRY[panel_type]


def validate_config_panels(panels: list[dict]) -> list[str]:
    """校验配置中所有panel的type是否已注册，返回错误列表"""
    errors = []
    for p in panels:
        if p["type"] not in _PANEL_REGISTRY:
            errors.append(f"Panel '{p['id']}': unknown type '{p['type']}'")
    return errors


def wrap_echarts_div(panel_id: str, option: dict, height: str = "100%") -> PanelRenderResult:
    """将ECharts option包装为div + option结构"""
    html = f'<div id="chart-{panel_id}" style="width:100%;height:{height};"></div>'
    return PanelRenderResult(html_content=html, echarts_option=option)
