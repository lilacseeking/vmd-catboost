"""
dashboard - 电力配电网物资需求预测看板模块

导出公共API：DashboardBuilder, ResultCollector
"""

try:
    from dashboard.collector import ResultCollector
    from dashboard.builder import DashboardBuilder
except ImportError as _e:
    # 延迟导入失败时提供明确错误
    raise ImportError(
        f"Dashboard模块初始化失败，请确认依赖已安装 (jinja2, jsonschema): {_e}"
    ) from _e

__all__ = ["DashboardBuilder", "ResultCollector"]
