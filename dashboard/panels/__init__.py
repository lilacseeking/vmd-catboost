"""
dashboard/panels/__init__.py - 自动导入所有panel模块

触发@register_panel装饰器执行注册。
新增panel类型时，只需在此目录新建文件并使用装饰器即可。
"""

import importlib
import pkgutil
import os
import warnings

_package_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([_package_dir]):
    try:
        importlib.import_module(f".{module_name}", package=__name__)
    except Exception as e:
        warnings.warn(
            f"[Dashboard] Panel模块 '{module_name}' 导入失败: {e}. "
            f"该窗口类型将不可用。",
            ImportWarning,
            stacklevel=2
        )
