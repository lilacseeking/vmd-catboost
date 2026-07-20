"""
dashboard/builder.py - DashboardBuilder 主构建器

编排整体流程：数据校验 -> 配置加载 -> 序列化 -> AI总结 -> repeat展开 -> 渲染 -> Jinja2输出
"""

import os
import json
import copy
from pathlib import Path

from dashboard.collector import ResultCollector
from dashboard.config_loader import ConfigLoader
from dashboard.data_binding import DataBindingResolver
from dashboard.summary import SummaryEngine
from dashboard.registry import get_renderer, PanelContext
from dashboard.serializer import NumpyEncoder

# 模块根目录（解决非项目根目录运行时相对路径失败问题）
_MODULE_DIR = Path(__file__).parent


class DashboardBuilder:
    """看板构建器 - 编排整体流程"""

    DEFAULT_CONFIG = _MODULE_DIR / "dashboard_config.json"
    DEFAULT_OUTPUT = Path("output") / "dashboard.html"

    def __init__(
        self,
        collector: ResultCollector,
        config_path: str | None = None,
        output_path: str | None = None,
        debug: bool = False
    ) -> None:
        self.collector = collector
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG
        self.output_path = Path(output_path) if output_path else self.DEFAULT_OUTPUT
        self.debug = debug

    def build(self) -> str:
        """
        执行完整构建流程，返回输出文件路径。

        流程：
        0. 前置检查（数据非空）
        1. 数据校验
        2. 加载并校验配置
        3. 序列化数据
        4. 生成AI总结（存入context层）
        5. 展开repeat配置
        6. 渲染所有panel（每个独立try/except）
        7. Jinja2模板渲染
        8. 写入HTML文件
        """
        # 确保panels已注册（触发自动导入）
        import dashboard.panels  # noqa: F401

        # Step 0: 前置检查
        if not self.collector.materials:
            print("[Dashboard] 警告：无预测数据，跳过看板生成")
            return ""

        # Step 1: 数据校验
        warnings_list = self.collector.validate()
        for w in warnings_list:
            print(f"[Dashboard] 数据警告: {w}")

        # Step 2: 配置
        config = ConfigLoader(str(self.config_path)).load()

        # Step 3: 数据
        data = self.collector.to_dashboard_data()

        # Step 4: AI总结 -> 独立context层
        context = {
            "summary": SummaryEngine().generate(data, mode="rule"),
            "generated_at": data["generated_at"],
        }

        # Step 5: 展开repeat配置
        panels_config = self._expand_repeats(config["panels"], data, config["layout"])

        # Step 6: 渲染panels
        rendered_panels = []
        for panel_cfg in panels_config:
            try:
                renderer = get_renderer(panel_cfg["type"])
                resolved = DataBindingResolver(data, context).resolve(
                    panel_cfg.get("data_binding", {})
                )
                ctx = PanelContext(
                    panel_config=panel_cfg,
                    data=data,
                    resolved_data=resolved,
                    panel_id=panel_cfg["id"]
                )
                result = renderer(ctx)
                rendered_panels.append({
                    **panel_cfg,
                    "html_content": result.html_content,
                    "echarts_option": result.echarts_option
                })
            except Exception as e:
                rendered_panels.append({
                    **panel_cfg,
                    "html_content": f'<div class="panel-error">渲染失败: {e}</div>',
                    "echarts_option": None
                })
                print(f"[Dashboard] Panel '{panel_cfg['id']}' 渲染失败: {e}")

        # Step 7: Jinja2渲染
        html = self._render_template(config, rendered_panels, data, context)

        # Step 8: 输出
        output_dir = self.output_path.parent
        os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)

        if self.debug:
            debug_path = output_dir / "dashboard_data.json"
            self._dump_debug_data(data, debug_path)

        print(f"[Dashboard] 看板已生成: {self.output_path}")
        return str(self.output_path)

    def _expand_repeats(self, panels: list[dict], data: dict, layout: dict) -> list[dict]:
        """
        展开含repeat配置的panel为多个实例，并执行顺序自动布局。

        布局算法（消除空白间隙）：
        - 第一个repeat panel之前的固定panel：保留硬编码行号
        - repeat panel及其后的所有panel：按配置顺序依次自动排列，
          紧跟在前一个panel之后，不依赖硬编码行号。
          这样物资数量变化时（如电抗器保护被跳过），后续panel自动上移，
          不会产生空白区域。
        """
        columns = layout.get("columns", 12)

        # Phase 1: 找到第一个repeat panel的位置，计算其之前固定panel的最大结束行
        first_repeat_idx = next(
            (i for i, p in enumerate(panels) if "repeat" in p), None
        )

        max_fixed_row = 0
        if first_repeat_idx is not None:
            for panel in panels[:first_repeat_idx]:
                grid = panel["grid"]
                row = grid.get("row", 1)
                if isinstance(row, int):
                    max_fixed_row = max(max_fixed_row, row + grid.get("row_span", 1) - 1)

        # Phase 2: 顺序展开 + 自动布局
        expanded = []
        auto_row_cursor = max_fixed_row + 1  # auto区域起始行

        idx_panel = 0
        total = len(panels)
        while idx_panel < total:
            panel = panels[idx_panel]

            # repeat之前的panel：保留硬编码行号
            if first_repeat_idx is None or idx_panel < first_repeat_idx:
                expanded.append(panel)
                idx_panel += 1
                continue

            # repeat panel：展开为多个实例
            if "repeat" in panel:
                repeat_cfg = panel["repeat"]
                items = DataBindingResolver(data, {}).resolve_path(repeat_cfg["foreach"])
                var_name = repeat_cfg["as"]
                id_suffix_key = repeat_cfg.get("id_suffix", "idx")
                col_span = panel["grid"]["col_span"]
                row_span = panel["grid"]["row_span"]
                items_per_row = max(1, columns // col_span)

                for idx, item in enumerate(items):
                    instance = self._substitute_template(panel, var_name, item, idx, id_suffix_key)
                    row_offset = (idx // items_per_row) * row_span
                    col_offset = (idx % items_per_row) * col_span
                    instance["grid"]["row"] = auto_row_cursor + row_offset
                    instance["grid"]["col"] = 1 + col_offset
                    expanded.append(instance)

                total_rows_used = ((len(items) - 1) // items_per_row + 1) * row_span
                auto_row_cursor += total_rows_used
                idx_panel += 1
                continue

            # repeat之后的非repeat panel：按原始row值分组，
            # 同一原始row的panel排在同一输出行（保留各自col位置，如1/5/9并排），
            # 行号统一重排到auto_row_cursor，消除空白间隙。
            orig_row = panel["grid"].get("row", "auto")
            group = [panel]
            j = idx_panel + 1
            while (j < total and "repeat" not in panels[j]
                   and panels[j]["grid"].get("row", "auto") == orig_row):
                group.append(panels[j])
                j += 1

            group_max_span = 0
            for p in group:
                new_grid = {**p["grid"], "row": auto_row_cursor}
                group_max_span = max(group_max_span, p["grid"]["row_span"])
                expanded.append({**p, "grid": new_grid})
            auto_row_cursor += group_max_span
            idx_panel = j

        return expanded

    def _substitute_template(self, panel: dict, var_name: str, value: str,
                              idx: int, id_suffix_key: str) -> dict:
        """递归替换配置中的${var}占位符"""
        result = copy.deepcopy(panel)
        # 替换id
        result["id"] = result["id"].replace(f"${{{id_suffix_key}}}", str(idx))
        # 替换title
        result["title"] = result["title"].replace(f"${{{var_name}}}", value)
        # 替换data_binding中的表达式
        for key, expr in result.get("data_binding", {}).items():
            if isinstance(expr, str):
                result["data_binding"][key] = expr.replace(f"${{{var_name}}}", value)
        # 移除repeat字段
        result.pop("repeat", None)
        return result

    def _render_template(self, config, panels, data, context) -> str:
        """加载Jinja2模板并渲染"""
        import jinja2
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_MODULE_DIR / "templates")),
            autoescape=False
        )
        template = env.get_template("base.html")

        # 读取ECharts源码用于内联
        echarts_path = _MODULE_DIR / "static" / "echarts.min.js"
        if echarts_path.exists():
            with open(echarts_path, "r", encoding="utf-8") as f:
                echarts_source = f.read()
        else:
            echarts_source = "/* ECharts not found */"
            print(f"[Dashboard] 警告: ECharts文件不存在: {echarts_path}")

        return template.render(
            title=config["title"],
            subtitle=config.get("subtitle", ""),
            layout=config["layout"],
            panels=panels,
            echarts_source=echarts_source,
            generated_at=context["generated_at"]
        )

    def _dump_debug_data(self, data: dict, path: Path) -> None:
        """输出调试用JSON数据"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
