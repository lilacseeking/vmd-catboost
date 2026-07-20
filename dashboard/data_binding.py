"""
dashboard/data_binding.py - 数据绑定表达式解析器

支持类JSONPath语法 + 管道操作：
  $.results.交流避雷器.*.y_pred
  $.aggregations.model_avg_metrics.*.R2 | mean | round(3)
"""

from typing import Any


class DataBindingResolver:
    """解析数据绑定表达式，从dashboard_data和context中提取数据"""

    def __init__(self, data: dict, context: dict | None = None) -> None:
        # $ 根节点包含业务数据和context层
        self._root = {**data, "__context__": context or {}}

    def resolve(self, bindings: dict) -> dict:
        """
        解析一组data_binding，返回解析后的数据字典。

        对于非表达式值（如固定字符串"R2"、数字5），直接透传。
        对于以"$."开头的表达式，执行解析。
        """
        result = {}
        for key, expr in bindings.items():
            if isinstance(expr, str) and expr.startswith("$."):
                result[key] = self._evaluate(expr)
            else:
                result[key] = expr  # 控制参数直接透传
        return result

    def resolve_path(self, expr: str) -> Any:
        """公开方法：解析单个路径表达式（供repeat展开使用）"""
        return self._evaluate(expr)

    def _evaluate(self, expr: str) -> Any:
        """
        解析单个表达式。

        示例：
            "$.results.交流避雷器.*.y_pred" -> {"LSTM": [...], "GRU": [...], ...}
            "$.aggregations.model_avg_metrics.*.R2 | mean" -> 0.87
            "$.__context__.summary" -> "综合R²表现..."
        """
        # 分离路径和管道
        parts = expr.split("|")
        path = parts[0].strip()
        pipes = [p.strip() for p in parts[1:]]

        # 解析路径
        value = self._resolve_path(path)

        # 应用管道
        for pipe in pipes:
            value = self._apply_pipe(value, pipe)

        return value

    def _resolve_path(self, path: str) -> Any:
        """解析$.a.b.*.c形式的路径"""
        segments = path.removeprefix("$.").split(".")
        return self._resolve_segments(self._root, segments)

    def _resolve_segments(self, current: Any, segments: list[str]) -> Any:
        """递归解析路径segments"""
        if not segments:
            return current

        seg = segments[0]
        remaining = segments[1:]

        if seg == "*":
            # 通配：对当前dict的每个value递归应用剩余segments
            if not isinstance(current, dict):
                raise ValueError(f"Cannot apply '*' to non-dict type: {type(current)}")
            if not remaining:
                # *是最后一个segment，返回所有value（跳过__前缀内部字段）
                return {k: v for k, v in current.items() if not k.startswith("__")}
            # 对每个子value递归
            result = {}
            for k, v in current.items():
                if k.startswith("__"):
                    continue  # 跳过__y_test__等内部字段
                try:
                    result[k] = self._resolve_segments(v, remaining)
                except (KeyError, TypeError):
                    continue  # 跳过无法解析的分支
            return result
        else:
            if isinstance(current, dict):
                if seg not in current:
                    raise KeyError(f"Key '{seg}' not found. Available: {list(current.keys())[:10]}")
                return self._resolve_segments(current[seg], remaining)
            else:
                raise KeyError(f"Cannot access '{seg}' on {type(current).__name__}")

    def _apply_pipe(self, value: Any, pipe: str) -> Any:
        """应用管道函数"""
        if pipe == "mean":
            values = list(value.values()) if isinstance(value, dict) else value
            if not values:
                return 0.0
            # 过滤None值
            numeric = [v for v in values if v is not None and isinstance(v, (int, float))]
            if not numeric:
                return 0.0
            return sum(numeric) / len(numeric)
        elif pipe.startswith("round("):
            decimals = int(pipe[6:-1])
            if isinstance(value, (int, float)):
                return round(value, decimals)
            return value
        elif pipe.startswith("top("):
            n = int(pipe[4:-1])
            if isinstance(value, dict):
                sorted_items = sorted(
                    value.items(),
                    key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                    reverse=True
                )
                return dict(sorted_items[:n])
            return value
        elif pipe == "sort_desc":
            if isinstance(value, dict):
                return dict(sorted(value.items(), key=lambda x: x[1], reverse=True))
            elif isinstance(value, list):
                return sorted(value, reverse=True)
            return value
        elif pipe == "minmax":
            values = list(value.values()) if isinstance(value, dict) else value
            if not values:
                return [0, 0]
            numeric = [v for v in values if v is not None and isinstance(v, (int, float))]
            if not numeric:
                return [0, 0]
            return [min(numeric), max(numeric)]
        return value
