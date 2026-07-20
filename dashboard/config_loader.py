"""
dashboard/config_loader.py - 配置加载器

加载JSON配置 + JSON Schema校验 + 语义校验。
"""

import json
from pathlib import Path

_MODULE_DIR = Path(__file__).parent


class ConfigValidationError(Exception):
    """配置校验失败"""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"配置校验失败: {'; '.join(errors)}")


class ConfigLoader:
    """加载并校验看板配置"""

    SCHEMA_PATH = _MODULE_DIR / "schemas" / "config_schema.json"

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def load(self) -> dict:
        """
        加载配置文件，执行Schema校验和语义校验。

        Raises:
            ConfigValidationError: 配置不合法时抛出
            FileNotFoundError: 配置文件不存在
        """
        # 1. 读取JSON
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 2. JSON Schema校验（如果jsonschema可用）
        try:
            import jsonschema
            if self.SCHEMA_PATH.exists():
                with open(self.SCHEMA_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                jsonschema.validate(config, schema)
        except ImportError:
            pass  # jsonschema不可用时跳过Schema校验
        except Exception as e:
            raise ConfigValidationError([f"Schema校验失败: {e}"])

        # 3. 语义校验：panel type是否已注册
        from dashboard.registry import validate_config_panels
        errors = validate_config_panels(config.get("panels", []))
        if errors:
            raise ConfigValidationError(errors)

        # 4. 语义校验：grid布局
        self._validate_grid(config)

        # 5. 填充默认值
        config = self._apply_defaults(config)

        return config

    def _validate_grid(self, config: dict) -> None:
        """检查grid定位不越界（警告级别，不阻断）"""
        columns = config.get("layout", {}).get("columns", 12)
        for panel in config.get("panels", []):
            grid = panel.get("grid", {})
            col = grid.get("col", 1)
            col_span = grid.get("col_span", 1)
            if isinstance(col, int) and col + col_span - 1 > columns:
                import warnings
                warnings.warn(
                    f"[Dashboard] Panel '{panel['id']}' grid越界: "
                    f"col={col} + col_span={col_span} > columns={columns}",
                    stacklevel=2
                )

    def _apply_defaults(self, config: dict) -> dict:
        """为缺失的可选字段填充默认值"""
        config.setdefault("theme", "dark")
        config.setdefault("subtitle", "")
        layout = config.setdefault("layout", {})
        layout.setdefault("columns", 12)
        layout.setdefault("row_height", 80)
        layout.setdefault("gap", 12)
        layout.setdefault("padding", 16)
        return config
