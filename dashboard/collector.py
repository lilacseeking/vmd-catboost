"""
dashboard/collector.py - 数据收集器

从main.py收集预测结果，numpy序列化，aggregations计算，统一数据结构。
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from dashboard.serializer import to_json_safe


@dataclass
class PredictionResult:
    """单个预测结果的标准化结构"""
    y_pred: list[float]
    metrics: dict[str, float]


class ResultCollector:
    """
    数据收集器 - 在预测循环中逐步收集结果。

    职责：
    - 收集：接收main.py的原始预测结果
    - 序列化：numpy -> JSON-safe Python对象
    - 聚合：计算aggregations（best_model, avg_metrics等）
    """

    # 必需指标列表
    REQUIRED_METRICS = ["MSE", "RMSE", "MAE", "R2", "sMAPE", "MASE", "WRMSSE", "zero_acc"]

    def __init__(self) -> None:
        self._data: dict[str, dict[str, PredictionResult]] = {}
        self._y_test: dict[str, list[float]] = {}
        self._meta: dict = {}

    def collect(self, material: str, model_name: str, raw_result: dict) -> None:
        """
        收集单个模型的预测结果。

        防御性编程：内部try/except保护，任何异常仅记录警告，
        不中断main.py的预测主循环。
        """
        try:
            if material not in self._data:
                self._data[material] = {}

            # 序列化numpy -> list（每物资存一次y_test）
            if material not in self._y_test and 'y_test' in raw_result:
                self._y_test[material] = to_json_safe(raw_result['y_test'])

            self._data[material][model_name] = PredictionResult(
                y_pred=to_json_safe(raw_result['y_pred']),
                metrics={k: to_json_safe(v) for k, v in raw_result['metrics'].items()}
            )
        except Exception as e:
            warnings.warn(
                f"[Dashboard] collect()失败 material={material}, model={model_name}: {e}",
                stacklevel=2
            )

    def set_meta(self, **kwargs) -> None:
        """设置元信息：time_labels, forecast_horizon等"""
        self._meta.update(kwargs)

    @property
    def materials(self) -> list[str]:
        """已收集的所有物资名称"""
        return list(self._data.keys())

    @property
    def models(self) -> list[str]:
        """已收集的所有模型名称（取第一个物资的模型列表）"""
        if not self._data:
            return []
        first_material = next(iter(self._data.values()))
        return list(first_material.keys())

    def to_dashboard_data(self) -> dict:
        """
        转换为dashboard_data标准结构。
        包含：meta, results（含__y_test__）, aggregations计算。
        """
        materials = self.materials
        models = self.models

        # 构建results
        results = {}
        for material in materials:
            results[material] = {"__y_test__": self._y_test.get(material, [])}
            for model_name, pred_result in self._data[material].items():
                results[material][model_name] = {
                    "y_pred": pred_result.y_pred,
                    "metrics": pred_result.metrics
                }

        # 计算aggregations
        aggregations = self._compute_aggregations(materials, models)

        # 构建完整结构
        tz = timezone(timedelta(hours=8))
        data = {
            "$schema": "dashboard_data_schema",
            "version": "1.0",
            "generated_at": datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "meta": {
                "project": "电力配电网物资需求预测",
                "materials": materials,
                "models": models,
                "metrics": self.REQUIRED_METRICS,
                "forecast_horizon": self._meta.get("forecast_horizon", 12),
                "time_labels": self._meta.get("time_labels", [
                    "1月", "2月", "3月", "4月", "5月", "6月",
                    "7月", "8月", "9月", "10月", "11月", "12月"
                ]),
            },
            "results": results,
            "aggregations": aggregations,
        }
        return data

    def _compute_aggregations(self, materials: list[str], models: list[str]) -> dict:
        """计算聚合指标"""
        # best_model_per_material: 每物资R2最高的模型
        best_model_per_material = {}
        for material in materials:
            best_model = None
            best_r2 = float('-inf')
            for model_name, pred_result in self._data[material].items():
                r2 = pred_result.metrics.get("R2")
                if r2 is not None and r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
            if best_model:
                best_model_per_material[material] = {
                    "model": best_model, "metric": "R2", "value": round(best_r2, 4)
                }

        # material_avg_metrics: 每物资所有模型的指标均值
        material_avg_metrics = {}
        for material in materials:
            metric_sums = {}
            count = 0
            for pred_result in self._data[material].values():
                count += 1
                for k, v in pred_result.metrics.items():
                    if v is not None and isinstance(v, (int, float)):
                        metric_sums[k] = metric_sums.get(k, 0) + v
            if count > 0:
                material_avg_metrics[material] = {
                    k: round(v / count, 4) for k, v in metric_sums.items()
                }

        # model_avg_metrics: 每模型所有物资的指标均值
        model_avg_metrics = {}
        for model in models:
            metric_sums = {}
            count = 0
            for material in materials:
                if model in self._data[material]:
                    count += 1
                    for k, v in self._data[material][model].metrics.items():
                        if v is not None and isinstance(v, (int, float)):
                            metric_sums[k] = metric_sums.get(k, 0) + v
            if count > 0:
                model_avg_metrics[model] = {
                    k: round(v / count, 4) for k, v in metric_sums.items()
                }

        # overall_best / overall_worst
        overall_best = {"material": "", "model": "", "R2": float('-inf')}
        overall_worst = {"material": "", "model": "", "R2": float('inf')}
        for material in materials:
            for model_name, pred_result in self._data[material].items():
                r2 = pred_result.metrics.get("R2")
                if r2 is None:
                    continue
                if r2 > overall_best["R2"]:
                    overall_best = {"material": material, "model": model_name, "R2": round(r2, 4)}
                if r2 < overall_worst["R2"]:
                    overall_worst = {"material": material, "model": model_name, "R2": round(r2, 4)}

        # 处理空数据边界
        if overall_best["R2"] == float('-inf'):
            overall_best = {"material": "", "model": "", "R2": 0}
        if overall_worst["R2"] == float('inf'):
            overall_worst = {"material": "", "model": "", "R2": 0}

        return {
            "best_model_per_material": best_model_per_material,
            "material_avg_metrics": material_avg_metrics,
            "model_avg_metrics": model_avg_metrics,
            "overall_best": overall_best,
            "overall_worst": overall_worst,
        }

    def validate(self) -> list[str]:
        """
        校验数据完整性，返回警告信息列表。
        """
        warns = []
        materials = self.materials
        if not materials:
            warns.append("无预测数据")
            return warns

        # 检查各物资模型数量一致性
        model_counts = {m: len(self._data[m]) for m in materials}
        unique_counts = set(model_counts.values())
        if len(unique_counts) > 1:
            warns.append(f"各物资模型数量不一致: {model_counts}")

        # 检查y_pred长度
        horizon = self._meta.get("forecast_horizon", 12)
        for material in materials:
            for model_name, pred_result in self._data[material].items():
                if len(pred_result.y_pred) != horizon:
                    warns.append(
                        f"{material}/{model_name}: y_pred长度{len(pred_result.y_pred)} != {horizon}"
                    )

        # 检查必需指标
        for material in materials:
            for model_name, pred_result in self._data[material].items():
                missing = [m for m in self.REQUIRED_METRICS if m not in pred_result.metrics]
                if missing:
                    warns.append(f"{material}/{model_name}: 缺少指标 {missing}")

        # 检查全NaN预测
        for material in materials:
            for model_name, pred_result in self._data[material].items():
                if all(v is None for v in pred_result.y_pred):
                    warns.append(f"{material}/{model_name}: y_pred全为None")

        return warns
