"""
dashboard/summary.py - AI总结规则引擎

基于规则引擎生成看板总结文本，支持rule/agent两种模式。
所有规则动态感知物资集合，禁止硬编码物资名。
"""

import json
from string import Template
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SummaryRule:
    """总结规则定义"""
    id: str
    name: str
    priority: int
    condition: Callable[[dict], bool]
    template: str
    extractor: Callable[[dict], dict]


# 规则注册表
_SUMMARY_RULES: list[SummaryRule] = []


def register_rule(rule: SummaryRule):
    _SUMMARY_RULES.append(rule)
    _SUMMARY_RULES.sort(key=lambda r: r.priority)


def _check_consensus(y_preds: list[list[float]], threshold: float = 0.95) -> bool:
    """判断Top3模型预测趋势是否一致（Pearson > threshold）"""
    for i in range(len(y_preds)):
        for j in range(i + 1, len(y_preds)):
            try:
                corr = np.corrcoef(y_preds[i], y_preds[j])[0, 1]
                if np.isnan(corr) or corr <= threshold:
                    return False
            except Exception:
                return False
    return True


# ============ 内置规则 ============

register_rule(SummaryRule(
    id="best_model",
    name="最佳模型",
    priority=10,
    condition=lambda data: bool(data.get("aggregations", {}).get("overall_best", {}).get("model")),
    template="综合R²表现，**${model}**以均值${value}位列第一，在${win_count}/${total}种物资上取得最优。",
    extractor=lambda data: {
        "model": data["aggregations"]["overall_best"]["model"],
        "value": f"{data['aggregations']['overall_best']['R2']:.2f}",
        "win_count": sum(
            1 for m in data["meta"]["materials"]
            if data["aggregations"]["best_model_per_material"].get(m, {}).get("model")
            == data["aggregations"]["overall_best"]["model"]
        ),
        "total": len(data["meta"]["materials"]),
    }
))

register_rule(SummaryRule(
    id="worst_material",
    name="最难预测物资",
    priority=20,
    condition=lambda data: bool(data.get("aggregations", {}).get("overall_worst", {}).get("material")),
    template="**${material}**预测难度最大，最佳模型R²仅为${value}，建议增加特征或调整预测策略。",
    extractor=lambda data: {
        "material": data["aggregations"]["overall_worst"]["material"],
        "value": f"{data['aggregations']['overall_worst']['R2']:.2f}",
    }
))


def _anomaly_condition(data: dict) -> bool:
    """存在R²<0.5的物资×模型组合"""
    for material in data.get("meta", {}).get("materials", []):
        results = data.get("results", {}).get(material, {})
        for model_name, model_data in results.items():
            if model_name.startswith("__"):
                continue
            r2 = model_data.get("metrics", {}).get("R2")
            if r2 is not None and r2 < 0.5:
                return True
    return False


def _anomaly_extractor(data: dict) -> dict:
    """提取第一个异常组合"""
    for material in data["meta"]["materials"]:
        results = data["results"].get(material, {})
        for model_name, model_data in results.items():
            if model_name.startswith("__"):
                continue
            r2 = model_data.get("metrics", {}).get("R2")
            if r2 is not None and r2 < 0.5:
                return {"model": model_name, "material": material, "value": f"{r2:.2f}"}
    return {"model": "", "material": "", "value": "N/A"}


register_rule(SummaryRule(
    id="anomaly_detect",
    name="异常检测",
    priority=30,
    condition=_anomaly_condition,
    template="检测到异常：**${model}**在**${material}**上R²=${value}，远低于均值，可能存在数据质量问题。",
    extractor=_anomaly_extractor,
))


def _zero_demand_condition(data: dict) -> bool:
    """存在zero_acc < 0.8的物资×模型组合"""
    for material in data.get("meta", {}).get("materials", []):
        results = data.get("results", {}).get(material, {})
        for model_name, model_data in results.items():
            if model_name.startswith("__"):
                continue
            za = model_data.get("metrics", {}).get("zero_acc")
            if za is not None and za < 0.8:
                return True
    return False


def _zero_demand_extractor(data: dict) -> dict:
    for material in data["meta"]["materials"]:
        results = data["results"].get(material, {})
        for model_name, model_data in results.items():
            if model_name.startswith("__"):
                continue
            za = model_data.get("metrics", {}).get("zero_acc")
            if za is not None and za < 0.8:
                return {"material": material, "model": model_name, "value": f"{za*100:.0f}%"}
    return {"material": "", "model": "", "value": "N/A"}


register_rule(SummaryRule(
    id="zero_demand",
    name="零需求特征",
    priority=40,
    condition=_zero_demand_condition,
    template="注意：**${material}**存在大量零需求月份，**${model}**的零值准确率仅${value}，建议使用零膨胀模型。",
    extractor=_zero_demand_extractor,
))


def _consensus_extractor(data: dict) -> dict:
    """动态遍历所有物资检测共识"""
    materials = data["meta"]["materials"]
    consensus_materials = []
    for material in materials:
        results = data["results"].get(material, {})
        # 按R2排序取Top3
        model_items = [
            (k, v) for k, v in results.items() if not k.startswith("__")
        ]
        model_items.sort(
            key=lambda x: x[1].get("metrics", {}).get("R2", float('-inf')),
            reverse=True
        )
        top3 = model_items[:3]
        if len(top3) < 3:
            continue
        y_preds = [item[1]["y_pred"] for item in top3]
        if _check_consensus(y_preds):
            consensus_materials.append(material)
    return {"materials": "、".join(consensus_materials) if consensus_materials else ""}


register_rule(SummaryRule(
    id="model_consensus",
    name="模型共识",
    priority=50,
    condition=lambda data: bool(_consensus_extractor(data).get("materials")),
    template="**${materials}**的Top3模型预测趋势高度一致（Pearson > 0.95）。",
    extractor=_consensus_extractor,
))


def _overall_quality_extractor(data: dict) -> dict:
    """整体质量评估"""
    model_avg = data.get("aggregations", {}).get("model_avg_metrics", {})
    if not model_avg:
        return {"r2_mean": "N/A", "smape_mean": "N/A", "total": "0", "above_06": "0"}

    r2_values = [v.get("R2", 0) for v in model_avg.values() if v.get("R2") is not None]
    smape_values = [v.get("sMAPE", 0) for v in model_avg.values() if v.get("sMAPE") is not None]

    r2_mean = sum(r2_values) / len(r2_values) if r2_values else 0
    smape_mean = sum(smape_values) / len(smape_values) if smape_values else 0

    # 统计R2>0.6的组合数
    total = 0
    above_06 = 0
    for material in data["meta"]["materials"]:
        results = data["results"].get(material, {})
        for model_name, model_data in results.items():
            if model_name.startswith("__"):
                continue
            total += 1
            r2 = model_data.get("metrics", {}).get("R2")
            if r2 is not None and r2 > 0.6:
                above_06 += 1

    return {
        "r2_mean": f"{r2_mean:.2f}",
        "smape_mean": f"{smape_mean:.1f}%",
        "total": str(total),
        "above_06": str(above_06),
    }


register_rule(SummaryRule(
    id="overall_quality",
    name="整体质量评估",
    priority=60,
    condition=lambda data: True,
    template="整体预测质量：R²均值${r2_mean}，sMAPE均值${smape_mean}，${above_06}/${total}组合R²>0.6。",
    extractor=_overall_quality_extractor,
))


class SummaryEngine:
    """AI总结引擎"""

    def generate(self, data: dict, mode: str = "rule") -> str:
        """
        生成AI总结文本。

        Args:
            data: dashboard_data完整结构
            mode: "rule"=规则引擎, "agent"=预留agent接口

        Returns:
            Markdown格式的总结文本
        """
        if mode == "rule":
            return self._rule_based_generate(data)
        elif mode == "agent":
            return self._agent_placeholder(data)
        else:
            raise ValueError(f"Unknown summary mode: '{mode}'. Expected 'rule' or 'agent'.")

    def _rule_based_generate(self, data: dict) -> str:
        sections = []
        for rule in _SUMMARY_RULES:
            try:
                if rule.condition(data):
                    variables = rule.extractor(data)
                    text = Template(rule.template).safe_substitute(variables)
                    sections.append(text)
            except Exception:
                continue  # 单条规则失败不影响其他
        return "\n\n".join(sections) if sections else "暂无分析数据。"

    def _agent_placeholder(self, data: dict) -> str:
        """agent模式：输出结构化数据供外部agent消费"""
        return json.dumps({
            "task": "generate_summary",
            "data_summary": {
                "materials": data["meta"]["materials"],
                "models": data["meta"]["models"],
                "overall_best": data.get("aggregations", {}).get("overall_best", {}),
                "overall_worst": data.get("aggregations", {}).get("overall_worst", {}),
            },
            "instruction": "请基于以上数据生成200字以内的预测效果总结"
        }, ensure_ascii=False)
