"""
dashboard/serializer.py - Numpy序列化工具模块

提供NumpyEncoder和to_json_safe，将numpy类型转为JSON-safe Python对象。
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.round(4).tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return round(float(obj), 4)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)


def to_json_safe(value):
    """将任意numpy值转为JSON-safe Python对象"""
    if isinstance(value, np.ndarray):
        return value.round(4).tolist()
    if isinstance(value, (np.float32, np.float64)):
        return round(float(value), 4)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    return value
