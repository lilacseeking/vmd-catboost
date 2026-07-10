"""
结构化运行时日志系统
为Route A/B/C脚本提供统一的步骤追踪和进度记录
"""
import os, time, logging
from datetime import datetime

class StepLogger:
    """带步骤编号、耗时、内存的结构化日志器"""

    def __init__(self, name, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        self.log_dir = log_dir
        self.name = name
        self.step_counter = 0
        self.step_start = None
        self.pipeline_start = time.time()

        # 文件日志: 完整信息
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

        self.file_logger = logging.getLogger(f'{name}_{timestamp}')
        self.file_logger.setLevel(logging.DEBUG)
        self.file_logger.handlers.clear()

        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.file_logger.addHandler(fh)
        self.log_file = log_file

    # ---- 管道级方法 ----
    def pipeline_start_log(self, title, description=''):
        """管道启动信息"""
        self.file_logger.info("=" * 70)
        self.file_logger.info(f"  PIPELINE: {title}")
        if description:
            self.file_logger.info(f"  {description}")
        self.file_logger.info(f"  日志文件: {self.log_file}")
        self.file_logger.info("=" * 70)
        print(f"\n[{self.name}] 启动: {title}")
        print(f"  日志: {os.path.basename(self.log_file)}")

    def pipeline_end_log(self, summary=''):
        """管道完成信息"""
        elapsed = time.time() - self.pipeline_start
        self.file_logger.info("")
        self.file_logger.info("=" * 70)
        self.file_logger.info(f"  PIPELINE COMPLETE ({elapsed:.0f}s)")
        if summary:
            self.file_logger.info(f"  {summary}")
        self.file_logger.info("=" * 70)
        print(f"\n[{self.name}] 完成 ({elapsed:.0f}s) {summary}")

    # ---- 步骤级方法 ----
    def step(self, title, detail=''):
        """标记一个新步骤，返回步骤编号"""
        self.step_counter += 1
        num = self.step_counter
        self.file_logger.info("")
        self.file_logger.info(f"[Step {num}] {title}")
        if detail:
            self.file_logger.info(f"       {detail}")
        print(f"  [{num}] {title}")
        self.step_start = time.time()
        return num

    def step_ok(self, detail=''):
        """步骤成功完成"""
        elapsed = time.time() - self.step_start if self.step_start else 0
        msg = f"  [Step {self.step_counter}] OK ({elapsed:.1f}s)"
        if detail:
            msg += f" | {detail}"
        self.file_logger.info(msg)
        self.file_logger.debug(f"       = {detail}")

    def step_warn(self, detail=''):
        """步骤完成但有警告"""
        self.file_logger.warning(f"       WARNING: {detail}")

    # ---- 子任务级方法 ----
    def task(self, detail):
        """子任务信息"""
        self.file_logger.info(f"       - {detail}")

    def task_detail(self, detail):
        """子任务详细信息 (DEBUG级别)"""
        self.file_logger.debug(f"         {detail}")

    # ---- 数据级方法 ----
    def data_table(self, title, headers, rows):
        """格式化表格输出到日志"""
        self.file_logger.info(f"       {title}")
        col_widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
        header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep = "-" * len(header_str)
        self.file_logger.info(f"       {sep}")
        for row in rows:
            line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
            self.file_logger.info(f"       {line}")
        self.file_logger.info(f"       {sep}")

    # ---- 指标级方法 ----
    def metrics(self, material, metrics_dict):
        """记录某物资的评估指标"""
        parts = [f"{k}={v}" for k, v in metrics_dict.items()]
        self.file_logger.info(f"       [{material}] {' | '.join(parts)}")

    def compare(self, label, old_val, new_val, better_is_higher=True):
        """对比新旧值"""
        if old_val is None:
            self.file_logger.info(f"       {label}: NEW={new_val:.4f}")
            return
        delta = new_val - old_val
        improved = (delta > 0) if better_is_higher else (delta < 0)
        marker = '+' if improved else '-'
        self.file_logger.info(f"       {label}: {old_val:.4f} -> {new_val:.4f} ({delta:+.4f}) {marker}")

    # ---- 错误级方法 ----
    def error(self, msg):
        self.file_logger.error(f"       ERROR: {msg}")

    def error_skip(self, material_name, reason):
        """跳过某物资并记录原因"""
        self.file_logger.warning(f"       SKIP [{material_name}]: {reason}")

    # ---- 辅助 ----
    def get_log_path(self):
        return self.log_file


def format_params(params, precision=3):
    """格式化参数字典为可读字符串"""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
