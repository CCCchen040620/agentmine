"""
时序趋势分析

按天/周聚合失败率，发现：
- 哪天上线了新版本后失败率暴涨
- 哪些失败模式在增长 vs 减少
- 自动标注"异常点"（失败率突变的日期）
"""
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from src.ingestion.schema import AgentTrace

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    时序趋势分析器

    输入：带 timestamp 的 Trace 列表
    输出：按天/周聚合的失败率序列 + 异常点检测
    """

    def __init__(self, granularity: str = "day"):
        """
        Args:
            granularity: "day" | "week"
        """
        self.granularity = granularity

    def analyze(
        self,
        traces: List[AgentTrace],
        failures: List[AgentTrace],
    ) -> dict:
        """
        分析时序趋势

        Returns:
            {
                "granularity": "day",
                "data_points": [
                    {"date": "2024-06-01", "total": 100, "failures": 8, "rate": 8.0},
                    ...
                ],
                "anomalies": [{"date": "2024-06-05", "rate": 23.0, "deviation": 2.8}],
                "trend_direction": "up" | "down" | "stable",
                "peak_date": "2024-06-05",
                "lowest_date": "2024-06-02",
            }
        """
        if not traces:
            return {"data_points": [], "anomalies": [], "trend_direction": "stable"}

        # ── 按日期分组 ──────────────────────────────────────
        groups = self._group_by_date(traces, failures)

        # ── 计算每日失败率 ──────────────────────────────────
        data_points = []
        for date_key in sorted(groups.keys()):
            day_traces = groups[date_key]
            day_failures = [t for t in day_traces if t.status.value == "failure"]
            total = len(day_traces)
            fail_count = len(day_failures)
            rate = round(fail_count / max(total, 1) * 100, 1)

            data_points.append({
                "date": date_key,
                "total": total,
                "failures": fail_count,
                "rate": rate,
            })

        if not data_points:
            return {"data_points": [], "anomalies": [], "trend_direction": "stable"}

        # ── 异常检测（简单：偏离均值 2 倍标准差） ──────────
        rates = [p["rate"] for p in data_points]
        mean_rate = sum(rates) / len(rates)
        std_rate = (sum((r - mean_rate) ** 2 for r in rates) / max(len(rates), 1)) ** 0.5

        anomalies = []
        for p in data_points:
            if std_rate > 1.0:  # 标准差太小说明没波动
                deviation = (p["rate"] - mean_rate) / max(std_rate, 0.01)
                if abs(deviation) > 2.0:
                    anomalies.append({
                        "date": p["date"],
                        "rate": p["rate"],
                        "deviation": round(deviation, 2),
                        "direction": "spike" if deviation > 0 else "drop",
                    })

        # ── 趋势方向 ────────────────────────────────────────
        if len(data_points) >= 3:
            first_half = rates[:len(rates)//2]
            second_half = rates[len(rates)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            diff = second_avg - first_avg
            if diff > 2:
                direction = "up"
            elif diff < -2:
                direction = "down"
            else:
                direction = "stable"
        else:
            direction = "stable"

        return {
            "granularity": self.granularity,
            "data_points": data_points,
            "anomalies": anomalies,
            "trend_direction": direction,
            "peak_date": max(data_points, key=lambda p: p["rate"])["date"] if data_points else "",
            "lowest_date": min(data_points, key=lambda p: p["rate"])["date"] if data_points else "",
        }

    def _group_by_date(
        self,
        traces: List[AgentTrace],
        failures: List[AgentTrace],
    ) -> Dict[str, List]:
        """按日期分组"""
        groups = defaultdict(list)

        for trace in traces:
            date_key = self._get_date_key(trace.timestamp)
            groups[date_key].append(trace)

        return groups

    def _get_date_key(self, timestamp: Optional[str]) -> str:
        """从 timestamp 提取日期/周标识"""
        if not timestamp:
            return "unknown"

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if self.granularity == "week":
                # 周一作为周标识
                monday = dt - timedelta(days=dt.weekday())
                return monday.strftime("%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            return "unknown"
