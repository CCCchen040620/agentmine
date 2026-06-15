"""
Agent 版本对比引擎

对比两个版本的 Agent 日志，发现：
- 回归 (Regression): v1 没事，v2 新出现的失败模式
- 改善 (Improvement): v1 存在，v2 消失/缩小的失败模式
- 稳定 (Stable): 两个版本都存在的失败模式
- 恶化 (Worsened): 两个版本都有，但 v2 更大的失败模式
"""
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.ingestion.schema import AgentTrace
from src.clustering.clusterer import ClusterInfo, ClusteringResult

logger = logging.getLogger(__name__)


@dataclass
class DiffEntry:
    """单条对比结果"""
    cluster_id: int
    label: str
    v1_size: int        # 版本1的大小（0表示新增）
    v2_size: int        # 版本2的大小（0表示已修复）
    change_type: str    # new / fixed / improved / worsened / stable
    change_pct: float   # 变化百分比
    keywords: List[str] = field(default_factory=list)
    canary_queries: List[str] = field(default_factory=list)


@dataclass
class DiffResult:
    """版本对比完整结果"""
    v1_total: int
    v1_failures: int
    v1_failure_rate: float
    v1_clusters: int

    v2_total: int
    v2_failures: int
    v2_failure_rate: float
    v2_clusters: int

    entries: List[DiffEntry]

    # 汇总
    new_failure_modes: int       # 新增失败模式
    fixed_failure_modes: int     # 已修复的失败模式
    improved_modes: int          # 改善的
    worsened_modes: int          # 恶化的
    stable_modes: int            # 稳定的

    @property
    def is_improvement(self) -> bool:
        """整体是否向好的方向变化"""
        return self.v2_failure_rate <= self.v1_failure_rate


class AgentDiffer:
    """
    Agent 版本对比器

    算法：
    1. 两个版本分别聚类
    2. 用关键词+语义相似度匹配两个版本的簇
    3. 对每个匹配对判断变化方向
    4. 未匹配的簇标记为 NEW 或 FIXED
    """

    def __init__(self, embedder=None):
        self.embedder = embedder

    def diff(
        self,
        v1_result: ClusteringResult,
        v2_result: ClusteringResult,
        v1_failures: List[AgentTrace],
        v2_failures: List[AgentTrace],
    ) -> DiffResult:
        """
        对比两个版本的失败模式

        Returns:
            DiffResult
        """
        v1_clusters = v1_result.clusters
        v2_clusters = v2_result.clusters

        # ── 建立簇匹配 ──────────────────────────────────────
        matches = self._match_clusters(v1_clusters, v2_clusters)

        entries = []
        matched_v1_ids = set()
        matched_v2_ids = set()

        # 已匹配的簇 → 判断变化方向
        for v1_id, v2_id, similarity in matches:
            v1 = next(c for c in v1_clusters if c.cluster_id == v1_id)
            v2 = next(c for c in v2_clusters if c.cluster_id == v2_id)
            matched_v1_ids.add(v1_id)
            matched_v2_ids.add(v2_id)

            change_pct = round((v2.size - v1.size) / max(v1.size, 1) * 100, 1)

            if abs(change_pct) <= 15:
                change_type = "stable"
            elif change_pct > 0:
                change_type = "worsened"
            else:
                change_type = "improved"

            entries.append(DiffEntry(
                cluster_id=v2.cluster_id,
                label=v2.label or v1.label or f"Cluster {v2.cluster_id}",
                v1_size=v1.size,
                v2_size=v2.size,
                change_type=change_type,
                change_pct=change_pct,
                keywords=v2.keywords or v1.keywords,
                canary_queries=v2.canary_queries,
            ))

        # v2 中独有的簇 → 新增失败 (regression)
        for v2 in v2_clusters:
            if v2.cluster_id not in matched_v2_ids:
                entries.append(DiffEntry(
                    cluster_id=v2.cluster_id,
                    label=v2.label or f"Cluster {v2.cluster_id}",
                    v1_size=0,
                    v2_size=v2.size,
                    change_type="new",
                    change_pct=100.0,
                    keywords=v2.keywords,
                    canary_queries=v2.canary_queries,
                ))

        # v1 中独有的簇 → 已修复 (improvement)
        for v1 in v1_clusters:
            if v1.cluster_id not in matched_v1_ids:
                entries.append(DiffEntry(
                    cluster_id=v1.cluster_id,
                    label=v1.label or f"Cluster {v1.cluster_id}",
                    v1_size=v1.size,
                    v2_size=0,
                    change_type="fixed",
                    change_pct=-100.0,
                    keywords=v1.keywords,
                    canary_queries=v1.canary_queries,
                ))

        # 按严重度排序：new > worsened > stable > improved > fixed
        priority = {"new": 0, "worsened": 1, "stable": 2, "improved": 3, "fixed": 4}
        entries.sort(key=lambda e: (priority.get(e.change_type, 99), -abs(e.change_pct)))

        # 统计
        types = {}
        for e in entries:
            types[e.change_type] = types.get(e.change_type, 0) + 1

        # ── 失败率计算 ──────────────────────────────────────
        v1_fail_count = len(v1_failures)
        v2_fail_count = len(v2_failures)

        return DiffResult(
            v1_total=v1_result.total_failures,
            v1_failures=v1_fail_count,
            v1_failure_rate=0.0,  # 由调用方填充
            v1_clusters=len(v1_clusters),
            v2_total=v2_result.total_failures,
            v2_failures=v2_fail_count,
            v2_failure_rate=0.0,
            v2_clusters=len(v2_clusters),
            entries=entries,
            new_failure_modes=types.get("new", 0),
            fixed_failure_modes=types.get("fixed", 0),
            improved_modes=types.get("improved", 0),
            worsened_modes=types.get("worsened", 0),
            stable_modes=types.get("stable", 0),
        )

    def _match_clusters(
        self,
        v1: List[ClusterInfo],
        v2: List[ClusterInfo],
    ) -> List[Tuple[int, int, float]]:
        """
        匹配两个版本的簇

        策略：关键词 Jaccard 相似度 + 语义相似度（可选）
        返回: [(v1_cluster_id, v2_cluster_id, similarity), ...]
        """
        matches = []

        for c1 in v1:
            best_match = None
            best_score = 0.0

            for c2 in v2:
                score = self._similarity(c1, c2)
                if score > best_score:
                    best_score = score
                    best_match = c2

            # 阈值 0.15：至少共享一些关键词
            if best_match and best_score >= 0.15:
                matches.append((c1.cluster_id, best_match.cluster_id, best_score))

        return matches

    def _similarity(self, c1: ClusterInfo, c2: ClusterInfo) -> float:
        """
        计算两个簇的相似度

        使用关键词 Jaccard 相似度
        """
        kw1 = set(k.lower() for k in (c1.keywords or []))
        kw2 = set(k.lower() for k in (c2.keywords or []))

        if not kw1 and not kw2:
            return 0.0
        if not kw1 or not kw2:
            return 0.1  # 至少给一点分数

        intersection = kw1 & kw2
        union = kw1 | kw2

        jaccard = len(intersection) / max(len(union), 1)

        # 如果有语义 embedding，可以加权组合
        return jaccard
