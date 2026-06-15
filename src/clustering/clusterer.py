"""
失败聚类引擎

使用 HDBSCAN 对失败 Trace 做语义聚类
配合 UMAP 降维用于可视化
"""
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from src.ingestion.schema import AgentTrace

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """聚类结果信息"""
    cluster_id: int
    size: int                                    # 簇大小
    percentage: float                            # 占总失败的比例
    label: str = ""                              # 簇标签（LLM 生成）
    summary: str = ""                            # 簇描述（LLM 生成）
    sample_traces: List[AgentTrace] = field(default_factory=list)  # 代表性样本
    root_cause: str = ""                         # 根因分析
    canary_queries: List[str] = field(default_factory=list)  # 金丝雀查询
    centroid_coords: Optional[np.ndarray] = None # 簇中心坐标（2D）
    keywords: List[str] = field(default_factory=list)  # 关键词


@dataclass
class ClusteringResult:
    """完整聚类结果"""
    clusters: List[ClusterInfo]
    noise_count: int                             # 未能聚类的噪声点
    total_failures: int
    embedding_dim: int
    umap_coords: Optional[np.ndarray] = None     # 2D 坐标（用于可视化）
    cluster_labels: Optional[np.ndarray] = None  # 每个点的簇ID


class FailureClusterer:
    """
    失败聚类器

    流程：
    1. 可选：UMAP 降维（高维 → 低维，加速聚类）
    2. HDBSCAN 聚类（自动发现簇数，处理噪声）
    3. 为每个簇抽取代表性样本
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        use_umap_reduce: bool = True,
        umap_n_components: int = 16,  # HDBSCAN 推荐先用 UMAP 降到 10-50 维
        random_state: int = 42,
    ):
        """
        Args:
            min_cluster_size: 最小簇大小（越小簇越多）
            min_samples: HDBSCAN min_samples 参数（默认=min_cluster_size）
            use_umap_reduce: 是否用 UMAP 做降维预处理
            umap_n_components: UMAP 降维目标维度
            random_state: 随机种子
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.use_umap_reduce = use_umap_reduce
        self.umap_n_components = umap_n_components
        self.random_state = random_state

    def cluster(
        self,
        traces: List[AgentTrace],
        embeddings: np.ndarray,
    ) -> ClusteringResult:
        """
        执行聚类

        Args:
            traces: 失败 Trace 列表
            embeddings: 对应的 embedding 向量 (n_traces, dim)

        Returns:
            ClusteringResult
        """
        n = len(traces)
        if n < self.min_cluster_size:
            logger.warning(f"Too few traces ({n}) for clustering (min={self.min_cluster_size})")
            return ClusteringResult(
                clusters=[],
                noise_count=n,
                total_failures=n,
                embedding_dim=embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            )

        logger.info(f"Clustering {n} traces (dim={embeddings.shape[1]})")

        # ── Step 1: UMAP 降维预处理（可选） ──────────────────
        if self.use_umap_reduce and embeddings.shape[1] > 32:
            reduced = self._umap_reduce(embeddings)
        else:
            reduced = embeddings

        # ── Step 2: HDBSCAN 聚类 ──────────────────────────────
        cluster_labels = self._hdbscan_cluster(reduced)

        # ── Step 3: 构建聚类信息 ─────────────────────────────
        clusters = self._build_clusters(traces, cluster_labels, reduced)

        # ── Step 4: UMAP 降到 2D 用于可视化 ──────────────────
        umap_2d = self._umap_2d(embeddings)

        # 统计
        unique_labels = set(cluster_labels)
        noise_count = sum(1 for l in cluster_labels if l == -1)
        cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)

        logger.info(
            f"Clustering done: {cluster_count} clusters, "
            f"{noise_count} noise points ({noise_count/max(n,1)*100:.1f}%)"
        )

        return ClusteringResult(
            clusters=clusters,
            noise_count=noise_count,
            total_failures=n,
            embedding_dim=embeddings.shape[1],
            umap_coords=umap_2d,
            cluster_labels=cluster_labels,
        )

    def _umap_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP 降维（高维 → 中维，用于加速聚类）"""
        try:
            import umap
            reducer = umap.UMAP(
                n_components=min(self.umap_n_components, embeddings.shape[1] - 1),
                metric="cosine",
                random_state=self.random_state,
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.0,  # 聚类友好
            )
            reduced = reducer.fit_transform(embeddings)
            logger.info(f"UMAP reduction: {embeddings.shape[1]} → {reduced.shape[1]}")
            return reduced
        except ImportError:
            logger.warning("umap-learn not installed, clustering on raw embeddings")
            return embeddings

    def _umap_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP 降到 2D（用于可视化）"""
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                metric="cosine",
                random_state=self.random_state,
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.1,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed, using random 2D coords")
            np.random.seed(self.random_state)
            return np.random.randn(embeddings.shape[0], 2)

    def _hdbscan_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """HDBSCAN 聚类"""
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                cluster_selection_epsilon=0.1,
                prediction_data=True,  # 允许后续预测新点
            )
            labels = clusterer.fit_predict(embeddings)
            logger.info(f"HDBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} clusters found")
            return labels
        except ImportError:
            logger.error("hdbscan not installed. Install with: pip install hdbscan")
            raise

    def _build_clusters(
        self,
        traces: List[AgentTrace],
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> List[ClusterInfo]:
        """构建 ClusterInfo 列表"""
        unique_labels = sorted(set(labels))
        total_failures = len(traces)
        clusters = []

        for label in unique_labels:
            if label == -1:
                continue  # 跳过噪声

            mask = labels == label
            cluster_indices = np.where(mask)[0]
            cluster_size = len(cluster_indices)

            # 选取代表性样本（离簇中心最近的 5 个点）
            cluster_embs = embeddings[cluster_indices]
            centroid = cluster_embs.mean(axis=0)
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)
            top_k = min(5, cluster_size)
            top_indices = cluster_indices[np.argsort(distances)[:top_k]]

            sample_traces = [traces[i] for i in top_indices]

            # 简单的关键词提取（高频词）
            keywords = self._extract_keywords(
                [traces[i].user_query for i in cluster_indices]
            )

            clusters.append(ClusterInfo(
                cluster_id=int(label),
                size=cluster_size,
                percentage=round(cluster_size / max(total_failures, 1) * 100, 1),
                sample_traces=sample_traces,
                centroid_coords=centroid[:2] if embeddings.shape[1] >= 2 else None,
                keywords=keywords,
            ))

        # 按簇大小降序排列
        clusters.sort(key=lambda c: c.size, reverse=True)

        # 重新编号（从 1 开始）
        for i, cluster in enumerate(clusters, 1):
            cluster.cluster_id = i
            # 同步更新原始 traces 的 cluster_id
            for trace in cluster.sample_traces:
                trace.failure_cluster_id = i

        return clusters

    def _extract_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        """
        简单关键词提取（基于词频）

        生产环境可以：
        - 用 jieba/KeyBERT 做中文关键词提取
        - 用 LLM 提取代表性关键词
        """
        # 简单的 unigram 统计
        word_count: Dict[str, int] = {}
        for text in texts:
            # 简单分词（中英文混合）
            words = text.replace("，", " ").replace("。", " ").replace("？", " ").split()
            for word in words:
                word = word.strip().lower()
                if len(word) >= 2:  # 过滤单字
                    word_count[word] = word_count.get(word, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
