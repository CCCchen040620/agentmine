"""
语义 Embedding 模块
将失败 Trace 的用户 query 转为稠密向量，用于聚类
"""
import logging
import numpy as np
from typing import List

from src.ingestion.schema import AgentTrace

logger = logging.getLogger(__name__)


class TraceEmbedder:
    """
    Trace 嵌入器

    将每条失败 Trace 转为向量表示，支持两种模式：
    1. Query-only: 仅用用户 query 做 embedding（快速，适合简单场景）
    2. Rich-context: 用 query + output + error 拼接做 embedding（更准确）

    默认使用 sentence-transformers 的 BGE-small-zh 模型（轻量，384维）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        mode: str = "query_only",
    ):
        """
        Args:
            model_name: sentence-transformers 模型名
            device: "cpu" | "cuda"
            mode: "query_only" | "rich_context"
        """
        self.model_name = model_name
        self.mode = mode
        self._model = None
        self._device = device

    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self._device)
                logger.info(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
        return self._model

    def embed(self, traces: List[AgentTrace]) -> np.ndarray:
        """
        对 Trace 列表做 embedding

        Args:
            traces: 失败 Trace 列表

        Returns:
            numpy array, shape (n_traces, embedding_dim)
        """
        if not traces:
            logger.warning("No traces to embed")
            return np.array([])

        # 提取文本
        texts = [self._get_text(trace) for trace in traces]

        # 批量编码
        logger.info(f"Embedding {len(texts)} traces...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,  # 余弦相似度，归一化
        )

        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def _get_text(self, trace: AgentTrace) -> str:
        """从 Trace 中提取用于 embedding 的文本"""
        if self.mode == "query_only":
            return trace.get_compact_query()
        else:
            return trace.get_diagnostic_text()[:1024]  # 限制长度

    def embed_single(self, text: str) -> np.ndarray:
        """对单个文本做 embedding"""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """对纯文本列表做 embedding"""
        logger.info(f"Embedding {len(texts)} texts...")
        return self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
            normalize_embeddings=True,
        )

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()
