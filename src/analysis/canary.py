"""
金丝雀查询生成器

为每种失败模式生成 3-5 个"金丝雀查询"——
这些查询能可靠地触发该失败模式，用于回归测试。
"""
import json
import logging
from typing import List, Optional

from src.clustering.clusterer import ClusterInfo

logger = logging.getLogger(__name__)


class CanaryGenerator:
    """
    金丝雀查询生成器

    灵感来源：煤矿工人带金丝雀下矿——鸟死了就知道有毒气。
    同理，金丝雀查询能第一时间告警"这个 Agent 版本有回归问题"。

    用法：
        generator = CanaryGenerator(llm)
        canaries = generator.generate(cluster)
        # canaries: ["请帮我计算...", "那个东西怎么...", ...]
    """

    CANARY_PROMPT = """你是一个 AI Agent 测试专家。请基于以下失败模式，生成金丝雀测试查询。

## 失败模式
- 标签: {cluster_label}
- 根因: {root_cause}
- 涉及关键词: {keywords}

## 失败案例参考
{sample_queries}

## 任务

生成 {n} 个"金丝雀查询"——这些查询能可靠地触发该失败模式。

金丝雀查询的要求：
1. 看起来像真实用户会问的问题
2. 包含该失败模式的特征要素
3. 覆盖不同的表述方式（口语化/正式/模糊/精确）
4. 单个查询一句话，不超过50字

输出 JSON 格式：
```json
{{
  "canary_queries": [
    "查询1",
    "查询2",
    ...
  ],
  "test_strategy": "针对这个失败模式，建议的测试策略（50字）"
}}
```
"""

    def __init__(self, llm=None, n_queries: int = 5):
        """
        Args:
            llm: LangChain LLM 实例
            n_queries: 每个簇生成的查询数量
        """
        self.llm = llm
        self.n_queries = n_queries

    def generate(self, cluster: ClusterInfo) -> dict:
        """
        为一个失败簇生成金丝雀查询

        Returns:
            {"canary_queries": [...], "test_strategy": "..."}
        """
        n = min(self.n_queries, max(3, cluster.size // 10))

        if self.llm:
            return self._llm_generate(cluster, n)
        else:
            return self._template_generate(cluster, n)

    def _llm_generate(self, cluster: ClusterInfo, n: int) -> dict:
        """LLM 生成金丝雀查询"""
        # 提取样本查询
        sample_queries = "\n".join([
            f"- {t.user_query}"
            for t in cluster.sample_traces[:8]
        ])

        # 解析根因
        root_cause_summary = "未知"
        try:
            rc = json.loads(cluster.root_cause) if isinstance(cluster.root_cause, str) else cluster.root_cause
            root_cause_summary = f"{rc.get('root_cause', '未知')} - {rc.get('root_cause_detail', '')}"
        except Exception:
            root_cause_summary = cluster.root_cause if isinstance(cluster.root_cause, str) else "未知"

        prompt = self.CANARY_PROMPT.format(
            cluster_label=cluster.label,
            root_cause=root_cause_summary,
            keywords=", ".join(cluster.keywords) if cluster.keywords else "无",
            sample_queries=sample_queries,
            n=n,
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content

            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content

            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"LLM canary generation failed: {e}")
            return self._template_generate(cluster, n)

    def _template_generate(self, cluster: ClusterInfo, n: int) -> dict:
        """
        基于模板的金丝雀查询生成（fallback）

        将样本查询做简单变换，生成变体
        """
        sample_queries = [t.user_query for t in cluster.sample_traces[:n]]

        # 简单变体策略
        variants = []
        for query in sample_queries:
            if query:
                # 加口语化前缀
                variants.append(f"帮我{query}" if not query.startswith("帮我") else query)
                # 变模糊一点
                if len(query) > 10:
                    variants.append(query[:len(query)//2] + "怎么办")

        # 去重，补足到 n 条
        variants = list(set(variants))[:n]
        while len(variants) < n:
            idx = len(variants) % len(sample_queries)
            variants.append(f"请问{sample_queries[idx] if sample_queries else '这个怎么处理'}？")

        return {
            "canary_queries": variants[:n],
            "test_strategy": f"建议在CI中加入这{n}个查询作为回归测试，确保每次Agent更新后都能通过",
        }

    def generate_all(self, clusters: List[ClusterInfo]) -> List[ClusterInfo]:
        """
        为所有簇生成金丝雀查询

        Returns:
            更新后的簇列表（原地修改 cluster.canary_queries）
        """
        total_queries = 0
        for cluster in clusters:
            try:
                result = self.generate(cluster)
                queries = result.get("canary_queries", [])
                cluster.canary_queries = queries
                total_queries += len(queries)
                logger.info(f"Cluster {cluster.cluster_id}: generated {len(queries)} canary queries")
            except Exception as e:
                logger.error(f"Canary generation failed for cluster {cluster.cluster_id}: {e}")
                cluster.canary_queries = []

        logger.info(f"Total canary queries generated: {total_queries}")
        return clusters
