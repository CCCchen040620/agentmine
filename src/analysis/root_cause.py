"""
根因分析模块
对每个失败簇做深度根因分析
"""
import json
import logging
from typing import List, Optional

from src.clustering.clusterer import ClusterInfo

logger = logging.getLogger(__name__)


class RootCauseAnalyzer:
    """
    LLM 驱动的根因分析器

    分析框架（5 Whys + Agent 专属维度）：
    1. 症状：这个簇的失败表现是什么？
    2. 直接原因：哪一步出了错？
    3. 根本原因：为什么那一步会出错？
    4. 系统缺陷：是 Prompt 设计问题？知识库盲区？工具选择策略？还是模型能力边界？
    5. 修复建议：具体的、可操作的修复方案
    """

    ROOT_CAUSE_PROMPT = """你是一个 AI Agent 故障根因分析专家。请对以下失败案例簇进行深度根因分析。

## 失败簇概况
- 失败数量: {cluster_size} 条
- 标签: {cluster_label}
- 关键词: {keywords}

## 代表性失败案例
{sample_cases}

## 分析要求

请从以下维度进行根因分析（遵循"5 Whys"原则）：

1. **症状 (Symptom)**: 这个簇的失败表现是什么？用户看到了什么？
2. **直接原因 (Direct Cause)**: 在 Agent 执行的哪一步出了问题？（意图理解？工具选择？工具调用？结果整合？）
3. **根本原因 (Root Cause)**: 为什么会出这个问题？属于以下哪类？
   - Prompt 缺陷（指令不清晰、约束不完整）
   - 知识库盲区（缺少关键信息）
   - 工具选择策略（选错工具或未调用必要工具）
   - 模型能力边界（LLM本身不擅长此类任务）
   - 系统设计缺陷（超时设置、权限控制、错误处理）
4. **影响评估**: 这个问题被修复后，能解决多大比例的失败？
5. **修复建议**: 给出3条具体、可操作的修复方案（按优先级排列）

## 输出格式

请严格按 JSON 格式输出：
```json
{{
  "symptom": "一句话描述症状",
  "direct_cause": "直接原因",
  "root_cause": "根本原因（选择: prompt_issue / knowledge_gap / tool_selection / model_capability / system_design）",
  "root_cause_detail": "详细的根因解释（100字内）",
  "priority": "high / medium / low",
  "fix_suggestions": [
    "修复建议1（具体可操作）",
    "修复建议2",
    "修复建议3"
  ]
}}
```
"""

    def __init__(self, llm=None):
        self.llm = llm

    def analyze(self, cluster: ClusterInfo) -> dict:
        """
        对单个失败簇做根因分析

        Returns:
            根因分析结果 dict
        """
        logger.info(f"Analyzing root cause for cluster {cluster.cluster_id} ({cluster.size} traces)")

        if not self.llm:
            return self._rule_based_analysis(cluster)

        return self._llm_analysis(cluster)

    def _llm_analysis(self, cluster: ClusterInfo) -> dict:
        """LLM 根因分析"""
        # 拼接样本
        sample_texts = []
        for i, trace in enumerate(cluster.sample_traces[:8], 1):
            sample_texts.append(f"### 案例 {i}\n{trace.get_rich_context()}")

        prompt = self.ROOT_CAUSE_PROMPT.format(
            cluster_size=cluster.size,
            cluster_label=cluster.label,
            keywords=", ".join(cluster.keywords) if cluster.keywords else "无",
            sample_cases="\n---\n".join(sample_texts)[:8000],
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            # 提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"LLM root cause analysis failed: {e}")
            return self._rule_based_analysis(cluster)

    def _rule_based_analysis(self, cluster: ClusterInfo) -> dict:
        """纯规则根因分析（fallback）"""
        # 统计失败类型
        type_counts = {}
        for trace in cluster.sample_traces:
            if trace.failure_type:
                ft = trace.failure_type.value
                type_counts[ft] = type_counts.get(ft, 0) + 1

        main_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"

        # 映射失败类型 → 根因
        type_to_root = {
            "tool_error": "tool_selection",
            "timeout": "system_design",
            "knowledge_gap": "knowledge_gap",
            "hallucination": "prompt_issue",
            "format_error": "prompt_issue",
            "context_loss": "system_design",
        }

        root_cause = type_to_root.get(main_type, "prompt_issue")

        return {
            "symptom": f"Agent在{cluster.label}场景下出现{main_type}",
            "direct_cause": f"涉及{main_type}的失败",
            "root_cause": root_cause,
            "root_cause_detail": f"基于{len(cluster.sample_traces)}条样本的统计，主要失败类型为{main_type}",
            "priority": "high" if cluster.percentage > 20 else "medium",
            "fix_suggestions": [
                f"检查与{main_type}相关的Agent配置和Prompt",
                f"增加{main_type}相关的错误处理逻辑",
                f"收集更多{main_type}失败样本，针对性优化",
            ],
        }

    def analyze_all(self, clusters: List[ClusterInfo]) -> List[ClusterInfo]:
        """
        对所有簇做根因分析（原地修改 cluster.root_cause）

        Returns:
            更新后的簇列表
        """
        for cluster in clusters:
            try:
                result = self.analyze(cluster)
                cluster.root_cause = json.dumps(result, ensure_ascii=False, indent=2)
                # 用根因结果更新 label（如果有更好的）
                if result.get("symptom") and not cluster.label:
                    cluster.label = result["symptom"]
            except Exception as e:
                logger.error(f"Root cause analysis failed for cluster {cluster.cluster_id}: {e}")
                cluster.root_cause = json.dumps({"error": str(e), "symptom": cluster.label}, ensure_ascii=False)

        return clusters
