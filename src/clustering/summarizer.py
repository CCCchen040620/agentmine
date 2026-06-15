"""
簇摘要生成器（使用 LLM）
"""
import json
import logging
from typing import List, Optional

from src.ingestion.schema import AgentTrace

logger = logging.getLogger(__name__)


class ClusterSummarizer:
    """
    用 LLM 为每个失败簇生成一句话描述 + 关键词

    策略：
    - 小簇（< 20条）：把全部 sample 送进去
    - 大簇（≥ 20条）：只送代表性样本（离簇中心最近的前 10 条）
    """

    SUMMARIZE_PROMPT = """你是一个 AI Agent 故障分析专家。请分析以下失败案例的共性，给出一个简洁的簇标签和描述。

失败案例：
{cases}

请按 JSON 格式输出：
```json
{{
  "label": "簇标签（10字以内，概括失败类型，如：金额计算错误）",
  "summary": "簇描述（50字以内，说明这类失败的共同特征）",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}
```
"""

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain LLM 实例（如果为 None，用纯规则方式）
        """
        self.llm = llm

    def summarize(self, cluster_samples: List[AgentTrace], cluster_size: int) -> dict:
        """
        为一个簇生成摘要

        Returns:
            {"label": "...", "summary": "...", "keywords": [...]}
        """
        # 限制 sample 数量
        max_samples = min(10, len(cluster_samples))
        samples = cluster_samples[:max_samples]

        # 拼接案例
        cases_text = "\n---\n".join([
            f"案例 {i+1}:\n{s.get_rich_context()}"
            for i, s in enumerate(samples)
        ])

        if self.llm:
            return self._llm_summarize(cases_text)
        else:
            return self._rule_summarize(samples, cluster_size)

    def _llm_summarize(self, cases_text: str) -> dict:
        """LLM 生成摘要"""
        prompt = self.SUMMARIZE_PROMPT.format(cases=cases_text[:6000])

        try:
            response = self.llm.invoke(prompt)
            # 提取 JSON
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return {"label": "未分类失败", "summary": "LLM 分析失败，请手动检查", "keywords": []}

    def _rule_summarize(self, samples: List[AgentTrace], cluster_size: int) -> dict:
        """纯规则摘要（fallback）"""
        # 统计错误类型
        error_types = {}
        for s in samples:
            if s.error_type:
                error_types[s.error_type] = error_types.get(s.error_type, 0) + 1

        # 统计涉及的 tool
        tool_names = set()
        for s in samples:
            for tc in s.tool_calls:
                tool_names.add(tc.tool_name)

        if error_types:
            main_type = max(error_types, key=error_types.get)
            label = f"{main_type}相关错误（{cluster_size}条）"
        elif tool_names:
            label = f"工具调用失败-涉及{', '.join(list(tool_names)[:2])}"
        else:
            label = f"未分类失败模式（{cluster_size}条）"

        return {
            "label": label,
            "summary": f"{cluster_size}条失败记录，主要涉及: {', '.join(list(tool_names)[:3]) or '未知'}",
            "keywords": list(tool_names),
        }
