"""
自动失败标注器
判断每条 Trace 是成功还是失败，并尝试分类失败类型
"""
import re
import logging
from typing import List, Tuple, Optional

from src.ingestion.schema import AgentTrace, TraceStatus, FailureType

logger = logging.getLogger(__name__)


class AutoLabeler:
    """
    自动标注器

    标注策略（按优先级）：
    1. 显式错误信号（error 字段、异常堆栈）→ FAILURE
    2. 用户负反馈（thumbs_down / 低评分）→ FAILURE
    3. 输出为空或明显异常 → FAILURE
    4. 关键词匹配（"抱歉"、"无法"、"I'm sorry"）→ 可能是 PARTIAL
    5. 其余 → SUCCESS

    可选：用 LLM-as-judge 做更精确的标注（configurable）
    """

    # ── 失败关键词（Agent 承认自己没做好时的常见表述） ──────
    FAILURE_SIGNALS_CN = [
        "抱歉，我无法",
        "对不起，我无法",
        "暂时无法处理",
        "系统繁忙",
        "请稍后重试",
        "出现错误",
        "接口异常",
        "数据查询失败",
        "未能找到",
        "目前不支持",
        "超出我的能力",
    ]

    FAILURE_SIGNALS_EN = [
        "I'm sorry, I cannot",
        "I apologize",
        "unable to process",
        "an error occurred",
        "please try again",
        "I don't have the ability",
        "I cannot answer",
        "no information found",
    ]

    def __init__(self, use_llm: bool = False, llm=None):
        """
        Args:
            use_llm: 是否用 LLM 做精标注（慢但准）
            llm: LangChain LLM 实例
        """
        self.use_llm = use_llm
        self.llm = llm

    def label(self, traces: List[AgentTrace]) -> List[AgentTrace]:
        """
        对一批 Trace 进行标注

        Returns:
            标注后的 Trace 列表（原地修改 + 返回）
        """
        stats = {"success": 0, "failure": 0, "partial": 0}

        for trace in traces:
            status, failure_type = self._label_one(trace)
            trace.status = status
            trace.failure_type = failure_type
            stats[status.value] += 1

        total = len(traces)
        logger.info(
            f"Labeled {total} traces: "
            f"{stats['success']} success ({stats['success']/max(total,1)*100:.1f}%), "
            f"{stats['failure']} failure ({stats['failure']/max(total,1)*100:.1f}%), "
            f"{stats['partial']} partial ({stats['partial']/max(total,1)*100:.1f}%)"
        )
        return traces

    def _label_one(self, trace: AgentTrace) -> Tuple[TraceStatus, Optional[FailureType]]:
        """标注单条 Trace"""

        # ── Rule 1: 显式错误 ──────────────────────────────
        if trace.error:
            return TraceStatus.FAILURE, self._classify_error_type(trace)

        # ── Rule 2: 工具调用失败 ──────────────────────────
        if trace.tool_calls:
            for tc in trace.tool_calls:
                if tc.error:
                    return TraceStatus.FAILURE, FailureType.TOOL_ERROR

        # ── Rule 3: 用户负反馈 ────────────────────────────
        if trace.user_feedback:
            feedback_lower = trace.user_feedback.lower()
            neg_signals = ["down", "bad", "no", "不", "差", "错", "误", "wrong", "incorrect"]
            if any(s in feedback_lower for s in neg_signals):
                return TraceStatus.FAILURE, FailureType.UNKNOWN

        # ── Rule 4: 低评分 ────────────────────────────────
        if trace.user_rating is not None and trace.user_rating <= 2:
            return TraceStatus.FAILURE, FailureType.UNKNOWN

        # ── Rule 5: 输出异常 ──────────────────────────────
        output = trace.final_output.strip()

        # 空输出
        if not output:
            return TraceStatus.FAILURE, FailureType.FORMAT_ERROR

        # 极短输出（<5字符且不含有意义内容）
        if len(output) < 5:
            return TraceStatus.FAILURE, FailureType.UNKNOWN

        # ── Rule 6: Agent 自己承认失败 ────────────────────
        all_signals = self.FAILURE_SIGNALS_CN + self.FAILURE_SIGNALS_EN
        for signal in all_signals:
            if signal.lower() in output.lower():
                return TraceStatus.PARTIAL, FailureType.UNKNOWN

        # ── Rule 7: 默认成功 ──────────────────────────────
        return TraceStatus.SUCCESS, None

    def _classify_error_type(self, trace: AgentTrace) -> FailureType:
        """
        根据错误信息自动分类失败类型

        使用规则匹配而非 LLM（快速）
        """
        error_text = (trace.error or "").lower()
        error_type = (trace.error_type or "").lower()

        # 超时
        timeout_keywords = ["timeout", "timed out", "超时", "timedout"]
        if any(kw in error_text for kw in timeout_keywords):
            return FailureType.TIMEOUT

        # 工具错误
        tool_keywords = ["tool", "function", "api", "调用失败", "接口", "connection refused"]
        if any(kw in error_text for kw in tool_keywords):
            return FailureType.TOOL_ERROR

        # 权限
        permission_keywords = ["permission", "unauthorized", "forbidden", "权限", "无权限", "401", "403"]
        if any(kw in error_text for kw in permission_keywords):
            return FailureType.PERMISSION

        # 格式错误
        format_keywords = ["parse", "json", "format", "格式", "解析"]
        if any(kw in error_text for kw in format_keywords):
            return FailureType.FORMAT_ERROR

        # 幻觉相关（输出校验失败）
        hallucination_keywords = ["hallucination", "factual", "contradict", "不准确", "编造", "幻觉"]
        if any(kw in error_text for kw in hallucination_keywords):
            return FailureType.HALLUCINATION

        return FailureType.UNKNOWN

    def label_with_llm(self, trace: AgentTrace) -> Tuple[TraceStatus, FailureType, str]:
        """
        使用 LLM 做精标注（用于关键 Trace）

        Returns:
            (status, failure_type, explanation)
        """
        if not self.llm:
            return trace.status, trace.failure_type or FailureType.UNKNOWN, "LLM not configured"

        prompt = f"""分析以下 AI Agent 交互是否失败，并分类：

交互记录：
{trace.get_rich_context()}

请按 JSON 格式输出：
```json
{{
  "is_failure": true/false,
  "failure_type": "hallucination|tool_error|knowledge_gap|logic_error|timeout|format_error|permission|context_loss|none",
  "explanation": "简短说明（50字内）"
}}
```
"""
        try:
            response = self.llm.invoke(prompt)
            import json as json_mod
            result = json_mod.loads(response.content)
            status = TraceStatus.FAILURE if result["is_failure"] else TraceStatus.SUCCESS
            failure_type = FailureType(result["failure_type"]) if result["failure_type"] != "none" else None
            return status, failure_type, result.get("explanation", "")
        except Exception as e:
            logger.warning(f"LLM labeling failed: {e}")
            return trace.status, trace.failure_type or FailureType.UNKNOWN, str(e)
