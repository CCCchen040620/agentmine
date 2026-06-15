"""
AgentMine 标准化数据模型

所有摄入的 Agent 日志统一转换为此 schema，屏蔽不同框架的格式差异。
支持: LangChain, LlamaIndex, LangFuse, 自定义 JSON/CSV
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class FailureType(str, Enum):
    """失败类型分类"""
    HALLUCINATION = "hallucination"       # 幻觉/编造
    TOOL_ERROR = "tool_error"              # 工具调用失败
    KNOWLEDGE_GAP = "knowledge_gap"        # 知识库盲区
    LOGIC_ERROR = "logic_error"            # 逻辑推理错误
    TIMEOUT = "timeout"                    # 超时
    FORMAT_ERROR = "format_error"          # 输出格式错误
    PERMISSION = "permission"              # 权限不足
    CONTEXT_LOSS = "context_loss"          # 上下文丢失
    UNKNOWN = "unknown"                    # 未知


class TraceStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # 部分成功


@dataclass
class ToolCall:
    """单次工具调用记录"""
    tool_name: str
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_output: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: Optional[str] = None


@dataclass
class AgentTrace:
    """
    标准化 Agent 执行轨迹

    一条 Trace 代表一次完整的用户-Agent 交互。
    """
    # ── 基础标识 ──────────────────────────────────────
    trace_id: str                                    # 唯一ID
    session_id: str = ""                             # 会话ID（多轮对话共享）
    agent_name: str = "unknown"                      # Agent 名称
    framework: str = "unknown"                       # 来源框架 (langchain/llamaindex/langfuse/custom)
    timestamp: Optional[str] = None                  # 交互时间 ISO8601

    # ── 用户交互 ──────────────────────────────────────
    user_query: str = ""                             # 用户原始问题
    user_feedback: Optional[str] = None              # 用户反馈 (thumbs_up/thumbs_down/具体文字)
    user_rating: Optional[int] = None                # 用户评分 (1-5)

    # ── Agent 执行 ──────────────────────────────────────
    final_output: str = ""                           # Agent 最终回复
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)  # 中间步骤
    tool_calls: List[ToolCall] = field(default_factory=list)  # 工具调用记录
    total_tokens: int = 0                            # 总 token 消耗
    total_latency_ms: float = 0.0                    # 总延迟

    # ── 错误信息 ──────────────────────────────────────
    error: Optional[str] = None                      # 错误消息
    error_type: Optional[str] = None                 # 错误类型
    stack_trace: Optional[str] = None                # 异常堆栈

    # ── 标注信息（分析过程中填充） ────────────────────
    status: TraceStatus = TraceStatus.SUCCESS        # 成功/失败
    failure_type: Optional[FailureType] = None       # 失败类型
    failure_cluster_id: int = -1                     # 所属失败簇
    root_cause_summary: str = ""                     # 根因摘要

    # ── 扩展字段 ──────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)  # 保留原始数据

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于序列化）"""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "framework": self.framework,
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "user_feedback": self.user_feedback,
            "user_rating": self.user_rating,
            "final_output": self.final_output,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
            "error_type": self.error_type,
            "status": self.status.value,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "failure_cluster_id": self.failure_cluster_id,
            "root_cause_summary": self.root_cause_summary,
            "tool_call_count": len(self.tool_calls),
            "tool_names": [tc.tool_name for tc in self.tool_calls],
        }

    def get_diagnostic_text(self) -> str:
        """
        拼接用于聚类和诊断的文本

        包含用户问题 + Agent 回复摘要 + 错误信息
        """
        parts = [f"Query: {self.user_query}"]

        if self.final_output:
            # 截断长回复
            output = self.final_output[:500] + "..." if len(self.final_output) > 500 else self.final_output
            parts.append(f"Output: {output}")

        if self.error:
            parts.append(f"Error: {self.error}")

        if self.error_type:
            parts.append(f"ErrorType: {self.error_type}")

        if self.tool_calls:
            tools_summary = ", ".join([tc.tool_name for tc in self.tool_calls])
            parts.append(f"ToolsUsed: {tools_summary}")

            # 工具失败信息
            for tc in self.tool_calls:
                if tc.error:
                    parts.append(f"ToolError({tc.tool_name}): {tc.error}")

        return "\n".join(parts)

    def get_compact_query(self) -> str:
        """获取紧凑版查询文本（仅用户问题，用于 embedding）"""
        return self.user_query.strip()

    def get_rich_context(self) -> str:
        """获取丰富上下文（用户问题 + 关键步骤，用于 LLM 分析）"""
        parts = [f"用户问题: {self.user_query}"]

        if self.tool_calls:
            parts.append("工具调用序列:")
            for i, tc in enumerate(self.tool_calls, 1):
                status = "❌" if tc.error else "✅"
                parts.append(f"  {status} Step {i}: {tc.tool_name}")
                parts.append(f"     Input: {str(tc.tool_input)[:200]}")
                if tc.error:
                    parts.append(f"     Error: {tc.error}")
                elif tc.tool_output:
                    parts.append(f"     Output: {tc.tool_output[:200]}")

        parts.append(f"\nAgent 最终回复:\n{self.final_output[:500]}")

        if self.error:
            parts.append(f"\n错误信息: {self.error}")

        if self.user_feedback:
            parts.append(f"\n用户反馈: {self.user_feedback}")

        return "\n".join(parts)
