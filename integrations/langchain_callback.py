"""
AgentMine × LangChain — 一行代码导出 Agent 日志

用法 1（零侵入）：
    from agentmine.integrations.langchain_callback import AgentMineCallback
    agent = create_agent(..., callbacks=[AgentMineCallback("agent_logs.jsonl")])

用法 2（装饰器）：
    from agentmine.integrations.langchain_callback import trace_to_agentmine
    @trace_to_agentmine("agent_logs.jsonl")
    def my_agent(query): ...

用法 3（手动记录）：
    from agentmine.integrations.langchain_callback import AgentMineLogger
    logger = AgentMineLogger("agent_logs.jsonl")
    logger.log(query="用户问题", output="Agent回复", error=None)
"""
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from functools import wraps

logger = logging.getLogger(__name__)


class AgentMineLogger:
    """
    AgentMine 日志记录器

    写入标准 JSONL 格式，AgentMine 的 parser 可以直接解析。
    不依赖 LangChain，任何 Agent 框架都可以用。
    """

    def __init__(self, output_path: str = "agent_logs.jsonl"):
        self.output_path = Path(output_path)

    def log(
        self,
        query: str,
        output: str = "",
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        tokens: int = 0,
        latency_ms: float = 0.0,
        feedback: Optional[str] = None,
        session_id: str = "",
        agent_name: str = "agent",
        **metadata,
    ) -> str:
        """
        记录一条 Agent 交互

        Args:
            query: 用户输入
            output: Agent 最终输出
            error: 错误信息（None 表示成功）
            tool_calls: 工具调用列表 [{"name": "...", "input": {...}, "output": "...", "error": null}]
            tokens: Token 消耗
            latency_ms: 延迟（毫秒）
            feedback: 用户反馈
        Returns:
            trace_id
        """
        trace_id = hashlib.md5(
            f"{query}{time.time()}".encode()
        ).hexdigest()[:12]

        record = {
            "trace_id": trace_id,
            "session_id": session_id or trace_id,
            "agent_name": agent_name,
            "framework": "langchain",
            "timestamp": datetime.now().isoformat(),
            "user_query": query,
            "final_output": output,
            "error": error,
            "error_type": error_type,
            "tool_calls": tool_calls or [],
            "total_tokens": tokens,
            "total_latency_ms": latency_ms,
            "user_feedback": feedback,
            "metadata": metadata,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return trace_id

    def log_batch(self, records: List[Dict]):
        """批量写入"""
        with open(self.output_path, "a", encoding="utf-8") as f:
            for r in records:
                if "trace_id" not in r:
                    r["trace_id"] = hashlib.md5(
                        f"{r.get('user_query', '')}{time.time()}".encode()
                    ).hexdigest()[:12]
                if "timestamp" not in r:
                    r["timestamp"] = datetime.now().isoformat()
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def trace_to_agentmine(output_path: str = "agent_logs.jsonl"):
    """
    装饰器：自动记录函数调用

    @trace_to_agentmine("my_agent_logs.jsonl")
    def my_agent(query: str) -> str:
        return llm.invoke(query)

    result = my_agent("hello")
    """
    logr = AgentMineLogger(output_path)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            query = args[0] if args else kwargs.get("query", str(kwargs))
            start = time.time()
            error = None
            output = ""
            try:
                output = func(*args, **kwargs)
                if not isinstance(output, str):
                    output = str(output)
            except Exception as e:
                error = str(e)
                raise
            finally:
                latency_ms = (time.time() - start) * 1000
                logr.log(
                    query=str(query),
                    output=str(output) if not error else "",
                    error=error,
                    latency_ms=latency_ms,
                    agent_name=func.__name__,
                )
            return output
        return wrapper
    return decorator


# ── LangChain Callback Handler ──────────────────────────

try:
    from langchain.callbacks.base import BaseCallbackHandler
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseCallbackHandler = object


if HAS_LANGCHAIN:

    class AgentMineCallback(BaseCallbackHandler):
        """
        LangChain Callback Handler

        自动捕获 Agent 执行的每一步，输出 AgentMine 兼容的 JSONL。

        用法：
            from agentmine.integrations.langchain_callback import AgentMineCallback
            callback = AgentMineCallback("agent_logs.jsonl")

            agent = create_openai_functions_agent(llm, tools, prompt)
            executor = AgentExecutor(agent, tools, callbacks=[callback])
            executor.invoke({"input": "hello"})

        每次 invoke 结束后自动写入一条完整的 Trace。
        """

        def __init__(self, output_path: str = "agent_logs.jsonl", agent_name: str = "langchain_agent"):
            self.logr = AgentMineLogger(output_path)
            self.agent_name = agent_name
            self._current_query = ""
            self._current_output = ""
            self._tool_calls: List[Dict] = []
            self._start_time = 0.0
            self._total_tokens = 0
            self._error = None

        def on_chain_start(self, serialized, inputs, **kwargs):
            self._start_time = time.time()
            self._tool_calls = []
            self._total_tokens = 0
            self._error = None
            # 提取用户输入
            self._current_query = str(inputs.get("input", inputs))

        def on_chain_end(self, outputs, **kwargs):
            self._current_output = str(outputs.get("output", outputs))
            latency_ms = (time.time() - self._start_time) * 1000

            self.logr.log(
                query=self._current_query,
                output=self._current_output,
                error=self._error,
                tool_calls=self._tool_calls,
                tokens=self._total_tokens,
                latency_ms=latency_ms,
                agent_name=self.agent_name,
            )

        def on_chain_error(self, error, **kwargs):
            self._error = str(error)

        def on_tool_start(self, serialized, input_str, **kwargs):
            self._current_tool = {
                "name": serialized.get("name", "unknown"),
                "input": {"query": str(input_str)} if isinstance(input_str, str) else input_str,
                "start_time": time.time(),
            }

        def on_tool_end(self, output, **kwargs):
            if hasattr(self, "_current_tool"):
                self._current_tool["output"] = str(output)[:500]
                self._current_tool["error"] = None
                self._tool_calls.append(self._current_tool)
                del self._current_tool

        def on_tool_error(self, error, **kwargs):
            if hasattr(self, "_current_tool"):
                self._current_tool["error"] = str(error)
                self._tool_calls.append(self._current_tool)
                del self._current_tool

        def on_llm_end(self, response, **kwargs):
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                self._total_tokens += token_usage.get("total_tokens", 0)

else:
    class AgentMineCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain is not installed. "
                "Use: pip install langchain\n"
                "Or use the framework-agnostic AgentMineLogger instead."
            )
