"""
AgentMine × LlamaIndex — 一行代码导出 Agent 日志

用法：
    from agentmine.integrations.llamaindex_observer import AgentMineObserver
    observer = AgentMineObserver("agent_logs.jsonl")

    agent = ReActAgent.from_tools(tools, llm=llm, callbacks=[observer])
    agent.chat("hello")
"""
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# 导入 AgentMineLogger（避免循环依赖）
try:
    from agentmine.integrations.langchain_callback import AgentMineLogger
except ImportError:
    from .langchain_callback import AgentMineLogger


try:
    from llama_index.core.callbacks import CallbackManager, CBEventType
    from llama_index.core.callbacks.base import BaseCallbackHandler
    from llama_index.core.callbacks.schema import EventPayload
    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False
    BaseCallbackHandler = object


if HAS_LLAMAINDEX:

    class AgentMineObserver(BaseCallbackHandler):
        """
        LlamaIndex Callback Handler

        自动捕获 LlamaIndex Agent 的每个事件，写入 AgentMine JSONL。

        用法：
            from agentmine.integrations.llamaindex_observer import AgentMineObserver
            import llama_index.core

            observer = AgentMineObserver("agent_logs.jsonl")
            llama_index.core.Settings.callback_manager = CallbackManager([observer])

            agent = ReActAgent.from_tools(tools, llm=llm)
            agent.chat("your question")
        """

        def __init__(self, output_path: str = "agent_logs.jsonl", agent_name: str = "llamaindex_agent"):
            self.logr = AgentMineLogger(output_path)
            self.agent_name = agent_name
            self._current_query = ""
            self._tool_calls: List[Dict] = []
            self._start_time = 0.0
            self._total_tokens = 0
            self._error = None
            self._output = ""

        def on_event_start(self, event_type, payload, **kwargs):
            if event_type == CBEventType.AGENT_STEP:
                if not self._start_time:
                    self._start_time = time.time()
                self._current_query = str(payload.get("input", ""))

            elif event_type == CBEventType.FUNCTION_CALL:
                tool_name = str(payload.get(EventPayload.TOOL_NAME, "unknown"))
                tool_input = payload.get(EventPayload.FUNCTION_CALL, {})
                self._tool_calls.append({
                    "name": tool_name,
                    "input": tool_input if isinstance(tool_input, dict) else {"query": str(tool_input)},
                    "start_time": time.time(),
                })

            elif event_type == CBEventType.LLM:
                pass  # 在 on_event_end 中收集 token

        def on_event_end(self, event_type, payload, **kwargs):
            if event_type == CBEventType.AGENT_STEP:
                response = payload.get(EventPayload.OUTPUT, "")
                if hasattr(response, "response"):
                    self._output = str(response.response)

            elif event_type == CBEventType.FUNCTION_CALL:
                # 更新最后一个工具调用的 output
                output = payload.get(EventPayload.FUNCTION_OUTPUT, "")
                if self._tool_calls:
                    self._tool_calls[-1]["output"] = str(output)[:500]
                    self._tool_calls[-1]["error"] = None

            elif event_type == CBEventType.LLM:
                token_usage = payload.get(EventPayload.TOKEN_USAGE, {})
                if hasattr(token_usage, "total_tokens"):
                    self._total_tokens += token_usage.total_tokens

        def on_event_error(self, event_type, payload, event_id, **kwargs):
            error_msg = str(payload.get("error", "Unknown error"))
            self._error = error_msg
            if self._tool_calls:
                self._tool_calls[-1]["error"] = error_msg

        def end_trace(self, query: str = ""):
            """
            手动结束当前 Trace 并写入日志。

            如果 LlamaIndex 没有在合适时机自动调用，可以在 agent.chat() 后手动调用。
            """
            latency_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0

            self.logr.log(
                query=self._current_query or query,
                output=self._output,
                error=self._error,
                tool_calls=self._tool_calls,
                tokens=self._total_tokens,
                latency_ms=latency_ms,
                agent_name=self.agent_name,
            )

            # 重置状态
            self._current_query = ""
            self._output = ""
            self._tool_calls = []
            self._start_time = 0.0
            self._total_tokens = 0
            self._error = None

else:
    class AgentMineObserver:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LlamaIndex is not installed. "
                "Use: pip install llama-index\n"
                "Or use the framework-agnostic AgentMineLogger instead."
            )
