"""
多格式 Agent 日志解析器

自动检测并解析：
- JSONL (每行一个 Trace)
- CSV
- LangSmith 导出格式
- LangFuse 导出格式
- 自定义 JSON 数组
"""
import json
import csv
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime

from .schema import AgentTrace, ToolCall, TraceStatus, FailureType

logger = logging.getLogger(__name__)


class LogParser:
    """
    通用 Agent 日志解析器

    用法:
        parser = LogParser()
        traces = parser.parse_file("agent_logs.jsonl")
        traces = parser.parse_file("agent_logs.csv")
        traces = parser.parse_file("langsmith_export.json")
    """

    # ── 格式检测 ────────────────────────────────────────

    @staticmethod
    def detect_format(file_path: str) -> str:
        """
        自动检测日志文件格式

        Returns:
            "jsonl" | "csv" | "langsmith" | "langfuse" | "json"
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # 先读前几行判断
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        # JSONL: 每行以 { 开头
        if ext == ".jsonl" and first_line.startswith("{"):
            # 进一步判断是否是 LangSmith/LangFuse 格式
            try:
                obj = json.loads(first_line)
                if "trace" in obj and "observations" in obj:
                    return "langfuse"
                if "id" in obj and "run_type" in obj:
                    return "langsmith"
            except json.JSONDecodeError:
                pass
            return "jsonl"

        # CSV
        if ext == ".csv":
            return "csv"

        # JSON 数组
        if ext == ".json" and first_line.strip().startswith("["):
            return "json"

        return "jsonl"  # 默认尝试 JSONL

    # ── 主入口 ────────────────────────────────────────────

    def parse_file(self, file_path: str) -> List[AgentTrace]:
        """解析日志文件，自动检测格式"""
        fmt = self.detect_format(file_path)
        logger.info(f"Detected format: {fmt} for file: {file_path}")

        if fmt == "jsonl":
            return self._parse_jsonl(file_path)
        elif fmt == "csv":
            return self._parse_csv(file_path)
        elif fmt == "langsmith":
            return self._parse_langsmith(file_path)
        elif fmt == "langfuse":
            return self._parse_langfuse(file_path)
        elif fmt == "json":
            return self._parse_json_array(file_path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def parse_raw(self, data: List[Dict[str, Any]], framework: str = "custom") -> List[AgentTrace]:
        """从字典列表直接解析"""
        traces = []
        for item in data:
            try:
                trace = self._dict_to_trace(item, framework)
                traces.append(trace)
            except Exception as e:
                logger.warning(f"Failed to parse trace item: {e}")
        return traces

    # ── 格式解析器 ────────────────────────────────────────

    def _parse_jsonl(self, file_path: str) -> List[AgentTrace]:
        """解析 JSONL 格式（每行一个 JSON 对象）"""
        traces = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    trace = self._dict_to_trace(obj, framework="custom")
                    traces.append(trace)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_no}: JSON parse error: {e}")
                except Exception as e:
                    logger.warning(f"Line {line_no}: Trace parse error: {e}")
        logger.info(f"Parsed {len(traces)} traces from JSONL")
        return traces

    def _parse_csv(self, file_path: str) -> List[AgentTrace]:
        """解析 CSV 格式"""
        traces = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace = self._csv_row_to_trace(row)
                traces.append(trace)
        logger.info(f"Parsed {len(traces)} traces from CSV")
        return traces

    def _parse_json_array(self, file_path: str) -> List[AgentTrace]:
        """解析 JSON 数组格式"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Parsed {len(data)} traces from JSON array")
        return self.parse_raw(data, framework="custom")

    def _parse_langsmith(self, file_path: str) -> List[AgentTrace]:
        """
        解析 LangSmith 导出格式

        LangSmith 导出是 JSONL，每条记录结构：
        {
            "id": "run-uuid",
            "run_type": "chain" | "llm" | "tool",
            "name": "...",
            "inputs": {"input": "..."},
            "outputs": {"output": "..."},
            "error": null | "...",
            "child_runs": [...],
            ...
        }
        """
        traces = []
        sessions: Dict[str, List[dict]] = {}  # session -> runs

        # 第一遍：按 session 分组
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    session_id = obj.get("session_id", obj.get("id", "unknown"))
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(obj)
                except json.JSONDecodeError:
                    pass

        # 第二遍：每组 session 聚合为一个 Trace
        for session_id, runs in sessions.items():
            trace = self._langsmith_runs_to_trace(session_id, runs)
            if trace:
                traces.append(trace)

        logger.info(f"Parsed {len(traces)} traces from LangSmith export")
        return traces

    def _parse_langfuse(self, file_path: str) -> List[AgentTrace]:
        """
        解析 LangFuse 导出格式

        LangFuse 导出结构：
        {
            "trace": {"id": "...", "userId": "...", "name": "..."},
            "observations": [
                {"type": "GENERATION", "input": "...", "output": "...", ...},
                {"type": "SPAN", "name": "tool_call", ...},
            ]
        }
        """
        traces = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    trace = self._langfuse_dict_to_trace(obj)
                    traces.append(trace)
                except json.JSONDecodeError:
                    pass

        logger.info(f"Parsed {len(traces)} traces from LangFuse export")
        return traces

    # ── 内部转换方法 ──────────────────────────────────────

    def _dict_to_trace(self, obj: Dict[str, Any], framework: str = "custom") -> AgentTrace:
        """将通用字典转为 AgentTrace"""

        # 生成或提取 trace_id
        trace_id = obj.get("trace_id") or obj.get("id") or obj.get("run_id") or \
                   obj.get("conversation_id") or self._generate_id(obj)

        # ── 工具调用 ──────────────────────────────────────
        tool_calls = []
        for tc_data in obj.get("tool_calls", []):
            if isinstance(tc_data, dict):
                tool_calls.append(ToolCall(
                    tool_name=tc_data.get("name") or tc_data.get("tool_name", "unknown"),
                    tool_input=tc_data.get("input") or tc_data.get("arguments", {}),
                    tool_output=str(tc_data.get("output", "")) if tc_data.get("output") else None,
                    error=tc_data.get("error"),
                    latency_ms=float(tc_data.get("latency_ms", 0)),
                ))

        # ── 判断状态 ──────────────────────────────────────
        error = obj.get("error") or obj.get("error_message")
        feedback = obj.get("user_feedback") or obj.get("feedback")
        rating = obj.get("rating") or obj.get("user_rating")

        if error:
            status = TraceStatus.FAILURE
        elif feedback and any(neg in str(feedback).lower() for neg in ["down", "bad", "不", "差", "错", "误"]):
            status = TraceStatus.FAILURE
        elif rating is not None and int(rating) <= 2:
            status = TraceStatus.FAILURE
        else:
            status = TraceStatus.SUCCESS

        return AgentTrace(
            trace_id=str(trace_id),
            session_id=str(obj.get("session_id", "")),
            agent_name=str(obj.get("agent_name", obj.get("name", "unknown"))),
            framework=framework,
            timestamp=obj.get("timestamp") or obj.get("created_at") or obj.get("time"),
            user_query=str(obj.get("user_query") or obj.get("query") or obj.get("input") or obj.get("question", "")),
            user_feedback=str(feedback) if feedback else None,
            user_rating=int(rating) if rating is not None else None,
            final_output=str(obj.get("final_output") or obj.get("output") or obj.get("response") or obj.get("answer", "")),
            intermediate_steps=obj.get("intermediate_steps", []),
            tool_calls=tool_calls,
            total_tokens=int(obj.get("total_tokens", 0)),
            total_latency_ms=float(obj.get("latency_ms", 0)),
            error=str(error) if error else None,
            error_type=obj.get("error_type"),
            stack_trace=obj.get("stack_trace"),
            status=status,
            metadata=obj.get("metadata", {}),
            raw_data=obj,
        )

    def _csv_row_to_trace(self, row: Dict[str, str]) -> AgentTrace:
        """CSV 行转 Trace"""
        trace_id = row.get("trace_id") or row.get("id") or self._generate_id(row)
        error = row.get("error") or row.get("error_message")

        return AgentTrace(
            trace_id=str(trace_id),
            session_id=row.get("session_id", ""),
            agent_name=row.get("agent_name", "unknown"),
            framework="csv",
            timestamp=row.get("timestamp"),
            user_query=row.get("user_query") or row.get("query") or row.get("input", ""),
            user_feedback=row.get("user_feedback"),
            user_rating=int(row["user_rating"]) if row.get("user_rating") else None,
            final_output=row.get("final_output") or row.get("output", ""),
            total_tokens=int(row.get("total_tokens", 0)),
            total_latency_ms=float(row.get("latency_ms", 0)),
            error=error,
            error_type=row.get("error_type"),
            status=TraceStatus.FAILURE if error else TraceStatus.SUCCESS,
            raw_data=dict(row),
        )

    def _langsmith_runs_to_trace(self, session_id: str, runs: List[dict]) -> Optional[AgentTrace]:
        """将 LangSmith 的多条 run 聚合为一条 Trace"""
        if not runs:
            return None

        # 找主 chain run
        main_run = None
        for run in runs:
            if run.get("run_type") == "chain" and run.get("parent_run_id") is None:
                main_run = run
                break
        if not main_run:
            main_run = runs[0]  # fallback

        # 提取工具调用
        tool_calls = []
        for run in runs:
            if run.get("run_type") == "tool":
                tool_calls.append(ToolCall(
                    tool_name=run.get("name", "unknown"),
                    tool_input=run.get("inputs", {}),
                    tool_output=str(run.get("outputs", {})) if run.get("outputs") else None,
                    error=run.get("error"),
                ))

        error = main_run.get("error")
        return AgentTrace(
            trace_id=main_run.get("id", session_id),
            session_id=session_id,
            agent_name=main_run.get("name", "unknown"),
            framework="langsmith",
            timestamp=main_run.get("start_time"),
            user_query=str(main_run.get("inputs", {}).get("input", "")),
            final_output=str(main_run.get("outputs", {}).get("output", "")),
            tool_calls=tool_calls,
            total_tokens=int(main_run.get("total_tokens", 0)),
            error=str(error) if error else None,
            status=TraceStatus.FAILURE if error else TraceStatus.SUCCESS,
            raw_data={"run_count": len(runs)},
        )

    def _langfuse_dict_to_trace(self, obj: dict) -> AgentTrace:
        """转换 LangFuse 导出格式"""
        trace_data = obj.get("trace", {})
        observations = obj.get("observations", [])

        trace_id = trace_data.get("id", self._generate_id(obj))

        # 分离 LLM 调用和工具调用
        user_query = ""
        final_output = ""
        tool_calls = []
        error = None

        for obs in observations:
            obs_type = obs.get("type", "")
            if obs_type == "GENERATION":
                # LLM 调用
                gen_input = obs.get("input", "")
                gen_output = obs.get("output", "")
                if isinstance(gen_input, str):
                    user_query = gen_input
                if isinstance(gen_output, str):
                    final_output = gen_output
                if obs.get("level") == "ERROR":
                    error = obs.get("statusMessage", "Generation error")

            elif obs_type == "SPAN" and obs.get("name", "").startswith("tool"):
                tool_calls.append(ToolCall(
                    tool_name=obs.get("name", "unknown"),
                    tool_input=obs.get("input", {}),
                    tool_output=str(obs.get("output", "")) if obs.get("output") else None,
                    error=obs.get("statusMessage"),
                ))

        return AgentTrace(
            trace_id=str(trace_id),
            session_id=str(trace_data.get("userId", "")),
            agent_name=str(trace_data.get("name", "unknown")),
            framework="langfuse",
            timestamp=trace_data.get("timestamp"),
            user_query=user_query,
            final_output=final_output,
            tool_calls=tool_calls,
            error=error,
            status=TraceStatus.FAILURE if error else TraceStatus.SUCCESS,
            raw_data=obj,
        )

    def _generate_id(self, obj: Any) -> str:
        """为没有 ID 的记录生成唯一 ID"""
        raw = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]
