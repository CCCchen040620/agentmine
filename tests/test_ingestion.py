"""
Tests for log ingestion and labeling pipeline
"""
import json
import tempfile
from pathlib import Path

from src.ingestion.parser import LogParser
from src.ingestion.schema import AgentTrace, TraceStatus, ToolCall
from src.labeling.auto_labeler import AutoLabeler


def test_parse_jsonl_basic():
    """Test parsing a basic JSONL file"""
    data = [
        {
            "trace_id": "test-001",
            "user_query": "公司年假政策是什么？",
            "final_output": "员工每年享有5天年假...",
            "error": None,
        },
        {
            "trace_id": "test-002",
            "user_query": "帮我查销售数据",
            "final_output": "",
            "error": "Connection timeout",
            "error_type": "timeout",
        },
    ]

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp_path = f.name

    try:
        parser = LogParser()
        traces = parser.parse_file(tmp_path)
        assert len(traces) == 2
        assert traces[0].trace_id == "test-001"
        assert traces[0].user_query == "公司年假政策是什么？"
        assert traces[1].error == "Connection timeout"
    finally:
        Path(tmp_path).unlink()


def test_parse_jsonl_with_tool_calls():
    """Test parsing traces with tool call records"""
    data = [{
        "trace_id": "test-003",
        "user_query": "华为订单情况",
        "final_output": "华为有2个订单...",
        "tool_calls": [
            {
                "name": "crm_query",
                "input": {"keyword": "华为"},
                "output": "found 2 orders",
                "error": None,
            }
        ],
    }]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        json.dump(data[0], f)
        f.write("\n")
        tmp_path = f.name

    try:
        parser = LogParser()
        traces = parser.parse_file(tmp_path)
        assert len(traces) == 1
        assert len(traces[0].tool_calls) == 1
        assert traces[0].tool_calls[0].tool_name == "crm_query"
    finally:
        Path(tmp_path).unlink()


def test_auto_labeler_failure_detection():
    """Test automatic failure labeling"""
    parser = LogParser()

    # Explicit error
    error_trace = AgentTrace(
        trace_id="err-001",
        user_query="test",
        error="Something went wrong",
    )

    # User negative feedback
    neg_feedback_trace = AgentTrace(
        trace_id="err-002",
        user_query="test",
        final_output="some output",
        user_feedback="thumbs_down",
    )

    # Success
    success_trace = AgentTrace(
        trace_id="ok-001",
        user_query="公司年假",
        final_output="员工享有5天年假，可以登录OA系统申请。",
    )

    labeler = AutoLabeler()
    traces = labeler.label([error_trace, neg_feedback_trace, success_trace])

    assert traces[0].status == TraceStatus.FAILURE
    assert traces[1].status == TraceStatus.FAILURE
    assert traces[2].status == TraceStatus.SUCCESS


def test_auto_labeler_tool_error():
    """Test labeling of tool call errors"""
    trace = AgentTrace(
        trace_id="tool-err-001",
        user_query="查询订单",
        final_output="",
        tool_calls=[
            ToolCall(
                tool_name="crm_query",
                tool_input={"keyword": "test"},
                error="Connection refused",
            )
        ],
    )

    labeler = AutoLabeler()
    labeler.label([trace])
    assert trace.status == TraceStatus.FAILURE


def test_detect_format():
    """Test log format auto-detection"""
    parser = LogParser()

    # JSONL
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        f.write('{"trace_id":"t1","user_query":"test"}\n')
        tmp = f.name
    assert parser.detect_format(tmp) in ("jsonl", "langsmith")
    Path(tmp).unlink()

    # CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("trace_id,user_query,final_output\n")
        f.write("t1,test,ok\n")
        tmp = f.name
    assert parser.detect_format(tmp) == "csv"
    Path(tmp).unlink()


def test_schema_diagnostic_text():
    """Test diagnostic text generation"""
    trace = AgentTrace(
        trace_id="diag-001",
        user_query="帮我算一下销售额",
        final_output="抱歉，计算出现了偏差",
        error="计算结果不准确",
        tool_calls=[
            ToolCall(tool_name="calculator", tool_input={"expr": "100+200"}, error=None),
        ],
    )

    diag_text = trace.get_diagnostic_text()
    assert "销售额" in diag_text
    assert "calculator" in diag_text

    rich_text = trace.get_rich_context()
    assert "抱歉" in rich_text or "计算" in rich_text


def test_demo_data_generation():
    """Test demo data generation produces valid traces"""
    from examples.demo_data import generate_demo_data
    import tempfile

    output_dir = tempfile.mkdtemp()
    try:
        filepath = generate_demo_data(n_traces=50, output_dir=output_dir)

        parser = LogParser()
        traces = parser.parse_file(filepath)
        assert len(traces) == 50

        labeler = AutoLabeler()
        traces = labeler.label(traces)

        failures = [t for t in traces if t.status.value == "failure"]
        successes = [t for t in traces if t.status.value == "success"]

        # Verify both success and failure traces exist
        assert len(failures) > 0, "Should have some failure traces"
        assert len(successes) > 0, "Should have some success traces"
        assert len(failures) + len(successes) == 50

    finally:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
