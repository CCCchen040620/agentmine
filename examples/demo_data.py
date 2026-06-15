"""
演示数据生成器

生成包含真实感的成功和失败案例的 Agent 日志，
用于快速体验 AgentMine 的分析能力。
"""
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any


# ── 失败模式模板 ──────────────────────────────────────────
# 5类系统性的失败模式 + 对应的用户查询模板

FAILURE_PATTERNS = {
    "计算错误": {
        "weight": 0.30,
        "queries": [
            "帮我算一下这个季度三个部门的平均销售额，A部门50万，B部门30万，C部门28万",
            "如果订单总额是89200元，税率13%，税后金额是多少？",
            "这三个产品的平均售价是多少？A:128 B:256 C:399",
            "上个月销售总额45.6万，环比增长12%，这个月预期是多少？",
            "如果每人分3875元，23个人需要多少？",
        ],
        "error_messages": [
            "计算结果不准确，Agent直接用心算而非调用计算器",
            "数值运算错误，LLM在无工具辅助下进行数学推理",
        ],
    },
    "上下文丢失": {
        "weight": 0.25,
        "queries": [
            "之前你帮我查的华为订单，那个合同的付款条款是什么来着？",
            "上面说的那个政策，第三条具体怎么规定的？",
            "你再详细说说刚才提到的那个方案",
            "上次你帮我做的那个分析，能再细化一下吗？",
            "前面讨论了三个方案，你推荐哪个？为什么？",
        ],
        "error_messages": [
            "Agent无法回溯对话历史，回复与上下文无关",
            "多轮对话中忘记用户初始目标",
        ],
    },
    "知识库盲区": {
        "weight": 0.20,
        "queries": [
            "那个东西怎么申请来着？就是那个绿色的表格",
            "今年的新规定是啥意思？跟去年比有什么变化？",
            "那个叫什么来着，就是经常提到的那个原则",
            "听说最近出了个新政策，你知道吗？",
            "那个谁说的那个，具体怎么操作？",
        ],
        "error_messages": [
            "用户用口语化/指代模糊表述，检索召回不相关文档",
            "知识库缺少最新政策信息，Agent无法回答",
        ],
    },
    "工具调用失败": {
        "weight": 0.15,
        "queries": [
            "帮我查一下明天下午有空会议室吗？大概10个人",
            "审批一下我这周提交的报销单",
            "帮我看看CRM系统里XX客户的联系方式",
            "给我拉一下这个月的销售报表",
            "帮我查一下仓库里还有多少库存",
        ],
        "error_messages": [
            "API接口返回超时/500错误",
            "工具调用参数格式错误，API拒绝请求",
        ],
    },
    "权限不足": {
        "weight": 0.10,
        "queries": [
            "帮我看一下全公司这个月的薪资表",
            "能把王总的审批权限临时转给我吗？",
            "帮我删掉那个已完成的订单记录",
            "我想查看其他部门的绩效数据",
            "帮我把这个文件分享给外部人员",
        ],
        "error_messages": [
            "用户权限不足，Agent未做友好提示而是报错",
            "敏感操作被拦截但错误信息不友好",
        ],
    },
}


# ── 成功查询模板 ─────────────────────────────────────────
SUCCESS_QUERIES = [
    # 知识问答类
    "公司年假政策是什么？",
    "入职需要准备哪些材料？",
    "出差报销标准是多少？",
    "加班费怎么计算？",
    "公司有哪些培训课程？",
    "社保公积金缴费比例是多少？",
    "怎么申请办公用品？",
    "公司的绩效考核周期是怎样的？",
    "IT部门联系方式是什么？",
    "公司都有哪些员工福利？",
    # 数据查询类
    "查一下上个月的销售额",
    "今天有哪些会议？",
    "最近一周的新增客户有哪些？",
    "项目A的进度如何？",
    "我这周的日程安排是什么？",
    # 闲聊类
    "你好，今天心情怎么样？",
    "谢谢你的帮助！",
    "能做个自我介绍吗？",
    "今天天气真不错",
    "周六加班的话调休怎么算？",
]


def _generate_id(seed: str) -> str:
    """生成唯一 ID"""
    return hashlib.md5(seed.encode()).hexdigest()[:12]


def generate_demo_data(
    n_traces: int = 200,
    failure_rate: float = 0.25,
    output_dir: str = "examples",
) -> str:
    """
    生成演示用 Agent 日志

    Args:
        n_traces: 生成的 trace 总数
        failure_rate: 失败比例（默认 25%）
        output_dir: 输出目录

    Returns:
        生成的 JSONL 文件路径
    """
    random.seed(42)  # 可复现
    traces = []

    n_failures = int(n_traces * failure_rate)
    n_successes = n_traces - n_failures

    # ── 生成失败 trace ──────────────────────────────────────
    pattern_names = list(FAILURE_PATTERNS.keys())
    pattern_weights = [FAILURE_PATTERNS[p]["weight"] for p in pattern_names]

    for i in range(n_failures):
        # 选择一个失败模式（按权重）
        pattern_name = random.choices(pattern_names, weights=pattern_weights, k=1)[0]
        pattern = FAILURE_PATTERNS[pattern_name]

        # 随机选一个查询模板并稍作变体
        query_template = random.choice(pattern["queries"])
        user_query = _add_variation(query_template)

        error_msg = random.choice(pattern["error_messages"])
        timestamp = (datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )).isoformat()

        trace = {
            "trace_id": _generate_id(f"failure_{i}_{timestamp}"),
            "session_id": _generate_id(f"session_{random.randint(0, n_traces//3)}"),
            "agent_name": "SmartOffice-Agent-v1.2",
            "timestamp": timestamp,
            "user_query": user_query,
            "final_output": _generate_failure_output(error_msg),
            "error": error_msg,
            "error_type": "agent_execution_error",
            "tool_calls": _generate_tool_calls(pattern_name, failed=True),
            "total_tokens": random.randint(200, 2000),
            "total_latency_ms": random.uniform(500, 8000),
            "user_feedback": random.choice([None, "thumbs_down", "不准确", "答非所问"]),
            "user_rating": random.choice([None, 1, 2]),
        }
        traces.append(trace)

    # ── 生成成功 trace ──────────────────────────────────────
    for i in range(n_successes):
        query = random.choice(SUCCESS_QUERIES)
        timestamp = (datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )).isoformat()

        trace = {
            "trace_id": _generate_id(f"success_{i}_{timestamp}"),
            "session_id": _generate_id(f"session_{random.randint(0, n_traces//3)}"),
            "agent_name": "SmartOffice-Agent-v1.2",
            "timestamp": timestamp,
            "user_query": query,
            "final_output": _generate_success_output(query),
            "error": None,
            "error_type": None,
            "tool_calls": _generate_tool_calls("random", failed=False),
            "total_tokens": random.randint(100, 1500),
            "total_latency_ms": random.uniform(200, 3000),
            "user_feedback": random.choice([None, "thumbs_up", "thumbs_up"]),
            "user_rating": random.choice([None, 4, 5]),
        }
        traces.append(trace)

    # 打乱顺序
    random.shuffle(traces)

    # ── 写入文件 ──────────────────────────────────────────
    output_path = Path(output_dir) / "demo_agent_logs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"Demo data generated: {output_path} ({len(traces)} traces, "
          f"{n_failures} failures, {n_successes} successes)")

    return str(output_path)


def _add_variation(query: str) -> str:
    """给查询模板添加自然变体"""
    prefixes = ["", "请问", "帮我", "我想问一下", "麻烦你", "那个"]
    suffixes = ["", "啊", "呢", "吧", "哦", "谢谢"]
    return random.choice(prefixes) + query + random.choice(suffixes)


def _generate_failure_output(error_msg: str) -> str:
    """生成失败的 Agent 回复"""
    templates = [
        f"抱歉，我暂时无法处理这个请求。{error_msg}",
        f"对不起，{error_msg}。请稍后重试或联系管理员。",
        f"我遇到了一个问题：{error_msg}。建议您换个方式提问。",
    ]
    return random.choice(templates)


def _generate_success_output(query: str) -> str:
    """生成成功的 Agent 回复"""
    return f"根据查询结果，您的问题已经处理完毕。相关信息如下：..."


def _generate_tool_calls(pattern_name: str, failed: bool = False) -> List[Dict]:
    """生成模拟的工具调用记录"""
    tools_pool = ["knowledge_base_search", "crm_query", "oa_leave_request",
                  "calculator", "calendar_query", "report_generator"]

    if failed:
        num_calls = random.randint(1, 3)
        calls = []
        for _ in range(num_calls):
            tool = random.choice(tools_pool)
            calls.append({
                "name": tool,
                "input": {"query": "..."},
                "output": None,
                "error": "Tool execution failed" if random.random() > 0.5 else None,
                "latency_ms": random.uniform(100, 5000),
            })
        return calls
    else:
        return [{
            "name": random.choice(tools_pool),
            "input": {"query": "..."},
            "output": "Success",
            "error": None,
            "latency_ms": random.uniform(50, 500),
        }]


# ── 直接运行生成演示数据 ──────────────────────────────
if __name__ == "__main__":
    generate_demo_data(n_traces=200, output_dir=".")
