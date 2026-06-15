"""
AgentMine CLI

用法：
    agentmine analyze agent_logs.jsonl              # 分析日志
    agentmine analyze agent_logs.jsonl --llm        # 用 LLM 做深度分析
    agentmine analyze agent_logs.jsonl -o report    # 指定输出目录
    agentmine ui                                    # 启动 Web 界面
    agentmine demo                                  # 生成演示数据并分析
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentmine")


# ── CLI 入口 ────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="AgentMine")
def main():
    """
    🔍 AgentMine — AI Agent 失效模式自动挖掘工具

    从 Agent 执行日志中自动发现系统性的失败模式。
    """
    pass


@main.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("-o", "--output", default="agentmine_report", help="输出目录或 HTML 文件路径")
@click.option("--llm/--no-llm", default=False, help="使用 LLM 做深度根因分析（需要配置 API Key）")
@click.option("--model", default=None, help="LLM 模型名称 (如 gpt-4o, claude-sonnet-4-6)")
@click.option("--min-cluster-size", default=5, help="最小簇大小 (默认 5)")
@click.option("--open", "open_browser", is_flag=True, default=True, help="生成后自动打开浏览器")
def analyze(log_file: str, output: str, llm: bool, model: Optional[str],
            min_cluster_size: int, open_browser: bool):
    """
    分析 Agent 日志，挖掘失败模式

    LOG_FILE: Agent 执行日志文件 (支持 JSONL/CSV/LangSmith/LangFuse 格式)
    """
    click.echo("")
    click.echo("🔍 AgentMine v0.1.0 — AI Agent Failure Pattern Miner")
    click.echo("─" * 55)

    # ── Step 1: 日志摄入 ──────────────────────────────────
    from src.ingestion.parser import LogParser

    parser = LogParser()
    click.echo(f"📂 Loading: {Path(log_file).name}...")
    traces = parser.parse_file(log_file)
    click.echo(f"   Loaded {len(traces)} conversation traces")

    if not traces:
        click.secho("❌ No traces found. Please check the log file format.", fg="red")
        return

    # ── Step 2: 自动标注 ──────────────────────────────────
    from src.labeling.auto_labeler import AutoLabeler

    labeler = AutoLabeler(use_llm=False)
    traces = labeler.label(traces)

    failures = [t for t in traces if t.status.value == "failure"]
    successes = [t for t in traces if t.status.value == "success"]
    partials = [t for t in traces if t.status.value == "partial"]

    failure_rate = len(failures) / max(len(traces), 1) * 100
    click.echo(
        f"🏷️  Auto-labeled: {len(failures)} failures, "
        f"{len(successes)} successes, {len(partials)} partial "
        f"({failure_rate:.1f}% failure rate)"
    )

    if not failures:
        click.secho("✅ No failures found! Your agent looks healthy.", fg="green")
        return

    # ── Step 3: 聚类 ──────────────────────────────────────
    from src.clustering.embedder import TraceEmbedder
    from src.clustering.clusterer import FailureClusterer

    click.echo("")
    click.echo("🧬 Clustering failures...")
    embedder = TraceEmbedder()
    embeddings = embedder.embed(failures)
    click.echo(f"   ✓ Generated {embeddings.shape[1]}-dim embeddings for {len(failures)} queries")

    clusterer = FailureClusterer(min_cluster_size=min_cluster_size)
    result = clusterer.cluster(failures, embeddings)
    click.echo(f"   ✓ HDBSCAN found {len(result.clusters)} failure clusters "
               f"({result.noise_count} noise points discarded)")

    if not result.clusters:
        click.secho("⚠️  No clear failure clusters found. Try with fewer traces or smaller min_cluster_size.", fg="yellow")
        return

    # ── Step 4: 簇摘要 ────────────────────────────────────
    from src.clustering.summarizer import ClusterSummarizer

    llm_instance = _get_llm(llm, model) if llm else None
    summarizer = ClusterSummarizer(llm=llm_instance)

    # ── Step 5: 根因分析 + 金丝雀（可选） ─────────────────
    if llm and llm_instance:
        click.echo("")
        click.echo("🔬 Analyzing root causes with LLM...")
        from src.analysis.root_cause import RootCauseAnalyzer
        from src.analysis.canary import CanaryGenerator

        analyzer = RootCauseAnalyzer(llm=llm_instance)
        result.clusters = analyzer.analyze_all(result.clusters)

        canary_gen = CanaryGenerator(llm=llm_instance)
        result.clusters = canary_gen.generate_all(result.clusters)
    else:
        # 纯规则模式，仅做摘要
        for cluster in result.clusters:
            summary = summarizer.summarize(cluster.sample_traces, cluster.size)
            cluster.label = summary.get("label", f"Cluster {cluster.cluster_id}")
            cluster.summary = summary.get("summary", "")

    # ── Step 6: 终端输出 ─────────────────────────────────
    click.echo("")
    click.echo("─" * 55)
    for cluster in result.clusters:
        emoji = "🔴" if cluster.percentage > 20 else "🟡" if cluster.percentage > 10 else "🟢"
        click.echo(f"  {emoji} Cluster {cluster.cluster_id}: {cluster.label} "
                   f"({cluster.size} cases, {cluster.percentage}%)")
        if cluster.summary:
            click.echo(f"     └─ {cluster.summary}")
        if cluster.canary_queries:
            click.echo(f"     └─ 🦜 {len(cluster.canary_queries)} canary queries generated")
    click.echo("─" * 55)

    # ── Step 7: 生成 HTML 报告 ────────────────────────────
    from src.report.html_generator import HTMLReportGenerator

    report_gen = HTMLReportGenerator()
    report_path = report_gen.generate(
        clustering_result=result,
        all_traces=traces,
        failures=failures,
        output_path=output,
    )

    if result.clusters:
        highest_impact = result.clusters[0]
        click.echo("")
        click.secho(f"💡 Fix Cluster {highest_impact.cluster_id} first — "
                     f"{highest_impact.percentage}% of failures, highest impact.", fg="yellow")

    click.echo("")
    click.secho(f"✅ Done! Report: {report_path}", fg="green")

    if open_browser:
        click.launch(report_path)


@main.command()
@click.option("--port", default=8765, help="Web UI 端口")
@click.option("--host", default="127.0.0.1", help="绑定地址")
def ui(port: int, host: str):
    """
    启动 Web UI（拖拽上传日志，在线分析）
    """
    click.echo(f"🌐 Starting AgentMine Web UI at http://{host}:{port}")
    # 启动 FastAPI
    import uvicorn
    uvicorn.run("src.web.app:app", host=host, port=port, reload=True)


@main.command()
@click.option("-o", "--output", default="agentmine_demo", help="输出目录")
@click.option("--traces", default=200, help="生成的 trace 数量")
@click.option("--llm/--no-llm", default=False, help="使用 LLM 深度分析")
def demo(output: str, traces: int, llm: bool):
    """
    生成演示数据并运行分析（快速体验）
    """
    from examples.demo_data import generate_demo_data

    click.echo("🎭 Generating demo data...")
    demo_file = generate_demo_data(n_traces=traces, output_dir=output)
    click.echo(f"   Generated {traces} demo traces")

    # 调用 analyze
    ctx = click.get_current_context()
    ctx.invoke(analyze, log_file=demo_file, output=output, llm=llm,
               min_cluster_size=3, open_browser=True)


# ── 辅助函数 ──────────────────────────────────────────────

def _get_llm(use_llm: bool, model: Optional[str] = None):
    """获取 LLM 实例"""
    if not use_llm:
        return None

    try:
        from src.utils.llm import create_llm
        return create_llm(model=model)
    except ImportError as e:
        click.secho(f"⚠️  LLM not available: {e}", fg="yellow")
        click.echo("   Running in rule-based mode (no LLM deep analysis)")
        return None


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    main()
