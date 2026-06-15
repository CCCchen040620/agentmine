"""
AgentMine CLI — Rich-powered terminal UI

用法：
    agentmine analyze agent_logs.jsonl              # 分析日志
    agentmine analyze agent_logs.jsonl --llm        # 用 LLM 做深度分析
    agentmine analyze agent_logs.jsonl -o report    # 指定输出目录
    agentmine ui                                    # 启动 Web 界面
    agentmine demo                                  # 生成演示数据并分析
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

# ── Windows UTF-8 兼容 ────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 屏蔽 sentence-transformers 内部的 tqdm
os.environ.setdefault("DISABLE_TQDM", "1")

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich import box

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="%H:%M:%S",
)
# 屏蔽第三方库的 verbose 日志和警告
for noisy_lib in ["sentence_transformers", "urllib3", "huggingface_hub", "httpx", "httpcore", "transformers"]:
    logging.getLogger(noisy_lib).setLevel(logging.WARNING)
logging.getLogger("src").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*n_jobs.*")
warnings.filterwarnings("ignore", message=".*get_sentence_embedding_dimension.*")
logger = logging.getLogger("agentmine")
console = Console()

# ── 依赖检查工具 ──────────────────────────────────────────

def _ensure_installed(package_name: str, pip_name: str = None, extra: str = None):
    """检查依赖是否安装，没有则给出友好提示"""
    if pip_name is None:
        pip_name = package_name
    try:
        __import__(package_name)
        return True
    except ImportError:
        msg = f"[bold red]❌ {pip_name}[/bold red] 未安装"
        if extra:
            cmd = f"pip install agentmine[{extra}]"
        else:
            cmd = f"pip install {pip_name}"
        console.print(f"{msg}\n   请执行: [bold cyan]{cmd}[/bold cyan]")
        return False


# ── CLI 入口 ──────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="AgentMine")
def main():
    """🔍 AgentMine — AI Agent 失效模式自动挖掘工具"""
    pass


@main.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("-o", "--output", default="agentmine_report", help="输出 HTML 文件路径")
@click.option("--llm/--no-llm", default=False, help="使用 LLM 做深度根因分析（需配置 API Key）")
@click.option("--model", default=None, help="LLM 模型名称 (如 gpt-4o, claude-sonnet-4-6)")
@click.option("--min-cluster-size", default=5, help="最小簇大小 (默认 5)")
@click.option("--open", "open_browser", is_flag=True, default=True, help="生成后自动打开浏览器")
def analyze(log_file: str, output: str, llm: bool, model: Optional[str],
            min_cluster_size: int, open_browser: bool):
    """
    分析 Agent 日志，挖掘失败模式

    LOG_FILE: Agent 执行日志文件 (JSONL / CSV / LangSmith / LangFuse)
    """
    # ── 依赖检查 ──────────────────────────────────────────
    deps_ok = True
    deps_ok &= _ensure_installed("sentence_transformers", "sentence-transformers")
    deps_ok &= _ensure_installed("hdbscan", "hdbscan")
    deps_ok &= _ensure_installed("umap", "umap-learn")
    deps_ok &= _ensure_installed("pandas", "pandas")
    deps_ok &= _ensure_installed("numpy", "numpy")
    deps_ok &= _ensure_installed("jinja2", "jinja2")
    if not deps_ok:
        sys.exit(1)

    # ── Header ────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold bright_cyan]🔍 AgentMine v0.1.0[/bold bright_cyan]\n"
        "[dim]AI Agent Failure Pattern Miner[/dim]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    # ── Step 1: 日志摄入 ──────────────────────────────────
    console.print(Rule("[bold]📂 日志摄入[/bold]", style="dim"))

    from src.ingestion.parser import LogParser
    parser = LogParser()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]正在解析 {Path(log_file).name}...", total=None)
        traces = parser.parse_file(log_file)

    if not traces:
        console.print("[bold red]❌ 未发现任何 Trace，请检查日志文件格式。[/bold red]")
        console.print("[dim]支持的格式: JSONL / CSV / LangSmith 导出 / LangFuse 导出[/dim]")
        return

    console.print(f"  [green]✓[/green] 加载 [bold]{len(traces)}[/bold] 条对话记录")

    # ── Step 2: 自动标注 ──────────────────────────────────
    console.print(Rule("[bold]🏷️  自动标注[/bold]", style="dim"))

    from src.labeling.auto_labeler import AutoLabeler
    labeler = AutoLabeler(use_llm=False)
    traces = labeler.label(traces)

    failures = [t for t in traces if t.status.value == "failure"]
    successes = [t for t in traces if t.status.value == "success"]

    failure_rate = len(failures) / max(len(traces), 1) * 100
    fg_color = "red" if failure_rate > 10 else "yellow" if failure_rate > 3 else "green"

    console.print(
        f"  成功 [green]{len(successes)}[/green]  ·  "
        f"失败 [{fg_color}]{len(failures)}[/{fg_color}]  ·  "
        f"失败率 [{fg_color}]{failure_rate:.1f}%[/{fg_color}]"
    )

    if not failures:
        console.print()
        console.print(Panel.fit(
            "[bold green]🎉 没有发现失败案例！你的 Agent 看起来很健康。[/bold green]",
            border_style="green",
        ))
        return

    # ── Step 3: 聚类分析 ──────────────────────────────────
    console.print(Rule("[bold]🧬 聚类分析[/bold]", style="dim"))

    from src.clustering.embedder import TraceEmbedder
    from src.clustering.clusterer import FailureClusterer

    embedder = TraceEmbedder()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]生成语义向量...", total=None)
        embeddings = embedder.embed(failures)

    console.print(f"  [green]✓[/green] 生成 [bold]{embeddings.shape[1]}维[/bold] 语义向量 ({len(failures)} 条)")

    clusterer = FailureClusterer(min_cluster_size=min_cluster_size)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]HDBSCAN 聚类中...", total=None)
        result = clusterer.cluster(failures, embeddings)

    cluster_count = len(result.clusters)
    noise_pct = result.noise_count / max(len(failures), 1) * 100
    console.print(
        f"  [green]✓[/green] 发现 [bold yellow]{cluster_count}[/bold yellow] 个失败簇 "
        f"([dim]{result.noise_count} 个噪声点, {noise_pct:.1f}%[/dim])"
    )

    if not result.clusters:
        console.print()
        console.print("[yellow]⚠️  未发现清晰的失败聚类。试试降低 --min-cluster-size 或用更多日志。[/yellow]")
        return

    # ── Step 4 & 5: 摘要 + 根因分析 ────────────────────────
    from src.clustering.summarizer import ClusterSummarizer
    llm_instance = _get_llm(llm, model) if llm else None
    summarizer = ClusterSummarizer(llm=llm_instance)

    if llm and llm_instance:
        console.print(Rule("[bold]🔬 LLM 根因分析[/bold]", style="dim"))

        from src.analysis.root_cause import RootCauseAnalyzer
        from src.analysis.canary import CanaryGenerator

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]LLM 分析中...", total=len(result.clusters))

            analyzer = RootCauseAnalyzer(llm=llm_instance)
            for cluster in result.clusters:
                analyzer.analyze(cluster)
                progress.advance(task)

            canary_gen = CanaryGenerator(llm=llm_instance)
            for cluster in result.clusters:
                canary_gen.generate(cluster)
    else:
        for cluster in result.clusters:
            summary = summarizer.summarize(cluster.sample_traces, cluster.size)
            cluster.label = summary.get("label", f"Cluster {cluster.cluster_id}")
            cluster.summary = summary.get("summary", "")

    # ── Step 6: 终端结果表格 ──────────────────────────────
    console.print(Rule("[bold]📊 失败模式[/bold]", style="dim"))

    table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold bright_cyan",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("严重度", width=6)
    table.add_column("簇标签", style="bold", min_width=25)
    table.add_column("数量", justify="right", width=8)
    table.add_column("占比", justify="right", width=8)
    table.add_column("金丝雀", justify="right", width=8)

    for cluster in result.clusters:
        if cluster.percentage > 20:
            severity = "[bold red]🔴 高[/bold red]"
        elif cluster.percentage > 10:
            severity = "[bold yellow]🟡 中[/bold yellow]"
        else:
            severity = "[green]🟢 低[/green]"

        canary_count = len(cluster.canary_queries)
        canary_str = f"[cyan]🦜 {canary_count}[/cyan]" if canary_count > 0 else "—"

        table.add_row(
            str(cluster.cluster_id),
            severity,
            cluster.label or f"Cluster {cluster.cluster_id}",
            str(cluster.size),
            f"{cluster.percentage}%",
            canary_str,
        )

    console.print(table)

    # 簇详情
    for cluster in result.clusters:
        if cluster.summary:
            console.print(f"  [dim]Cluster {cluster.cluster_id}:[/dim] {cluster.summary}")

    # ── Step 7: 生成 HTML 报告 ────────────────────────────
    console.print(Rule("[bold]📄 报告生成[/bold]", style="dim"))

    from src.report.html_generator import HTMLReportGenerator
    report_gen = HTMLReportGenerator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]生成 HTML 报告...", total=None)
        report_path = report_gen.generate(
            clustering_result=result,
            all_traces=traces,
            failures=failures,
            output_path=output,
        )

    console.print(f"  [green]✓[/green] 报告已生成: [bold bright_cyan]{report_path}[/bold bright_cyan]")

    # ── 总结 ──────────────────────────────────────────────
    console.print()
    if result.clusters:
        highest = result.clusters[0]
        console.print(Panel.fit(
            f"[bold yellow]💡 优先修复 Cluster {highest.cluster_id}[/bold yellow]\n"
            f"[dim]{highest.label} 占了 {highest.percentage}% 的失败，影响最大。[/dim]",
            border_style="yellow",
            padding=(1, 2),
        ))

    console.print(f"[bold green]✅ 分析完成！[/bold green]")
    console.print(f"[dim]   HTML 报告: {report_path}[/dim]")
    console.print(f"[dim]   失败模式: {len(result.clusters)} 类[/dim]")
    console.print(f"[dim]   金丝雀查询: {sum(len(c.canary_queries) for c in result.clusters)} 条[/dim]")
    console.print()

    if open_browser:
        click.launch(report_path)


@main.command()
@click.option("--port", default=8765, help="Web UI 端口")
@click.option("--host", default="127.0.0.1", help="绑定地址")
def ui(port: int, host: str):
    """启动 Web UI（拖拽上传日志，在线分析）"""
    # 依赖检查
    if not _ensure_installed("fastapi", "fastapi"):
        return
    if not _ensure_installed("uvicorn", "uvicorn"):
        return
    if not _ensure_installed("sentence_transformers", "sentence-transformers"):
        return

    console.print()
    console.print(Panel.fit(
        f"[bold bright_cyan]🌐 AgentMine Web UI[/bold bright_cyan]\n\n"
        f"[bold]打开浏览器访问:[/bold] [underline cyan]http://{host}:{port}[/underline cyan]\n"
        f"[dim]拖拽日志文件即可分析 · 按 Ctrl+C 停止[/dim]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))
    console.print()

    import uvicorn
    uvicorn.run("src.web.app:app", host=host, port=port, reload=False, log_level="warning")


@main.command()
@click.option("-o", "--output", default="agentmine_demo", help="输出目录")
@click.option("--traces", default=200, help="生成的 trace 数量")
@click.option("--llm/--no-llm", default=False, help="使用 LLM 深度分析")
def demo(output: str, traces: int, llm: bool):
    """生成演示数据并运行分析（快速体验）"""
    from examples.demo_data import generate_demo_data

    console.print()
    console.print("[bold cyan]🎭 正在生成演示数据...[/bold cyan]")
    demo_file = generate_demo_data(n_traces=traces, output_dir=output)
    console.print(f"  [green]✓[/green] 生成了 [bold]{traces}[/bold] 条模拟对话")
    console.print()

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
        console.print(f"[yellow]⚠️  LLM 不可用: {e}[/yellow]")
        console.print("[dim]   运行在纯规则模式（无 LLM 深度分析）[/dim]")
        return None


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    main()
