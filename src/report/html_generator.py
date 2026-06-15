"""
HTML 报告生成器
生成包含交互式聚类可视化、根因卡片、趋势图的单文件 HTML 报告
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List

from src.ingestion.schema import AgentTrace
from src.clustering.clusterer import ClusteringResult

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """生成交互式 HTML 报告"""

    def generate(
        self,
        clustering_result: ClusteringResult,
        all_traces: List[AgentTrace],
        failures: List[AgentTrace],
        output_path: str = "agentmine_report",
        trend_data: dict = None,
    ) -> str:
        """生成 HTML 报告。trend_data 可选，来自 TrendAnalyzer.analyze()。"""
        if not output_path.endswith(".html"):
            output_path = output_path.rstrip("/") + ".html"

        total = len(all_traces)
        fail_count = len(failures)
        failure_rate = round(fail_count / max(total, 1) * 100, 1)

        # ── 构建集群卡片 HTML ─────────────────────────────
        clusters_html = self._build_cluster_cards(clustering_result)
        clusters_json = json.dumps([
            {
                "id": c.cluster_id,
                "label": c.label or f"Cluster {c.cluster_id}",
                "size": c.size,
                "percentage": c.percentage,
                "keywords": c.keywords[:5] if c.keywords else [],
                "sample_queries": [t.user_query[:80] for t in c.sample_traces[:5]],
                "canary_queries": c.canary_queries[:5],
            }
            for c in clustering_result.clusters
        ], ensure_ascii=False)

        # ── 构建统计卡片 HTML ─────────────────────────────
        stats_html = f"""
        <div class="stat-card">
            <div class="value">{total}</div>
            <div class="label">Total Traces</div>
        </div>
        <div class="stat-card failure">
            <div class="value">{fail_count}</div>
            <div class="label">Failures ({failure_rate}%)</div>
        </div>
        <div class="stat-card clusters">
            <div class="value">{len(clustering_result.clusters)}</div>
            <div class="label">Failure Clusters</div>
        </div>
        <div class="stat-card">
            <div class="value" style="color:#8b949e;">{clustering_result.noise_count}</div>
            <div class="label">Noise Points</div>
        </div>"""

        # ── 构建散点图数据 ────────────────────────────────
        scatter_json = self._build_scatter_data(clustering_result, failures)

        # ── 构建趋势图数据 ────────────────────────────────
        trend_json = "null"
        trend_section_html = ""
        if trend_data and trend_data.get("data_points"):
            trend_json = json.dumps(trend_data["data_points"], ensure_ascii=False)
            anomalies_json = json.dumps(trend_data.get("anomalies", []), ensure_ascii=False)
            direction = trend_data.get("trend_direction", "stable")
            trend_section_html = f"""
            <div class="viz-section">
                <h2>📈 失败率趋势 ({trend_data.get('granularity', 'day')})</h2>
                <div id="trend-plot" style="width:100%;height:350px;"></div>
            </div>
            <script>
            (function() {{
                var trendData = {trend_json};
                var anomalies = {anomalies_json};
                var anomalyDates = new Set(anomalies.map(function(a) {{ return a.date; }}));
                var anomalyRates = {{}};
                anomalies.forEach(function(a) {{ anomalyRates[a.date] = a.rate; }});

                var dates = trendData.map(function(d) {{ return d.date; }});
                var rates = trendData.map(function(d) {{ return d.rate; }});
                var totals = trendData.map(function(d) {{ return d.total; }});

                var normalX = [], normalY = [], normalText = [];
                var anomalyX = [], anomalyY = [], anomalyText = [];
                dates.forEach(function(d, i) {{
                    if (anomalyDates.has(d)) {{
                        anomalyX.push(d); anomalyY.push(rates[i]);
                        anomalyText.push(d + ': ' + rates[i] + '% (' + totals[i] + ' traces) ⚠️');
                    }} else {{
                        normalX.push(d); normalY.push(rates[i]);
                        normalText.push(d + ': ' + rates[i] + '% (' + totals[i] + ' traces)');
                    }}
                }});

                var traces = [];
                if (normalX.length) traces.push({{
                    x: normalX, y: normalY, text: normalText,
                    mode: 'lines+markers', type: 'scatter', name: 'Failure Rate',
                    line: {{ color: '#58a6ff', width: 2 }},
                    marker: {{ size: 5, color: '#58a6ff' }},
                    hovertemplate: '%{{text}}<extra></extra>'
                }});
                if (anomalyX.length) traces.push({{
                    x: anomalyX, y: anomalyY, text: anomalyText,
                    mode: 'markers', type: 'scatter', name: '⚠️ Anomaly',
                    marker: {{ size: 12, color: '#f85149', symbol: 'x' }},
                    hovertemplate: '%{{text}}<extra></extra>'
                }});

                Plotly.newPlot('trend-plot', traces, {{
                    plot_bgcolor: '#0d1117', paper_bgcolor: '#0d1117',
                    font: {{ color: '#c9d1d9' }},
                    xaxis: {{ gridcolor: '#21262d', title: '' }},
                    yaxis: {{ gridcolor: '#21262d', title: 'Failure Rate (%)' }},
                    legend: {{ x: 0.01, y: 0.99 }},
                    hovermode: 'closest',
                    margin: {{ l: 50, r: 30, t: 20, b: 60 }}
                }});
            }})();
            </script>"""

        # ── 组装完整 HTML ─────────────────────────────────
        html = self._build_full_html(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_traces=total,
            total_failures=fail_count,
            failure_rate=failure_rate,
            cluster_count=len(clustering_result.clusters),
            noise_count=clustering_result.noise_count,
            stats_html=stats_html,
            clusters_json=clusters_json,
            clusters_html=clusters_html,
            scatter_json=scatter_json,
            trend_section_html=trend_section_html,
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding="utf-8")

        logger.info(f"HTML report generated: {output_file} ({len(html)} bytes)")
        return str(output_file.absolute())

    def generate_diff(self, diff_result, output_path: str = "agentmine_diff") -> str:
        """生成版本对比 HTML 报告"""
        if not output_path.endswith(".html"):
            output_path = output_path.rstrip("/") + ".html"

        # 构建 diff 卡片
        cards = []
        for e in diff_result.entries:
            if e.change_type == "new":
                border = "4px solid #f85149"
                badge = '<span class="card-badge badge-high">NEW ⚠️</span>'
                v1_str = "—"
                v2_str = f'<span style="color:#f85149;font-weight:bold;">{e.v2_size}</span>'
                trend = f'<span style="color:#f85149;">+100%</span>'
            elif e.change_type == "fixed":
                border = "4px solid #3fb950"
                badge = '<span class="card-badge badge-low">✅ FIXED</span>'
                v1_str = f'<span style="color:#f85149;">{e.v1_size}</span>'
                v2_str = "—"
                trend = '<span style="color:#3fb950;">已修复</span>'
            elif e.change_type == "worsened":
                border = "4px solid #d2991d"
                badge = f'<span class="card-badge badge-medium">📈 +{e.change_pct}%</span>'
                v1_str = str(e.v1_size)
                v2_str = f'<span style="color:#d2991d;font-weight:bold;">{e.v2_size}</span>'
                trend = f'<span style="color:#d2991d;">+{e.change_pct}%</span>'
            elif e.change_type == "improved":
                border = "4px solid #58a6ff"
                badge = f'<span class="card-badge" style="background:rgba(88,166,255,0.15);color:#58a6ff;">📉 {e.change_pct}%</span>'
                v1_str = str(e.v1_size)
                v2_str = str(e.v2_size)
                trend = f'<span style="color:#58a6ff;">{e.change_pct}%</span>'
            else:
                border = "4px solid #30363d"
                badge = '<span class="card-badge" style="background:rgba(139,148,158,0.15);color:#8b949e;">➖ STABLE</span>'
                v1_str = str(e.v1_size)
                v2_str = str(e.v2_size)
                trend = f'<span style="color:#8b949e;">{e.change_pct}%</span>'

            kw_html = " ".join(
                f'<span class="canary-chip" style="font-size:0.75rem;">{kw}</span>'
                for kw in (e.keywords or [])[:5]
            )

            cards.append(f"""
            <div class="cluster-card" style="border-left:{border};">
                <div class="card-header">
                    <h3>{e.label}</h3>
                    {badge}
                </div>
                <div class="card-body">
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;text-align:center;margin-bottom:1rem;">
                        <div>
                            <div style="color:#8b949e;font-size:0.8rem;">V1 (基线)</div>
                            <div style="font-size:1.5rem;font-weight:700;">{v1_str}</div>
                        </div>
                        <div>
                            <div style="color:#8b949e;font-size:0.8rem;">V2 (新版本)</div>
                            <div style="font-size:1.5rem;font-weight:700;">{v2_str}</div>
                        </div>
                        <div>
                            <div style="color:#8b949e;font-size:0.8rem;">变化</div>
                            <div style="font-size:1.2rem;font-weight:700;">{trend}</div>
                        </div>
                    </div>
                    {f'<div style="margin-top:0.5rem;">{kw_html}</div>' if kw_html else ''}
                </div>
            </div>""")

        # 统计概览
        rate_change = diff_result.v2_failure_rate - diff_result.v1_failure_rate
        if rate_change <= 0:
            rate_html = f'<div class="value" style="color:#3fb950;">📉 {abs(rate_change):.1f}%</div><div class="label">失败率下降</div>'
        else:
            rate_html = f'<div class="value" style="color:#f85149;">📈 +{rate_change:.1f}%</div><div class="label">失败率上升</div>'

        stats_html = f"""
        <div class="stat-card"><div class="value">{diff_result.v1_failure_rate}%</div><div class="label">V1 失败率</div></div>
        <div class="stat-card"><div class="value">{diff_result.v2_failure_rate}%</div><div class="label">V2 失败率</div></div>
        <div class="stat-card">{rate_html}</div>
        <div class="stat-card failure"><div class="value">{diff_result.new_failure_modes}</div><div class="label">新增失败 ⚠️</div></div>
        <div class="stat-card success"><div class="value">{diff_result.fixed_failure_modes}</div><div class="label">已修复 ✅</div></div>
        <div class="stat-card clusters"><div class="value">{diff_result.worsened_modes}</div><div class="label">恶化 📈</div></div>
        <div class="stat-card"><div class="value">{diff_result.improved_modes}</div><div class="label">改善 📉</div></div>
        """

        html = _DIFF_HTML_SHELL
        html = html.replace("__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        html = html.replace("__STATS_HTML__", stats_html)
        html = html.replace("__CARDS_HTML__", "\n".join(cards))
        html = html.replace("__V1_FAILURES__", str(diff_result.v1_failures))
        html = html.replace("__V1_TOTAL__", str(diff_result.v1_total))
        html = html.replace("__V1_RATE__", str(diff_result.v1_failure_rate))
        html = html.replace("__V1_CLUSTERS__", str(diff_result.v1_clusters))
        html = html.replace("__V2_FAILURES__", str(diff_result.v2_failures))
        html = html.replace("__V2_TOTAL__", str(diff_result.v2_total))
        html = html.replace("__V2_RATE__", str(diff_result.v2_failure_rate))
        html = html.replace("__V2_CLUSTERS__", str(diff_result.v2_clusters))
        html = html.replace("__NEW_MODES__", str(diff_result.new_failure_modes))
        html = html.replace("__FIXED_MODES__", str(diff_result.fixed_failure_modes))

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding="utf-8")

        logger.info(f"Diff HTML report generated: {output_file}")
        return str(output_file.absolute())

    def _build_cluster_cards(self, result: ClusteringResult) -> str:
        """Python 构建集群卡片 HTML"""
        cards = []
        for cluster in result.clusters:
            if cluster.percentage > 20:
                impact_class = "high-impact"
                priority = "high"
                badge_class = "badge-high"
            elif cluster.percentage > 10:
                impact_class = "medium-impact"
                priority = "medium"
                badge_class = "badge-medium"
            else:
                impact_class = "low-impact"
                priority = "low"
                badge_class = "badge-low"

            root_cause_summary = ""
            fix_suggestions = []
            try:
                if cluster.root_cause:
                    rc = json.loads(cluster.root_cause) if isinstance(cluster.root_cause, str) else cluster.root_cause
                    root_cause_summary = rc.get("root_cause_detail", rc.get("symptom", ""))
                    fix_suggestions = rc.get("fix_suggestions", [])
            except Exception:
                pass

            # 样本查询
            sample_html = ""
            for t in cluster.sample_traces[:5]:
                q = t.user_query[:80].replace("<", "&lt;").replace(">", "&gt;")
                sample_html += f'<div class="query-item">"{q}"</div>'

            # 关键词
            kw_html = " ".join(
                f'<span class="canary-chip">{kw}</span>'
                for kw in (cluster.keywords[:5] if cluster.keywords else [])
            )

            # 金丝雀
            canary_html = ""
            if cluster.canary_queries:
                chips = "".join(
                    f'<span class="canary-chip">{cq}</span>'
                    for cq in cluster.canary_queries[:5]
                )
                canary_html = f"""
                <div class="canary-section">
                    <div class="info-row"><span class="key">🦜 Canary Queries</span></div>
                    <div class="canary-chips">{chips}</div>
                </div>"""

            # 修复建议
            fixes_html = ""
            if fix_suggestions:
                li_items = "".join(f"<li>{f}</li>" for f in fix_suggestions)
                fixes_html = f"""
                <div class="info-row"><span class="key">Fix Suggestions</span></div>
                <ul class="fix-list">{li_items}</ul>"""

            cards.append(f"""
            <div class="cluster-card {impact_class}">
                <div class="card-header">
                    <h3>Cluster {cluster.cluster_id}: {cluster.label or f'Cluster {cluster.cluster_id}'}</h3>
                    <span class="card-badge {badge_class}">{priority.upper()}</span>
                </div>
                <div class="card-body">
                    <div class="info-row">
                        <span class="key">Size</span>
                        <span class="val">{cluster.size} cases ({cluster.percentage}% of failures)</span>
                    </div>
                    {f'''<div class="info-row"><span class="key">Root Cause</span><span class="val">{root_cause_summary}</span></div>''' if root_cause_summary else ""}
                    {f'''<div class="info-row"><span class="key">Keywords</span><span class="val">{kw_html}</span></div>''' if kw_html else ""}
                    {f'''<div class="info-row"><span class="key">Samples</span></div><div class="sample-queries">{sample_html}</div>''' if sample_html else ""}
                    {fixes_html}
                    {canary_html}
                </div>
            </div>""")

        return "\n".join(cards)

    def _build_scatter_data(self, result: ClusteringResult, failures: List[AgentTrace]) -> str:
        """构建散点图 JSON"""
        if result.umap_coords is None or result.cluster_labels is None:
            return "[]"

        colors = [
            "#58a6ff", "#f85149", "#3fb950", "#d2991d", "#bc8cff",
            "#ff7b72", "#79c0ff", "#56d364", "#e3b341", "#d2a8ff",
        ]
        points = []
        for i in range(len(result.umap_coords)):
            label = int(result.cluster_labels[i])
            if label == -1:
                color = "#484f58"
                name = "Noise"
            else:
                color = colors[label % len(colors)]
                name = f"Cluster {label}"
            query = failures[i].user_query[:60].replace("'", "\\'").replace('"', '\\"') if i < len(failures) else ""
            points.append({
                "x": round(float(result.umap_coords[i, 0]), 4),
                "y": round(float(result.umap_coords[i, 1]), 4),
                "text": query,
                "name": name,
                "color": color,
            })
        return json.dumps(points, ensure_ascii=False)

    def _build_full_html(self, **kwargs) -> str:
        """组装完整 HTML（避免 .format() 与 CSS 花括号冲突）"""
        # 使用字符串模板 + 替换
        template = _HTML_SHELL
        for key, value in kwargs.items():
            template = template.replace(f"__{key.upper()}__", str(value))
        return template


# ── HTML 外壳模板 ─────────────────────────────────────────
# 使用 __VARIABLE__ 占位符，避免与 CSS/JS 花括号冲突

_HTML_SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentMine — Failure Pattern Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.6;
}
.container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
.header { text-align: center; padding: 3rem 0; border-bottom: 1px solid #21262d; margin-bottom: 2rem; }
.header h1 { font-size: 2.5rem; color: #58a6ff; margin-bottom: 0.5rem; }
.header .subtitle { color: #8b949e; font-size: 1.1rem; }
.header .meta { color: #484f58; font-size: 0.85rem; margin-top: 1rem; }

.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
.stat-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.5rem; text-align: center; }
.stat-card .value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.stat-card .label { color: #8b949e; font-size: 0.85rem; margin-top: 0.25rem; }
.stat-card.failure .value { color: #f85149; }
.stat-card.success .value { color: #3fb950; }
.stat-card.clusters .value { color: #d2991d; }

.viz-section { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }
.viz-section h2 { color: #58a6ff; margin-bottom: 1rem; font-size: 1.3rem; }
#cluster-plot { width: 100%; height: 500px; }
#bar-plot { width: 100%; height: 350px; }

.cluster-cards { display: grid; gap: 1.5rem; margin-bottom: 2rem; }
.cluster-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; transition: border-color 0.2s; }
.cluster-card:hover { border-color: #58a6ff; }
.cluster-card.high-impact { border-left: 4px solid #f85149; }
.cluster-card.medium-impact { border-left: 4px solid #d2991d; }
.cluster-card.low-impact { border-left: 4px solid #3fb950; }

.card-header { padding: 1.25rem 1.5rem; background: #1c2128; border-bottom: 1px solid #21262d; display: flex; justify-content: space-between; align-items: center; }
.card-header h3 { font-size: 1.1rem; color: #e6edf3; }
.card-badge { padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-high { background: rgba(248,81,73,0.15); color: #f85149; }
.badge-medium { background: rgba(210,153,29,0.15); color: #d2991d; }
.badge-low { background: rgba(63,185,80,0.15); color: #3fb950; }

.card-body { padding: 1.5rem; }
.info-row { display: flex; margin-bottom: 0.75rem; }
.info-row .key { color: #8b949e; min-width: 100px; font-size: 0.85rem; }
.info-row .val { color: #c9d1d9; font-size: 0.9rem; }

.sample-queries { background: #0d1117; border-radius: 6px; padding: 1rem; margin: 0.75rem 0; }
.sample-queries .query-item { padding: 0.4rem 0; border-bottom: 1px solid #161b22; color: #8b949e; font-size: 0.85rem; font-style: italic; }
.sample-queries .query-item:last-child { border: none; }

.fix-list { list-style: none; }
.fix-list li { padding: 0.5rem 0; border-bottom: 1px solid #161b22; font-size: 0.9rem; }
.fix-list li::before { content: "→ "; color: #58a6ff; font-weight: bold; }

.canary-section { margin-top: 1rem; }
.canary-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.canary-chip { background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2); border-radius: 20px; padding: 0.3rem 0.8rem; font-size: 0.8rem; color: #58a6ff; }

.footer { text-align: center; padding: 2rem; color: #484f58; font-size: 0.8rem; border-top: 1px solid #21262d; margin-top: 2rem; }
@media (max-width: 768px) { .container { padding: 1rem; } .header h1 { font-size: 1.8rem; } .stats-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>🔍 AgentMine</h1>
        <div class="subtitle">AI Agent Failure Pattern Report</div>
        <div class="meta">
            Generated: __GENERATED_AT__<br>
            Total Traces: __TOTAL_TRACES__ | Failures: __TOTAL_FAILURES__ | Clusters: __CLUSTER_COUNT__
        </div>
    </div>

    <div class="stats-grid">__STATS_HTML__</div>

    <div class="viz-section">
        <h2>📊 Failure Clusters (UMAP 2D)</h2>
        <div id="cluster-plot"></div>
    </div>

    __TREND_SECTION_HTML__

    <div class="viz-section">
        <h2>📈 Failure Distribution</h2>
        <div id="bar-plot"></div>
    </div>

    <h2 style="color: #58a6ff; margin-bottom: 1rem; font-size: 1.3rem;">🔬 Root Cause Analysis</h2>
    <div class="cluster-cards">__CLUSTERS_HTML__</div>

    <div class="footer">
        Generated by <strong>AgentMine</strong> v0.1.0 ·
        <a href="https://github.com/CCCchen040620/agentmine" style="color: #58a6ff;">GitHub</a>
    </div>
</div>

<script>
var clusters = __CLUSTERS_JSON__;
var scatterPoints = __SCATTER_JSON__;

// UMAP scatter plot
var scatterTraces = [];
var seenNames = new Set();
scatterPoints.forEach(function(p) {
    var showLegend = !seenNames.has(p.name);
    seenNames.add(p.name);
    scatterTraces.push({
        x: [p.x], y: [p.y],
        text: p.text,
        name: p.name,
        mode: 'markers',
        type: 'scatter',
        marker: { size: 7, opacity: 0.75, color: p.color },
        hovertemplate: '%{text}<extra></extra>',
        showlegend: showLegend
    });
});

Plotly.newPlot('cluster-plot', scatterTraces, {
    plot_bgcolor: '#0d1117', paper_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    xaxis: { gridcolor: '#21262d', zeroline: false },
    yaxis: { gridcolor: '#21262d', zeroline: false },
    legend: { itemclick: 'toggleothers' },
    hovermode: 'closest',
    margin: { l: 40, r: 40, t: 20, b: 40 }
});

// Bar chart
var colors = ['#f85149', '#d2991d', '#3fb950', '#58a6ff', '#bc8cff', '#ff7b72', '#79c0ff'];
Plotly.newPlot('bar-plot', [{
    x: clusters.map(function(c) { return c.label.substring(0, 20); }),
    y: clusters.map(function(c) { return c.size; }),
    type: 'bar',
    marker: { color: clusters.map(function(c, i) { return colors[i % colors.length]; }) },
    text: clusters.map(function(c) { return c.size + ' cases (' + c.percentage + '%)'; }),
    textposition: 'auto',
    hovertemplate: '%{text}<extra></extra>'
}], {
    plot_bgcolor: '#0d1117', paper_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    xaxis: { gridcolor: '#21262d', tickangle: -25 },
    yaxis: { gridcolor: '#21262d', title: 'Count' },
    margin: { l: 50, r: 40, t: 20, b: 100 }
});
</script>
</body>
</html>"""


# ── Diff 报告模板 ─────────────────────────────────────────

_DIFF_HTML_SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentMine — Version Diff Report</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.6;
}
.container { max-width: 1000px; margin: 0 auto; padding: 2rem; }
.header { text-align: center; padding: 3rem 0; border-bottom: 1px solid #21262d; margin-bottom: 2rem; }
.header h1 { font-size: 2.5rem; color: #58a6ff; margin-bottom: 0.5rem; }
.header .subtitle { color: #8b949e; font-size: 1.1rem; }
.header .meta { color: #484f58; font-size: 0.85rem; margin-top: 1rem; }

.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin-bottom: 2rem; }
.stat-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.2rem; text-align: center; }
.stat-card .value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
.stat-card .label { color: #8b949e; font-size: 0.75rem; margin-top: 0.25rem; }
.stat-card.failure .value { color: #f85149; }
.stat-card.success .value { color: #3fb950; }
.stat-card.clusters .value { color: #d2991d; }

.version-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
.version-box {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.5rem; text-align: center;
}
.version-box.v1 { border-color: #484f58; }
.version-box.v2 { border-color: #58a6ff; }
.version-box h3 { color: #e6edf3; margin-bottom: 0.75rem; }
.version-box .rate { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.25rem; }
.version-box.v1 .rate { color: #8b949e; }
.version-box.v2 .rate { color: #58a6ff; }
.version-box .detail { color: #8b949e; font-size: 0.85rem; }

.cluster-cards { display: grid; gap: 1rem; margin-bottom: 2rem; }
.cluster-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }
.card-header { padding: 1rem 1.25rem; background: #1c2128; border-bottom: 1px solid #21262d; display: flex; justify-content: space-between; align-items: center; }
.card-header h3 { font-size: 1rem; color: #e6edf3; }
.card-badge { padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-high { background: rgba(248,81,73,0.15); color: #f85149; }
.badge-medium { background: rgba(210,153,29,0.15); color: #d2991d; }
.badge-low { background: rgba(63,185,80,0.15); color: #3fb950; }

.card-body { padding: 1.25rem; }
.canary-chip { display: inline-block; background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2); border-radius: 14px; padding: 0.2rem 0.6rem; font-size: 0.75rem; color: #58a6ff; margin: 0.15rem; }

.footer { text-align: center; padding: 2rem; color: #484f58; font-size: 0.8rem; border-top: 1px solid #21262d; margin-top: 2rem; }
@media (max-width: 768px) { .container { padding: 1rem; } .header h1 { font-size: 1.8rem; } .version-compare { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>🔄 AgentMine Diff</h1>
        <div class="subtitle">Agent 版本对比报告</div>
        <div class="meta">Generated: __GENERATED_AT__</div>
    </div>

    <div class="version-compare">
        <div class="version-box v1">
            <h3>📦 V1 (基线)</h3>
            <div class="rate">__V1_RATE__%</div>
            <div class="detail">__V1_FAILURES__ / __V1_TOTAL__ traces · __V1_CLUSTERS__ clusters</div>
        </div>
        <div class="version-box v2">
            <h3>🚀 V2 (新版本)</h3>
            <div class="rate">__V2_RATE__%</div>
            <div class="detail">__V2_FAILURES__ / __V2_TOTAL__ traces · __V2_CLUSTERS__ clusters</div>
        </div>
    </div>

    <div class="stats-grid">__STATS_HTML__</div>

    <h2 style="color: #58a6ff; margin-bottom: 1rem; font-size: 1.2rem;">📊 失败模式变化</h2>
    <div class="cluster-cards">__CARDS_HTML__</div>

    <div class="footer">
        Generated by <strong>AgentMine</strong> v0.1.0 ·
        <a href="https://github.com/CCCchen040620/agentmine" style="color: #58a6ff;">GitHub</a>
    </div>
</div>
</body>
</html>"""
