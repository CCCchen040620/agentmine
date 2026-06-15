"""
HTML 报告生成器
生成包含交互式聚类可视化、根因卡片、趋势图的单文件 HTML 报告
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.ingestion.schema import AgentTrace
from src.clustering.clusterer import ClusteringResult

logger = logging.getLogger(__name__)

# ── 嵌入式 HTML 模板（无需外部文件，报告自包含） ────────────

REPORT_TEMPLATE = """<!DOCTYPE html>
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
.container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
.header {
    text-align: center; padding: 3rem 0; border-bottom: 1px solid #21262d; margin-bottom: 2rem;
}
.header h1 { font-size: 2.5rem; color: #58a6ff; margin-bottom: 0.5rem; }
.header .subtitle { color: #8b949e; font-size: 1.1rem; }
.header .meta { color: #484f58; font-size: 0.85rem; margin-top: 1rem; }

/* 统计卡片 */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
.stat-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.5rem;
    text-align: center;
}
.stat-card .value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.stat-card .label { color: #8b949e; font-size: 0.85rem; margin-top: 0.25rem; }
.stat-card.success .value { color: #3fb950; }
.stat-card.failure .value { color: #f85149; }
.stat-card.clusters .value { color: #d2991d; }

/* 可视化区域 */
.viz-section {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 1.5rem; margin-bottom: 2rem;
}
.viz-section h2 { color: #58a6ff; margin-bottom: 1rem; font-size: 1.3rem; }
#cluster-plot { width: 100%; height: 500px; }
#bar-plot { width: 100%; height: 300px; }

/* 根因卡片 */
.cluster-cards { display: grid; gap: 1.5rem; margin-bottom: 2rem; }
.cluster-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px; overflow: hidden;
    transition: border-color 0.2s;
}
.cluster-card:hover { border-color: #58a6ff; }
.cluster-card.high-impact { border-left: 4px solid #f85149; }
.cluster-card.medium-impact { border-left: 4px solid #d2991d; }
.cluster-card.low-impact { border-left: 4px solid #3fb950; }

.card-header {
    padding: 1.25rem 1.5rem; background: #1c2128; border-bottom: 1px solid #21262d;
    display: flex; justify-content: space-between; align-items: center;
}
.card-header h3 { font-size: 1.1rem; color: #e6edf3; }
.card-badge {
    padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
}
.badge-high { background: rgba(248,81,73,0.15); color: #f85149; }
.badge-medium { background: rgba(210,153,29,0.15); color: #d2991d; }
.badge-low { background: rgba(63,185,80,0.15); color: #3fb950; }

.card-body { padding: 1.5rem; }
.info-row { display: flex; margin-bottom: 0.75rem; }
.info-row .key { color: #8b949e; min-width: 100px; font-size: 0.85rem; }
.info-row .val { color: #c9d1d9; font-size: 0.9rem; }

.sample-queries {
    background: #0d1117; border-radius: 6px; padding: 1rem; margin: 0.75rem 0;
}
.sample-queries .query-item {
    padding: 0.4rem 0; border-bottom: 1px solid #161b22; color: #8b949e; font-size: 0.85rem;
    font-style: italic;
}
.sample-queries .query-item:last-child { border: none; }

.fix-list { list-style: none; }
.fix-list li {
    padding: 0.5rem 0; border-bottom: 1px solid #161b22; font-size: 0.9rem;
    counter-increment: fix-counter;
}
.fix-list li::before {
    content: "→ "; color: #58a6ff; font-weight: bold;
}

/* 金丝雀查询 */
.canary-section { margin-top: 1rem; }
.canary-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.canary-chip {
    background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2);
    border-radius: 20px; padding: 0.3rem 0.8rem; font-size: 0.8rem; color: #58a6ff;
}

/* Footer */
.footer {
    text-align: center; padding: 2rem; color: #484f58; font-size: 0.8rem;
    border-top: 1px solid #21262d; margin-top: 2rem;
}

/* 响应式 */
@media (max-width: 768px) {
    .container { padding: 1rem; }
    .header h1 { font-size: 1.8rem; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <h1>🔍 AgentMine</h1>
        <div class="subtitle">AI Agent 失效模式分析报告</div>
        <div class="meta">
            Generated: {{ generated_at }} | Source: {{ source_file }}<br>
            Total Traces: {{ total_traces }} | Failures: {{ total_failures }} | Clusters: {{ cluster_count }}
        </div>
    </div>

    <!-- Stats -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{{ total_traces }}</div>
            <div class="label">Total Traces</div>
        </div>
        <div class="stat-card failure">
            <div class="value">{{ total_failures }}</div>
            <div class="label">Failures ({{ failure_rate }}%)</div>
        </div>
        <div class="stat-card clusters">
            <div class="value">{{ cluster_count }}</div>
            <div class="label">Failure Clusters</div>
        </div>
        <div class="stat-card success">
            <div class="value">{{ noise_count }}</div>
            <div class="label">Noise / Unique Failures</div>
        </div>
    </div>

    <!-- Cluster Visualization -->
    <div class="viz-section">
        <h2>📊 失败聚类可视化 (UMAP 2D)</h2>
        <div id="cluster-plot"></div>
    </div>

    <!-- Bar Chart -->
    <div class="viz-section">
        <h2>📈 失败分布 (按簇)</h2>
        <div id="bar-plot"></div>
    </div>

    <!-- Root Cause Cards -->
    <h2 style="color: #58a6ff; margin-bottom: 1rem; font-size: 1.3rem;">
        🔬 根因分析
    </h2>
    <div class="cluster-cards">
        {% for cluster in clusters %}
        <div class="cluster-card {{ cluster.impact_class }}">
            <div class="card-header">
                <h3>Cluster {{ cluster.id }}: {{ cluster.label }}</h3>
                <span class="card-badge badge-{{ cluster.priority }}">
                    {{ cluster.priority | upper }}
                </span>
            </div>
            <div class="card-body">
                <div class="info-row">
                    <span class="key">规模</span>
                    <span class="val">{{ cluster.size }} cases ({{ cluster.percentage }}% of failures)</span>
                </div>
                {% if cluster.root_cause_summary %}
                <div class="info-row">
                    <span class="key">根因</span>
                    <span class="val">{{ cluster.root_cause_summary }}</span>
                </div>
                {% endif %}
                {% if cluster.keywords %}
                <div class="info-row">
                    <span class="key">关键词</span>
                    <span class="val">
                        {% for kw in cluster.keywords %}
                        <span class="canary-chip">{{ kw }}</span>
                        {% endfor %}
                    </span>
                </div>
                {% endif %}
                {% if cluster.sample_queries %}
                <div class="info-row">
                    <span class="key">代表性案例</span>
                </div>
                <div class="sample-queries">
                    {% for q in cluster.sample_queries %}
                    <div class="query-item">"{{ q }}"</div>
                    {% endfor %}
                </div>
                {% endif %}
                {% if cluster.fix_suggestions %}
                <div class="info-row">
                    <span class="key">修复建议</span>
                </div>
                <ul class="fix-list">
                    {% for fix in cluster.fix_suggestions %}
                    <li>{{ fix }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
                {% if cluster.canary_queries %}
                <div class="canary-section">
                    <div class="info-row">
                        <span class="key">🦜 金丝雀查询</span>
                    </div>
                    <div class="canary-chips">
                        {% for cq in cluster.canary_queries %}
                        <span class="canary-chip">{{ cq }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </div>

    <!-- Footer -->
    <div class="footer">
        Generated by <strong>AgentMine</strong> v0.1.0 ·
        <a href="https://github.com/yourusername/agentmine" style="color: #58a6ff;">GitHub</a>
    </div>
</div>

<!-- Plotly Charts -->
<script>
// Cluster scatter plot
const scatterData = [{% for trace in scatter_data %}
{
    x: {{ trace.x }}, y: {{ trace.y }},
    text: '{{ trace.query | replace("'", "\\'") | truncate(60) }}',
    name: '{{ trace.label }}',
    mode: 'markers',
    type: 'scatter',
    marker: { size: 6, opacity: 0.7 },
    hovertemplate: '%{text}<extra></extra>'
},
{% endfor %}];

Plotly.newPlot('cluster-plot', scatterData, {
    title: 'Failure Clusters (UMAP Projection)',
    plot_bgcolor: '#0d1117', paper_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    xaxis: { gridcolor: '#21262d', zeroline: false, title: '' },
    yaxis: { gridcolor: '#21262d', zeroline: false, title: '' },
    legend: { itemclick: 'toggleothers' },
    hovermode: 'closest',
    margin: { l: 40, r: 40, t: 40, b: 40 }
});

// Bar chart
Plotly.newPlot('bar-plot', [{
    x: [{% for c in clusters %}'{{ c.label | truncate(20) }}'{% if not loop.last %},{% endif %}{% endfor %}],
    y: [{% for c in clusters %}{{ c.size }}{% if not loop.last %},{% endif %}{% endfor %}],
    type: 'bar',
    marker: {
        color: [{% for c in clusters %}'{{ c.bar_color }}'{% if not loop.last %},{% endif %}{% endfor %}],
    },
    text: [{% for c in clusters %}'{{ c.size }} cases ({{ c.percentage }}%)'{% if not loop.last %},{% endif %}{% endfor %}],
    textposition: 'auto',
    hovertemplate: '%{text}<extra></extra>',
}], {
    title: 'Failure Distribution by Cluster',
    plot_bgcolor: '#0d1117', paper_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    xaxis: { gridcolor: '#21262d', tickangle: -30 },
    yaxis: { gridcolor: '#21262d', title: 'Count' },
    margin: { l: 50, r: 40, t: 40, b: 100 }
});
</script>
</body>
</html>"""


class HTMLReportGenerator:
    """生成交互式 HTML 报告"""

    def generate(
        self,
        clustering_result: ClusteringResult,
        all_traces: List[AgentTrace],
        failures: List[AgentTrace],
        output_path: str = "agentmine_report",
    ) -> str:
        """
        生成 HTML 报告

        Args:
            clustering_result: 聚类结果
            all_traces: 所有 traces
            failures: 失败 traces
            output_path: 输出文件路径（不含扩展名）

        Returns:
            生成的 HTML 文件路径
        """
        # 确保输出路径以 .html 结尾
        if not output_path.endswith(".html"):
            output_path = output_path.rstrip("/") + ".html"

        total = len(all_traces)
        fail_count = len(failures)
        failure_rate = round(fail_count / max(total, 1) * 100, 1)

        # ── 构建集群卡片数据 ─────────────────────────────
        clusters_data = []
        for cluster in clustering_result.clusters:
            # 优先级判断
            if cluster.percentage > 20:
                priority = "high"
                impact_class = "high-impact"
                bar_color = "#f85149"
            elif cluster.percentage > 10:
                priority = "medium"
                impact_class = "medium-impact"
                bar_color = "#d2991d"
            else:
                priority = "low"
                impact_class = "low-impact"
                bar_color = "#3fb950"

            # 解析根因
            root_cause_summary = ""
            fix_suggestions = []
            try:
                if cluster.root_cause:
                    rc = json.loads(cluster.root_cause) if isinstance(cluster.root_cause, str) else cluster.root_cause
                    root_cause_summary = rc.get("root_cause_detail", rc.get("symptom", ""))
                    fix_suggestions = rc.get("fix_suggestions", [])
            except Exception:
                pass

            clusters_data.append({
                "id": cluster.cluster_id,
                "label": cluster.label or f"Cluster {cluster.cluster_id}",
                "size": cluster.size,
                "percentage": cluster.percentage,
                "priority": priority,
                "impact_class": impact_class,
                "bar_color": bar_color,
                "root_cause_summary": root_cause_summary,
                "keywords": cluster.keywords[:5] if cluster.keywords else [],
                "sample_queries": [t.user_query[:80] for t in cluster.sample_traces[:5]],
                "fix_suggestions": fix_suggestions,
                "canary_queries": cluster.canary_queries[:5],
            })

        # ── 构建散点图数据 ────────────────────────────────
        scatter_data = []
        if clustering_result.umap_coords is not None and clustering_result.cluster_labels is not None:
            from src.clustering.clusterer import ClusterInfo

            # 构建 label → cluster_id 的映射
            colors = [
                "#58a6ff", "#f85149", "#3fb950", "#d2991d", "#bc8cff",
                "#ff7b72", "#79c0ff", "#56d364", "#e3b341", "#d2a8ff",
            ]

            for i in range(len(clustering_result.umap_coords)):
                cluster_label_int = int(clustering_result.cluster_labels[i])

                if cluster_label_int == -1:
                    # 噪声点
                    label = "Noise"
                    color = "#484f58"
                else:
                    # 找到对应的 cluster info
                    matched = [c for c in clustering_result.clusters if c.cluster_id == cluster_label_int]
                    if matched:
                        # 我们已经重新编号了（从1开始），需要找到原始标签
                        # 简化处理：用颜色区分
                        color_idx = cluster_label_int % len(colors)
                        label = f"Cluster {cluster_label_int}"
                        color = colors[color_idx]
                    else:
                        label = f"Cluster {cluster_label_int}"
                        color = colors[cluster_label_int % len(colors)]

                scatter_data.append({
                    "x": round(float(clustering_result.umap_coords[i, 0]), 4),
                    "y": round(float(clustering_result.umap_coords[i, 1]), 4),
                    "query": failures[i].user_query[:80] if i < len(failures) else "",
                    "label": label,
                    "color": color,
                })

        # ── 渲染 HTML ─────────────────────────────────────
        html = REPORT_TEMPLATE.format(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_file="agent_logs",
            total_traces=total,
            total_failures=fail_count,
            cluster_count=len(clustering_result.clusters),
            noise_count=clustering_result.noise_count,
            failure_rate=failure_rate,
            clusters=clusters_data,
            scatter_data=scatter_data,
        )

        # 写入文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding="utf-8")

        logger.info(f"HTML report generated: {output_file} ({len(html)} bytes)")
        return str(output_file.absolute())
