"""
AgentMine Web UI

FastAPI 应用，提供：
- 拖拽上传日志文件
- 在线分析
- 报告预览
"""
import os
import tempfile
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.ingestion.parser import LogParser
from src.labeling.auto_labeler import AutoLabeler
from src.clustering.embedder import TraceEmbedder
from src.clustering.clusterer import FailureClusterer
from src.report.html_generator import HTMLReportGenerator

logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="AgentMine Web UI",
    description="AI Agent Failure Pattern Miner — Upload logs, discover failure patterns",
    version="0.1.0",
)

# 静态文件
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── 首页 ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 — 拖拽上传界面"""
    return UPLOAD_PAGE


# ── 分析 API ──────────────────────────────────────────────

@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile = File(...),
    min_cluster_size: int = Form(5),
):
    """
    上传日志文件并进行分析

    Returns:
        分析结果 JSON
    """
    # 保存上传文件
    content = await file.read()
    suffix = Path(file.filename).suffix or ".jsonl"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── 分析 Pipeline ──────────────────────────────────
        parser = LogParser()
        traces = parser.parse_file(tmp_path)

        labeler = AutoLabeler()
        traces = labeler.label(traces)

        failures = [t for t in traces if t.status.value == "failure"]
        if len(failures) < min_cluster_size:
            return JSONResponse({
                "error": f"Not enough failures ({len(failures)}) for clustering. Need at least {min_cluster_size}.",
                "total_traces": len(traces),
                "failure_count": len(failures),
            })

        embedder = TraceEmbedder()
        embeddings = embedder.embed(failures)

        clusterer = FailureClusterer(min_cluster_size=min_cluster_size)
        result = clusterer.cluster(failures, embeddings)

        # 生成报告
        report_gen = HTMLReportGenerator()
        report_path = report_gen.generate(
            clustering_result=result,
            all_traces=traces,
            failures=failures,
            output_path=tempfile.mktemp(suffix=".html"),
        )

        # 读取报告内容
        report_html = Path(report_path).read_text(encoding="utf-8")

        # 构建 JSON 响应
        clusters_json = []
        for cluster in result.clusters:
            clusters_json.append({
                "id": cluster.cluster_id,
                "label": cluster.label,
                "size": cluster.size,
                "percentage": cluster.percentage,
                "keywords": cluster.keywords,
                "sample_queries": [t.user_query[:100] for t in cluster.sample_traces[:3]],
                "canary_queries": cluster.canary_queries[:5],
            })

        return JSONResponse({
            "total_traces": len(traces),
            "failure_count": len(failures),
            "failure_rate": round(len(failures) / max(len(traces), 1) * 100, 1),
            "cluster_count": len(result.clusters),
            "noise_count": result.noise_count,
            "clusters": clusters_json,
            "report_html": report_html,
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── 上传页面 HTML ─────────────────────────────────────────

UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentMine — AI Agent Failure Pattern Miner</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; min-height: 100vh;
}
.container { max-width: 900px; margin: 0 auto; padding: 3rem 1.5rem; }

/* Header */
.header { text-align: center; margin-bottom: 3rem; }
.header h1 { font-size: 2.5rem; color: #58a6ff; margin-bottom: 0.5rem; }
.header p { color: #8b949e; font-size: 1.1rem; }
.header .demo-link { margin-top: 1rem; }
.header .demo-link a {
    color: #d2991d; text-decoration: none; font-size: 0.9rem;
    border: 1px solid #d2991d; padding: 0.4rem 1rem; border-radius: 20px;
    transition: all 0.2s;
}
.header .demo-link a:hover { background: rgba(210,153,29,0.1); }

/* Upload Zone */
.upload-zone {
    background: #161b22; border: 2px dashed #30363d; border-radius: 12px;
    padding: 3rem 2rem; text-align: center; margin-bottom: 2rem;
    transition: border-color 0.3s, background 0.3s; cursor: pointer;
    position: relative;
}
.upload-zone:hover, .upload-zone.drag-over {
    border-color: #58a6ff; background: rgba(88,166,255,0.05);
}
.upload-zone .icon { font-size: 3rem; margin-bottom: 1rem; }
.upload-zone h3 { color: #e6edf3; margin-bottom: 0.5rem; }
.upload-zone p { color: #8b949e; font-size: 0.9rem; }
.upload-zone input[type="file"] {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    opacity: 0; cursor: pointer;
}
.upload-zone .formats {
    margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap;
}
.upload-zone .format-tag {
    background: #21262d; border-radius: 4px; padding: 0.2rem 0.6rem;
    font-size: 0.75rem; color: #8b949e; font-family: monospace;
}

/* Settings */
.settings { display: flex; gap: 1rem; align-items: center; justify-content: center; margin-bottom: 2rem; }
.settings label { color: #8b949e; font-size: 0.9rem; }
.settings input {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 0.5rem 1rem; color: #c9d1d9; width: 80px; text-align: center;
}

/* Progress */
.progress-container { display: none; margin-bottom: 2rem; }
.progress-bar {
    width: 100%; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden;
}
.progress-fill {
    height: 100%; background: linear-gradient(90deg, #58a6ff, #3fb950);
    border-radius: 3px; transition: width 0.5s ease;
    width: 0%;
}
.progress-text { text-align: center; color: #8b949e; font-size: 0.85rem; margin-top: 0.5rem; }

/* Results */
.results { display: none; }
.result-header {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;
}
.result-header h2 { color: #58a6ff; }
.download-btn {
    background: #238636; color: #fff; border: none; padding: 0.6rem 1.2rem;
    border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600;
}
.download-btn:hover { background: #2ea043; }

/* Stats */
.stats {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem;
}
.stat {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 1rem; text-align: center;
}
.stat .num { font-size: 1.8rem; font-weight: 700; }
.stat .lbl { color: #8b949e; font-size: 0.8rem; margin-top: 0.25rem; }
.stat.succ .num { color: #3fb950; }
.stat.fail .num { color: #f85149; }
.stat.clust .num { color: #d2991d; }

/* Cluster list */
.clusters { display: grid; gap: 1rem; }
.cluster-item {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 1.25rem; display: flex; gap: 1rem; align-items: flex-start;
    transition: border-color 0.2s;
}
.cluster-item:hover { border-color: #58a6ff; }
.cluster-rank {
    font-size: 2rem; font-weight: 800; color: #58a6ff;
    min-width: 50px; text-align: center;
}
.cluster-info { flex: 1; }
.cluster-info h4 { color: #e6edf3; margin-bottom: 0.25rem; }
.cluster-info .meta { color: #8b949e; font-size: 0.85rem; margin-bottom: 0.5rem; }
.cluster-samples { font-size: 0.85rem; color: #8b949e; font-style: italic; }
.cluster-samples span { display: block; padding: 0.15rem 0; }

/* Footer */
.footer { text-align: center; padding: 2rem; color: #484f58; font-size: 0.8rem; margin-top: 3rem; }

@media (max-width: 600px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .header h1 { font-size: 1.8rem; }
}
</style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🔍 AgentMine</h1>
        <p>AI Agent Failure Pattern Miner</p>
        <p style="color: #8b949e; font-size: 0.9rem; margin-top: 0.5rem;">
            上传 Agent 日志 → 自动发现系统性失败模式
        </p>
        <div class="demo-link">
            <a href="#" onclick="loadDemo(event)">🎭 加载演示数据</a>
        </div>
    </div>

    <!-- Upload Zone -->
    <div class="upload-zone" id="uploadZone">
        <div class="icon">📂</div>
        <h3>拖拽日志文件到此处，或点击上传</h3>
        <p>支持 JSONL / CSV / LangSmith / LangFuse 格式</p>
        <div class="formats">
            <span class="format-tag">.jsonl</span>
            <span class="format-tag">.csv</span>
            <span class="format-tag">langsmith</span>
            <span class="format-tag">langfuse</span>
        </div>
        <input type="file" id="fileInput" accept=".jsonl,.csv,.json" />
    </div>

    <!-- Settings -->
    <div class="settings">
        <label>最小簇大小:</label>
        <input type="number" id="minClusterSize" value="5" min="2" max="50" />
    </div>

    <!-- Progress -->
    <div class="progress-container" id="progress">
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        <div class="progress-text" id="progressText">Analyzing...</div>
    </div>

    <!-- Results -->
    <div class="results" id="results">
        <div class="result-header">
            <h2>📊 分析结果</h2>
            <button class="download-btn" onclick="downloadReport()">📥 下载 HTML 报告</button>
        </div>
        <div class="stats" id="statsContainer"></div>
        <div class="clusters" id="clustersContainer"></div>
    </div>

    <div class="footer">
        AgentMine v0.1.0 · Open Source on <a href="https://github.com/yourusername/agentmine" style="color: #58a6ff;">GitHub</a>
    </div>
</div>

<script>
let reportHtml = '';
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const progress = document.getElementById('progress');
const results = document.getElementById('results');

// Drag & Drop
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) analyzeFile(files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) analyzeFile(fileInput.files[0]);
});

// Load demo data
async function loadDemo(e) {
    e.preventDefault();
    progress.style.display = 'block';
    document.getElementById('progressText').textContent = 'Generating demo data...';

    // Fetch demo data from API
    try {
        const resp = await fetch('/api/demo?traces=200&min_cluster_size=' + document.getElementById('minClusterSize').value);
        const data = await resp.json();
        if (data.error) { alert(data.error); return; }
        showResults(data);
    } catch (err) {
        alert('Failed to load demo: ' + err.message);
    }
    progress.style.display = 'none';
}

// Analyze file
async function analyzeFile(file) {
    progress.style.display = 'block';
    results.style.display = 'none';
    document.getElementById('progressText').textContent = 'Uploading & analyzing...';
    updateProgress(20);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('min_cluster_size', document.getElementById('minClusterSize').value);

    try {
        updateProgress(40);
        const resp = await fetch('/api/analyze', { method: 'POST', body: formData });
        updateProgress(80);
        const data = await resp.json();

        if (data.error) {
            alert('Analysis error: ' + data.error);
            progress.style.display = 'none';
            return;
        }

        updateProgress(100);
        reportHtml = data.report_html;
        showResults(data);
    } catch (err) {
        alert('Request failed: ' + err.message);
        progress.style.display = 'none';
    }
}

function updateProgress(pct) {
    document.getElementById('progressFill').style.width = pct + '%';
}

function showResults(data) {
    document.getElementById('progressText').textContent = 'Done!';
    setTimeout(() => { progress.style.display = 'none'; }, 1000);

    // Stats
    document.getElementById('statsContainer').innerHTML = `
        <div class="stat succ"><div class="num">${data.total_traces}</div><div class="lbl">Total Traces</div></div>
        <div class="stat fail"><div class="num">${data.failure_count}</div><div class="lbl">Failures (${data.failure_rate}%)</div></div>
        <div class="stat clust"><div class="num">${data.cluster_count}</div><div class="lbl">Clusters</div></div>
        <div class="stat"><div class="num" style="color:#8b949e;">${data.noise_count}</div><div class="lbl">Noise Points</div></div>
    `;

    // Clusters
    let html = '';
    data.clusters.forEach((c, i) => {
        const samples = (c.sample_queries || []).map(q => `<span>"${q}"</span>`).join('');
        const canaries = (c.canary_queries || []).map(cq => `<span style="background:rgba(88,166,255,0.1);padding:0.15rem 0.5rem;border-radius:12px;font-size:0.75rem;color:#58a6ff;">${cq}</span>`).join(' ');
        const impactColor = c.percentage > 20 ? '#f85149' : c.percentage > 10 ? '#d2991d' : '#3fb950';
        html += `
            <div class="cluster-item" style="border-left: 4px solid ${impactColor};">
                <div class="cluster-rank">#${i+1}</div>
                <div class="cluster-info">
                    <h4>${c.label || 'Cluster ' + c.id}</h4>
                    <div class="meta">${c.size} cases (${c.percentage}% of failures) · Keywords: ${(c.keywords||[]).join(', ') || 'N/A'}</div>
                    <div class="cluster-samples">${samples}</div>
                    ${canaries ? '<div style="margin-top:0.5rem;">🦜 ' + canaries + '</div>' : ''}
                </div>
            </div>`;
    });
    document.getElementById('clustersContainer').innerHTML = html;
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth' });
}

function downloadReport() {
    if (!reportHtml) { alert('No report available'); return; }
    const blob = new Blob([reportHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'agentmine_report.html'; a.click();
    URL.revokeObjectURL(url);
}
</script>
</body>
</html>"""


# ── Demo API ──────────────────────────────────────────────

@app.get("/api/demo")
async def api_demo(traces: int = 200, min_cluster_size: int = 5):
    """生成演示数据并分析"""
    from examples.demo_data import generate_demo_data
    import tempfile
    from src.ingestion.parser import LogParser
    from src.labeling.auto_labeler import AutoLabeler
    from src.clustering.embedder import TraceEmbedder
    from src.clustering.clusterer import FailureClusterer
    from src.report.html_generator import HTMLReportGenerator

    # 生成演示数据
    demo_file = generate_demo_data(n_traces=traces, output_dir=tempfile.gettempdir())

    # 分析
    parser = LogParser()
    all_traces = parser.parse_file(demo_file)

    labeler = AutoLabeler()
    all_traces = labeler.label(all_traces)

    failures = [t for t in all_traces if t.status.value == "failure"]

    embedder = TraceEmbedder()
    embeddings = embedder.embed(failures)

    clusterer = FailureClusterer(min_cluster_size=min_cluster_size)
    result = clusterer.cluster(failures, embeddings)

    # 规则标注簇标签
    for cluster in result.clusters:
        keywords = ", ".join(cluster.keywords[:3]) if cluster.keywords else "unknown"
        cluster.label = f"{keywords}相关失败"

    # 生成报告
    report_gen = HTMLReportGenerator()
    report_path = report_gen.generate(
        clustering_result=result,
        all_traces=all_traces,
        failures=failures,
        output_path=tempfile.mktemp(suffix=".html"),
    )
    report_html = Path(report_path).read_text(encoding="utf-8")

    clusters_json = []
    for cluster in result.clusters:
        clusters_json.append({
            "id": cluster.cluster_id,
            "label": cluster.label,
            "size": cluster.size,
            "percentage": cluster.percentage,
            "keywords": cluster.keywords,
            "sample_queries": [t.user_query[:100] for t in cluster.sample_traces[:3]],
            "canary_queries": cluster.canary_queries[:5],
        })

    return {
        "total_traces": len(all_traces),
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / max(len(all_traces), 1) * 100, 1),
        "cluster_count": len(result.clusters),
        "noise_count": result.noise_count,
        "clusters": clusters_json,
        "report_html": report_html,
    }

    finally:
        try:
            os.unlink(demo_file)
        except Exception:
            pass
