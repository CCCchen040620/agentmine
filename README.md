# 🔍 AgentMine — AI Agent Failure Pattern Miner

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/cool--factor-over%209000-ff69b4.svg" alt="Cool">
</p>

<p align="center">
  <strong>你的 Agent 处理了 10000 次对话，它到底在哪些问题上系统性翻车？</strong><br>
  丢入日志 → AgentMine 自动告诉你答案。
</p>

---

## 💡 为什么需要 AgentMine？

你花了两周调 Prompt、优化 RAG、修 bug，Agent 终于上线了。

一周后，你只知道"有 8% 的对话失败了"。但：

- ❓ 这 800 条失败是同一类问题，还是几十种不同的问题？
- ❓ 是知识库的锅、Prompt 的锅、还是模型能力的锅？
- ❓ 修了其中一个 bug，能解决多少比例的失败？
- ❓ 有没有一种"金丝雀查询"能在 CI 里自动发现回归？

**LangSmith 让你看单棵树，AgentMine 让你看到整片森林。**

---

## 🎬 30 秒演示

<p align="center">
  <img src="docs/demo.gif" alt="AgentMine CLI Demo" width="800">
</p>

```bash
$ agentmine analyze agent_logs.jsonl

🔍 AgentMine v0.1.0 — AI Agent Failure Pattern Miner
─────────────────────────────────────────────────────
📂 Loaded: 5,234 conversations from agent_logs.jsonl
🏷️  Auto-labeled: 412 failures (7.9% failure rate)

🧬 Clustering failures...
   ✓ Generated 384-dim embeddings for 412 queries
   ✓ HDBSCAN found 5 failure clusters (38 noise points)

🔬 Analyzing root causes...
   Cluster 1 (127 cases, 30.8%): 🔴 金额/数字计算错误
     └─ Agent在涉及计算时未调用计算器工具，LLM心算出错
   Cluster 2 (89 cases, 21.6%):  🟡 多步骤推理丢失上下文
     └─ 超过3步的推理链中Agent忘记初始目标
   Cluster 3 (72 cases, 17.5%):  🟡 知识库检索召回不相关
     └─ 用户口语化表述导致语义检索失效
   Cluster 4 (54 cases, 13.1%):  🟢 权限校验错误提示不友好
     └─ 错误处理Prompt缺乏具体话术
   Cluster 5 (32 cases, 7.8%):   🟢 时间范围模糊导致超时
     └─ Agent未要求用户明确时间就发起全表扫描

🦜 19 canary queries generated (cover all 5 failure patterns)

📊 Report: agentmine_report.html (open in browser)
─────────────────────────────────────────────────────
💡 Fix Cluster 1 first — 30.8% of failures, highest impact.
```

---

## 🏗 工作原理

```
 Agent 日志                       AgentMine Pipeline              输出
 ───────────                     ──────────────────            ──────────
 JSONL / CSV              ┌─ 1. 日志标准化 + 格式检测
 LangSmith 导出    ───►    ├─ 2. 自动成功/失败标注              🎯 失败聚类
 LangFuse 导出             ├─ 3. 语义 Embedding (BGE)           🔬 根因分析
 自定义格式                ├─ 4. HDBSCAN 聚类 + UMAP 降维       🦜 金丝雀查询
                          ├─ 5. LLM 根因分析 (可选)             📊 HTML 报告
                          └─ 6. 金丝雀查询生成
```

### 核心技术

| 模块 | 技术 | 说明 |
|------|------|------|
| **Embedding** | BGE-small-zh-v1.5 | 384维中文语义向量，轻量高效 |
| **聚类** | HDBSCAN | 自动发现簇数量，天然处理噪声点 |
| **降维** | UMAP | 保留全局结构，可视化友好 |
| **根因分析** | LLM (GPT-4o/Claude) | 可选的深度根因分析 |
| **报告** | 自包含 HTML + Plotly.js | 单文件，可分享，交互式可视化 |

---

## 🚀 快速开始

### 安装

```bash
pip install agentmine

# 或从源码安装
git clone https://github.com/CCCchen040620/agentmine.git
cd agentmine
pip install -e .
```

### 基本用法

```bash
# 分析日志文件
agentmine analyze agent_logs.jsonl

# 使用 LLM 做深度根因分析
agentmine analyze agent_logs.jsonl --llm --model gpt-4o

# 启动 Web UI（拖拽上传）
agentmine ui

# 生成演示数据体验
agentmine demo
```

### 支持的日志格式

| 格式 | 说明 |
|------|------|
| **JSONL** | 每行一个 JSON，自动检测字段映射 |
| **CSV** | 自动映射列名到内部 Schema |
| **LangSmith** | 导出格式自动解析 |
| **LangFuse** | 导出格式自动解析 |

自定义日志只要包含 `user_query`、`final_output`、`error` 字段即可。

---

## 📊 HTML 报告预览

生成的报告是**单文件 HTML**，包含：

- 📈 **统计概览** — 总对话、失败数、失败率、簇数量
- 🗺 **UMAP 聚类可视化** — 交互式散点图，hover 看原始 query
- 📊 **失败分布柱状图** — 一眼看出哪个簇最严重
- 🔬 **根因分析卡片** — 每个簇的症状、直接原因、根本原因、修复建议
- 🦜 **金丝雀查询** — 可直接加入 CI 的回归测试用例
- 🌓 **暗色主题** — GitHub 风格，开发者友好

---

## 🎯 适用场景

- ✅ 你的 AI Agent 上线了，想系统性地了解失败模式
- ✅ 迭代了 Prompt/RAG/工具，想对比新旧版本的失败分布
- ✅ 要给老板汇报"Agent 当前的质量状况和改进计划"
- ✅ 想建立 Agent 的质量基线和回归测试体系
- ✅ 排查线上问题时，想快速定位是哪类问题

---

## 🔧 进阶用法

### 版本对比（回归检测）

改了 Prompt？换了模型？升级了工具？对比两个版本的日志，一眼看出变化：

```bash
agentmine diff v1_logs.jsonl v2_logs.jsonl
```

输出：
```
失败率变化: 📉 下降 2.3%
🆕 1 新增失败模式      ← 回归，需要修
✅ 2 已修复            ← 改好了
📈 0 恶化
📉 1 改善              ← 有进步
➖ 3 稳定
```

### 自定义最小簇大小

```bash
# 小数据集，降低最小簇大小
agentmine analyze small_logs.jsonl --min-cluster-size 3

# 大数据集，提高最小簇大小（减少噪音簇）
agentmine analyze large_logs.jsonl --min-cluster-size 20
```

### Web UI

```bash
agentmine ui --port 8765
# 浏览器打开 http://127.0.0.1:8765
# 拖拽日志文件 → 自动分析 → 在线预览报告
```

### Python API

```python
from src.ingestion.parser import LogParser
from src.labeling.auto_labeler import AutoLabeler
from src.clustering.embedder import TraceEmbedder
from src.clustering.clusterer import FailureClusterer

# 解析日志
parser = LogParser()
traces = parser.parse_file("agent_logs.jsonl")

# 标注
labeler = AutoLabeler()
traces = labeler.label(traces)

# 聚类
failures = [t for t in traces if t.status.value == "failure"]
embedder = TraceEmbedder()
embeddings = embedder.embed(failures)

clusterer = FailureClusterer()
result = clusterer.cluster(failures, embeddings)

# 查看结果
for cluster in result.clusters:
    print(f"Cluster {cluster.cluster_id}: {cluster.label} ({cluster.size} cases)")
```

---

## 🗺 路线图

- [x] 多格式日志解析 (JSONL/CSV/LangSmith/LangFuse)
- [x] 自动失败标注
- [x] HDBSCAN 语义聚类 + UMAP 可视化
- [x] 交互式 HTML 报告
- [x] Web UI (拖拽上传)
- [x] LLM 根因分析
- [x] 金丝雀查询生成
- [x] 演示数据生成器
- [ ] CI/CD 集成 (GitHub Action)
- [ ] 时序趋势分析（按天/周对比失败分布）
- [ ] 多 Agent 对比（A/B test 两个版本的失败模式差异）
- [ ] 实时监控模式（接入 Agent 的 webhook）
- [ ] VS Code 扩展

---

## 🤝 贡献

欢迎 PR！特别是以下方向：

1. 支持更多日志格式
2. 改进中文聚类效果
3. 添加更多失败模式模板
4. 前端美化

---

## 📝 License

MIT License — 随意使用、修改、分发。

---

## ⭐ Star 历史

如果这个项目对你有帮助，请点个 Star ⭐ 支持一下！

<p align="center">
  <strong>Made with ❤️ for AI Agent developers who are tired of guessing.</strong>
</p>
