# Changelog

All notable changes to AgentMine will be documented in this file.

## [0.1.0] — 2024-06-16

### 🚀 First Release

**AgentMine** is the first open-source tool that automatically discovers systemic failure patterns from AI agent execution logs.

### ✨ Features

- **Multi-format log parsing** — JSONL, CSV, LangSmith, LangFuse auto-detection
- **Automatic failure labeling** — Rule-based + optional LLM classification
- **Semantic clustering** — BGE embeddings + HDBSCAN + UMAP visualization
- **LLM root cause analysis** — GPT-4o / Claude powered deep analysis
- **Canary query generation** — Auto-generate regression test queries
- **Version diff** — Compare two agent versions, detect regressions & improvements
- **Trend analysis** — Failure rate over time with anomaly detection
- **Rich terminal UI** — Beautiful CLI output with panels, tables, progress bars
- **Interactive HTML report** — Self-contained Plotly.js visualization
- **Web UI** — Drag-and-drop log upload
- **Framework integrations** — LangChain callback + LlamaIndex observer
- **7 tests passing**, ~5000 lines of Python code
