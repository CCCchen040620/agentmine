# Contributing to AgentMine

Thanks for your interest in contributing! 🎉

## Quick Start

```bash
git clone https://github.com/yourusername/agentmine.git
cd agentmine
pip install -e ".[dev]"
```

## Development Flow

1. Fork and clone the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Run lint: `ruff check src/`
6. Commit and push
7. Open a Pull Request

## What to Contribute

### 🟢 Good First Issues
- Add support for more log formats (e.g., Arize, Helicone)
- Improve failure pattern templates in `examples/demo_data.py`
- Add more test cases in `tests/`
- Improve documentation

### 🟡 Intermediate
- Improve Chinese text clustering (custom embedding, better tokenization)
- Add time-series trend analysis
- Implement multi-agent comparison mode
- Add more failure type classifiers in `src/labeling/auto_labeler.py`

### 🔴 Advanced
- Real-time monitoring mode (webhook integration)
- VS Code extension
- GPU-accelerated clustering for very large datasets
- Multi-language support (English, Japanese, etc.)

## Project Structure

See [README.md](README.md) for the full architecture.

Key files:
- `src/ingestion/schema.py` — Standardized trace data model
- `src/clustering/clusterer.py` — HDBSCAN clustering logic
- `src/analysis/root_cause.py` — LLM root cause analysis prompts
- `src/report/html_generator.py` — HTML report template

## Code Style

- Python 3.10+, type hints encouraged
- Black for formatting (line length 100)
- Ruff for linting
- Google-style docstrings

## Questions?

Open an issue or start a discussion!
