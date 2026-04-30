# compare-models

CLI tool for automated LLM model comparison using multiple evaluation data sources.

## Data Sources

- **Chatbot Arena** — Human preference ratings from head-to-head blind votes (Bradley-Terry ratings across 27 categories)
- **Artificial Analysis** — Automated benchmarks (Intelligence Index), plus speed, latency, and pricing data

More sources can be added via the provider pattern (e.g., Every Eval Ever).

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# Compare specific models across all sources
uv run compare-models -m "trinity-large-preview,qwen3-235b-a22b" -o comparison.md

# Compare entire model families
uv run compare-models -m "trinity,qwen" --families -o comparison.md

# Use only specific sources
uv run compare-models -m "trinity-large-preview,qwen3-235b-a22b" --sources arena
```

## Development

```bash
uv sync --extra dev
make lint       # Ruff linter
make format     # Ruff auto-format
make typecheck  # Mypy type checking
make test       # Run all tests
```

## Adding a New Data Source

1. Create a new module in `src/compare_models/sources/`
2. Implement the `DataSource` protocol
3. Register it in `sources/__init__.py`
4. Add a Jinja2 template section in `templates/`

## License

Apache-2.0
