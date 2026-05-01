# compare-models

Automated LLM model comparison tool that generates detailed reports from
multiple evaluation sources. It pulls data from
[Arena](https://lmarena.ai/) (human preference ratings) and
[Artificial Analysis](https://artificialanalysis.ai/) (automated benchmarks,
speed, latency, and pricing), then produces a report with
global rankings, category breakdowns, head-to-head comparisons, and key
findings.

The tool works in two modes: a **CLI** that generates data-driven reports with
deterministic findings, and a **Claude skill** (`/compare-models`) that runs
the CLI and then layers on narrative analysis with an Overall Conclusions
section.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/anfredette/compare-models.git
cd compare-models
uv sync
```

**Optional -- PDF output** requires [pandoc](https://pandoc.org/installing.html)
and a LaTeX engine (pdflatex, xelatex, or lualatex):

```bash
# macOS
brew install pandoc basictex          # or: brew install pandoc mactex-no-gui

# Debian/Ubuntu
sudo apt install pandoc texlive-latex-recommended

# Fedora
sudo dnf install pandoc texlive-latex
```

## Usage

### Claude Skill (recommended)

The best results come from running inside a
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) session. The
`/compare-models` skill provides a natural language interface -- just describe
what you want to compare and Claude handles the rest. It builds the right CLI
command, reads the generated report, then enhances it with interpretive Key
Findings and a full Overall Conclusions section covering positioning, value
proposition, quality profiles, and a side-by-side summary table.

From the `compare-models` project directory:

```bash
cd /path/to/compare-models
claude
```

Then use the skill:

```
/compare-models Compare Trinity vs Qwen model families
/compare-models How does Gemini 3 stack up against Gemma 4?
/compare-models Compare just Arena data for trinity and qwen, and generate a PDF
/compare-models Add Claude Opus 4.7 to the AA data
```

Claude will:
1. Parse the natural language request and build the appropriate CLI command
2. Run the CLI to generate data tables, head-to-head comparisons, and findings
3. Enhance the Key Findings with narrative interpretation
4. Write an Overall Conclusions section with positioning analysis, a summary
   comparison table, and a bottom-line recommendation
5. Summarize the key findings in the chat

You can also ask follow-up questions about specific sections after the report
is generated.

### CLI

Use the CLI directly if you don't have a Claude Code session or don't need
the narrative interpretation.

```bash
# Compare specific models -- report auto-named in reports/
uv run compare-models -m "trinity-large-preview,qwen3-235b-a22b"

# Compare entire model families
uv run compare-models -m "trinity,qwen" --families

# Use only specific sources
uv run compare-models -m "trinity-large-preview,qwen3-235b-a22b" --sources arena

# Generate a PDF alongside the markdown report
uv run compare-models -m "trinity-large-preview,qwen3-235b-a22b" --pdf

# Override the output path (skips auto-naming)
uv run compare-models -m "trinity,qwen" --families -o custom_report.md
```

Reports are saved to `reports/` with auto-generated names based on the models
compared, date, and a sequence number (e.g., `qwen_trinity_2026_05_01_00.md`).
Use `-o` to override the path.

## Output Structure

The generated markdown report includes:

- **Global Rankings** -- Consolidated leaderboard showing each subject model's
  neighborhood (+-5 models), with `[N models not shown]` gap markers for distant models
- **Category Ratings** -- General capabilities (7 categories) and industry categories
  (7 categories) side-by-side
- **Head-to-Head** -- Pairwise comparison across all 14 categories with deltas and
  winner per category
- **Win/Loss Summary** -- Cross-matchup overview
- **Key Findings** -- Analytical findings (positioning, strengths, weaknesses,
  profile characterization)
- **Overall Conclusions** -- Narrative analysis (added by the `/compare-models`
  skill)

## Updating AA Data

AA data lives in `data/artificial_analysis.json`. To refresh it:

```bash
uv run python scripts/scrape_aa.py
```

This scrapes all model pages from artificialanalysis.ai via their sitemap (takes
~6 minutes with 1s delay between requests). You can also add individual models
via the `/compare-models` skill -- ask Claude to "Add [model name] to the AA data".

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
2. Implement the `DataSource` protocol (see `sources/__init__.py`)
3. Call `register_source("name", YourSourceClass)` at module level
4. Import the module in `cli.py` to trigger registration

## License

Not open source -- contains proprietary Artificial Analysis data.
