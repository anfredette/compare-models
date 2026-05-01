from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import click

import compare_models.sources.arena  # noqa: F401
import compare_models.sources.artificial_analysis  # noqa: F401
from compare_models.models import ComparisonResult
from compare_models.renderer import render_comparison
from compare_models.sources import get_available_sources, get_source

REPORTS_DIR = Path("reports")


def generate_output_path(model_names: list[str]) -> Path:
    """Generate an auto-named report path in reports/.

    Format: reports/{short1}_{short2}_{YYYY}_{MM}_{DD}_{NN}.md
    where NN increments from the highest existing file with the same prefix.
    """
    parts = [name.split("-")[0] for name in model_names]
    counts = Counter(parts)
    if any(c > 1 for c in counts.values()):
        parts = []
        for name in model_names:
            tokens = name.split("-")
            parts.append(tokens[1] if len(tokens) > 1 else tokens[0])

    prefix = "_".join(sorted(set(parts)))
    today = date.today().strftime("%Y_%m_%d")
    base = f"{prefix}_{today}"

    REPORTS_DIR.mkdir(exist_ok=True)

    existing = list(REPORTS_DIR.glob(f"{base}_*.md"))
    max_n = -1
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.md$")
    for p in existing:
        m = pattern.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))

    return REPORTS_DIR / f"{base}_{max_n + 1:02d}.md"


@click.group()
def main() -> None:
    """Compare LLM models using evaluation data from multiple sources."""


@main.command()
@click.option(
    "--models",
    "-m",
    required=True,
    help="Comma-separated model or family names to compare.",
)
@click.option(
    "--families",
    is_flag=True,
    default=False,
    help="Treat model names as family prefixes (e.g., 'qwen' matches all qwen* models).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output markdown file path. Auto-generates in reports/ if not set.",
)
@click.option(
    "--sources",
    "-s",
    default=None,
    help=f"Comma-separated source names. Available: {', '.join(get_available_sources())}",
)
@click.option(
    "--aa-data",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom Artificial Analysis JSON data file (bypasses cache).",
)
@click.option("--pdf", is_flag=True, default=False, help="Also generate a PDF via pandoc.")
@click.option(
    "--check-api",
    is_flag=True,
    default=False,
    help="If models aren't found in the AA cache, check the live API.",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def compare(
    models: str,
    families: bool,
    output: str,
    sources: str | None,
    aa_data: str | None,
    pdf: bool,
    check_api: bool,
    verbose: bool,
) -> None:
    """Compare LLM models across evaluation sources."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    model_names = [m.strip() for m in models.split(",") if m.strip()]
    if not model_names:
        raise click.UsageError("No model names provided.")

    source_names = (
        [s.strip() for s in sources.split(",") if s.strip()] if sources else get_available_sources()
    )

    result = ComparisonResult(model_names=model_names)

    for source_name in source_names:
        kwargs: dict[str, Any] = {}
        if source_name == "artificial_analysis":
            if aa_data:
                kwargs["data_path"] = Path(aa_data)
            if check_api:
                kwargs["check_api"] = True

        try:
            source = get_source(source_name, **kwargs)
        except ValueError as e:
            click.echo(f"Warning: {e}", err=True)
            continue

        source_data = source.fetch_and_compare(model_names, families=families)
        result.sources.append(source_data)

        found_count = len(source_data.models_found)
        not_found_count = len(source_data.models_not_found)
        click.echo(f"  Found {found_count} models, {not_found_count} not found.")

        for name, similar in source_data.suggestions.items():
            click.echo(f'  Model "{name}" not found. Similar models: {", ".join(similar)}')

    if not result.sources:
        raise click.ClickException("No data sources produced results.")

    output_path = Path(output) if output else generate_output_path(model_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_comparison(result, output_path)
    click.echo(f"Comparison written to {output_path}")

    if pdf:
        if not shutil.which("pandoc"):
            raise click.ClickException(
                "pandoc is required for --pdf. Install: brew install pandoc (macOS) or apt install pandoc (Linux)."
            )
        pdf_path = output_path.with_suffix(".pdf")
        proc = subprocess.run(
            ["pandoc", str(output_path), "-o", str(pdf_path), "-V", "geometry:margin=1.5cm"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise click.ClickException(
                f"pandoc failed — a LaTeX engine is required for PDF output.\n"
                f"Install one: brew install mactex-no-gui (macOS) or apt install texlive-xetex (Linux).\n"
                f"pandoc error: {proc.stderr.strip()}"
            )
        click.echo(f"PDF written to {pdf_path}")


@main.command("sync-aa")
@click.option(
    "--api-key",
    envvar="AA_API_KEY",
    required=True,
    help="Artificial Analysis API key (or set AA_API_KEY env var).",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def sync_aa(api_key: str, verbose: bool) -> None:
    """Sync Artificial Analysis model data from the API."""
    from compare_models import aa_client

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo("Fetching models from Artificial Analysis API...")
    try:
        count, cache_path = aa_client.sync(api_key)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Synced {count} models to {cache_path}")


@main.command("sync-arena")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def sync_arena(verbose: bool) -> None:
    """Sync Arena leaderboard data from HuggingFace."""
    from compare_models import arena_client

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo("Fetching Arena leaderboard from HuggingFace...")
    try:
        count, cache_path = arena_client.sync()
    except Exception as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Synced {count} rows to {cache_path}")
