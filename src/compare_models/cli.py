from __future__ import annotations

import logging
from pathlib import Path

import click

import compare_models.sources.arena  # noqa: F401
import compare_models.sources.artificial_analysis  # noqa: F401
from compare_models.models import ComparisonResult
from compare_models.renderer import render_comparison
from compare_models.sources import get_available_sources, get_source


@click.command()
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
    default="comparison.md",
    type=click.Path(),
    help="Output markdown file path.",
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
    help="Path to custom Artificial Analysis JSON data file.",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def main(
    models: str,
    families: bool,
    output: str,
    sources: str | None,
    aa_data: str | None,
    verbose: bool,
) -> None:
    """Compare LLM models using evaluation data from multiple sources."""
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
        kwargs = {}
        if source_name == "artificial_analysis" and aa_data:
            kwargs["data_path"] = Path(aa_data)

        try:
            source = get_source(source_name, **kwargs)
        except ValueError as e:
            click.echo(f"Warning: {e}", err=True)
            continue

        click.echo(f"Fetching data from {source.name}...")
        source_data = source.fetch_and_compare(model_names, families=families)
        result.sources.append(source_data)

        found_count = len(source_data.models_found)
        not_found_count = len(source_data.models_not_found)
        click.echo(f"  Found {found_count} models, {not_found_count} not found.")

    if not result.sources:
        raise click.ClickException("No data sources produced results.")

    output_path = Path(output)
    render_comparison(result, output_path)
    click.echo(f"Comparison written to {output_path}")
