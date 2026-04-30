from __future__ import annotations

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from compare_models.models import ComparisonResult

TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_comparison(result: ComparisonResult, output_path: Path) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template("comparison.md.j2")

    content = template.render(
        model_names=result.model_names,
        sources=result.sources,
        overall_conclusions=result.overall_conclusions,
        date=date.today().isoformat(),
    )

    content = _clean_blank_lines(content)

    output_path.write_text(content)
    return content


def _clean_blank_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    while cleaned and cleaned[-1].strip() == "":
        cleaned.pop()
    cleaned.append("")

    return "\n".join(cleaned)
