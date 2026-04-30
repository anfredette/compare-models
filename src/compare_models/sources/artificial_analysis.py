from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from compare_models.models import ComparisonTable, SourceData
from compare_models.sources import register_source

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "artificial_analysis.json"

METHODOLOGY = """\
[Artificial Analysis](https://artificialanalysis.ai/) evaluates models using \
**automated benchmark suites** -- standardized tests with known correct answers, \
run programmatically. Their Intelligence Index (v4.0) aggregates scores from 10 \
evaluations:

- **GPQA Diamond** -- graduate-level science questions
- **Humanity's Last Exam (HLE)** -- extremely difficult cross-domain questions
- **SciCode** -- scientific coding problems
- **Terminal-Bench Hard** -- complex terminal/CLI tasks
- **IFBench** -- instruction following
- **AA-LCR** -- long-context retrieval
- **AA-Omniscience** -- broad knowledge assessment
- **GDPval-AA** -- GDP prediction (quantitative reasoning)
- **tau2-Bench Telecom** -- domain-specific agent tasks
- **CritPt** -- critical thinking

Unlike Arena's human preference votes, these benchmarks have **objectively \
correct answers**. This makes AA scores more precise for measurable capabilities \
(coding, math, factual recall) but less reflective of subjective qualities like \
writing style, helpfulness, or conversational fluency.

AA also independently measures **speed** (output tokens/sec), **latency** (time \
to first token), and **pricing** across API providers, providing a practical \
deployment perspective."""


class AAModel(BaseModel):
    name: str
    slug: str
    organization: str
    intelligence_index: int
    speed_tps: float | None = None
    ttft_s: float | None = None
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None
    context_window: int | None = None
    params_total_b: float | None = None
    params_active_b: float | None = None
    reasoning: bool = False
    url: str | None = None
    accessed_date: str | None = None

    @property
    def blended_price(self) -> float | None:
        if self.input_price_per_1m is not None and self.output_price_per_1m is not None:
            return round((3 * self.input_price_per_1m + self.output_price_per_1m) / 4, 2)
        return None

    @property
    def params_display(self) -> str:
        if self.params_total_b is None:
            return "proprietary"
        total = f"{self.params_total_b:g}B"
        if self.params_active_b and self.params_active_b != self.params_total_b:
            active = f"{self.params_active_b:g}B"
            return f"{total} / {active}"
        return total


def _load_models(data_path: Path) -> list[AAModel]:
    with open(data_path) as f:
        raw = json.load(f)
    return [AAModel(**m) for m in raw]


def _match_models(
    models: list[AAModel], names: list[str], *, families: bool = False
) -> tuple[list[AAModel], list[str]]:
    if families:
        matched: list[AAModel] = []
        for name in names:
            name_lower = name.lower()
            matched.extend(m for m in models if name_lower in m.name.lower())
        return list({m.name: m for m in matched}.values()), []

    name_lower_map = {n.lower(): n for n in names}
    found: list[AAModel] = []
    found_names: set[str] = set()

    for model in models:
        for search_name, orig_name in name_lower_map.items():
            if search_name in model.name.lower() or search_name in model.slug.lower():
                found.append(model)
                found_names.add(orig_name)
                break

    not_found = [n for n in names if n not in found_names]
    return found, not_found


def _comparison_table(
    models: list[AAModel], *, title: str, include_params: bool = True
) -> ComparisonTable:
    sorted_models = sorted(models, key=lambda m: m.intelligence_index, reverse=True)

    if include_params:
        headers = [
            "Model",
            "Params (total/active)",
            "AA Intelligence",
            "Speed (t/s)",
            "TTFT (s)",
            "Price ($/1M blend)",
            "Context",
        ]
    else:
        headers = [
            "Model",
            "AA Intelligence",
            "Speed (t/s)",
            "TTFT (s)",
            "Price ($/1M blend)",
            "Context",
        ]

    rows: list[list[str]] = []
    for m in sorted_models:
        speed = f"{m.speed_tps:.1f}" if m.speed_tps else "--"
        ttft = f"{m.ttft_s:.2f}" if m.ttft_s else "--"
        price = f"${m.blended_price:.2f}" if m.blended_price else "--"
        ctx = f"{m.context_window // 1000}k" if m.context_window else "--"

        if include_params:
            rows.append(
                [
                    m.name,
                    m.params_display,
                    str(m.intelligence_index),
                    speed,
                    ttft,
                    price,
                    ctx,
                ]
            )
        else:
            rows.append(
                [
                    m.name,
                    str(m.intelligence_index),
                    speed,
                    ttft,
                    price,
                    ctx,
                ]
            )

    return ComparisonTable(
        title=title,
        headers=headers,
        rows=rows,
        alignments=["left"] + ["right"] * (len(headers) - 1),
    )


def _global_ranking_table(
    all_models: list[AAModel], target_name: str, window: int = 5
) -> ComparisonTable:
    sorted_all = sorted(all_models, key=lambda m: m.intelligence_index, reverse=True)

    target_idx = None
    for i, m in enumerate(sorted_all):
        if target_name.lower() in m.name.lower():
            target_idx = i
            break

    if target_idx is None:
        return ComparisonTable(
            title=f"{target_name} (not found in AA data)",
            headers=[],
            rows=[],
        )

    target_model = sorted_all[target_idx]
    rank = target_idx + 1
    total = len(sorted_all)
    start = max(0, target_idx - window)
    end = min(total, target_idx + window + 1)

    rows: list[list[str]] = []
    for i in range(start, end):
        m = sorted_all[i]
        r = i + 1
        is_target = m.name == target_model.name
        fmt_rank = f"**{r}**" if is_target else str(r)
        fmt_name = f"**{m.name}**" if is_target else m.name
        fmt_score = f"**{m.intelligence_index}**" if is_target else str(m.intelligence_index)
        rows.append([fmt_rank, fmt_name, fmt_score])

    return ComparisonTable(
        title=f"{target_model.name} (score {target_model.intelligence_index}, ~rank {rank} of {total})",
        headers=["Rank", "Model", "AA Intelligence"],
        rows=rows,
        alignments=["right", "left", "center"],
    )


def _compute_findings(matched: list[AAModel]) -> list[str]:
    findings: list[str] = []

    if not matched:
        return ["No matching models found in AA data."]

    orgs: dict[str, list[AAModel]] = {}
    for m in matched:
        orgs.setdefault(m.organization, []).append(m)

    for org, models in orgs.items():
        best = max(models, key=lambda m: m.intelligence_index)
        findings.append(
            f"**{org} top model:** {best.name} (Intelligence Index: {best.intelligence_index})"
        )

    if len(orgs) >= 2:
        org_list = list(orgs.values())
        a_best = max(org_list[0], key=lambda m: m.intelligence_index)
        b_best = max(org_list[1], key=lambda m: m.intelligence_index)

        if a_best.speed_tps is not None and b_best.speed_tps is not None:
            faster = a_best if a_best.speed_tps > b_best.speed_tps else b_best
            slower = b_best if faster == a_best else a_best
            assert faster.speed_tps is not None and slower.speed_tps is not None
            ratio = faster.speed_tps / slower.speed_tps
            findings.append(
                f"**Speed:** {faster.name} is {ratio:.1f}x faster "
                f"({faster.speed_tps:.0f} vs {slower.speed_tps:.0f} t/s)"
            )

        if a_best.blended_price is not None and b_best.blended_price is not None:
            cheaper = a_best if a_best.blended_price < b_best.blended_price else b_best
            pricier = b_best if cheaper == a_best else a_best
            assert pricier.blended_price is not None and cheaper.blended_price is not None
            ratio = pricier.blended_price / cheaper.blended_price
            findings.append(
                f"**Price:** {cheaper.name} is {ratio:.1f}x cheaper "
                f"(${cheaper.blended_price:.2f} vs ${pricier.blended_price:.2f}/1M tokens)"
            )

        if a_best.ttft_s is not None and b_best.ttft_s is not None:
            faster_ttft = a_best if a_best.ttft_s < b_best.ttft_s else b_best
            slower_ttft = b_best if faster_ttft == a_best else a_best
            assert slower_ttft.ttft_s is not None and faster_ttft.ttft_s is not None
            ratio = slower_ttft.ttft_s / faster_ttft.ttft_s
            findings.append(
                f"**Latency:** {faster_ttft.name} has {ratio:.1f}x lower TTFT "
                f"({faster_ttft.ttft_s:.2f}s vs {slower_ttft.ttft_s:.2f}s)"
            )

    return findings


class ArtificialAnalysisSource:
    """Artificial Analysis data source."""

    def __init__(self, data_path: Path | None = None):
        self._data_path = data_path or DEFAULT_DATA_PATH

    @property
    def name(self) -> str:
        return "Artificial Analysis"

    @property
    def description(self) -> str:
        return "Automated Benchmarks"

    def fetch_and_compare(
        self,
        model_names: list[str],
        *,
        families: bool = False,
        **kwargs: Any,
    ) -> SourceData:
        logger.info(f"Loading AA data from {self._data_path}")
        all_models = _load_models(self._data_path)
        matched, not_found = _match_models(all_models, model_names, families=families)

        if not matched:
            return SourceData(
                source_name=self.name,
                source_description=self.description,
                methodology=METHODOLOGY,
                models_found=[],
                models_not_found=not_found,
                findings=["No matching models found in AA data."],
            )

        orgs: dict[str, list[AAModel]] = {}
        for m in matched:
            orgs.setdefault(m.organization, []).append(m)

        global_rankings: list[ComparisonTable] = []
        for org_models in orgs.values():
            best = max(org_models, key=lambda m: m.intelligence_index)
            global_rankings.append(_global_ranking_table(all_models, best.name))

        comparison_tables: list[ComparisonTable] = []

        reasoning = [m for m in matched if m.reasoning]
        non_reasoning = [m for m in matched if not m.reasoning]

        if reasoning:
            comparison_tables.append(_comparison_table(reasoning, title="Reasoning Models"))
        if non_reasoning:
            comparison_tables.append(_comparison_table(non_reasoning, title="Non-Reasoning Models"))

        if reasoning and non_reasoning:
            comparison_tables.append(
                _comparison_table(matched, title="All Models", include_params=False)
            )

        findings = _compute_findings(matched)

        return SourceData(
            source_name=self.name,
            source_description=self.description,
            methodology=METHODOLOGY,
            global_rankings=global_rankings,
            comparison_tables=comparison_tables,
            findings=findings,
            models_found=[m.name for m in matched],
            models_not_found=not_found,
        )


register_source("artificial_analysis", ArtificialAnalysisSource)
