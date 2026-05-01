from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankEntry:
    """A model's position in a ranking."""

    rank: int
    model_name: str
    score: float
    extra: dict[str, str | float] = field(default_factory=dict)


@dataclass
class ComparisonTable:
    """A table comparing models across dimensions."""

    title: str
    headers: list[str]
    rows: list[list[str]]
    alignments: list[str] | None = None


@dataclass
class HeadToHead:
    """Head-to-head comparison between two models."""

    model_a: str
    model_b: str
    dimensions: list[str]
    a_scores: list[float]
    b_scores: list[float]
    deltas: list[float]
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0


@dataclass
class SourceData:
    """All data produced by a single data source for a comparison."""

    source_name: str
    source_description: str
    methodology: str
    global_rankings: list[ComparisonTable] = field(default_factory=list)
    comparison_tables: list[ComparisonTable] = field(default_factory=list)
    head_to_heads: list[HeadToHead] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    models_found: list[str] = field(default_factory=list)
    models_not_found: list[str] = field(default_factory=list)
    suggestions: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Complete comparison result from all sources."""

    model_names: list[str]
    sources: list[SourceData] = field(default_factory=list)
    overall_conclusions: list[str] = field(default_factory=list)
