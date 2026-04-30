from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from datasets import load_dataset

from compare_models.models import ComparisonTable, HeadToHead, SourceData
from compare_models.sources import register_source

logger = logging.getLogger(__name__)

KEY_CATEGORIES = [
    "overall",
    "coding",
    "math",
    "creative_writing",
    "instruction_following",
    "hard_prompts",
    "if_expert",
    "multi_turn",
    "long_query",
    "software_engineering_it",
    "legal",
    "science",
    "industry_mathematical",
    "writing",
]

CATEGORY_SHORT = {
    "creative_writing": "creative",
    "instruction_following": "instruct",
    "hard_prompts": "hard",
    "if_expert": "expert",
    "software_engineering_it": "sw/it",
    "industry_mathematical": "ind_math",
}

GENERAL_CATEGORIES = [
    "overall",
    "coding",
    "math",
    "creative_writing",
    "instruction_following",
    "hard_prompts",
    "if_expert",
]

INDUSTRY_CATEGORIES = [
    "multi_turn",
    "long_query",
    "software_engineering_it",
    "legal",
    "science",
    "industry_mathematical",
    "writing",
]

METHODOLOGY = """\
[Chatbot Arena](https://lmarena.ai/) (formerly LMSYS) ranks models using \
**head-to-head human preference votes**. Real users submit prompts to two \
anonymous models side-by-side, then choose which response they prefer. These \
pairwise outcomes are aggregated into Bradley-Terry ratings (similar to Elo in \
chess) -- higher is better, with scores typically ranging from ~900 to ~1550.

The `text_style_control` variant adjusts for verbosity bias, so models don't get \
rewarded simply for producing longer responses.

Ratings are broken down into **27 topic-based categories** (coding, math, \
creative writing, legal, etc.) based on conversation content, giving a detailed \
profile of where each model excels relative to the full field. Because these are \
human judgments rather than automated test suites, they reflect what real users \
find helpful -- but they also carry the biases of the voter population (skewed \
toward AI-savvy early adopters) and the preference-vs-correctness gap (a \
confident but wrong answer can still win votes)."""


def _short_cat(cat: str) -> str:
    return CATEGORY_SHORT.get(cat, cat)


def _fetch_arena() -> pd.DataFrame:
    ds = load_dataset("lmarena-ai/leaderboard-dataset", "text_style_control", split="latest")
    return pd.DataFrame(ds)


def _resolve_family_models(df: pd.DataFrame, family_names: list[str]) -> list[str]:
    overall = df[df["category"] == "overall"]
    matched: list[str] = []
    for family in family_names:
        family_lower = family.lower()
        matches = overall[overall["model_name"].str.lower().str.contains(family_lower)]
        matched.extend(matches["model_name"].tolist())
    return sorted(set(matched))


def _find_models(df: pd.DataFrame, model_names: list[str]) -> tuple[list[str], list[str]]:
    overall = df[df["category"] == "overall"]
    available = set(overall["model_name"].tolist())
    found = [m for m in model_names if m in available]
    not_found = [m for m in model_names if m not in available]
    return found, not_found


def _global_ranking_table(df: pd.DataFrame, model_name: str, window: int = 5) -> ComparisonTable:
    overall = (
        df[df["category"] == "overall"]
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    total = len(overall)

    idx = overall[overall["model_name"] == model_name].index
    if len(idx) == 0:
        return ComparisonTable(
            title=f"{model_name} (not found in Arena)",
            headers=[],
            rows=[],
        )

    pos = idx[0]
    rank = pos + 1
    start = max(0, pos - window)
    end = min(total, pos + window + 1)
    neighborhood = overall.iloc[start:end]

    rows: list[list[str]] = []
    for i, row in neighborhood.iterrows():
        r = int(i) + 1  # type: ignore[call-overload]
        name = row["model_name"]
        is_target = name == model_name
        fmt_rank = f"**{r}**" if is_target else str(r)
        fmt_name = f"**{name}**" if is_target else name
        fmt_rating = f"**{row['rating']:.1f}**" if is_target else f"{row['rating']:.1f}"
        fmt_votes = (
            f"**{int(row['vote_count']):,}**" if is_target else f"{int(row['vote_count']):,}"
        )
        rows.append([fmt_rank, fmt_name, fmt_rating, fmt_votes])

    return ComparisonTable(
        title=f"{model_name} (rank {rank} of {total})",
        headers=["Rank", "Model", "Rating", "Votes"],
        rows=rows,
        alignments=["right", "left", "right", "right"],
    )


def _subset_ranking(df: pd.DataFrame, model_names: list[str]) -> ComparisonTable:
    overall = df[df["category"] == "overall"]
    subset = overall[overall["model_name"].isin(model_names)].sort_values("rating", ascending=False)

    rows: list[list[str]] = []
    for rank, (_, row) in enumerate(subset.iterrows(), 1):
        rows.append(
            [
                str(rank),
                row["model_name"],
                f"{int(row['vote_count']):,}",
                f"{row['rating']:.1f}",
            ]
        )

    return ComparisonTable(
        title=f"Subset Rankings ({len(rows)} models)",
        headers=["Rank", "Model", "Votes", "Overall"],
        rows=rows,
        alignments=["right", "left", "right", "right"],
    )


def _category_table(
    df: pd.DataFrame, model_names: list[str], categories: list[str]
) -> ComparisonTable:
    subset = df[(df["model_name"].isin(model_names)) & (df["category"].isin(categories))]

    pivot = subset.pivot_table(
        index="model_name", columns="category", values="rating", aggfunc="first"
    )

    if "overall" in categories:
        pivot = pivot.sort_values("overall", ascending=False)
    else:
        pivot = pivot.sort_values(pivot.columns[0], ascending=False)

    headers = ["Model"] + [_short_cat(c) for c in categories if c in pivot.columns]
    rows: list[list[str]] = []
    for model_name in pivot.index:
        row_data: list[str] = [model_name]
        for cat in categories:
            if cat in pivot.columns:
                val = pivot.loc[model_name, cat]
                row_data.append(f"{val:.1f}" if pd.notna(val) else "—")
        rows.append(row_data)

    return ComparisonTable(
        title="Category Ratings",
        headers=headers,
        rows=rows,
        alignments=["left"] + ["right"] * (len(headers) - 1),
    )


def _head_to_head(
    df: pd.DataFrame, model_a: str, model_b: str, categories: list[str]
) -> HeadToHead:
    a_data = df[df["model_name"] == model_a].set_index("category")
    b_data = df[df["model_name"] == model_b].set_index("category")

    dims: list[str] = []
    a_scores: list[float] = []
    b_scores: list[float] = []
    deltas: list[float] = []

    for cat in categories:
        if cat in a_data.index and cat in b_data.index:
            a_val = float(a_data.loc[cat, "rating"])  # type: ignore[arg-type]
            b_val = float(b_data.loc[cat, "rating"])  # type: ignore[arg-type]
            dims.append(_short_cat(cat))
            a_scores.append(a_val)
            b_scores.append(b_val)
            deltas.append(round(a_val - b_val, 1))

    sorted_indices = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)
    dims = [dims[i] for i in sorted_indices]
    a_scores = [a_scores[i] for i in sorted_indices]
    b_scores = [b_scores[i] for i in sorted_indices]
    deltas = [deltas[i] for i in sorted_indices]

    a_wins = sum(1 for d in deltas if d > 0)
    b_wins = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)

    return HeadToHead(
        model_a=model_a,
        model_b=model_b,
        dimensions=dims,
        a_scores=a_scores,
        b_scores=b_scores,
        deltas=deltas,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
    )


def _win_loss_table(h2hs: list[HeadToHead]) -> ComparisonTable:
    rows: list[list[str]] = []
    for h in sorted(h2hs, key=lambda h: h.deltas[0] if h.deltas else 0, reverse=True):
        overall_delta = next(
            (d for dim, d in zip(h.dimensions, h.deltas, strict=False) if dim == "overall"),
            0.0,
        )
        gap_str = f"+{overall_delta:.1f}" if overall_delta > 0 else f"{overall_delta:.1f}"
        rows.append(
            [
                h.model_b,
                str(h.a_wins),
                str(h.b_wins),
                gap_str,
            ]
        )

    return ComparisonTable(
        title="Win/Loss Summary",
        headers=["vs Model", f"{h2hs[0].model_a} Wins", "Opponent Wins", "Overall Gap"],
        rows=rows,
        alignments=["left", "center", "center", "center"],
    )


def _compute_findings(
    model_names: list[str],
    h2hs: list[HeadToHead],
    global_tables: list[ComparisonTable],
) -> list[str]:
    findings: list[str] = []

    for gt in global_tables:
        findings.append(f"**Global ranking:** {gt.title}")

    if h2hs:
        total_cats = len(h2hs[0].dimensions)
        for h in h2hs:
            if h.a_wins > h.b_wins:
                findings.append(
                    f"**{h.model_a} vs {h.model_b}:** Wins {h.a_wins} of {total_cats} categories"
                )
            elif h.b_wins > h.a_wins:
                findings.append(
                    f"**{h.model_a} vs {h.model_b}:** Loses {h.b_wins} of {total_cats} categories"
                )
            else:
                findings.append(f"**{h.model_a} vs {h.model_b}:** Tied at {h.a_wins}-{h.b_wins}")

        all_dims: dict[str, list[float]] = {}
        for h in h2hs:
            for dim, delta in zip(h.dimensions, h.deltas, strict=True):
                all_dims.setdefault(dim, []).append(delta)

        strengths = [
            (dim, sum(ds) / len(ds))
            for dim, ds in all_dims.items()
            if all(d > 0 for d in ds) and dim != "overall"
        ]
        weaknesses = [
            (dim, sum(ds) / len(ds))
            for dim, ds in all_dims.items()
            if all(d < 0 for d in ds) and dim != "overall"
        ]

        if strengths:
            strengths.sort(key=lambda x: x[1], reverse=True)
            top = [f"{s[0]} (avg +{s[1]:.1f})" for s in strengths[:3]]
            findings.append(f"**Consistent strengths:** {', '.join(top)}")

        if weaknesses:
            weaknesses.sort(key=lambda x: x[1])
            bottom = [f"{w[0]} (avg {w[1]:.1f})" for w in weaknesses[:3]]
            findings.append(f"**Consistent weaknesses:** {', '.join(bottom)}")

    return findings


def _select_h2h_pairs(df: pd.DataFrame, model_names: list[str]) -> list[tuple[str, str]]:
    overall = df[(df["category"] == "overall") & (df["model_name"].isin(model_names))].sort_values(
        "rating", ascending=False
    )

    orgs: dict[str, list[str]] = {}
    for _, row in overall.iterrows():
        name = row["model_name"]
        org = row.get("organization", "unknown")
        orgs.setdefault(org, []).append(name)

    org_groups = list(orgs.values())
    if len(org_groups) < 2:
        models_sorted = overall["model_name"].tolist()
        pairs: list[tuple[str, str]] = []
        for i in range(len(models_sorted)):
            for j in range(i + 1, len(models_sorted)):
                pairs.append((models_sorted[i], models_sorted[j]))
        return pairs[:8]

    pairs = []
    for i, group_a in enumerate(org_groups):
        for group_b in org_groups[i + 1 :]:
            for a in group_a[:3]:
                for b in group_b[:3]:
                    pairs.append((a, b))

    return pairs[:8]


class ArenaSource:
    """Chatbot Arena data source."""

    @property
    def name(self) -> str:
        return "Chatbot Arena"

    @property
    def description(self) -> str:
        return "Human Preference Ratings"

    def fetch_and_compare(
        self,
        model_names: list[str],
        *,
        families: bool = False,
        **kwargs: Any,
    ) -> SourceData:
        logger.info("Fetching Arena leaderboard data...")
        df = _fetch_arena()

        if families:
            model_names = _resolve_family_models(df, model_names)
            logger.info(f"Resolved {len(model_names)} models from family names")

        found, not_found = _find_models(df, model_names)

        if not found:
            return SourceData(
                source_name=self.name,
                source_description=self.description,
                methodology=METHODOLOGY,
                models_found=[],
                models_not_found=not_found,
                findings=["No matching models found in Arena leaderboard."],
            )

        global_rankings = [_global_ranking_table(df, m) for m in found]
        subset = _subset_ranking(df, found)

        general = _category_table(df, found, GENERAL_CATEGORIES)
        general.title = "Part 1: General capabilities"
        industry = _category_table(df, found, INDUSTRY_CATEGORIES)
        industry.title = "Part 2: Conversational and industry categories"

        h2h_pairs = _select_h2h_pairs(df, found)
        h2hs = [_head_to_head(df, a, b, KEY_CATEGORIES) for a, b in h2h_pairs]

        win_loss = _win_loss_table(h2hs) if h2hs else None

        findings = _compute_findings(found, h2hs, global_rankings)

        comparison_tables = [subset, general, industry]
        if win_loss:
            comparison_tables.append(win_loss)

        return SourceData(
            source_name=self.name,
            source_description=self.description,
            methodology=METHODOLOGY,
            global_rankings=global_rankings,
            comparison_tables=comparison_tables,
            head_to_heads=h2hs,
            findings=findings,
            models_found=found,
            models_not_found=not_found,
        )


register_source("arena", ArenaSource)
