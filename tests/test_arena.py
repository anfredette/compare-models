from __future__ import annotations

import pandas as pd
import pytest

from compare_models.sources.arena import (
    GENERAL_CATEGORIES,
    INDUSTRY_CATEGORIES,
    KEY_CATEGORIES,
    _category_table,
    _find_models,
    _global_ranking_table,
    _head_to_head,
    _subset_ranking,
    _win_loss_table,
)


@pytest.mark.unit
class TestFindModels:
    def test_finds_existing_models(self, sample_arena_df: pd.DataFrame) -> None:
        found, not_found = _find_models(sample_arena_df, ["model-alpha", "model-beta"])
        assert found == ["model-alpha", "model-beta"]
        assert not_found == []

    def test_reports_missing_models(self, sample_arena_df: pd.DataFrame) -> None:
        found, not_found = _find_models(sample_arena_df, ["model-alpha", "nonexistent"])
        assert found == ["model-alpha"]
        assert not_found == ["nonexistent"]


@pytest.mark.unit
class TestGlobalRanking:
    def test_returns_neighborhood(self, sample_arena_df: pd.DataFrame) -> None:
        table = _global_ranking_table(sample_arena_df, "model-gamma", window=2)
        assert "model-gamma" in table.title
        assert len(table.rows) > 0
        names = [row[1] for row in table.rows]
        assert any("model-gamma" in n for n in names)

    def test_handles_missing_model(self, sample_arena_df: pd.DataFrame) -> None:
        table = _global_ranking_table(sample_arena_df, "nonexistent")
        assert "not found" in table.title


@pytest.mark.unit
class TestSubsetRanking:
    def test_ranks_models_correctly(self, sample_arena_df: pd.DataFrame) -> None:
        table = _subset_ranking(sample_arena_df, ["model-alpha", "model-beta", "model-gamma"])
        assert len(table.rows) == 3
        assert table.rows[0][1] == "model-alpha"
        assert table.rows[1][1] == "model-beta"
        assert table.rows[2][1] == "model-gamma"


@pytest.mark.unit
class TestCategoryTable:
    def test_produces_table_with_correct_columns(self, sample_arena_df: pd.DataFrame) -> None:
        table = _category_table(
            sample_arena_df,
            ["model-alpha", "model-beta"],
            ["overall", "coding", "math"],
        )
        assert "Model" in table.headers
        assert len(table.rows) == 2


@pytest.mark.unit
class TestHeadToHead:
    def test_computes_deltas(self, sample_arena_df: pd.DataFrame) -> None:
        h2h = _head_to_head(
            sample_arena_df,
            "model-alpha",
            "model-beta",
            ["overall", "coding", "math", "creative_writing"],
        )
        assert h2h.model_a == "model-alpha"
        assert h2h.model_b == "model-beta"
        assert len(h2h.deltas) == 4
        assert all(d == 50.0 for d in h2h.deltas)

    def test_symmetry(self, sample_arena_df: pd.DataFrame) -> None:
        h2h_ab = _head_to_head(sample_arena_df, "model-alpha", "model-beta", ["overall"])
        h2h_ba = _head_to_head(sample_arena_df, "model-beta", "model-alpha", ["overall"])
        assert h2h_ab.deltas[0] == -h2h_ba.deltas[0]

    def test_win_loss_counts(self, sample_arena_df: pd.DataFrame) -> None:
        h2h = _head_to_head(
            sample_arena_df,
            "model-alpha",
            "model-beta",
            ["overall", "coding", "math", "creative_writing"],
        )
        assert h2h.a_wins + h2h.b_wins + h2h.ties == 4


@pytest.mark.unit
class TestWinLossTable:
    def test_produces_table(self, sample_arena_df: pd.DataFrame) -> None:
        h2h = _head_to_head(
            sample_arena_df,
            "model-alpha",
            "model-beta",
            ["overall", "coding"],
        )
        table = _win_loss_table([h2h])
        assert len(table.rows) == 1
        assert table.headers[0] == "vs Model"


@pytest.mark.unit
class TestKeyCategories:
    def test_general_and_industry_cover_all(self) -> None:
        combined = set(GENERAL_CATEGORIES + INDUSTRY_CATEGORIES)
        assert combined == set(KEY_CATEGORIES)

    def test_no_overlap(self) -> None:
        overlap = set(GENERAL_CATEGORIES) & set(INDUSTRY_CATEGORIES)
        assert overlap == set()
