from __future__ import annotations

import json
from pathlib import Path

import pytest

from compare_models.sources.artificial_analysis import (
    AAModel,
    _comparison_table,
    _compute_findings,
    _consolidated_ranking_table,
    _load_models,
    _match_models,
)


@pytest.mark.unit
class TestAAModel:
    def test_blended_price_computed(self) -> None:
        m = AAModel(
            name="Test",
            slug="test",
            organization="Org",
            intelligence_index=30,
            input_price_per_1m=0.20,
            output_price_per_1m=0.80,
        )
        assert m.blended_price == 0.35

    def test_blended_price_api_preferred(self) -> None:
        m = AAModel(
            name="Test",
            slug="test",
            organization="Org",
            intelligence_index=30,
            input_price_per_1m=0.20,
            output_price_per_1m=0.80,
            blended_price_api=0.50,
        )
        assert m.blended_price == 0.50

    def test_blended_price_none(self) -> None:
        m = AAModel(
            name="Test",
            slug="test",
            organization="Org",
            intelligence_index=30,
        )
        assert m.blended_price is None

    def test_params_display_moe(self) -> None:
        m = AAModel(
            name="Test",
            slug="test",
            organization="Org",
            intelligence_index=30,
            params_total_b=399,
            params_active_b=13,
        )
        assert m.params_display == "399B / 13B"

    def test_params_display_dense(self) -> None:
        m = AAModel(
            name="Test",
            slug="test",
            organization="Org",
            intelligence_index=30,
            params_total_b=72,
            params_active_b=72,
        )
        assert m.params_display == "72B"

    def test_params_display_proprietary(self) -> None:
        m = AAModel(
            name="Test",
            slug="test",
            organization="Org",
            intelligence_index=30,
        )
        assert m.params_display == "proprietary"


@pytest.mark.unit
class TestLoadModels:
    def test_loads_from_file(self, tmp_path: Path) -> None:
        data = [
            {
                "name": "Test Model",
                "slug": "test-model",
                "organization": "Org",
                "intelligence_index": 30,
            }
        ]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        models, status = _load_models(p)
        assert len(models) == 1
        assert models[0].name == "Test Model"
        assert "loaded from" in status


@pytest.mark.unit
class TestMatchModels:
    def test_exact_match(self, sample_aa_models: list[AAModel]) -> None:
        found, not_found = _match_models(sample_aa_models, ["Alpha Thinking"])
        assert len(found) == 1
        assert found[0].name == "Alpha Thinking"

    def test_slug_match(self, sample_aa_models: list[AAModel]) -> None:
        found, _ = _match_models(sample_aa_models, ["beta-large"])
        assert len(found) == 1
        assert found[0].name == "Beta Large"

    def test_family_match(self, sample_aa_models: list[AAModel]) -> None:
        found, _ = _match_models(sample_aa_models, ["Beta"], families=True)
        assert len(found) == 2

    def test_not_found(self, sample_aa_models: list[AAModel]) -> None:
        _, not_found = _match_models(sample_aa_models, ["nonexistent"])
        assert not_found == ["nonexistent"]


@pytest.mark.unit
class TestComparisonTable:
    def test_sorted_by_intelligence(self, sample_aa_models: list[AAModel]) -> None:
        table = _comparison_table(sample_aa_models, title="Test")
        assert table.rows[0][0] == "Alpha Thinking"

    def test_headers_with_params(self, sample_aa_models: list[AAModel]) -> None:
        table = _comparison_table(sample_aa_models, title="Test", include_params=True)
        assert "Params (total/active)" in table.headers

    def test_headers_without_params(self, sample_aa_models: list[AAModel]) -> None:
        table = _comparison_table(sample_aa_models, title="Test", include_params=False)
        assert "Params (total/active)" not in table.headers


@pytest.mark.unit
class TestGlobalRanking:
    def test_returns_consolidated_table(self, sample_aa_models: list[AAModel]) -> None:
        targets = [m for m in sample_aa_models if m.name == "Beta Large"]
        table = _consolidated_ranking_table(sample_aa_models, targets, window=2)
        assert "Global AA Rankings" in table.title
        names = [row[1] for row in table.rows]
        assert any("Beta Large" in n for n in names)

    def test_no_models_found(self, sample_aa_models: list[AAModel]) -> None:
        table = _consolidated_ranking_table(sample_aa_models, [], window=2)
        assert "no models found" in table.title


@pytest.mark.unit
class TestFindings:
    def test_produces_findings(self, sample_aa_models: list[AAModel]) -> None:
        findings = _compute_findings(sample_aa_models, sample_aa_models)
        assert len(findings) > 0

    def test_empty_input(self) -> None:
        findings = _compute_findings([], [])
        assert findings == ["No matching models found in AA data."]
