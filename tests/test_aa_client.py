from __future__ import annotations

from pathlib import Path

import pytest

from compare_models.aa_client import (
    _infer_reasoning,
    _map_api_model,
    cache_age_display,
    load_cache,
    save_cache,
)


@pytest.mark.unit
class TestMapApiModel:
    def test_basic_mapping(self) -> None:
        api_obj = {
            "name": "Test Model",
            "slug": "test-model",
            "model_creator": {"name": "TestOrg"},
            "evaluations": {
                "artificial_analysis_intelligence_index": 42,
                "artificial_analysis_coding_index": 35,
                "artificial_analysis_math_index": 28,
            },
            "median_output_tokens_per_second": 100.5,
            "median_time_to_first_token_seconds": 0.85,
            "pricing": {
                "price_1m_input_tokens": 1.0,
                "price_1m_output_tokens": 3.0,
                "price_1m_blended_3_to_1": 1.5,
            },
        }
        result = _map_api_model(api_obj)
        assert result["name"] == "Test Model"
        assert result["slug"] == "test-model"
        assert result["organization"] == "TestOrg"
        assert result["intelligence_index"] == 42
        assert result["coding_index"] == 35
        assert result["math_index"] == 28
        assert result["speed_tps"] == 100.5
        assert result["ttft_s"] == 0.85
        assert result["input_price_per_1m"] == 1.0
        assert result["output_price_per_1m"] == 3.0
        assert result["blended_price_api"] == 1.5
        assert result["url"] == "https://artificialanalysis.ai/models/test-model"

    def test_missing_fields(self) -> None:
        api_obj = {"name": "Minimal", "slug": "minimal"}
        result = _map_api_model(api_obj)
        assert result["name"] == "Minimal"
        assert result["organization"] == "Unknown"
        assert result["intelligence_index"] is None
        assert result["coding_index"] is None
        assert result["speed_tps"] is None

    def test_reasoning_from_api(self) -> None:
        api_obj = {"name": "Test", "slug": "test", "reasoning": True}
        result = _map_api_model(api_obj)
        assert result["reasoning"] is True

    def test_reasoning_inferred(self) -> None:
        api_obj = {"name": "GPT-5 Thinking", "slug": "gpt-5-thinking"}
        result = _map_api_model(api_obj)
        assert result["reasoning"] is True


@pytest.mark.unit
class TestInferReasoning:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("GPT-5 Thinking", True),
            ("Claude Reasoning", True),
            ("o1 (high)", True),
            ("o1 (low)", True),
            ("o1 (xhigh)", True),
            ("GPT-4o", False),
            ("Gemini Pro", False),
        ],
    )
    def test_patterns(self, name: str, expected: bool) -> None:
        assert _infer_reasoning(name) == expected


@pytest.mark.unit
class TestCacheRoundTrip:
    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPARE_MODELS_CACHE_DIR", str(tmp_path))
        models = [
            {"name": "Test", "slug": "test", "organization": "Org", "intelligence_index": 30}
        ]
        path = save_cache(models)
        assert path.exists()

        loaded, fetched_at = load_cache()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "Test"
        assert fetched_at is not None

    def test_load_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPARE_MODELS_CACHE_DIR", str(tmp_path))
        loaded, fetched_at = load_cache()
        assert loaded == []
        assert fetched_at is None

    def test_load_corrupt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPARE_MODELS_CACHE_DIR", str(tmp_path))
        cache_file = tmp_path / "aa_models.json"
        cache_file.write_text("not json{{{")
        loaded, fetched_at = load_cache()
        assert loaded == []
        assert fetched_at is None


@pytest.mark.unit
class TestCacheAgeDisplay:
    def test_just_now(self) -> None:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        assert cache_age_display(now) == "just now"

    def test_invalid(self) -> None:
        assert cache_age_display("not-a-date") == "unknown age"

    def test_hours_ago(self) -> None:
        from datetime import UTC, datetime, timedelta

        two_hours_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert cache_age_display(two_hours_ago) == "2 hours ago"

    def test_days_ago(self) -> None:
        from datetime import UTC, datetime, timedelta

        three_days_ago = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        assert cache_age_display(three_days_ago) == "3 days ago"
