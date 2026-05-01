from __future__ import annotations

from pathlib import Path

import pytest

from compare_models.arena_client import load_cache, save_cache


@pytest.mark.unit
class TestCacheRoundTrip:
    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import compare_models.aa_client as mod

        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        rows = [
            {"model_name": "test-model", "category": "overall", "rating": 1400.0, "vote_count": 100}
        ]
        path = save_cache(rows)
        assert path.exists()

        loaded, fetched_at = load_cache()
        assert len(loaded) == 1
        assert loaded[0]["model_name"] == "test-model"
        assert fetched_at is not None

    def test_load_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import compare_models.aa_client as mod

        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        loaded, fetched_at = load_cache()
        assert loaded == []
        assert fetched_at is None

    def test_load_corrupt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import compare_models.aa_client as mod

        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        cache_dir = tmp_path / ".model_cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "arena_models.json"
        cache_file.write_text("not json{{{")
        loaded, fetched_at = load_cache()
        assert loaded == []
        assert fetched_at is None
