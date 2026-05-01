from __future__ import annotations

import pytest

from compare_models.resolver import suggest_similar


@pytest.mark.unit
class TestSuggestSimilar:
    def test_suggests_close_matches(self) -> None:
        known = ["trinity-large-preview", "qwen3-235b-a22b", "qwen3-32b"]
        suggestions = suggest_similar("trinity-large", known)
        assert "trinity-large-preview" in suggestions

    def test_no_matches(self) -> None:
        suggestions = suggest_similar("zzzzz", ["alpha", "beta"])
        assert suggestions == []
