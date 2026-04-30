from __future__ import annotations

import json
from pathlib import Path

import pytest

from compare_models.resolver import (
    is_family,
    load_aliases,
    resolve_names_for_source,
    suggest_similar,
)


@pytest.fixture
def aliases_file(tmp_path: Path) -> Path:
    data = {
        "trinity": {
            "arena": ["trinity-large-preview"],
            "artificial_analysis": ["Trinity Large Thinking"],
            "family": True,
        },
        "qwen3-235b-a22b": {
            "arena": ["qwen3-235b-a22b"],
            "artificial_analysis": ["Qwen3 235B A22B"],
        },
    }
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(data))
    return p


@pytest.mark.unit
class TestLoadAliases:
    def test_loads_file(self, aliases_file: Path) -> None:
        aliases = load_aliases(aliases_file)
        assert "trinity" in aliases

    def test_missing_file(self, tmp_path: Path) -> None:
        aliases = load_aliases(tmp_path / "nonexistent.json")
        assert aliases == {}


@pytest.mark.unit
class TestResolveNames:
    def test_resolves_known_alias(self, aliases_file: Path) -> None:
        aliases = load_aliases(aliases_file)
        result = resolve_names_for_source(["trinity"], "arena", aliases)
        assert result == ["trinity-large-preview"]

    def test_passes_through_unknown(self, aliases_file: Path) -> None:
        aliases = load_aliases(aliases_file)
        result = resolve_names_for_source(["unknown-model"], "arena", aliases)
        assert result == ["unknown-model"]


@pytest.mark.unit
class TestIsFamily:
    def test_family_true(self, aliases_file: Path) -> None:
        aliases = load_aliases(aliases_file)
        assert is_family("trinity", aliases) is True

    def test_family_false(self, aliases_file: Path) -> None:
        aliases = load_aliases(aliases_file)
        assert is_family("qwen3-235b-a22b", aliases) is False


@pytest.mark.unit
class TestSuggestSimilar:
    def test_suggests_close_matches(self) -> None:
        known = ["trinity-large-preview", "qwen3-235b-a22b", "qwen3-32b"]
        suggestions = suggest_similar("trinity-large", known)
        assert "trinity-large-preview" in suggestions

    def test_no_matches(self) -> None:
        suggestions = suggest_similar("zzzzz", ["alpha", "beta"])
        assert suggestions == []
