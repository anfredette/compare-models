from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

DEFAULT_ALIASES_PATH = Path(__file__).parent.parent.parent / "data" / "model_aliases.json"


def load_aliases(path: Path | None = None) -> dict[str, Any]:
    aliases_path = path or DEFAULT_ALIASES_PATH
    if not aliases_path.exists():
        return {}
    with open(aliases_path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def resolve_names_for_source(
    user_inputs: list[str],
    source_name: str,
    aliases: dict[str, Any] | None = None,
) -> list[str]:
    if aliases is None:
        aliases = load_aliases()

    resolved: list[str] = []
    for name in user_inputs:
        name_lower = name.lower()
        if name_lower in aliases:
            entry = aliases[name_lower]
            source_names = entry.get(source_name, [])
            if source_names:
                resolved.extend(source_names)
            elif entry.get("family"):
                resolved.append(name)
            else:
                resolved.append(name)
        else:
            resolved.append(name)

    return resolved


def is_family(name: str, aliases: dict[str, Any] | None = None) -> bool:
    if aliases is None:
        aliases = load_aliases()
    entry = aliases.get(name.lower(), {})
    return bool(entry.get("family"))


def suggest_similar(name: str, known_names: list[str], n: int = 3) -> list[str]:
    return difflib.get_close_matches(name.lower(), [k.lower() for k in known_names], n=n)
