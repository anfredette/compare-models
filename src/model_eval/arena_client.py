"""Arena leaderboard cache management."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from model_eval.aa_client import get_cache_dir

logger = logging.getLogger(__name__)


def get_cache_path() -> Path:
    return get_cache_dir() / "arena_models.json"


def fetch_from_hf() -> list[dict]:
    """Fetch the Arena leaderboard dataset from HuggingFace."""
    ds = load_dataset("lmarena-ai/leaderboard-dataset", "text_style_control", split="latest")
    df = pd.DataFrame(ds)
    return df.to_dict(orient="records")  # type: ignore[return-value]


def save_cache(rows: list[dict]) -> Path:
    """Write rows to cache file using atomic write."""
    cache_path = get_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "rows": rows,
    }

    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(envelope, f)
        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return cache_path


def load_cache() -> tuple[list[dict], str | None]:
    """Load rows from cache. Returns (row_dicts, fetched_at) or ([], None) if no cache."""
    cache_path = get_cache_path()
    if not cache_path.exists():
        return [], None
    try:
        with open(cache_path) as f:
            envelope = json.load(f)
        return envelope.get("rows", []), envelope.get("fetched_at")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Corrupt cache file %s: %s", cache_path, e)
        return [], None


def sync() -> tuple[int, Path]:
    """Fetch from HuggingFace and save to cache. Returns (row_count, cache_path)."""
    logger.info("Fetching Arena leaderboard from HuggingFace...")
    rows = fetch_from_hf()
    logger.info("Received %d rows", len(rows))
    cache_path = save_cache(rows)
    return len(rows), cache_path
