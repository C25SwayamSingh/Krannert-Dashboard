"""
Shared utilities for the Krannert dash project.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from dateutil import parser as date_parser

# Event category normalization -------------------------------------------------

CATEGORY_MAP = {
    "musical": "Theatre",
    "musicals": "Theatre",
    "opera": "Theatre",
    "ballet": "Theatre",
    "theatre": "Theatre",
    "play": "Theatre",
    "talk": "Talks",
    "lecture": "Talks",
    "panel": "Talks",
    "concert": "Concerts",
    "orchestra": "Concerts",
    "symphony": "Concerts",
    "choir": "Concerts",
    "chorus": "Concerts",
    "festival": "Festivals",
    "film": "Film",
}
CATEGORY_KEYWORDS = tuple(CATEGORY_MAP.items())
DEFAULT_CATEGORY = "Other (Unmapped)"


def categorize_event(raw: Optional[str], fallback: Optional[str] = None) -> str:
    """
    Collapse disparate event descriptors into broad, audience-friendly buckets.
    """
    if raw:
        candidate = raw.strip()
        if not candidate:
            return DEFAULT_CATEGORY
        key = candidate.lower()
        mapped = CATEGORY_MAP.get(key)
        if mapped:
            return mapped
        return candidate.title()

    if fallback:
        text = fallback.strip().lower()
        if not text:
            return DEFAULT_CATEGORY
        for keyword, mapped in CATEGORY_KEYWORDS:
            if keyword in text:
                return mapped
    return DEFAULT_CATEGORY


# Backwards compatibility for earlier naming
normalize_category = categorize_event


# Date helpers -----------------------------------------------------------------

PRE_WINDOW_END = datetime(2020, 3, 1)
POST_WINDOW_START = datetime(2021, 7, 1)


def coerce_datetime(value) -> Optional[datetime]:
    """Best-effort parsing that tolerates empty values."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return date_parser.parse(str(value))


def is_pre_window(sale_date: datetime) -> bool:
    return sale_date < PRE_WINDOW_END


def is_post_window(sale_date: datetime) -> bool:
    return sale_date >= POST_WINDOW_START


def window_label(sale_date: datetime) -> str:
    if is_pre_window(sale_date):
        return "pre"
    if is_post_window(sale_date):
        return "post"
    return "bridge"


# File system helpers ----------------------------------------------------------

def available_data_files(data_dir: Path, suffixes: Iterable[str] = (".csv",)) -> list[Path]:
    """
    Enumerate data files the analyst can pick instead of using the uploader.
    """
    if not data_dir.exists():
        return []
    suffix_set = {s.lower() for s in suffixes}
    return sorted(
        path for path in data_dir.iterdir() if path.suffix.lower() in suffix_set and path.is_file()
    )

