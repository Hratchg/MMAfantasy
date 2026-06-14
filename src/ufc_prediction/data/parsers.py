"""Parsing utilities for Kaggle UFC CSV data formats.

Handles all string-to-typed-value conversions needed by the rajeevw/ufcdata
and mdabbert/ultimate-ufc-dataset Kaggle datasets.
"""

from __future__ import annotations

import re
from datetime import date, datetime

# ── Sentinel values treated as missing ────────────────────────────────────────

_MISSING = frozenset({"", "---", "--", "nan", "None"})


def _is_missing(value: str | None) -> bool:
    """Return True if value is None or a known missing-data sentinel."""
    if value is None:
        return True
    return value.strip() in _MISSING


# ── Strike / attempt parsers ─────────────────────────────────────────────────

_X_OF_Y_RE = re.compile(r"(\d+)\s+of\s+(\d+)")


def parse_x_of_y(value: str | None) -> tuple[int | None, int | None]:
    """Parse 'X of Y' format into (landed, attempted).

    Examples:
        parse_x_of_y("69 of 101")  -> (69, 101)
        parse_x_of_y("0 of 0")     -> (0, 0)
        parse_x_of_y("")           -> (None, None)
        parse_x_of_y("---")        -> (None, None)
    """
    if value is None or _is_missing(value):
        return (None, None)
    match = _X_OF_Y_RE.search(value)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (None, None)


# ── Time parsers ─────────────────────────────────────────────────────────────


def parse_control_time(value: str | None) -> int | None:
    """Parse 'M:SS' control time to total seconds.

    Examples:
        parse_control_time("2:15")  -> 135
        parse_control_time("0:00")  -> 0
        parse_control_time("")      -> None
    """
    if value is None or _is_missing(value):
        return None
    parts = value.strip().split(":")
    if len(parts) == 2:
        try:
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        except ValueError:
            return None
    return None


# ── Percentage parser ────────────────────────────────────────────────────────


def parse_percentage(value: str | None) -> float | None:
    """Parse percentage string to decimal float.

    Examples:
        parse_percentage("65%")  -> 0.65
        parse_percentage("0%")   -> 0.0
        parse_percentage("")     -> None
    """
    if value is None or _is_missing(value):
        return None
    cleaned = value.strip().rstrip("%")
    if not cleaned:
        return None
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return None


# ── Physical measurement parsers ─────────────────────────────────────────────

_HEIGHT_RE = re.compile(r"(\d+)'\s*(\d+)")


def parse_height_to_inches(value: str | None) -> float | None:
    """Parse height string like '5\\' 10\"' to inches.

    Examples:
        parse_height_to_inches("5' 10\\"")  -> 70.0
        parse_height_to_inches("6' 2\\"")   -> 74.0
        parse_height_to_inches("--")        -> None
    """
    if value is None or _is_missing(value):
        return None
    match = _HEIGHT_RE.search(value)
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return float(feet * 12 + inches)
    return None


def parse_reach_to_inches(value: str | None) -> float | None:
    """Parse reach string like '72\"' or '72.0' to float inches.

    Examples:
        parse_reach_to_inches('72\"')  -> 72.0
        parse_reach_to_inches("72.0") -> 72.0
        parse_reach_to_inches("")     -> None
    """
    if value is None or _is_missing(value):
        return None
    cleaned = value.strip().rstrip('"')
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def cm_to_inches(cm: float) -> float:
    """Convert centimeters to inches.

    Used for the mdabbert dataset which stores height/reach in cm.

    Examples:
        cm_to_inches(180.0)  -> ~70.87
        cm_to_inches(0.0)    -> 0.0
    """
    return cm / 2.54


# ── Weight class parser ──────────────────────────────────────────────────────

# All 14 valid UFC weight classes, ordered longest-name-first for greedy matching
_WEIGHT_CLASSES = [
    "Women's Strawweight",
    "Women's Featherweight",
    "Women's Bantamweight",
    "Women's Flyweight",
    "Light Heavyweight",
    "Catch Weight",
    "Open Weight",
    "Featherweight",
    "Bantamweight",
    "Welterweight",
    "Middleweight",
    "Heavyweight",
    "Lightweight",
    "Flyweight",
]


def parse_weight_class(fight_type: str | None) -> tuple[str, bool]:
    """Parse the Fight_type column into (weight_class, is_title_fight).

    Strips 'UFC' prefix, detects 'Title' keyword, and matches against
    known weight classes.

    Examples:
        parse_weight_class("UFC Heavyweight Title Bout")     -> ("Heavyweight", True)
        parse_weight_class("Women's Strawweight Bout")       -> ("Women's Strawweight", False)
        parse_weight_class("Middleweight Bout")              -> ("Middleweight", False)
    """
    if fight_type is None:
        return ("Unknown", False)

    text = fight_type.strip()
    is_title = "title" in text.lower()

    for wc in _WEIGHT_CLASSES:
        if wc.lower() in text.lower():
            return (wc, is_title)

    return ("Unknown", is_title)


# ── Date parsers ─────────────────────────────────────────────────────────────

_DATE_FORMATS = ["%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"]


def parse_fight_date(value: str | None) -> date | None:
    """Parse fight date from multiple formats.

    Supports:
        - "March 28, 2026"  (rajeevw dataset)
        - "2026-03-28"      (mdabbert dataset)
        - "Jan 01, 2020"    (abbreviated month)

    Returns None on failure.
    """
    if value is None or _is_missing(value):
        return None
    cleaned = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def parse_dob(value: str | None) -> date | None:
    """Parse date of birth from raw_fighter_details.csv.

    Supports:
        - "Jul 13, 1978"    (abbreviated month)
        - "January 28, 1986" (full month)
        - "1990-05-15"      (ISO)

    Returns None on failure.
    """
    if value is None or _is_missing(value):
        return None
    cleaned = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


# ── Round format parser ──────────────────────────────────────────────────────

_ROUNDS_RE = re.compile(r"(\d+)\s+Rnd")


def parse_num_rounds(format_str: str | None) -> int:
    """Parse round format string to number of rounds.

    Examples:
        parse_num_rounds("3 Rnd (5-5-5)")       -> 3
        parse_num_rounds("5 Rnd (5-5-5-5-5)")   -> 5
        parse_num_rounds("No Time Limit")        -> 3 (default)

    Returns 3 as default when format is unparseable.
    """
    if format_str is None:
        return 3
    match = _ROUNDS_RE.search(format_str)
    if match:
        return int(match.group(1))
    return 3
