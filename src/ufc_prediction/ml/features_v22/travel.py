"""v2.2 TRAVEL features (haversine miles + signed-continuous tz_shift).

Phase 23 Plan 23-02 TRAVEL-V22-02 + TRAVEL-V22-03. Six per-fight TRAVEL
cols at FEATURE_COLUMNS_V22 indices 75-80:
    - travel_distance_miles_red / blue / diff
    - tz_shift_red_signed / blue_signed / diff_signed

Pure-Python compute helpers. NO geopy/pytz/timezonefinder runtime imports
per CONTEXT D-05. ``zoneinfo`` (PEP 615 stdlib, Python 3.9+) + ``tzdata``
runtime fallback (Phase 22 D-07; the system tz database on macOS / Linux
suffices in normal deployment).

First-fight fighter sentinel = 0 for both travel_distance and tz_shift per
CONTEXT D-04 (``is_debut_*`` explicit flags deferred to v2.3+ Backlog).

Both ``feature_matrix.py`` (training path) and ``inference_features.py``
(live-prediction path) import from this module so train/inference math is
byte-identical (Pitfall #12 LIVE-03 parity per CONTEXT D-10).

Pitfall #4 (DST evaluation): ``tz_offset_hours`` evaluates the IANA tz at
``event_date`` (NOT ``date.today()``) — DST transitions are correctly
resolved per-event. Cross-DST fights (e.g. an October LA fight following
a March LA fight) produce a non-zero ``tz_shift_signed`` because the UTC
offset of the same venue can differ across calendar dates.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

EARTH_RADIUS_MILES: float = 3958.7613  # Earth mean radius in statute miles


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles. Pure-Python (``math`` stdlib only).

    Per CONTEXT D-05 + TRAVEL-V22-02 requirements: no ``geopy`` runtime dep
    (Phase 22 used ``geopy.Nominatim`` for the one-off geocode backfill;
    that is a build-time-only dep). Pre-fight as-of-date discipline is the
    CALLER's responsibility; this function just computes raw distance.

    Earth radius constant = 3958.7613 mi (mean radius); do NOT substitute
    equatorial (3963.19) or polar (3949.90) — the mean is the standard
    haversine convention.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def tz_offset_hours(tz_iana: str, event_date: date) -> float:
    """UTC offset for ``tz_iana`` at ``event_date`` in signed hours.

    ``zoneinfo`` + ``tzdata`` stdlib path per CONTEXT D-05. DST evaluated AT
    ``event_date`` (Pitfall #4): naive datetime at midnight of ``event_date``.
    Returns hours as float (e.g. ``-7.0`` PDT, ``-8.0`` PST, ``+9.0`` JST,
    ``+5.5`` IST).
    """
    naive_midnight = datetime.combine(event_date, datetime.min.time())
    offset = ZoneInfo(tz_iana).utcoffset(naive_midnight)
    if offset is None:  # pragma: no cover — closed-set venue tz_iana
        raise ValueError(f"ZoneInfo({tz_iana!r}) has no utcoffset for {event_date}")
    return offset.total_seconds() / 3600.0


def compute_tz_shift_signed(
    prior_tz_iana: str | None,
    prior_event_date: date | None,
    curr_tz_iana: str,
    curr_event_date: date,
) -> float:
    """Signed continuous tz_shift in hours; positive = traveling east.

    First-fight fighter (``prior_tz_iana is None`` OR ``prior_event_date is
    None``) returns 0.0 sentinel per CONTEXT D-04.

    Sign convention: ``curr_offset - prior_offset``. Eastward travel (larger
    UTC offset) → positive. Westward (smaller / more-negative offset) →
    negative.
    """
    if prior_tz_iana is None or prior_event_date is None:
        return 0.0
    prior_offset = tz_offset_hours(prior_tz_iana, prior_event_date)
    curr_offset = tz_offset_hours(curr_tz_iana, curr_event_date)
    return curr_offset - prior_offset


def compute_travel_features(
    prior_venue: dict[str, Any] | None,
    current_venue: dict[str, Any],
    event_date: date,
) -> dict[str, float]:
    """Compose haversine + tz_shift for one fighter's leg.

    Args:
        prior_venue: ``{"lat", "lon", "timezone_iana", "event_date"}`` from
            the fighter's most recent prior fight (Q1 RESOLVED in
            RESEARCH.md). ``None`` = first UFC fight for this fighter →
            both values = 0 per CONTEXT D-04.
        current_venue: ``{"lat", "lon", "timezone_iana"}`` from this
            fight's event.
        event_date: This fight's event date (used for DST resolution of
            ``current_venue.timezone_iana``).

    Returns:
        Dict with 2 keys: ``travel_distance_miles`` and ``tz_shift_signed``.
        Caller composes the 3 per-side cols (red/blue/diff) by calling this
        helper twice (once per fighter) and subtracting.
    """
    if prior_venue is None:
        return {"travel_distance_miles": 0.0, "tz_shift_signed": 0.0}
    return {
        "travel_distance_miles": haversine_miles(
            prior_venue["lat"],
            prior_venue["lon"],
            current_venue["lat"],
            current_venue["lon"],
        ),
        "tz_shift_signed": compute_tz_shift_signed(
            prior_venue.get("timezone_iana"),
            prior_venue.get("event_date"),
            current_venue["timezone_iana"],
            event_date,
        ),
    }


# ── Phase 42 Plan 42-01 v2.5 TRAVEL additive siblings ─────────────────────
#
# Phase 42 D-21 (TRAVEL-V25-01) + D-22 (TRAVEL-V25-02): NEW primitives beyond
# the v2.2 TRAVEL block — kilometer-scale Haversine (NOT miles) and ±12-clipped
# UTC tz_shift (NOT continuous), both with NaN-debut sentinels (NOT 0.0).
#
# Sibling-not-replacement discipline: the v2.2 exports above (haversine_miles,
# tz_offset_hours, compute_tz_shift_signed, compute_travel_features, plus the
# EARTH_RADIUS_MILES constant) MUST stay byte-stable — they're inside
# meta_v2.joblib's input space (FEATURE_COLUMNS_V22 indices 75-80). Touching
# them would invalidate AUDIT-01 SHA invariants.
#
# These new helpers are consumed only by the candidate META-V22+CALIB+TRAVEL
# blender (Plan 42-02) — NOT by xgb_v2 or meta_v2 at inference time.

EARTH_RADIUS_KM: float = 6371.0088  # IUGG mean radius in kilometers
TZ_SHIFT_CAP_HOURS: float = 12.0  # CONTEXT D-22 — symmetric ±12 clip


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers. Pure-Python (``math`` stdlib only).

    Per CONTEXT D-21 + TRAVEL-V25-01 locked decision: kilometers (NOT miles).
    Pre-fight as-of-date discipline is the CALLER's responsibility.

    Earth radius constant = 6371.0088 km (IUGG mean radius); do NOT substitute
    equatorial (6378.137) or polar (6356.752) — the mean is the standard
    haversine convention. Mirror of EARTH_RADIUS_MILES discipline.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def compute_tz_shift_hours_clipped(
    prior_tz_iana: str | None,
    prior_event_date: date | None,
    curr_tz_iana: str,
    curr_event_date: date,
) -> float:
    """Clipped signed tz_shift in hours; NaN sentinel for debut.

    Per CONTEXT D-22:
        - Sign convention: ``curr_offset - prior_offset``. Positive = traveling
          east (clock advances). Matches circadian-research convention.
        - Cap: clip to ``[-12.0, +12.0]`` hours. Anything beyond is unphysical
          / data error.
        - Debut (``prior_tz_iana is None`` OR ``prior_event_date is None``)
          returns ``math.nan`` (NOT 0.0 — this is the v2.5 difference from
          v2.2 ``compute_tz_shift_signed`` which returns 0.0 per Phase 23 D-04).
        - DST evaluated per ``event_date`` (correct circadian-research
          semantics; mirrors existing ``tz_offset_hours`` discipline).
    """
    if prior_tz_iana is None or prior_event_date is None:
        return math.nan
    prior_offset = tz_offset_hours(prior_tz_iana, prior_event_date)
    curr_offset = tz_offset_hours(curr_tz_iana, curr_event_date)
    raw = curr_offset - prior_offset
    if raw > TZ_SHIFT_CAP_HOURS:
        return TZ_SHIFT_CAP_HOURS
    if raw < -TZ_SHIFT_CAP_HOURS:
        return -TZ_SHIFT_CAP_HOURS
    return raw


def compute_travel_v25_features(
    prior_venue: dict[str, Any] | None,
    current_venue: dict[str, Any],
    event_date: date,
) -> dict[str, float]:
    """v2.5 TRAVEL features (km + clipped hours) for one fighter's leg.

    Per CONTEXT D-21 (TRAVEL-V25-01) + D-22 (TRAVEL-V25-02):
        - kilometers (not miles), NaN debut sentinel (not 0.0).
        - Clip tz_shift to ±12 hours.

    Args mirror the v2.2 ``compute_travel_features`` API: ``prior_venue`` is
    a dict ``{"lat", "lon", "timezone_iana", "event_date"}`` or ``None``
    (debut); ``current_venue`` is the same shape (event_date optional —
    not used); ``event_date`` is the current fight date for DST resolution.

    Returns:
        ``{"travel_distance_km": float, "tz_shift_hours": float}`` —
        both ``math.nan`` when ``prior_venue is None``.
    """
    if prior_venue is None:
        return {
            "travel_distance_km": math.nan,
            "tz_shift_hours": math.nan,
        }
    return {
        "travel_distance_km": haversine_km(
            prior_venue["lat"],
            prior_venue["lon"],
            current_venue["lat"],
            current_venue["lon"],
        ),
        "tz_shift_hours": compute_tz_shift_hours_clipped(
            prior_venue.get("timezone_iana"),
            prior_venue.get("event_date"),
            current_venue["timezone_iana"],
            event_date,
        ),
    }
