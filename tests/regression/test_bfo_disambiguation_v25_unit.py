"""Unit-tier regression for BFO disambiguation anomaly (BFO-V25-03).

Validates the Plan 41-02 fix to ``BFOOddsIngester._match_fighters``: when a
``BFOFighterName`` row carries a canonical ``database_id`` (sourced from the
operator-curated alias substrate in ``data/bfo/fighters_names.csv``), the
ingester MUST use it directly as the DB Fighter.id rather than re-running
fuzzy name matching. Fuzzy fallback is preserved only when ``database_id`` is
blank/None.

This regression freezes the post-fix behavior against ≥5 fixture rows drawn
from the Plan 41-01 cascade diagnosis (Sec 2.6 ``bfo_source_cascade``):
1+ row per anomaly year (2011 / 2013 / 2019 / 2020) plus control rows from
non-anomaly years (2018 / 2022) for no-regression assertion.

Pure-Python — no DB, no network, no rapidfuzz randomness. The session is
mocked to return the candidate pool deterministically.
"""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester, IngestSummary
from ufc_prediction.scraper.bfo_models import BFOFighterName

# ── Fixture loading ─────────────────────────────────────────────────────────

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "bfo_disambiguation_v25"
_ANOMALY_ROWS: list[dict[str, Any]] = json.loads((_FIXTURE_DIR / "anomaly_rows.json").read_text())
_CONTROL_ROWS: list[dict[str, Any]] = json.loads((_FIXTURE_DIR / "control_rows.json").read_text())


def _build_ingester_with_pool(
    candidate_pool: list[list[Any]],
) -> BFOOddsIngester:
    """Construct a BFOOddsIngester whose session returns ``candidate_pool``.

    The fixture stores ``candidate_pool`` as JSON arrays ``[[id, name], ...]``.
    SQLAlchemy returns ``Row`` objects from ``session.execute(...).all()`` that
    support attribute access (``row.id``/``row.name``), so mock with a namedtuple
    rather than a plain tuple (which only supports index access).
    """
    session = MagicMock()
    _Row = namedtuple("_Row", ["id", "name"])
    pool_rows = [_Row(*row) for row in candidate_pool]
    session.execute.return_value.all.return_value = pool_rows

    # data_folder is unused in _match_fighters; tmp_path-like dummy is fine.
    ingester = BFOOddsIngester(
        session=session,
        data_folder=Path("/tmp/bfo-disambig-unit-test-unused"),
    )
    return ingester


def _row_to_bfo_fighter_name(row: dict[str, Any]) -> BFOFighterName:
    """Construct a single BFOFighterName from a fixture row."""
    return BFOFighterName(
        fighter_id=row["bfo_fighter_id"],
        database=row["database"],
        name=row["bfo_name"],
        database_id=row.get("database_id"),
    )


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "row",
    _ANOMALY_ROWS,
    ids=[f"anomaly-{r['year']}-{r['bfo_fighter_id']}" for r in _ANOMALY_ROWS],
)
def test_anomaly_row_resolves_to_canonical_fighter_id(
    row: dict[str, Any],
) -> None:
    """Anomaly fixture: post-fix MUST resolve to expected_canonical_fighter_id.

    For rows where ``database_id`` is present, the fix uses it directly,
    bypassing the fuzzy matcher. For the fallback fixture
    (``database_id`` is None), the fuzzy matcher is exercised and MUST
    still return the expected id.
    """
    ingester = _build_ingester_with_pool(row["candidate_pool"])
    summary = IngestSummary()
    bfo_names = {
        row["bfo_fighter_id"]: [_row_to_bfo_fighter_name(row)],
    }

    bfo_to_db = ingester._match_fighters(bfo_names, summary)

    assert bfo_to_db.get(row["bfo_fighter_id"]) == row["expected_canonical_fighter_id"], (
        f"Anomaly fixture year={row['year']} bfo_fighter_id={row['bfo_fighter_id']} "
        f"expected DB id={row['expected_canonical_fighter_id']} but got "
        f"{bfo_to_db.get(row['bfo_fighter_id'])!r}. Notes: {row['notes']}"
    )


@pytest.mark.parametrize(
    "row",
    _CONTROL_ROWS,
    ids=[f"control-{r['year']}-{r['bfo_fighter_id']}" for r in _CONTROL_ROWS],
)
def test_control_row_resolves_identically_post_fix(
    row: dict[str, Any],
) -> None:
    """Control fixture (non-anomaly year): post-fix MUST NOT regress.

    The control rows have a canonical ``database_id`` that already resolves
    to the right DB id pre-fix via fuzzy matching (since these are healthy
    post-2018 cohorts). The fix MUST preserve the same result.
    """
    ingester = _build_ingester_with_pool(row["candidate_pool"])
    summary = IngestSummary()
    bfo_names = {
        row["bfo_fighter_id"]: [_row_to_bfo_fighter_name(row)],
    }

    bfo_to_db = ingester._match_fighters(bfo_names, summary)

    assert bfo_to_db.get(row["bfo_fighter_id"]) == row["expected_canonical_fighter_id"], (
        f"Control fixture year={row['year']} bfo_fighter_id={row['bfo_fighter_id']} "
        f"REGRESSED post-fix: expected DB id={row['expected_canonical_fighter_id']} "
        f"but got {bfo_to_db.get(row['bfo_fighter_id'])!r}. Notes: {row['notes']}"
    )


def test_canonical_path_bypasses_fuzzy_matcher_when_database_id_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavior assertion: when database_id is present, match_bfo_name is NOT called.

    This is the core fix — Plan 41-01 root cause is that _match_fighters re-fuzz-matches
    despite the canonical column being present. Post-fix, the fuzzy matcher MUST be
    skipped entirely on the canonical path.
    """
    call_counter = {"calls": 0}

    def _spy_match_bfo_name(*args: Any, **kwargs: Any) -> Any:
        call_counter["calls"] += 1
        return None

    monkeypatch.setattr("ufc_prediction.scraper.bfo_ingest.match_bfo_name", _spy_match_bfo_name)

    # Pick the first anomaly fixture with a non-null database_id.
    row = next(r for r in _ANOMALY_ROWS if r.get("database_id"))
    ingester = _build_ingester_with_pool(row["candidate_pool"])
    summary = IngestSummary()
    bfo_names = {row["bfo_fighter_id"]: [_row_to_bfo_fighter_name(row)]}

    bfo_to_db = ingester._match_fighters(bfo_names, summary)

    assert bfo_to_db[row["bfo_fighter_id"]] == row["expected_canonical_fighter_id"]
    assert call_counter["calls"] == 0, (
        "Post-fix: match_bfo_name MUST NOT be invoked when canonical "
        f"database_id is present. Got {call_counter['calls']} call(s)."
    )


def test_fallback_path_uses_fuzzy_matcher_when_database_id_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavior assertion: when database_id is None, match_bfo_name IS called.

    Verifies the fix does NOT short-circuit cases where the alias substrate
    has no canonical id — those rows still need the existing fuzzy fallback.
    """
    call_counter = {"calls": 0}

    def _spy_match_bfo_name(
        bfo_name: str, candidates: list[tuple[int, str]], threshold: int = 80
    ) -> tuple[int, str] | None:
        call_counter["calls"] += 1
        # Mimic real matcher: return first candidate (deterministic).
        if candidates:
            return candidates[0]
        return None

    monkeypatch.setattr("ufc_prediction.scraper.bfo_ingest.match_bfo_name", _spy_match_bfo_name)

    row = next(r for r in _ANOMALY_ROWS if r.get("database_id") is None)
    ingester = _build_ingester_with_pool(row["candidate_pool"])
    summary = IngestSummary()
    bfo_names = {row["bfo_fighter_id"]: [_row_to_bfo_fighter_name(row)]}

    bfo_to_db = ingester._match_fighters(bfo_names, summary)

    assert bfo_to_db[row["bfo_fighter_id"]] == row["expected_canonical_fighter_id"]
    assert call_counter["calls"] == 1, (
        "Fuzzy-fallback path: match_bfo_name MUST be invoked exactly once when "
        f"canonical database_id is None. Got {call_counter['calls']} call(s)."
    )


def test_ingest_summary_tracks_canonical_vs_fuzzy_split() -> None:
    """IngestSummary observability: canonical-path and fuzzy-path counts split.

    The Plan 41-01 implementation contract requires an IngestSummary counter
    so Plan 41-03 backfill can assert canonical-path dominance on anomaly years.
    """
    # Combine one canonical-path anomaly row + one fallback row.
    canonical_row = next(r for r in _ANOMALY_ROWS if r.get("database_id"))
    fallback_row = next(r for r in _ANOMALY_ROWS if r.get("database_id") is None)

    merged_pool = canonical_row["candidate_pool"] + fallback_row["candidate_pool"]
    ingester = _build_ingester_with_pool(merged_pool)
    summary = IngestSummary()
    bfo_names = {
        canonical_row["bfo_fighter_id"]: [_row_to_bfo_fighter_name(canonical_row)],
        fallback_row["bfo_fighter_id"]: [_row_to_bfo_fighter_name(fallback_row)],
    }

    ingester._match_fighters(bfo_names, summary)

    assert summary.fighters_matched == 2, (
        f"Expected 2 fighters matched; got {summary.fighters_matched}"
    )
    # Per Plan 41-01 contract: split canonical vs fuzzy counts.
    assert hasattr(summary, "fighters_matched_canonical"), (
        "IngestSummary missing 'fighters_matched_canonical' counter (Plan 41-01 contract)"
    )
    assert hasattr(summary, "fighters_matched_fuzzy"), (
        "IngestSummary missing 'fighters_matched_fuzzy' counter (Plan 41-01 contract)"
    )
    assert summary.fighters_matched_canonical == 1, (
        f"Expected 1 canonical-path match; got {summary.fighters_matched_canonical}"
    )
    assert summary.fighters_matched_fuzzy == 1, (
        f"Expected 1 fuzzy-path match; got {summary.fighters_matched_fuzzy}"
    )
