"""BestFightOdds ingestion orchestrator.

Reads ufcscraper's CSV output from ``data_folder``, fuzzy-matches BFO
fighter names to the database, computes proportional vig-removed
implied probabilities (D-02), and upserts per-(fight, fighter) rows
into the ``fight_odds`` table.

Design notes:
- Name matching reuses ``bfo_matcher`` (Plan 01) — threshold 80 default.
- Fight matching uses the fighter PAIR + closest-date matching against
  the BFO row's ``event_date`` (parsed from the composite ``fight_id``
  prefix), constrained by ``date_window_days`` (Pitfall 1, fix 999.4 —
  before this fix, every BFO row for a rematch pair resolved to the
  most-recent fight and silently overwrote earlier bouts on the
  ``pk_fight_odds`` upsert conflict target).
- Upsert uses PostgreSQL ``INSERT ... ON CONFLICT DO UPDATE`` to remain
  idempotent across re-runs.
- Missing moneylines flow through as SQL NULL — D-07 explicit (no
  zero substitution).

Threat mitigations:
- T-13-02-02: every CSV row passes through Pydantic (BFOOddsRow /
  BFOFighterName) before SQL.
- T-13-02-06: all DB writes via parameterized SQLAlchemy statements.
- T-13-02-07: fighter-pair collisions resolved by ``min |db_date -
  bfo_date|`` (deterministic); rows outside ``date_window_days`` return
  ``None`` and are skipped rather than mis-attributed.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.bfo_matcher import match_bfo_name
from ufc_prediction.scraper.bfo_math import (
    InvalidMoneylineError,
    devig_closing_range,
    devig_proportional,
)
from ufc_prediction.scraper.bfo_models import BFOFighterName, BFOOddsRow

logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    """Outcome of a single ``BFOOddsIngester.ingest_all()`` run.

    ``coverage_pct`` is ``fights_matched / bfo_fights_scanned`` when the
    scanned count is > 0, else 0.0. RESEARCH Pitfall 7 flags sub-60%
    coverage as a weak-signal situation — the CLI surfaces a warning.

    ``fighters_matched_canonical`` / ``fighters_matched_fuzzy`` split the
    ``fighters_matched`` total by resolution path (Plan 41-02 BFO-V25-02
    observability hook). Canonical-path counts confirm the alias substrate
    is being honored; a non-zero fuzzy count on anomaly years indicates
    residual rows where ``fighters_names.csv.database_id`` is blank.
    """

    bfo_fights_scanned: int = 0
    fights_matched: int = 0
    fighters_matched: int = 0
    fighters_matched_canonical: int = 0
    fighters_matched_fuzzy: int = 0
    fighters_unmatched: int = 0
    rows_upserted: int = 0
    rows_skipped_bad_range: int = 0
    unmatched_bfo_names: list[str] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        if self.bfo_fights_scanned == 0:
            return 0.0
        return self.fights_matched / self.bfo_fights_scanned


class BFOOddsIngester:
    """Orchestrator for CSV -> validate -> match -> devig -> upsert.

    Args:
        session: SQLAlchemy session bound to the target DB.
        data_folder: Directory containing ufcscraper's output CSVs
            (``BestFightOdds_odds.csv``, ``fighters_names.csv``). Also
            accepts the Plan 01 fixture filenames
            (``bestfightodds_sample.csv``,
            ``bestfightodds_fighters_names_sample.csv``) — tested via
            ``--skip-scrape --data-folder tests/scraper/fixtures``.
        match_threshold: rapidfuzz ratio threshold for BFO -> DB name
            match. Default 80 (RESEARCH Pitfall 6 — lower than Sherdog's
            85 because the BFO candidate pool is tiny and
            ``normalize_name`` is aggressive).
        date_window_days: Tolerance (in days) for matching a BFO row's
            ``event_date`` to a DB ``Event.date`` when resolving the
            fight for a fighter pair. ``_resolve_fight`` picks the
            closest-date candidate; if the closest is more than this
            many days off, returns ``None`` (skip rather than mis-attach).
            Default 14 (Pitfall 1 — two fighters fighting twice within
            ±14 days is not a real-world case in UFC due to medical
            suspensions; fix 999.4).
    """

    _ODDS_CSV = "BestFightOdds_odds.csv"
    _NAMES_CSV = "fighters_names.csv"
    _FIXTURE_ODDS_CSV = "bestfightodds_sample.csv"
    _FIXTURE_NAMES_CSV = "bestfightodds_fighters_names_sample.csv"

    def __init__(
        self,
        session: Session,
        data_folder: Path,
        match_threshold: int = 80,
        date_window_days: int = 14,
    ) -> None:
        self._session = session
        self._data_folder = Path(data_folder)
        self._match_threshold = match_threshold
        self._date_window = timedelta(days=date_window_days)

    # ── Public API ──────────────────────────────────────────────────────

    def ingest_all(self) -> IngestSummary:
        """Read CSVs, validate, match, devig, upsert. Commits on success."""
        summary = IngestSummary()

        names_csv = self._resolve_csv(self._NAMES_CSV, self._FIXTURE_NAMES_CSV)
        odds_csv = self._resolve_csv(self._ODDS_CSV, self._FIXTURE_ODDS_CSV)

        bfo_names = self._load_fighter_names(names_csv)
        bfo_to_db = self._match_fighters(bfo_names, summary)
        odds_by_fight = self._load_odds_rows(odds_csv)
        summary.bfo_fights_scanned = len(odds_by_fight)

        for bfo_fight_id, rows in odds_by_fight.items():
            if len(rows) != 2:
                logger.warning(
                    "BFO fight %s has %d rows (expected 2), skipping",
                    bfo_fight_id,
                    len(rows),
                )
                continue

            row_a, row_b = rows
            db_fighter_a = bfo_to_db.get(row_a.fighter_id)
            db_fighter_b = bfo_to_db.get(row_b.fighter_id)
            if db_fighter_a is None or db_fighter_b is None:
                continue  # partial-match; skip (T-13-02-07 logged already)

            # 999.4: parse the BFO event_date out of row_a.fight_id (the
            # composite ``YYYY-MM-DD|<bfo_a>|<bfo_b>`` key). Both rows in
            # a (row_a, row_b) pair share the same fight_id by construction
            # (they are grouped by it in _load_odds_rows). Bad rows skip
            # rather than fail the whole ingest — a single malformed
            # CSV line should not kill thousands of valid rows.
            try:
                bfo_event_date = row_a.event_date
            except ValueError as exc:
                logger.warning("Skipping BFO fight %s: %s", bfo_fight_id, exc)
                continue

            db_fight_id = self._resolve_fight(db_fighter_a, db_fighter_b, bfo_event_date)
            if db_fight_id is None:
                logger.debug(
                    "No DB fight for fighter pair (%s, %s) within %s of "
                    "BFO event_date %s (BFO fight %s)",
                    db_fighter_a,
                    db_fighter_b,
                    self._date_window,
                    bfo_event_date,
                    bfo_fight_id,
                )
                continue

            summary.fights_matched += 1
            self._upsert_row(db_fight_id, db_fighter_a, row_a, row_b)
            self._upsert_row(db_fight_id, db_fighter_b, row_b, row_a)
            summary.rows_upserted += 2

        self._session.commit()
        return summary

    # ── Internals ───────────────────────────────────────────────────────

    def _resolve_csv(self, primary: str, fallback: str) -> Path:
        """Prefer real ufcscraper filename; fall back to Plan 01 fixture."""
        candidate = self._data_folder / primary
        if candidate.exists():
            return candidate
        fixture = self._data_folder / fallback
        if fixture.exists():
            return fixture
        raise FileNotFoundError(f"Neither {primary} nor {fallback} found in {self._data_folder}")

    def _load_fighter_names(self, names_csv: Path) -> dict[str, list[BFOFighterName]]:
        bfo_names: dict[str, list[BFOFighterName]] = {}
        with names_csv.open() as fh:
            for raw in csv.DictReader(fh):
                try:
                    name_row = BFOFighterName(**raw)
                except Exception as exc:
                    logger.warning("Invalid row in %s: %r (%s)", names_csv, raw, exc)
                    continue
                bfo_names.setdefault(name_row.fighter_id, []).append(name_row)
        return bfo_names

    def _load_odds_rows(self, odds_csv: Path) -> dict[str, list[BFOOddsRow]]:
        odds_by_fight: dict[str, list[BFOOddsRow]] = {}
        with odds_csv.open() as fh:
            for raw in csv.DictReader(fh):
                try:
                    odds_row = BFOOddsRow(**raw)  # type: ignore[arg-type]
                except Exception as exc:
                    logger.warning("Invalid row in %s: %r (%s)", odds_csv, raw, exc)
                    continue
                odds_by_fight.setdefault(odds_row.fight_id, []).append(odds_row)
        return odds_by_fight

    def _match_fighters(
        self,
        bfo_names: dict[str, list[BFOFighterName]],
        summary: IngestSummary,
    ) -> dict[str, int]:
        """Resolve each BFO fighter_id to a DB Fighter.id.

        Plan 41-02 (BFO-V25-02) — honors the canonical ``database_id``
        column from ``data/bfo/fighters_names.csv`` when present + int-
        parseable, short-circuiting the fuzzy matcher. Falls back to
        ``match_bfo_name`` only when ``database_id`` is blank/None on the
        preferred ``ufcstats`` linkage row. Counts canonical vs fuzzy
        resolutions on the IngestSummary so Plan 41-03 can audit the split.
        """
        db_candidates: list[tuple[int, str]] = [
            (row.id, row.name)
            for row in self._session.execute(select(Fighter.id, Fighter.name)).all()
        ]
        # Existence + semantics guards for the canonical path (see below). The
        # two producers of fighters_names.csv disagree on database_id semantics:
        # refresh_fighters_names_v26.py writes the int Fighter.id PK, while the
        # operator-curated fixture stores ufcstats hex source_id hashes. Accept
        # BOTH — but only when the referenced fighter actually exists — so a
        # bogus/stale id can never be trusted as a PK (was: cast to int and used
        # unchecked, silently mis-attributing or undercounting canonical hits).
        valid_ids: set[int] = {fid for fid, _ in db_candidates}
        sourceid_to_id: dict[str, int] = {
            src_id: fid
            for fid, src_id in self._session.execute(
                select(Fighter.id, Fighter.source_id).where(Fighter.source == "ufcstats")
            ).all()
            if src_id is not None
        }

        bfo_to_db: dict[str, int] = {}
        for bfo_id, name_rows in bfo_names.items():
            # Prefer the 'ufcstats' linkage row if present (exact source).
            preferred_row = next(
                (nr for nr in name_rows if nr.database == "ufcstats"),
                name_rows[0] if name_rows else None,
            )
            if preferred_row is None:
                continue

            # Canonical path (BFO-V25-02): trust the operator-curated alias
            # substrate ONLY when database_id resolves to a real fighter — either
            # as an int PK that exists, or as a ufcstats source_id hash.
            if preferred_row.database_id is not None:
                canonical_id: int | None = None
                try:
                    as_int = int(preferred_row.database_id)
                except (TypeError, ValueError):
                    as_int = None
                if as_int is not None and as_int in valid_ids:
                    canonical_id = as_int  # int PK, verified to exist
                elif str(preferred_row.database_id) in sourceid_to_id:
                    canonical_id = sourceid_to_id[str(preferred_row.database_id)]
                if canonical_id is not None:
                    bfo_to_db[bfo_id] = canonical_id
                    summary.fighters_matched += 1
                    summary.fighters_matched_canonical += 1
                    continue
                # database_id present but did not resolve to a real fighter →
                # do NOT trust it; fall through to the fuzzy matcher.

            # Fuzzy fallback: alias substrate had no canonical id for this row.
            match = match_bfo_name(
                preferred_row.name,
                db_candidates,
                threshold=self._match_threshold,
            )
            if match is None:
                summary.fighters_unmatched += 1
                summary.unmatched_bfo_names.append(preferred_row.name)
                logger.warning("No DB match for BFO fighter: %s", preferred_row.name)
                continue
            bfo_to_db[bfo_id] = match[0]
            summary.fighters_matched += 1
            summary.fighters_matched_fuzzy += 1
        return bfo_to_db

    def _resolve_fight(
        self,
        db_fighter_a: int,
        db_fighter_b: int,
        event_date: date,
    ) -> int | None:
        """Resolve the BFO row to the DB fight whose ``Event.date`` is
        closest to ``event_date``, within ``self._date_window`` tolerance.

        Disambiguates rematches (999.4): without the date filter, every
        BFO row for a rematch pair resolves to the most-recent fight and
        upserts overwrite each other on the ``pk_fight_odds`` conflict
        target. With the date filter:

          - All candidate fights for the pair (either order) are loaded.
          - The candidate with ``min |candidate.date - event_date|`` wins.
          - If that minimum exceeds ``self._date_window``, returns
            ``None`` — refuse to attach odds to a fight that's not
            actually the right one (skip > mis-attribute).

        Args:
            db_fighter_a: One side of the pair (DB ``Fighter.id``).
            db_fighter_b: The other side (DB ``Fighter.id``).
            event_date: The BFO row's ``event_date`` (parsed from the
                composite ``fight_id`` prefix on the BFOOddsRow).

        Returns:
            The matched ``Fight.id``, or ``None`` if no candidate is
            within the date window.
        """
        stmt = (
            select(Fight.id, Event.date)
            .join(Event, Fight.event_id == Event.id)
            .where(
                ((Fight.fighter_a_id == db_fighter_a) & (Fight.fighter_b_id == db_fighter_b))
                | ((Fight.fighter_a_id == db_fighter_b) & (Fight.fighter_b_id == db_fighter_a))
            )
        )
        candidates = list(self._session.execute(stmt).all())
        if not candidates:
            return None

        best_fight_id, best_dt = min(
            candidates,
            key=lambda row: abs(row.date - event_date),
        )
        if abs(best_dt - event_date) > self._date_window:
            logger.debug(
                "No DB fight within %s of BFO event_date %s for pair (%s, %s); closest was %s",
                self._date_window,
                event_date,
                db_fighter_a,
                db_fighter_b,
                best_dt,
            )
            return None
        return int(best_fight_id)

    def _upsert_row(
        self,
        fight_id: int,
        this_fighter_id: int,
        this_row: BFOOddsRow,
        other_row: BFOOddsRow,
    ) -> None:
        """Compute implied probs for ``this_row`` normalized against
        ``other_row``, then upsert into ``fight_odds`` idempotently.
        """
        # Bad-data resilience (review #12): the devig helpers now raise
        # InvalidMoneylineError on an out-of-domain moneyline (|ml| < 100). BFOOddsRow does
        # not validate moneylines, so a single malformed feed cell must NOT abort
        # the whole batch (bfo_ingest invariant: one bad row ≠ thousands lost).
        # Catch per-field and leave that prob NULL, mirroring the missing-data path.
        opening_prob: float | None = None
        if this_row.opening is not None and other_row.opening is not None:
            try:
                p_this, _ = devig_proportional(this_row.opening, other_row.opening)
                opening_prob = p_this
            except InvalidMoneylineError as exc:
                logger.warning(
                    "skipping opening odds for fight_id=%s fighter_id=%s: %s",
                    fight_id,
                    this_fighter_id,
                    exc,
                )

        closing_prob: float | None = None
        if (
            this_row.closing_range_min is not None
            and this_row.closing_range_max is not None
            and other_row.closing_range_min is not None
            and other_row.closing_range_max is not None
        ):
            try:
                p_this, _ = devig_closing_range(
                    this_row.closing_range_min,
                    this_row.closing_range_max,
                    other_row.closing_range_min,
                    other_row.closing_range_max,
                )
                closing_prob = p_this
            except InvalidMoneylineError as exc:
                logger.warning(
                    "skipping closing odds for fight_id=%s fighter_id=%s: %s",
                    fight_id,
                    this_fighter_id,
                    exc,
                )

        stmt = pg_insert(FightOdds).values(
            fight_id=fight_id,
            fighter_id=this_fighter_id,
            opening_ml=this_row.opening,
            closing_range_min_ml=this_row.closing_range_min,
            closing_range_max_ml=this_row.closing_range_max,
            opening_implied_prob=opening_prob,
            closing_implied_prob=closing_prob,
            source="bestfightodds",
        )
        upsert = stmt.on_conflict_do_update(
            constraint="pk_fight_odds",
            set_={
                "opening_ml": stmt.excluded.opening_ml,
                "closing_range_min_ml": stmt.excluded.closing_range_min_ml,
                "closing_range_max_ml": stmt.excluded.closing_range_max_ml,
                "opening_implied_prob": stmt.excluded.opening_implied_prob,
                "closing_implied_prob": stmt.excluded.closing_implied_prob,
            },
        )
        self._session.execute(upsert)
