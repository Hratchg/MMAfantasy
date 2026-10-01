"""DB persistence for the Sherdog debutant Elo seed inputs (``debutant_seed_inputs``).

Operator decision D3 option C (2026-09-29): the table is the single source of
truth for debutant seeds. ``ufc elo compute`` and the serve path
(``ml.inference_features``) call :func:`load_seeds_from_db`; the untracked
``data/sherdog/pre_ufc_records.csv`` only feeds ``ufc db backfill-pre-ufc-seeds``
(via :func:`parse_seed_csv` + :func:`upsert_seed_rows`) and stays an export of
``scripts/ingest_pre_ufc_records_v25.py``, which now writes the table too.

The seed formula itself stays in :mod:`ufc_prediction.elo.seed` (pure,
operator-locked). Rows are converted with the same types ``load_seeds`` feeds
``derive_seed`` (``int`` fight count, ``float`` win rate, ``str`` tier), and
``win_rate`` is stored as float8, so ``load_seeds_from_db`` returns the same
dict, float for float, as ``load_seeds(csv)`` over the same records.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import literal_column, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ufc_prediction.elo.seed import SeedDerivationError, derive_seed
from ufc_prediction.models.debutant_seed_input import ORG_TIERS, DebutantSeedInput
from ufc_prediction.models.fighter import Fighter

SEED_TABLE: str = DebutantSeedInput.__tablename__

_OPTIONAL_INT_COLUMNS: tuple[str, ...] = (
    "wins",
    "losses",
    "draws",
    "nc_dq",
    "kos",
    "submissions",
    "decisions",
)

# Rows per INSERT … ON CONFLICT statement (14 bind params each, well under
# Postgres' 65,535-parameter limit).
_UPSERT_CHUNK: int = 1000


@dataclass(frozen=True)
class UpsertResult:
    """Outcome of :func:`upsert_seed_rows`; ``unchanged`` rows were not rewritten."""

    inserted: int
    updated: int
    unchanged: int


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def seed_row_from_csv(row: Mapping[str, object], *, row_idx: int | None = None) -> dict[str, Any]:
    """Convert one ``pre_ufc_records.csv`` row into a ``debutant_seed_inputs`` row.

    Accepts the CSV's text values or the native values the ingest script builds
    (``build_csv_row``). Fails closed with :class:`SeedDerivationError` on any
    row ``load_seeds`` would reject, on an out-of-domain seed input, and on
    missing provenance (``sherdog_url``, ``scraped_at``). Blank optional columns
    become ``NULL``; a naive ``scraped_at`` is read as UTC.
    """
    prefix = f"Row {row_idx}" if row_idx is not None else "Seed row"
    fid_text = _text(row.get("fighter_id"))
    if not fid_text:
        raise SeedDerivationError(f"{prefix}: required column 'fighter_id' is empty.")
    try:
        fighter_id = int(fid_text)
    except ValueError as exc:
        raise SeedDerivationError(
            f"{prefix}: column 'fighter_id'={fid_text!r} is not an integer."
        ) from exc
    where = f"{prefix} fighter_id={fighter_id}"

    def required(col: str) -> str:
        val = _text(row.get(col))
        if not val:
            raise SeedDerivationError(f"{where}: required column {col!r} is empty.")
        return val

    def as_int(col: str, text: str) -> int:
        try:
            return int(text)
        except ValueError as exc:
            msg = f"{where}: column {col!r}={text!r} is not an integer."
            raise SeedDerivationError(msg) from exc

    n_text = required("n_pre_ufc_fights")
    n_pre_ufc_fights = as_int("n_pre_ufc_fights", n_text)
    if n_pre_ufc_fights < 0:
        raise SeedDerivationError(f"{where}: column 'n_pre_ufc_fights'={n_text!r} is negative.")

    wr_text = required("win_rate")
    try:
        win_rate = float(wr_text)
    except ValueError as exc:
        msg = f"{where}: column 'win_rate'={wr_text!r} is not a number."
        raise SeedDerivationError(msg) from exc
    if not math.isfinite(win_rate) or not 0.0 <= win_rate <= 1.0:
        raise SeedDerivationError(f"{where}: column 'win_rate'={wr_text!r} is outside [0, 1].")

    # Not stripped: derive_seed matches the tier verbatim, so neither does this.
    required("org_tier")
    org_tier = str(row.get("org_tier"))
    if org_tier not in ORG_TIERS:
        raise SeedDerivationError(
            f"{where}: column 'org_tier'={org_tier!r} is not one of {list(ORG_TIERS)}."
        )

    sherdog_url = required("sherdog_url")
    scraped_text = required("scraped_at")
    try:
        scraped_at = datetime.fromisoformat(scraped_text)
    except ValueError as exc:
        raise SeedDerivationError(
            f"{where}: column 'scraped_at'={scraped_text!r} is not an ISO-8601 timestamp."
        ) from exc
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.replace(tzinfo=UTC)

    optional_ints: dict[str, int | None] = {}
    for col in _OPTIONAL_INT_COLUMNS:
        text = _text(row.get(col))
        optional_ints[col] = as_int(col, text) if text else None

    last_org_raw = row.get("last_organization")
    last_organization = None if not _text(last_org_raw) else str(last_org_raw)

    return {
        "fighter_id": fighter_id,
        "sherdog_url": sherdog_url,
        "n_pre_ufc_fights": n_pre_ufc_fights,
        "wins": optional_ints["wins"],
        "losses": optional_ints["losses"],
        "draws": optional_ints["draws"],
        "nc_dq": optional_ints["nc_dq"],
        "win_rate": win_rate,
        "kos": optional_ints["kos"],
        "submissions": optional_ints["submissions"],
        "decisions": optional_ints["decisions"],
        "last_organization": last_organization,
        "org_tier": org_tier,
        "scraped_at": scraped_at,
    }


def parse_seed_csv(csv_path: Path | str) -> list[dict[str, Any]]:
    """Read and validate a whole ``pre_ufc_records.csv`` into table rows.

    Every row is validated before the caller writes anything. Duplicate
    ``fighter_id`` rows resolve last-wins, as in ``load_seeds``. Rows are
    returned sorted by ``fighter_id``. Raises ``FileNotFoundError`` for a
    missing file and :class:`SeedDerivationError` (numbered like
    ``load_seeds``: the header is row 1) for a malformed row.
    """
    path = Path(csv_path)
    by_id: dict[int, dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for row_idx, row in enumerate(csv.DictReader(fh), start=2):
            rec = seed_row_from_csv(row, row_idx=row_idx)
            by_id[rec["fighter_id"]] = rec
    return [by_id[fid] for fid in sorted(by_id)]


def unknown_fighter_ids(session: Session, fighter_ids: Iterable[int]) -> set[int]:
    """The subset of ``fighter_ids`` with no ``fighters`` row (would violate the FK)."""
    ids = set(fighter_ids)
    if not ids:
        return set()
    found = set(session.scalars(select(Fighter.id).where(Fighter.id.in_(ids))))
    return ids - found


def upsert_seed_rows(session: Session, rows: Iterable[Mapping[str, Any]]) -> UpsertResult:
    """Idempotently INSERT … ON CONFLICT (fighter_id) DO UPDATE the given rows.

    A row identical to the stored one is left untouched (not rewritten), so
    re-running a backfill reports every row ``unchanged``. Duplicate
    ``fighter_id`` values resolve last-wins. Does not commit.
    """
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        by_id[int(row["fighter_id"])] = row
    ordered = [dict(by_id[fid]) for fid in sorted(by_id)]

    table = DebutantSeedInput.__table__
    value_cols = [c.name for c in table.columns if c.name != "fighter_id"]
    inserted = updated = 0
    for start in range(0, len(ordered), _UPSERT_CHUNK):
        insert_stmt = pg_insert(DebutantSeedInput).values(ordered[start : start + _UPSERT_CHUNK])
        excluded = insert_stmt.excluded
        upsert = insert_stmt.on_conflict_do_update(
            index_elements=[table.c.fighter_id],
            set_={col: excluded[col] for col in value_cols},
            where=or_(*(table.c[col].is_distinct_from(excluded[col]) for col in value_cols)),
        ).returning(table.c.fighter_id, literal_column("(xmax = 0)").label("inserted"))
        for _fighter_id, was_inserted in session.execute(upsert):
            if was_inserted:
                inserted += 1
            else:
                updated += 1
    return UpsertResult(
        inserted=inserted, updated=updated, unchanged=len(ordered) - inserted - updated
    )


def load_seeds_from_db(session: Session) -> dict[int, float]:
    """``{fighter_id: seed}`` derived from every ``debutant_seed_inputs`` row.

    Same contract as ``load_seeds(csv)`` for the same records. Returns ``{}``
    for an empty table; a missing table raises (``ProgrammingError``) so each
    caller decides how to fail.
    """
    stmt = select(
        DebutantSeedInput.fighter_id,
        DebutantSeedInput.n_pre_ufc_fights,
        DebutantSeedInput.win_rate,
        DebutantSeedInput.org_tier,
    )
    return {
        fighter_id: derive_seed(
            {"n_pre_ufc_fights": n_fights, "win_rate": win_rate, "org_tier": org_tier}
        )
        for fighter_id, n_fights, win_rate, org_tier in session.execute(stmt)
    }


def seeded_fighter_ids(session: Session) -> set[int]:
    """Fighter ids that already have a ``debutant_seed_inputs`` row."""
    return set(session.scalars(select(DebutantSeedInput.fighter_id)))


__all__ = [
    "SEED_TABLE",
    "UpsertResult",
    "load_seeds_from_db",
    "parse_seed_csv",
    "seed_row_from_csv",
    "seeded_fighter_ids",
    "unknown_fighter_ids",
    "upsert_seed_rows",
]
