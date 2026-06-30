"""Pydantic models for ufcscraper CSV output validation.

Schemas mirror ufcscraper v1.1.0 CSV columns:
  - BestFightOdds_odds.csv: fight_id, fighter_id, opening,
    closing_range_min, closing_range_max
  - fighters_names.csv:     fighter_id, database, name, database_id

All downstream ingestion MUST pass CSV rows through these models before
touching the DB (T-13-01 boundary validation; CSV injection / type
confusion guard).

Design notes:
- ``strict: False`` allows CSV-native string->int coercion by Pydantic v2
  (e.g. ``"-200"`` -> ``-200``). Floats / non-numeric strings still raise
  ``ValidationError``, which is the correctness requirement.
- A pre-validator maps empty strings to ``None`` for nullable numeric and
  string fields. ufcscraper emits blank cells for missing values; we
  handle that defensively rather than relying on undocumented behavior
  (Open Questions 1 & 2 in 13-RESEARCH.md).
"""

from __future__ import annotations

from datetime import date
from functools import cached_property

from pydantic import BaseModel, Field, field_validator


def _blank_to_none(v: object) -> object:
    """Coerce blank/whitespace-only strings to ``None``.

    Returns the original value otherwise (Pydantic's type coercion then
    handles numeric strings -> int).
    """
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


class BFOOddsRow(BaseModel):
    """One row of ufcscraper's BestFightOdds_odds.csv.

    Missing odds are valid per D-07 (XGBoost handles NaN natively).

    Columns from ufcscraper v1.1.0:
      - fight_id: str (composite "YYYY-MM-DD|<bfo_a>|<bfo_b>")
      - fighter_id: str (BFO-native)
      - opening: int | None (American moneyline)
      - closing_range_min: int | None (American ML, min across books)
      - closing_range_max: int | None (American ML, max across books)

    Computed:
      - event_date: typed ``date`` parsed from ``fight_id`` prefix.
        See the ``event_date`` property docstring for the rationale on
        raising ``ValueError`` rather than returning ``None`` (999.4).
    """

    fight_id: str = Field(min_length=1)
    fighter_id: str = Field(min_length=1)
    opening: int | None = None
    closing_range_min: int | None = None
    closing_range_max: int | None = None

    # ``ignored_types`` keeps Pydantic from treating ``cached_property`` as a
    # model field. ``strict: False`` keeps CSV string -> int coercion.
    model_config = {
        "strict": False,
        "ignored_types": (cached_property,),
    }

    @field_validator("opening", "closing_range_min", "closing_range_max", mode="before")
    @classmethod
    def _coerce_blank_ml(cls, v: object) -> object:
        return _blank_to_none(v)

    @cached_property
    def event_date(self) -> date:
        """Parse the ``YYYY-MM-DD`` prefix from the composite ``fight_id``.

        The real BFO CSV uses ``YYYY-MM-DD|<bfo_a>|<bfo_b>`` (see
        ``data/bfo/BestFightOdds_odds.csv``). The ingester needs this date
        to disambiguate rematches in ``BFOOddsIngester._resolve_fight``
        (backlog 999.4) — without it, every BFO row for a rematch pair
        resolves to the most-recent fight via ``ORDER BY date DESC ...
        LIMIT 1`` and the ``pk_fight_odds`` upsert silently overwrites
        the earlier bouts.

        Raises:
            ValueError: if ``fight_id`` is missing the ``|`` separator or
                if the prefix is not an ISO date. We refuse to silently
                fall back to ``None`` here — silent fallback in the
                ingester would re-introduce the H1 fall-back-to-most-recent
                bug. Bad rows are caught + logged at the ``ingest_all``
                call site (``bfo_ingest.py``) so a single malformed row
                cannot kill the whole ingest.
        """
        if "|" not in self.fight_id:
            raise ValueError(f"BFO fight_id missing '|' separator: {self.fight_id!r}")
        head, _, _ = self.fight_id.partition("|")
        if not head:
            raise ValueError(f"BFO fight_id has empty date prefix: {self.fight_id!r}")
        return date.fromisoformat(head)


class BFOFighterName(BaseModel):
    """One row of ufcscraper's fighters_names.csv cross-database linkage.

    Columns:
      - fighter_id: str (BFO-native)
      - database: str (source name e.g. 'ufcstats', 'sherdog')
      - name: str (display name in that source)
      - database_id: str | None (ID in that source; may be blank)
    """

    fighter_id: str = Field(min_length=1)
    database: str = Field(min_length=1)
    name: str = Field(min_length=1)
    database_id: str | None = None

    @field_validator("database_id", mode="before")
    @classmethod
    def _coerce_blank_db_id(cls, v: object) -> object:
        return _blank_to_none(v)
