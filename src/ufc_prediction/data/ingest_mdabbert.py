"""Ingestor for the mdabbert/ultimate-ufc-dataset Kaggle dataset (supplement).

Reads ufc-master.csv to supplement rajeevw data: fills missing fighter profiles,
adds 2021-2026 fights not in rajeevw, and enriches fighter physical attributes.
Does NOT ingest career averages or pre-computed features (temporal leakage risk).

Uses upsert logic (D-03) and skip-and-log validation (D-06).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from ufc_prediction.data.parsers import (
    cm_to_inches,
    parse_fight_date,
)
from ufc_prediction.data.schemas import FighterRow, FightRow, IngestResult
from ufc_prediction.data.upsert import upsert_event
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter

logger = logging.getLogger(__name__)

SOURCE = "kaggle-mdabbert"
_RAJEEVW_SOURCE = "kaggle-rajeevw"

# Batch commit interval
_BATCH_SIZE = 500


# ── Method mapping ──────────────────────────────────────────────────────────────

_METHOD_MAP = {
    "KO/TKO": "KO/TKO",
    "Submission": "Submission",
    "Decision": "Decision",
    "DQ": "DQ",
    "Overturned": "Other",
}


def _map_method(finish: str | None) -> str | None:
    """Map mdabbert 'finish' column to normalized method."""
    if not finish or pd.isna(finish):
        return None
    finish = str(finish).strip()
    return _METHOD_MAP.get(finish, finish)


# ── Fighter supplement logic ────────────────────────────────────────────────────


def _supplement_or_create_fighter(
    session: Session,
    name: str,
    height_inches: float | None,
    reach_inches: float | None,
    stance: str | None,
    approx_dob: date | None,
) -> Fighter:
    """Supplement existing rajeevw fighter or create new mdabbert fighter.

    Logic:
    a. If fighter exists with source="kaggle-rajeevw", fill NULL fields only.
    b. If fighter does NOT exist with any source, create with source="kaggle-mdabbert".
    c. If fighter exists only with source="kaggle-mdabbert", update non-null fields.

    Returns Fighter instance (flushed, has id).
    """
    # Check rajeevw first
    fighter = (
        session.query(Fighter)
        .filter(Fighter.name == name, Fighter.source == _RAJEEVW_SOURCE)
        .first()
    )
    if fighter is not None:
        # Supplement: fill NULL fields only (don't overwrite rajeevw data)
        if fighter.height_inches is None and height_inches is not None:
            fighter.height_inches = height_inches
        if fighter.reach_inches is None and reach_inches is not None:
            fighter.reach_inches = reach_inches
        if fighter.stance is None and stance is not None:
            fighter.stance = stance
        if fighter.date_of_birth is None and approx_dob is not None:
            fighter.date_of_birth = approx_dob
        return fighter

    # Check mdabbert
    fighter = session.query(Fighter).filter(Fighter.name == name, Fighter.source == SOURCE).first()
    if fighter is not None:
        # Update non-null fields
        if height_inches is not None:
            fighter.height_inches = height_inches
        if reach_inches is not None:
            fighter.reach_inches = reach_inches
        if stance is not None:
            fighter.stance = stance
        if approx_dob is not None and fighter.date_of_birth is None:
            fighter.date_of_birth = approx_dob
        return fighter

    # Create new fighter
    fighter = Fighter(
        name=name,
        height_inches=height_inches,
        reach_inches=reach_inches,
        stance=stance,
        date_of_birth=approx_dob,
        source=SOURCE,
    )
    session.add(fighter)
    session.flush()
    return fighter


# ── Helper parsers ──────────────────────────────────────────────────────────────


def _parse_cm_to_inches(value: object) -> float | None:
    """Parse a cm value to inches, handling NaN/None/zero."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        cm = float(value)  # type: ignore[arg-type]
        if cm <= 0:
            return None
        return round(cm_to_inches(cm), 2)
    except (ValueError, TypeError):
        return None


def _parse_bool(value: object) -> bool:
    """Parse a boolean-like value from mdabbert CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


def _parse_int_safe(value: object) -> int | None:
    """Safely parse an integer from a potentially messy value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


def _approximate_dob(age: object, event_date: date) -> date | None:
    """Approximate DOB from age at event date.

    Only used when fighter has no existing DOB.
    Computes: event_date - timedelta(days=int(age * 365.25))
    """
    if age is None or (isinstance(age, float) and pd.isna(age)):
        return None
    try:
        age_float = float(age)  # type: ignore[arg-type]
        if age_float <= 0:
            return None
        days = int(age_float * 365.25)
        return event_date - timedelta(days=days)
    except (ValueError, TypeError):
        return None


# ── Main ingestor ───────────────────────────────────────────────────────────────


def ingest_mdabbert(csv_path: Path, session: Session) -> IngestResult:
    """Ingest fights and fighter profiles from mdabbert ufc-master.csv.

    Supplements rajeevw data:
    - Fills missing fighter profile fields (height, reach, stance, DOB)
    - Adds 2021-2026 fights not in rajeevw
    - Skips fights that already exist from any source

    Does NOT ingest: career averages (avg_SIG_STR_*, avg_TD_*),
    career records (wins/losses), pre-computed differentials (*_dif),
    rankings, or betting odds. These all have temporal leakage risk.

    Args:
        csv_path: Path to ufc-master.csv
        session: SQLAlchemy session for database writes

    Returns:
        IngestResult with accepted/updated/rejected counts
    """
    result = IngestResult()
    df = pd.read_csv(csv_path, sep=",")
    logger.info("Reading %d rows from %s", len(df), csv_path)

    for idx, row in df.iterrows():
        try:
            raw = row.to_dict()

            # 1. Parse fight metadata
            event_date = parse_fight_date(str(raw["date"]) if pd.notna(raw.get("date")) else None)
            if event_date is None:
                # Per D-07: skip row if date is unparseable
                result.log_rejection(idx, raw, "Unparseable date (D-07)")
                logger.warning("Skipping row %d: unparseable date", idx)
                continue

            weight_class_raw = raw.get("weight_class")
            weight_class = (
                str(weight_class_raw).strip() if pd.notna(weight_class_raw) else "Unknown"
            )

            is_title_fight = _parse_bool(raw.get("title_bout"))
            num_rounds_raw = _parse_int_safe(raw.get("no_of_rounds"))
            num_rounds = num_rounds_raw if num_rounds_raw in (3, 5) else 3

            method = _map_method(raw.get("finish"))
            method_detail_raw = raw.get("finish_details")
            method_detail = str(method_detail_raw).strip() if pd.notna(method_detail_raw) else None

            round_finished = _parse_int_safe(raw.get("finish_round"))
            time_raw = raw.get("finish_round_time")
            time_finished = str(time_raw).strip() if pd.notna(time_raw) else None

            # Fighter names
            r_fighter = str(raw.get("R_fighter", "")).strip()
            b_fighter = str(raw.get("B_fighter", "")).strip()

            # Winner mapping
            winner_raw = str(raw.get("Winner", "")).strip()
            if winner_raw == "Red":
                winner_name = r_fighter
            elif winner_raw == "Blue":
                winner_name = b_fighter
            else:
                winner_name = None  # Draw / NC

            # Location
            location_raw = raw.get("location")
            location = str(location_raw).strip() if pd.notna(location_raw) else None

            # 2. Extract fighter profiles (per D-02: DATA-02)
            # DO NOT ingest: career averages, career records,
            # differentials, rankings, betting odds
            r_height = _parse_cm_to_inches(raw.get("R_Height_cms"))
            r_reach = _parse_cm_to_inches(raw.get("R_Reach_cms"))
            r_stance_raw = raw.get("R_Stance")
            r_stance = str(r_stance_raw).strip() if pd.notna(r_stance_raw) else None
            r_dob = _approximate_dob(raw.get("R_age"), event_date)

            b_height = _parse_cm_to_inches(raw.get("B_Height_cms"))
            b_reach = _parse_cm_to_inches(raw.get("B_Reach_cms"))
            b_stance_raw = raw.get("B_Stance")
            b_stance = str(b_stance_raw).strip() if pd.notna(b_stance_raw) else None
            b_dob = _approximate_dob(raw.get("B_age"), event_date)

            # Validate fighter profiles through Pydantic
            FighterRow(
                name=r_fighter,
                height_inches=r_height,
                reach_inches=r_reach,
                stance=r_stance,
                date_of_birth=r_dob,
            )
            FighterRow(
                name=b_fighter,
                height_inches=b_height,
                reach_inches=b_reach,
                stance=b_stance,
                date_of_birth=b_dob,
            )

            # Event name
            event_name = f"UFC Event - {event_date.isoformat()}"

            # Validate fight through Pydantic
            FightRow(
                event_name=event_name,
                event_date=event_date,
                location=location,
                fighter_a_name=r_fighter,
                fighter_b_name=b_fighter,
                winner_name=winner_name,
                weight_class=weight_class,
                method=method,
                method_detail=method_detail,
                round_finished=round_finished,
                time_finished=time_finished,
                is_title_fight=is_title_fight,
                num_rounds=num_rounds,
            )

            # 3. Supplement or create fighters
            fighter_a = _supplement_or_create_fighter(
                session, r_fighter, r_height, r_reach, r_stance, r_dob
            )
            fighter_b = _supplement_or_create_fighter(
                session, b_fighter, b_height, b_reach, b_stance, b_dob
            )

            # 4. Upsert Event
            event = upsert_event(session, event_name, event_date, location, SOURCE)

            # 5. Check if fight already exists from ANY source
            existing_fight = (
                session.query(Fight)
                .filter(
                    Fight.fighter_a_id == fighter_a.id,
                    Fight.fighter_b_id == fighter_b.id,
                    Fight.event_id.in_(session.query(Event.id).filter(Event.date == event_date)),
                )
                .first()
            )
            # Also check reversed fighter order
            if existing_fight is None:
                existing_fight = (
                    session.query(Fight)
                    .filter(
                        Fight.fighter_a_id == fighter_b.id,
                        Fight.fighter_b_id == fighter_a.id,
                        Fight.event_id.in_(
                            session.query(Event.id).filter(Event.date == event_date)
                        ),
                    )
                    .first()
                )

            if existing_fight is not None:
                # Fight already exists (from rajeevw or earlier mdabbert)
                result.updated += 1
                if (idx + 1) % _BATCH_SIZE == 0:
                    session.commit()
                    logger.info("Processed %d rows...", idx + 1)
                continue

            # Determine winner_id
            winner_id = None
            if winner_name == r_fighter:
                winner_id = fighter_a.id
            elif winner_name == b_fighter:
                winner_id = fighter_b.id

            # Insert new fight (only for fights not already in DB)
            fight = Fight(
                event_id=event.id,
                fighter_a_id=fighter_a.id,
                fighter_b_id=fighter_b.id,
                winner_id=winner_id,
                weight_class=weight_class,
                method=method,
                method_detail=method_detail,
                round_finished=round_finished,
                time_finished=time_finished,
                is_title_fight=is_title_fight,
                num_rounds=num_rounds,
                source=SOURCE,
            )
            session.add(fight)
            session.flush()
            result.accepted += 1

            # NOTE: mdabbert does NOT have per-fight performance stats.
            # No RoundStats records created. Fight-level stats come from
            # rajeevw (pre-2021) or Phase 5 scraper (all fights).

            # Batch commit for performance
            if (idx + 1) % _BATCH_SIZE == 0:
                session.commit()
                logger.info("Committed %d rows...", idx + 1)

        except Exception as exc:
            # Per D-06: catch ANY exception, log rejection, continue
            session.rollback()
            result.log_rejection(idx, raw, str(exc))
            logger.warning("Rejected row %d: %s", idx, exc)
            continue

    # Final commit
    session.commit()
    logger.info(result.summary())
    return result
