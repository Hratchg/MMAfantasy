"""Ingestor for the rajeevw/ufcdata Kaggle dataset (primary source).

Reads raw_fighter_details.csv (fighter profiles) and raw_total_fight_data.csv
(fight results + fight-level aggregate stats). Uses upsert logic (D-03) and
skip-and-log validation (D-06).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ufc_prediction.data.parsers import (
    parse_control_time,
    parse_dob,
    parse_fight_date,
    parse_height_to_inches,
    parse_num_rounds,
    parse_reach_to_inches,
    parse_weight_class,
    parse_x_of_y,
)
from ufc_prediction.data.schemas import FighterRow, FightRow, FightStats, IngestResult
from ufc_prediction.data.upsert import upsert_event, upsert_fighter
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats

logger = logging.getLogger(__name__)

SOURCE = "kaggle-rajeevw"

# Batch commit interval
_BATCH_SIZE = 500


# ── Method parsing ──────────────────────────────────────────────────────────────


def _parse_method(win_by: str | None) -> tuple[str | None, str | None]:
    """Parse the win_by column into (method, method_detail).

    Examples:
        "Decision - Unanimous" -> ("Decision", "Unanimous")
        "KO/TKO" -> ("KO/TKO", None)
        "Could Not Continue" -> ("Other", "Could Not Continue")
    """
    if not win_by or pd.isna(win_by):
        return (None, None)

    win_by = str(win_by).strip()

    if " - " in win_by:
        parts = win_by.split(" - ", 1)
        method = parts[0].strip()
        detail = parts[1].strip()
        # Normalize Decision variants
        if method == "Decision":
            return ("Decision", detail)
        return (method, detail)

    method_map = {
        "KO/TKO": "KO/TKO",
        "Submission": "Submission",
        "Could Not Continue": "Other",
        "Overturned": "Other",
        "DQ": "DQ",
    }
    mapped = method_map.get(win_by)
    if mapped is not None:
        if mapped == "Other":
            return (mapped, win_by)
        return (mapped, None)

    return (win_by, None)


# ── Fighter ingestor ────────────────────────────────────────────────────────────


def ingest_rajeevw_fighters(csv_path: Path, session: Session) -> IngestResult:
    """Ingest fighter profiles from raw_fighter_details.csv.

    Reads the comma-delimited CSV (14 columns), validates through FighterRow,
    and upserts into the fighters table. Career stats columns are skipped
    (temporal leakage risk).

    Args:
        csv_path: Path to raw_fighter_details.csv
        session: SQLAlchemy session for database writes

    Returns:
        IngestResult with accepted/updated/rejected counts
    """
    result = IngestResult()
    df = pd.read_csv(csv_path, sep=",")
    logger.info("Reading %d fighter rows from %s", len(df), csv_path)

    for idx, row in df.iterrows():
        try:
            raw = row.to_dict()

            # Parse fields - skip career stats (SLpM, Str_Acc, etc.)
            name = str(raw.get("fighter_name", "")).strip()
            height = parse_height_to_inches(
                str(raw["Height"]) if pd.notna(raw.get("Height")) else None
            )
            reach = parse_reach_to_inches(str(raw["Reach"]) if pd.notna(raw.get("Reach")) else None)
            dob = parse_dob(str(raw["DOB"]) if pd.notna(raw.get("DOB")) else None)
            stance_raw = raw.get("Stance")
            stance = str(stance_raw).strip() if pd.notna(stance_raw) else None

            # Validate through Pydantic
            fighter_row = FighterRow(
                name=name,
                height_inches=height,
                reach_inches=reach,
                stance=stance,
                date_of_birth=dob,
            )

            # Upsert fighter
            existing = (
                session.query(Fighter)
                .filter(Fighter.name == fighter_row.name, Fighter.source == SOURCE)
                .first()
            )
            if existing is not None:
                # Update only non-null new values
                if fighter_row.height_inches is not None:
                    existing.height_inches = fighter_row.height_inches
                if fighter_row.reach_inches is not None:
                    existing.reach_inches = fighter_row.reach_inches
                if fighter_row.stance is not None:
                    existing.stance = fighter_row.stance
                if fighter_row.date_of_birth is not None:
                    existing.date_of_birth = fighter_row.date_of_birth
                result.updated += 1
            else:
                fighter = Fighter(
                    name=fighter_row.name,
                    height_inches=fighter_row.height_inches,
                    reach_inches=fighter_row.reach_inches,
                    stance=fighter_row.stance,
                    date_of_birth=fighter_row.date_of_birth,
                    source=SOURCE,
                )
                session.add(fighter)
                result.accepted += 1

            if (idx + 1) % _BATCH_SIZE == 0:
                session.flush()
                logger.info("Processed %d fighter rows...", idx + 1)

        except (ValidationError, ValueError, KeyError) as exc:
            result.log_rejection(idx, row.to_dict(), str(exc))
            logger.warning("Rejected fighter row %d: %s", idx, exc)
            continue

    session.flush()
    logger.info(result.summary())
    return result


# ── Fight ingestor ──────────────────────────────────────────────────────────────


def _parse_int_safe(value: object) -> int | None:
    """Safely parse an integer from a potentially messy value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


def _build_fight_stats(row: dict[str, object], prefix: str) -> FightStats | None:
    """Build FightStats from a row dict for a given corner prefix (R_ or B_).

    Per D-04: if ALL key stats are 0 or None, set everything to None
    (missing data, not "nothing happened").
    """
    sig_landed, sig_attempted = parse_x_of_y(
        str(row.get(f"{prefix}SIG_STR.")) if pd.notna(row.get(f"{prefix}SIG_STR.")) else None
    )
    td_landed, td_attempted = parse_x_of_y(
        str(row.get(f"{prefix}TD")) if pd.notna(row.get(f"{prefix}TD")) else None
    )
    knockdowns = _parse_int_safe(row.get(f"{prefix}KD"))
    sub_attempts = _parse_int_safe(row.get(f"{prefix}SUB_ATT"))
    reversals = _parse_int_safe(row.get(f"{prefix}REV"))
    ctrl_time = parse_control_time(
        str(row.get(f"{prefix}CTRL")) if pd.notna(row.get(f"{prefix}CTRL")) else None
    )

    # Positional strikes - only landed values used
    head_landed, _ = parse_x_of_y(
        str(row.get(f"{prefix}HEAD")) if pd.notna(row.get(f"{prefix}HEAD")) else None
    )
    body_landed, _ = parse_x_of_y(
        str(row.get(f"{prefix}BODY")) if pd.notna(row.get(f"{prefix}BODY")) else None
    )
    leg_landed, _ = parse_x_of_y(
        str(row.get(f"{prefix}LEG")) if pd.notna(row.get(f"{prefix}LEG")) else None
    )
    distance_landed, _ = parse_x_of_y(
        str(row.get(f"{prefix}DISTANCE")) if pd.notna(row.get(f"{prefix}DISTANCE")) else None
    )
    clinch_landed, _ = parse_x_of_y(
        str(row.get(f"{prefix}CLINCH")) if pd.notna(row.get(f"{prefix}CLINCH")) else None
    )
    ground_landed, _ = parse_x_of_y(
        str(row.get(f"{prefix}GROUND")) if pd.notna(row.get(f"{prefix}GROUND")) else None
    )

    # Per D-04: if all key stats are 0/None, treat as missing data
    key_stats = [sig_landed, td_landed, knockdowns, sub_attempts]
    if all(v is None or v == 0 for v in key_stats):
        return FightStats()  # All None fields

    return FightStats(
        sig_strikes_landed=sig_landed,
        sig_strikes_attempted=sig_attempted,
        takedowns_landed=td_landed,
        takedowns_attempted=td_attempted,
        submission_attempts=sub_attempts,
        reversals=reversals,
        control_time_seconds=ctrl_time,
        knockdowns=knockdowns,
        head_strikes_landed=head_landed,
        body_strikes_landed=body_landed,
        leg_strikes_landed=leg_landed,
        distance_strikes_landed=distance_landed,
        clinch_strikes_landed=clinch_landed,
        ground_strikes_landed=ground_landed,
    )


def _upsert_round_stats(
    session: Session,
    fight_id: int,
    fighter_id: int,
    round_number: int,
    stats: FightStats,
) -> None:
    """Upsert a RoundStats record by natural key (fight_id, fighter_id, round_number)."""
    existing = (
        session.query(RoundStats)
        .filter(
            RoundStats.fight_id == fight_id,
            RoundStats.fighter_id == fighter_id,
            RoundStats.round_number == round_number,
        )
        .first()
    )

    stat_fields = {
        "sig_strikes_landed": stats.sig_strikes_landed,
        "sig_strikes_attempted": stats.sig_strikes_attempted,
        "takedowns_landed": stats.takedowns_landed,
        "takedowns_attempted": stats.takedowns_attempted,
        "submission_attempts": stats.submission_attempts,
        "reversals": stats.reversals,
        "control_time_seconds": stats.control_time_seconds,
        "knockdowns": stats.knockdowns,
        "head_strikes_landed": stats.head_strikes_landed,
        "body_strikes_landed": stats.body_strikes_landed,
        "leg_strikes_landed": stats.leg_strikes_landed,
        "distance_strikes_landed": stats.distance_strikes_landed,
        "clinch_strikes_landed": stats.clinch_strikes_landed,
        "ground_strikes_landed": stats.ground_strikes_landed,
    }

    if existing is not None:
        for field_name, value in stat_fields.items():
            setattr(existing, field_name, value)
    else:
        rs = RoundStats(
            fight_id=fight_id,
            fighter_id=fighter_id,
            round_number=round_number,
            **stat_fields,
        )
        session.add(rs)


def ingest_rajeevw_fights(csv_path: Path, session: Session) -> IngestResult:
    """Ingest fight results and stats from raw_total_fight_data.csv.

    Reads the SEMICOLON-delimited CSV (41 columns). Creates Events, Fights,
    and RoundStats (round_number=0 for fight-level aggregates).

    Args:
        csv_path: Path to raw_total_fight_data.csv
        session: SQLAlchemy session for database writes

    Returns:
        IngestResult with accepted/updated/rejected counts
    """
    result = IngestResult()
    df = pd.read_csv(csv_path, sep=";")
    logger.info("Reading %d fight rows from %s", len(df), csv_path)

    for idx, row in df.iterrows():
        try:
            raw = row.to_dict()

            # 1. Parse fight metadata
            event_date = parse_fight_date(str(raw["date"]) if pd.notna(raw.get("date")) else None)
            if event_date is None:
                # Per D-07: skip row if date is unparseable
                result.log_rejection(idx, raw, "Unparseable date (D-07)")
                logger.warning("Skipping fight row %d: unparseable date", idx)
                continue

            weight_class, is_title_fight = parse_weight_class(
                str(raw.get("Fight_type", "")) if pd.notna(raw.get("Fight_type")) else None
            )
            num_rounds = parse_num_rounds(
                str(raw.get("Format", "")) if pd.notna(raw.get("Format")) else None
            )

            method, method_detail = _parse_method(raw.get("win_by"))

            round_finished = _parse_int_safe(raw.get("last_round"))
            time_finished_raw = raw.get("last_round_time")
            time_finished = str(time_finished_raw).strip() if pd.notna(time_finished_raw) else None

            # Fighter names
            r_fighter = str(raw.get("R_fighter", "")).strip()
            b_fighter = str(raw.get("B_fighter", "")).strip()

            # Winner mapping — CSV uses fighter names, not "Red"/"Blue"
            winner_raw = str(raw.get("Winner", "")).strip()
            if winner_raw == r_fighter:
                winner_name = r_fighter
            elif winner_raw == b_fighter:
                winner_name = b_fighter
            elif winner_raw in ("Draw", "NC", "No Contest", ""):
                winner_name = None  # Draw / NC
            else:
                winner_name = None  # Unrecognized winner value

            # Location
            location_raw = raw.get("location")
            location = str(location_raw).strip() if pd.notna(location_raw) else None

            # Event name
            event_name = raw.get("event")
            if pd.isna(event_name) or not event_name:
                event_name = f"UFC Event - {event_date.isoformat()}"
            else:
                event_name = str(event_name).strip()

            # Build fight-level stats
            fighter_a_stats = _build_fight_stats(raw, "R_")
            fighter_b_stats = _build_fight_stats(raw, "B_")

            # Validate through Pydantic
            fight_row = FightRow(
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
                fighter_a_stats=fighter_a_stats,
                fighter_b_stats=fighter_b_stats,
            )

            # 2. Upsert Event
            event = upsert_event(
                session,
                name=fight_row.event_name,
                event_date=fight_row.event_date,
                location=fight_row.location,
                source=SOURCE,
            )

            # 3. Upsert Fighters (create minimal if not found)
            fighter_a = upsert_fighter(session, fight_row.fighter_a_name, SOURCE)
            fighter_b = upsert_fighter(session, fight_row.fighter_b_name, SOURCE)

            # Determine winner_id
            winner_id = None
            if fight_row.winner_name == fight_row.fighter_a_name:
                winner_id = fighter_a.id
            elif fight_row.winner_name == fight_row.fighter_b_name:
                winner_id = fighter_b.id

            # 4. Upsert Fight (natural key: fighter_a_id, fighter_b_id, event_id)
            existing_fight = (
                session.query(Fight)
                .filter(
                    Fight.fighter_a_id == fighter_a.id,
                    Fight.fighter_b_id == fighter_b.id,
                    Fight.event_id == event.id,
                )
                .first()
            )

            if existing_fight is not None:
                existing_fight.winner_id = winner_id
                existing_fight.weight_class = fight_row.weight_class
                existing_fight.method = fight_row.method
                existing_fight.method_detail = fight_row.method_detail
                existing_fight.round_finished = fight_row.round_finished
                existing_fight.time_finished = fight_row.time_finished
                existing_fight.is_title_fight = fight_row.is_title_fight
                existing_fight.num_rounds = fight_row.num_rounds
                fight = existing_fight
                result.updated += 1
            else:
                fight = Fight(
                    event_id=event.id,
                    fighter_a_id=fighter_a.id,
                    fighter_b_id=fighter_b.id,
                    winner_id=winner_id,
                    weight_class=fight_row.weight_class,
                    method=fight_row.method,
                    method_detail=fight_row.method_detail,
                    round_finished=fight_row.round_finished,
                    time_finished=fight_row.time_finished,
                    is_title_fight=fight_row.is_title_fight,
                    num_rounds=fight_row.num_rounds,
                    source=SOURCE,
                )
                session.add(fight)
                session.flush()
                result.accepted += 1

            # 5. Upsert fight-level stats as RoundStats with round_number=0
            if fight_row.fighter_a_stats is not None:
                _upsert_round_stats(session, fight.id, fighter_a.id, 0, fight_row.fighter_a_stats)
            if fight_row.fighter_b_stats is not None:
                _upsert_round_stats(session, fight.id, fighter_b.id, 0, fight_row.fighter_b_stats)

            # Batch commit for performance
            if (idx + 1) % _BATCH_SIZE == 0:
                session.commit()
                logger.info("Committed %d fight rows...", idx + 1)

        except Exception as exc:
            # Per D-06: catch ANY exception, log rejection, continue
            session.rollback()
            result.log_rejection(idx, row.to_dict(), str(exc))
            logger.warning("Rejected fight row %d: %s", idx, exc)
            continue

    # Final commit
    session.commit()
    logger.info(result.summary())
    return result
