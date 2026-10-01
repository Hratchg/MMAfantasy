"""Pure (no-DB) tests for the Sherdog seed-input row conversion in ``elo.seed_store``.

``seed_row_from_csv`` turns one ``pre_ufc_records.csv`` row (text from the
file, or the native values ``scripts/ingest_pre_ufc_records_v25.py`` builds)
into a ``debutant_seed_inputs`` row. It must fail closed on anything
``load_seeds`` would reject, and on missing provenance, so a bad row never
lands in the table the seeds are read from.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ufc_prediction.elo.seed import SeedDerivationError, load_seeds
from ufc_prediction.elo.seed_store import parse_seed_csv, seed_row_from_csv

_HEADER = [
    "fighter_id",
    "sherdog_url",
    "n_pre_ufc_fights",
    "wins",
    "losses",
    "draws",
    "nc_dq",
    "win_rate",
    "kos",
    "submissions",
    "decisions",
    "last_organization",
    "org_tier",
    "scraped_at",
]


def _csv_row(**overrides: str) -> dict[str, str]:
    row = {
        "fighter_id": "4282",
        "sherdog_url": "https://www.sherdog.com/fighter/Dan-Henderson-195",
        "n_pre_ufc_fights": "9",
        "wins": "7",
        "losses": "2",
        "draws": "0",
        "nc_dq": "0",
        "win_rate": "0.7777777777777778",
        "kos": "0",
        "submissions": "6",
        "decisions": "1",
        "last_organization": "Brazil Open - '97",
        "org_tier": "local",
        "scraped_at": "2026-07-03T23:38:19.320540+00:00",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_text_row_converts_to_typed_db_row() -> None:
    out = seed_row_from_csv(_csv_row())

    assert out == {
        "fighter_id": 4282,
        "sherdog_url": "https://www.sherdog.com/fighter/Dan-Henderson-195",
        "n_pre_ufc_fights": 9,
        "wins": 7,
        "losses": 2,
        "draws": 0,
        "nc_dq": 0,
        "win_rate": 0.7777777777777778,
        "kos": 0,
        "submissions": 6,
        "decisions": 1,
        "last_organization": "Brazil Open - '97",
        "org_tier": "local",
        "scraped_at": datetime(2026, 7, 3, 23, 38, 19, 320540, tzinfo=UTC),
    }
    # Exact float, not approximately equal: the seed formula consumes it.
    assert out["win_rate"] == float("0.7777777777777778")


def test_native_values_from_the_ingest_script_are_accepted() -> None:
    out = seed_row_from_csv(
        {
            "fighter_id": 7,
            "sherdog_url": "https://www.sherdog.com/fighter/X-7",
            "n_pre_ufc_fights": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "nc_dq": 0,
            "win_rate": 0.0,
            "kos": 0,
            "submissions": 0,
            "decisions": 0,
            "last_organization": "",
            "org_tier": "none",
            "scraped_at": "2026-07-04T01:06:50.276709+00:00",
        }
    )
    assert out["fighter_id"] == 7
    assert out["win_rate"] == 0.0
    assert out["org_tier"] == "none"
    # Blank optional text becomes NULL, not an empty string.
    assert out["last_organization"] is None


def test_blank_optional_counts_become_null() -> None:
    out = seed_row_from_csv(_csv_row(wins="", kos="", last_organization=""))
    assert out["wins"] is None
    assert out["kos"] is None
    assert out["last_organization"] is None


@pytest.mark.parametrize(
    "column",
    ["fighter_id", "n_pre_ufc_fights", "win_rate", "org_tier", "sherdog_url", "scraped_at"],
)
def test_blank_required_column_raises(column: str) -> None:
    with pytest.raises(SeedDerivationError, match=column):
        seed_row_from_csv(_csv_row(**{column: " "}), row_idx=5)


def test_error_names_the_row() -> None:
    with pytest.raises(SeedDerivationError, match="Row 12"):
        seed_row_from_csv(_csv_row(win_rate=""), row_idx=12)


def test_unknown_org_tier_raises() -> None:
    with pytest.raises(SeedDerivationError, match="org_tier"):
        seed_row_from_csv(_csv_row(org_tier="Major"))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("win_rate", "1.5"),
        ("win_rate", "-0.1"),
        ("win_rate", "nan"),
        ("win_rate", "abc"),
        ("n_pre_ufc_fights", "-1"),
        ("n_pre_ufc_fights", "3.5"),
        ("fighter_id", "x"),
        ("scraped_at", "yesterday"),
    ],
)
def test_out_of_range_or_unparseable_value_raises(column: str, value: str) -> None:
    with pytest.raises(SeedDerivationError, match=column):
        seed_row_from_csv(_csv_row(**{column: value}))


def test_naive_scraped_at_is_read_as_utc() -> None:
    out = seed_row_from_csv(_csv_row(scraped_at="2026-07-03T23:38:19"))
    assert out["scraped_at"] == datetime(2026, 7, 3, 23, 38, 19, tzinfo=UTC)


def test_parse_seed_csv_matches_load_seeds_keys_and_keeps_last_duplicate(
    tmp_path: Path,
) -> None:
    """Duplicate fighter_ids resolve last-wins, exactly like ``load_seeds``,
    so the backfilled table seeds the same value the CSV would have."""
    path = _write_csv(
        tmp_path / "pre_ufc_records.csv",
        [
            _csv_row(fighter_id="2", win_rate="0.5", org_tier="major"),
            _csv_row(fighter_id="1"),
            _csv_row(fighter_id="2", win_rate="0.25", org_tier="none"),
        ],
    )

    rows = parse_seed_csv(path)

    assert [r["fighter_id"] for r in rows] == [1, 2]
    assert rows[1]["win_rate"] == 0.25
    assert rows[1]["org_tier"] == "none"
    assert {r["fighter_id"] for r in rows} == set(load_seeds(path))


def test_parse_seed_csv_reports_header_row_offset(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "pre_ufc_records.csv",
        [_csv_row(fighter_id="1"), _csv_row(fighter_id="2", org_tier="")],
    )
    # Row 1 is the header, so the second data row is row 3 (load_seeds' numbering).
    with pytest.raises(SeedDerivationError, match="Row 3"):
        parse_seed_csv(path)


def test_parse_seed_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_seed_csv(tmp_path / "absent.csv")


def test_table_tier_set_matches_the_locked_seed_formula() -> None:
    """The DB CHECK / converter tiers must be exactly the tiers derive_seed prices."""
    from ufc_prediction.elo.seed import _TIER_BONUSES
    from ufc_prediction.models.debutant_seed_input import ORG_TIERS

    assert set(ORG_TIERS) == set(_TIER_BONUSES)
