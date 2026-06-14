"""RED-GREEN unit tests for Sherdog debutant Elo seed derivation (DEBUT-V25-02).

Pins the operator-locked seed formula per Phase 43 CONTEXT decisions:

    seed = 1500
         + 100 * (win_rate - 0.5)                    # +/-50 around mean
         + tier_bonus(org_tier)                      # major +50; regional 0; local -25; none -50
         + clip(n_pre_ufc_fights / 20 * 25, 0, 25)   # experience floor
    seed = clip(seed, 1300, 1700)                    # outer bounds

Any drift in constants will break these tests. See docs/elo_seed_v25.md for rationale.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ufc_prediction.elo.seed import (
    SeedDerivationError,
    derive_seed,
    load_seeds,
    tier_bonus,
)

# Canonical 14-column header for pre_ufc_records.csv (Plan 43-01 substrate).
_CSV_HEADER = [
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


def _row(
    *,
    fighter_id: int,
    n_pre_ufc_fights: int,
    win_rate: float,
    org_tier: str,
    last_organization: str = "Some Promotion",
) -> dict[str, str]:
    """Build a canonical 14-column CSV row dict for fixture construction."""
    wins = int(round(win_rate * n_pre_ufc_fights)) if n_pre_ufc_fights else 0
    losses = n_pre_ufc_fights - wins
    return {
        "fighter_id": str(fighter_id),
        "sherdog_url": f"https://www.sherdog.com/fighter/Fixture-{fighter_id}",
        "n_pre_ufc_fights": str(n_pre_ufc_fights),
        "wins": str(wins),
        "losses": str(losses),
        "draws": "0",
        "nc_dq": "0",
        "win_rate": str(win_rate),
        "kos": "0",
        "submissions": "0",
        "decisions": str(wins),
        "last_organization": last_organization,
        "org_tier": org_tier,
        "scraped_at": "2026-06-02T18:00:00+00:00",
    }


# ----------------------------------------------------------------------------
# derive_seed formula anchor tests
# ----------------------------------------------------------------------------


def test_1_formula_center_neutral_debutant_equals_1500() -> None:
    """Neutral debutant (regional, 0.5 win_rate, 0 fights) lands on the 1500 center."""
    record = {"win_rate": 0.5, "org_tier": "regional", "n_pre_ufc_fights": 0}
    assert derive_seed(record) == 1500.0


def test_2_tier_major_high_winrate_full_experience_equals_1625() -> None:
    """Best-legitimate combo: 1.0 win_rate + major + 20 pre-UFC fights == 1625."""
    record = {"win_rate": 1.0, "org_tier": "major", "n_pre_ufc_fights": 20}
    assert derive_seed(record) == 1625.0


def test_3_tier_none_low_winrate_zero_experience_equals_1400() -> None:
    """Worst-legitimate combo: 0.0 win_rate + none + 0 fights == 1400."""
    record = {"win_rate": 0.0, "org_tier": "none", "n_pre_ufc_fights": 0}
    assert derive_seed(record) == 1400.0


def test_4_outer_clip_lower_bound_not_reached_by_worst_legit_case() -> None:
    """Lower clip (1300) is dormant for legitimate inputs; worst case stops at 1400."""
    worst = derive_seed(
        {"win_rate": 0.0, "org_tier": "none", "n_pre_ufc_fights": 0}
    )
    assert worst == 1400.0
    assert worst > 1300.0


def test_5_outer_clip_upper_bound_not_reached_by_best_legit_case() -> None:
    """Upper clip (1700) is dormant for legitimate inputs; best case stops at 1625."""
    best = derive_seed(
        {"win_rate": 1.0, "org_tier": "major", "n_pre_ufc_fights": 20}
    )
    assert best == 1625.0
    assert best < 1700.0


def test_6_experience_floor_saturates_at_plus_25() -> None:
    """n=100 pre-UFC fights -> raw component 125, clipped to +25."""
    record = {"win_rate": 0.5, "org_tier": "regional", "n_pre_ufc_fights": 100}
    assert derive_seed(record) == 1525.0


def test_7_experience_floor_partial_linear_at_n_equals_10() -> None:
    """n=10 -> +12.5 experience component (linear before the +25 cap)."""
    record = {"win_rate": 0.5, "org_tier": "regional", "n_pre_ufc_fights": 10}
    assert derive_seed(record) == pytest.approx(1512.5)


# ----------------------------------------------------------------------------
# tier_bonus tests
# ----------------------------------------------------------------------------


def test_8_tier_bonus_table_pins_all_four_tiers_and_rejects_unknown() -> None:
    """Tier bonuses are integer constants {50, 0, -25, -50}; unknown tier raises ValueError."""
    assert tier_bonus("major") == 50
    assert tier_bonus("regional") == 0
    assert tier_bonus("local") == -25
    assert tier_bonus("none") == -50
    with pytest.raises(ValueError, match="Unknown org_tier"):
        tier_bonus("unknown")


# ----------------------------------------------------------------------------
# load_seeds tests
# ----------------------------------------------------------------------------


def test_9_load_seeds_happy_path_three_rows_three_tiers(tmp_path: Path) -> None:
    """Build a 3-row CSV (one each of major/none/regional) and verify
    load_seeds returns the expected fighter_id -> derive_seed mapping."""
    csv_path = tmp_path / "pre_ufc_records.csv"
    rows = [
        _row(
            fighter_id=1,
            n_pre_ufc_fights=20,
            win_rate=1.0,
            org_tier="major",
            last_organization="Bellator",
        ),
        _row(
            fighter_id=2,
            n_pre_ufc_fights=0,
            win_rate=0.0,
            org_tier="none",
            last_organization="",
        ),
        _row(
            fighter_id=3,
            n_pre_ufc_fights=10,
            win_rate=0.5,
            org_tier="regional",
            last_organization="LFA",
        ),
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    seeds = load_seeds(csv_path)

    assert set(seeds.keys()) == {1, 2, 3}
    assert all(isinstance(k, int) for k in seeds.keys())
    assert all(isinstance(v, float) for v in seeds.values())
    assert seeds[1] == 1625.0
    assert seeds[2] == 1400.0
    assert seeds[3] == pytest.approx(1512.5)


def test_10_load_seeds_missing_file_returns_empty_dict() -> None:
    """Plan 43-03 graceful-fallback contract: missing CSV -> {} (no exception)."""
    missing = Path("/nonexistent/path/should/not/exist/pre_ufc_records.csv")
    assert load_seeds(missing) == {}


def test_11_load_seeds_malformed_row_raises_seed_derivation_error(
    tmp_path: Path,
) -> None:
    """An empty required column (org_tier) raises SeedDerivationError naming the row + column."""
    csv_path = tmp_path / "pre_ufc_records.csv"
    bad_row = _row(
        fighter_id=42,
        n_pre_ufc_fights=5,
        win_rate=0.6,
        org_tier="",
        last_organization="MysteryFC",
    )
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        writer.writerow(bad_row)

    with pytest.raises(SeedDerivationError) as exc:
        load_seeds(csv_path)

    msg = str(exc.value)
    assert "42" in msg
    assert "org_tier" in msg


# ----------------------------------------------------------------------------
# determinism test
# ----------------------------------------------------------------------------


def test_12_derive_seed_is_deterministic_bit_exact() -> None:
    """Same input -> same float, every time. No randomness, no side effects."""
    record = {"win_rate": 0.7, "org_tier": "major", "n_pre_ufc_fights": 15}
    first = derive_seed(record)
    second = derive_seed(record)
    third = derive_seed(record)
    assert first == second == third
