"""Sherdog debutant Elo seed derivation (DEBUT-V25-02).

Formula is OPERATOR-LOCKED per Phase 43 CONTEXT decisions.
Do NOT modify the constants or formula shape without operator approval.
See docs/elo_seed_v25.md (elo_seed_v25) for the canonical rationale + bounds
analysis. Any divergence between code constants here and the doc is a defect.

This module is pure-function and stdlib-only by design:
- `derive_seed` is referentially transparent — no I/O, no state, no randomness.
- `load_seeds` does a single pass over a CSV produced by Plan 43-01; raises on
  malformed rows (no silent skipping) so the operator gets a load-time error
  rather than a wrong-seed-at-runtime.

Consumer contract (Plan 43-03 EloEngine init):

    SEEDS: dict[int, float] = load_seeds(Path("data/sherdog/pre_ufc_records.csv"))
    rating = SEEDS.get(fighter_id, config.initial_rating)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locked per Phase 43 CONTEXT decisions D — DO NOT MODIFY without operator
# approval. Drift here breaks tests/unit/elo/test_seed_v25.py and the
# bounds-analysis table in docs/elo_seed_v25.md.
# ---------------------------------------------------------------------------

_TIER_BONUSES: dict[str, int] = {
    "major": 50,
    "regional": 0,
    "local": -25,
    "none": -50,
}

_SEED_LOWER_BOUND: float = 1300.0
_SEED_UPPER_BOUND: float = 1700.0
_DEFAULT_SEED: float = 1500.0
_WIN_RATE_SCALE: float = 100.0
_EXPERIENCE_DIVISOR: float = 20.0
_EXPERIENCE_BONUS_MAX: float = 25.0

# Required columns enforced PER ROW in load_seeds (not header-only) so future
# schema extensions can append columns without breaking ingestion here.
_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"fighter_id", "n_pre_ufc_fights", "win_rate", "org_tier"}
)


class SeedDerivationError(ValueError):
    """Raised when a pre-UFC record row is malformed for seed derivation.

    Carries enough context (row number, fighter_id, offending column) for the
    operator to locate and fix the substrate row immediately.
    """


def tier_bonus(tier: str) -> int:
    """Return the locked tier bonus for the given org-tier label.

    Raises ValueError on unknown tier — no silent fallback (typo guard).
    """
    if tier not in _TIER_BONUSES:
        raise ValueError(f"Unknown org_tier: {tier!r}. Expected one of {sorted(_TIER_BONUSES)}.")
    return _TIER_BONUSES[tier]


def derive_seed(record: dict[str, Any]) -> float:
    """Apply the locked seed formula to a pre-UFC record dict.

    Required fields:
        n_pre_ufc_fights: int -- experience-floor input
        win_rate: float in [0.0, 1.0] -- 100x scaled around 0.5
        org_tier: str in {"major", "regional", "local", "none"}

    Returns:
        Seed value clipped to [1300.0, 1700.0].

    Formula (per Phase 43 CONTEXT D — LOCKED):
        seed = 1500
             + 100 * (win_rate - 0.5)
             + tier_bonus(org_tier)
             + clip(n_pre_ufc_fights / 20 * 25, 0, 25)
        seed = clip(seed, 1300, 1700)
    """
    win_rate = float(record["win_rate"])
    n_fights = int(record["n_pre_ufc_fights"])
    tier = str(record["org_tier"])

    win_component = _WIN_RATE_SCALE * (win_rate - 0.5)
    tier_component = float(tier_bonus(tier))

    # Experience floor: linear up to 20 fights, capped at +25.
    exp_raw = (n_fights / _EXPERIENCE_DIVISOR) * _EXPERIENCE_BONUS_MAX
    exp_component = max(0.0, min(_EXPERIENCE_BONUS_MAX, exp_raw))

    seed = _DEFAULT_SEED + win_component + tier_component + exp_component
    # Outer clip — dormant for legitimate inputs; safety guard for anomalies.
    seed = max(_SEED_LOWER_BOUND, min(_SEED_UPPER_BOUND, seed))
    return seed


def load_seeds(csv_path: Path | str) -> dict[int, float]:
    """Load `pre_ufc_records.csv` and return ``{fighter_id: seed_value}``.

    Returns an empty dict if `csv_path` does not exist — this is the graceful
    fallback contract that Plan 43-03's EloEngine dispatch logic depends on.

    Raises SeedDerivationError if any row is missing one of the required
    columns ({fighter_id, n_pre_ufc_fights, win_rate, org_tier}) or holds an
    empty/blank value for one.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}

    seeds: dict[int, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):  # row 1 is header
            fighter_id_str = (row.get("fighter_id") or "").strip()
            if not fighter_id_str:
                raise SeedDerivationError(f"Row {row_idx}: required column 'fighter_id' is empty.")
            try:
                fighter_id = int(fighter_id_str)
            except ValueError as exc:
                raise SeedDerivationError(
                    f"Row {row_idx} fighter_id={fighter_id_str!r}: not an integer ({exc})."
                ) from exc

            for col in _REQUIRED_COLUMNS:
                val = row.get(col)
                if val is None or str(val).strip() == "":
                    raise SeedDerivationError(
                        f"Row {row_idx} fighter_id={fighter_id}: required column {col!r} is empty."
                    )

            seeds[fighter_id] = derive_seed(row)
    return seeds


__all__ = [
    "SeedDerivationError",
    "derive_seed",
    "load_seeds",
    "tier_bonus",
]
