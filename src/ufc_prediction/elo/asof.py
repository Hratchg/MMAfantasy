"""Point-in-time Elo reconstruction from stored snapshots.

The training matrix reads ``EloSnapshot.elo_before`` for the fight being
labelled: the engine's per-division raw rating AFTER the pre-fight
inactivity regression and division transfer, BEFORE the fight's own delta
and before any display shrinkage. The serve path has no future fight row
to read ``elo_before`` from, so it must reconstruct the same quantity from
the fighter's stored history. :func:`pre_fight_rating` replays the engine's
bookkeeping over that history and applies the same pre-fight adjustments
at the as-of date.

Parity with the engines is pinned by ``tests/unit/elo/test_asof_parity.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import _NON_TRANSFER_DIVISIONS


@dataclass(frozen=True)
class RatedFight:
    """One stored snapshot of a fighter in one Elo type."""

    fight_date: date
    division: str
    elo_after: float


def _regress(
    ratings: dict[str, float],
    last_date: date | None,
    current_date: date,
    config: EloConfig,
) -> None:
    """Mirror ``EloEngine._apply_inactivity_regression`` for one fighter."""
    if last_date is None:
        return
    days_inactive = (current_date - last_date).days
    if days_inactive <= config.inactivity_threshold_days:
        return
    months_past = (days_inactive - config.inactivity_threshold_days) / 30.0
    regression_pct = min(
        months_past * config.inactivity_regression_rate,
        config.inactivity_regression_cap,
    )
    initial = config.initial_rating
    for division in list(ratings):
        delta = ratings[division] - initial
        ratings[division] = ratings[division] - delta * regression_pct


def _transfer(
    ratings: dict[str, float],
    last_division: str | None,
    current_division: str,
    config: EloConfig,
) -> None:
    """Mirror ``EloEngine._check_division_transfer`` for one fighter."""
    if current_division in _NON_TRANSFER_DIVISIONS:
        return
    if last_division is None or last_division == current_division:
        return
    if last_division in _NON_TRANSFER_DIVISIONS:
        return
    if current_division not in ratings:
        old_elo = ratings.get(last_division, config.initial_rating)
        old_delta = old_elo - config.initial_rating
        ratings[current_division] = config.initial_rating + config.division_transfer_pct * old_delta


def pre_fight_rating(
    history: Sequence[RatedFight],
    as_of: date,
    division: str | None,
    *,
    config: EloConfig | None = None,
    seed: float | None = None,
    regress: bool = True,
    transfer: bool = True,
) -> float:
    """Return the rating the engine would use as ``elo_before`` for a fight.

    Args:
        history: The fighter's stored snapshots for ONE ``elo_type``, all
            dated strictly before ``as_of``. Any order; sorted here.
        as_of: Date of the upcoming fight.
        division: Weight class of the upcoming fight. ``None`` means unknown
            and falls back to the fighter's most recent transferable
            division (or the last one fought if none is transferable).
        config: Elo configuration; defaults to ``EloConfig()``.
        seed: Debutant seed for a ``(fighter, division)`` key with no rating,
            exactly like ``EloEngine._lookup_initial_rating``. ``None`` means
            ``config.initial_rating``. Domain Elo never seeds.
        regress: Apply inactivity regression. Every Elo type (overall,
            striking, grappling) applies it; the flag exists so callers can
            replay legacy snapshot sets written before ``DomainEloComputer``
            kept per-domain bookkeeping.
        transfer: Apply division transfer (same scope as ``regress``).
    """
    cfg = config or EloConfig()
    ratings: dict[str, float] = {}
    last_date: date | None = None
    last_division: str | None = None

    for snap in sorted(history, key=lambda s: s.fight_date):
        if snap.fight_date >= as_of:
            raise ValueError(
                f"history contains a snapshot dated {snap.fight_date}, not before as_of={as_of}"
            )
        if regress:
            _regress(ratings, last_date, snap.fight_date, cfg)
        if transfer:
            _transfer(ratings, last_division, snap.division, cfg)
        ratings[snap.division] = snap.elo_after
        last_date = snap.fight_date
        if snap.division not in _NON_TRANSFER_DIVISIONS:
            last_division = snap.division
        elif last_division is None:
            # Never fought a transferable division yet; remember something so
            # an unknown upcoming division still resolves to a real rating.
            last_division = None

    if division is None:
        division = last_division
        if division is None and history:
            division = max(history, key=lambda s: s.fight_date).division
    if division is None:
        return seed if seed is not None else cfg.initial_rating

    if regress:
        _regress(ratings, last_date, as_of, cfg)
    if transfer:
        _transfer(ratings, last_division, division, cfg)
    if division in ratings:
        return ratings[division]
    return seed if seed is not None else cfg.initial_rating
