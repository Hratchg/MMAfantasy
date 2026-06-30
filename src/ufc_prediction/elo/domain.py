"""Domain-specific Elo computation -- striking and grappling ratings.

Second-pass computation that reads overall Elo deltas and round-by-round
performance stats to produce domain-attributed Elo snapshots. Runs after
the overall EloEngine computation (D-09).

Attribution model:
  - KO/TKO: 80% striking / 20% grappling (D-01)
  - Submission: 20% striking / 80% grappling (D-01)
  - Decision/DQ/Draw/other: stats-based ratio from round data (D-02)
  - No round data: skip domain update (D-06)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import FightRecord, SnapshotRecord

# Weight classes that never trigger division transfers (mirrors engine.py)
_NON_TRANSFER_DIVISIONS = frozenset({"Catch Weight", "Open Weight"})

# Fixed attribution ratios for finish methods (D-01, validated Phase 9.1) [E]
# Backtest optimal: KO -> 100% striking / 0% grappling; Sub -> 0% / 100%
# Brier score 0.183334 (best of 25 configs); was (0.8, 0.2) / (0.2, 0.8) [L]
# Mapping: method -> (striking_ratio, grappling_ratio)
# Includes scraper-format aliases "TKO" and "SUB" to match UFCStats method strings.
_FINISH_RATIOS: dict[str, tuple[float, float]] = {
    "KO/TKO": (1.0, 0.0),  # was (0.8, 0.2) [L]; backtest optimal: full striking
    "TKO": (1.0, 0.0),  # scraper-format alias for KO/TKO
    "Submission": (0.0, 1.0),  # was (0.2, 0.8) [L]; backtest optimal: full grappling
    "SUB": (0.0, 1.0),  # scraper-format alias for Submission
}


@dataclass(frozen=True)
class DomainAttribution:
    """Attribution ratios for splitting overall Elo delta between domains."""

    fight_id: int
    striking_ratio: float
    grappling_ratio: float


def compute_attribution(
    method: str | None,
    round_stats: list[dict[str, object]] | None,
    fight_id: int,
) -> DomainAttribution | None:
    """Compute domain attribution ratios for a fight.

    Returns None if round_stats is None or empty (D-06: skip domain update).

    For KO/TKO and Submission, uses fixed ratios (D-01).
    For all other methods, computes from round stats (D-02).
    """
    if not round_stats:
        return None

    # Check for fixed finish ratios (D-01)
    if method in _FINISH_RATIOS:
        striking_r, grappling_r = _FINISH_RATIOS[method]
        return DomainAttribution(
            fight_id=fight_id,
            striking_ratio=striking_r,
            grappling_ratio=grappling_r,
        )

    # Stats-based ratio for Decision, DQ, Draw, and other methods (D-02)
    striking_signal = 0.0
    grappling_signal = 0.0

    for entry in round_stats:
        sig_str: float = float(entry.get("sig_str_landed") or 0)  # type: ignore[arg-type]
        td: float = float(entry.get("takedowns_landed") or 0)  # type: ignore[arg-type]
        ctrl: float = float(entry.get("ctrl_time_seconds") or 0)  # type: ignore[arg-type]
        striking_signal += sig_str
        grappling_signal += td + ctrl

    total = striking_signal + grappling_signal

    # Zero signal fallback: 50/50 per D-02
    if total == 0:
        return DomainAttribution(
            fight_id=fight_id,
            striking_ratio=0.5,
            grappling_ratio=0.5,
        )

    striking_ratio = striking_signal / total
    return DomainAttribution(
        fight_id=fight_id,
        striking_ratio=striking_ratio,
        grappling_ratio=1.0 - striking_ratio,
    )


class DomainEloComputer:
    """Computes domain-specific (striking/grappling) Elo ratings.

    Operates as a second pass after overall Elo computation (D-09).
    Reads overall Elo deltas and round stats to produce domain snapshots.

    Domain ratings start at 1500.0 with independent K-factor decay,
    Bayesian shrinkage, inactivity regression, and division transfers
    (D-08).
    """

    def __init__(self, config: EloConfig) -> None:
        self.config = config
        # (fighter_id, division) -> current raw domain Elo
        self._striking_ratings: dict[tuple[int, str], float] = {}
        self._grappling_ratings: dict[tuple[int, str], float] = {}
        # Independent fight counts for domain K-factor decay (Pitfall 5 / D-08)
        self._domain_fight_counts: dict[int, int] = {}
        # Independent last_fight_date for domain inactivity regression (Pitfall 4 / A2)
        self._domain_last_fight_date: dict[int, date] = {}
        # Last division for domain-specific division transfers
        self._domain_last_division: dict[int, str] = {}

    def compute_all(
        self,
        fights: list[FightRecord],
        overall_snapshots: list[SnapshotRecord],
        round_stats_by_fight: dict[int, list[dict[str, object]]],
    ) -> list[SnapshotRecord]:
        """Process all fights and produce domain Elo snapshots.

        Args:
            fights: Chronologically sorted fight records (same list as overall engine).
            overall_snapshots: SnapshotRecords from overall Elo computation.
            round_stats_by_fight: Dict mapping fight_id -> list of round stat dicts.

        Returns:
            List of SnapshotRecords with elo_type "striking" or "grappling".
        """
        # Build overall snapshot index: (fight_id, fighter_id) -> SnapshotRecord
        overall_index: dict[tuple[int, int], SnapshotRecord] = {}
        for snap in overall_snapshots:
            overall_index[(snap.fight_id, snap.fighter_id)] = snap

        domain_snapshots: list[SnapshotRecord] = []

        for fight in fights:
            # Look up overall snapshots for both fighters
            snap_a = overall_index.get((fight.fight_id, fight.fighter_a_id))
            snap_b = overall_index.get((fight.fight_id, fight.fighter_b_id))

            # If either is missing, fight was skipped by overall engine
            if snap_a is None or snap_b is None:
                continue

            # Get round stats for this fight
            round_stats = round_stats_by_fight.get(fight.fight_id)

            # Compute attribution
            attribution = compute_attribution(fight.method, round_stats, fight.fight_id)

            # If attribution is None (no round data per D-06), skip domain update
            if attribution is None:
                continue

            # Process domain fight
            fight_snaps = self._process_domain_fight(
                fight,
                snap_a,
                snap_b,
                attribution,
            )
            domain_snapshots.extend(fight_snaps)

        return domain_snapshots

    def _process_domain_fight(
        self,
        fight: FightRecord,
        overall_snap_a: SnapshotRecord,
        overall_snap_b: SnapshotRecord,
        attribution: DomainAttribution,
    ) -> list[SnapshotRecord]:
        """Process a single fight for both domains, producing 4 SnapshotRecords."""
        snapshots: list[SnapshotRecord] = []
        division = fight.weight_class

        domain_configs = [
            ("striking", attribution.striking_ratio, self._striking_ratings),
            ("grappling", attribution.grappling_ratio, self._grappling_ratings),
        ]

        fighter_snaps = [
            (fight.fighter_a_id, overall_snap_a),
            (fight.fighter_b_id, overall_snap_b),
        ]

        for domain_name, ratio, ratings_dict in domain_configs:
            for fighter_id, overall_snap in fighter_snaps:
                # Pre-fight adjustments
                self._apply_domain_inactivity_regression(
                    fighter_id,
                    fight.event_date,
                    ratings_dict,
                )
                self._check_domain_division_transfer(
                    fighter_id,
                    division,
                    ratings_dict,
                )

                # Get current domain rating (default 1500.0)
                key = (fighter_id, division)
                elo_before = ratings_dict.get(key, self.config.initial_rating)

                # Compute domain delta = overall_delta * ratio (D-03)
                overall_delta = overall_snap.elo_after - overall_snap.elo_before
                domain_delta = overall_delta * ratio

                # Compute new domain rating
                elo_after = elo_before + domain_delta
                ratings_dict[key] = elo_after

                # Increment domain fight count
                self._domain_fight_counts[fighter_id] = (
                    self._domain_fight_counts.get(fighter_id, 0) + 1
                )
                count_after = self._domain_fight_counts[fighter_id]

                # Update domain last_fight_date
                self._domain_last_fight_date[fighter_id] = fight.event_date

                # Update domain last_division (skip Catch Weight / Open Weight)
                if division not in _NON_TRANSFER_DIVISIONS:
                    self._domain_last_division[fighter_id] = division

                # Compute shrinkage (same formula as EloEngine, per D-08)
                elo_after_shrinkage = self._compute_shrinkage(elo_after, count_after)

                # k_factor_used = overall_snap.k_factor_used * ratio
                k_factor_used = overall_snap.k_factor_used * ratio

                snapshots.append(
                    SnapshotRecord(
                        fighter_id=fighter_id,
                        fight_id=fight.fight_id,
                        division=division,
                        elo_type=domain_name,
                        elo_before=elo_before,
                        elo_after=elo_after,
                        elo_after_shrinkage=elo_after_shrinkage,
                        k_factor_used=k_factor_used,
                        fight_date=fight.event_date,
                    ),
                )

        return snapshots

    def _apply_domain_inactivity_regression(
        self,
        fighter_id: int,
        current_date: date,
        ratings_dict: dict[tuple[int, str], float],
    ) -> None:
        """Regress domain rating toward initial if fighter inactive > threshold.

        Same logic as EloEngine._apply_inactivity_regression but operates
        on domain rating dict using domain_last_fight_date (D-08).
        """
        last_date = self._domain_last_fight_date.get(fighter_id)
        if last_date is None:
            return  # first domain fight, no regression

        days_inactive = (current_date - last_date).days
        if days_inactive <= self.config.inactivity_threshold_days:
            return

        months_past = (days_inactive - self.config.inactivity_threshold_days) / 30.0
        regression_pct = min(
            months_past * self.config.inactivity_regression_rate,
            self.config.inactivity_regression_cap,
        )

        initial = self.config.initial_rating
        for key in list(ratings_dict):
            if key[0] == fighter_id:
                current_elo = ratings_dict[key]
                delta = current_elo - initial
                ratings_dict[key] = current_elo - delta * regression_pct

    def _check_domain_division_transfer(
        self,
        fighter_id: int,
        current_division: str,
        ratings_dict: dict[tuple[int, str], float],
    ) -> None:
        """Carry 75% of domain Elo delta to new division if fighter changed divisions.

        Same logic as EloEngine._check_division_transfer but operates on
        domain rating dict using domain_last_division (D-08).
        """
        if current_division in _NON_TRANSFER_DIVISIONS:
            return

        last_div = self._domain_last_division.get(fighter_id)
        if last_div is None or last_div == current_division:
            return

        if last_div in _NON_TRANSFER_DIVISIONS:
            return

        new_key = (fighter_id, current_division)
        if new_key not in ratings_dict:
            old_key = (fighter_id, last_div)
            old_elo = ratings_dict.get(old_key, self.config.initial_rating)
            old_delta = old_elo - self.config.initial_rating
            new_elo = self.config.initial_rating + self.config.division_transfer_pct * old_delta
            ratings_dict[new_key] = new_elo

    def _compute_shrinkage(self, raw_elo: float, fight_count: int) -> float:
        """Compute Bayesian-shrunk display rating (same formula as EloEngine, per D-08).

        shrinkage_factor = min(fight_count / shrinkage_min_fights, 1.0)
        elo_after_shrinkage = initial + (raw - initial) * shrinkage_factor
        """
        shrinkage_factor = min(fight_count / self.config.shrinkage_min_fights, 1.0)
        initial = self.config.initial_rating
        return initial + (raw_elo - initial) * shrinkage_factor
