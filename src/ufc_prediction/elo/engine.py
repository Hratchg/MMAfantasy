"""Core Elo rating engine -- pure-Python, database-free computation.

Processes a chronological list of FightRecord objects and produces
SnapshotRecord objects with correct Elo ratings, K-factor decay,
MOV weighting, division transfers, Bayesian shrinkage, and inactivity
regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ufc_prediction.elo.config import EloConfig

# Weight classes that never trigger division transfers (per D-06)
_NON_TRANSFER_DIVISIONS = frozenset({"Catch Weight", "Open Weight"})

# Methods that indicate a no-contest / should be skipped (per D-02)
_SKIP_METHODS = frozenset({None, "Other", "No Contest"})


@dataclass
class FightRecord:
    """Plain data transfer object representing a fight for Elo processing."""

    fight_id: int
    event_date: date
    fighter_a_id: int
    fighter_b_id: int
    winner_id: int | None  # None = draw or no-contest
    weight_class: str
    method: str | None  # "KO/TKO", "Submission", "Decision", "DQ", "Other", etc.
    method_detail: str | None  # "Unanimous", "Split", "Majority" for decisions


@dataclass
class SnapshotRecord:
    """Plain data object for an Elo snapshot result."""

    fighter_id: int
    fight_id: int
    division: str
    elo_type: str  # Always "overall" in this phase
    elo_before: float
    elo_after: float
    elo_after_shrinkage: float
    k_factor_used: float
    fight_date: date


class EloEngine:
    """Stateless Elo computation engine.

    Takes an EloConfig and operates on plain Python data structures.
    No database dependency -- receives fight data as a list of FightRecord
    objects and returns SnapshotRecord objects.
    """

    def __init__(
        self,
        config: EloConfig,
        seeds: dict[int, float] | None = None,
    ) -> None:
        """Construct an EloEngine.

        Args:
            config: EloConfig with all tunable parameters.
            seeds: Optional ``fighter_id -> initial Elo`` override per
                DEBUT-V25-03 (Phase 43). Empty dict (or None) means use
                ``config.initial_rating`` uniformly (pre-Phase-43 behavior —
                a bit-exact regression invariant guarded by
                ``tests/unit/elo/test_engine_seed_dispatch.py::test_3``).

                When provided, the seed is consulted ONLY on first-encounter
                of a fighter in a division (i.e. when ``(fighter_id, division)
                not in self._ratings``). Once a fighter is rated in a
                division, subsequent lookups return the current rating and
                the seed is dormant for that ``(fighter_id, division)`` key.
        """
        self.config = config
        # Plan 43-03 -- DEBUT-V25-03 substrate. Empty dict by default keeps
        # the engine's pre-Phase-43 behavior bit-exact (Test 3 invariant).
        self.seeds: dict[int, float] = dict(seeds) if seeds else {}
        # (fighter_id, division) -> current raw Elo
        self._ratings: dict[tuple[int, str], float] = {}
        # fighter_id -> total fight count across all divisions (per D-04)
        self._fight_counts: dict[int, int] = {}
        # fighter_id -> date of last fight (for inactivity regression)
        self._last_fight_date: dict[int, date] = {}
        # fighter_id -> last division fought in (for division transfer detection)
        self._last_division: dict[int, str] = {}

    def compute_all(self, fights: list[FightRecord]) -> list[SnapshotRecord]:
        """Process all fights in chronological order, return snapshots.

        Fights MUST be pre-sorted by (event_date, fight_id).
        """
        snapshots: list[SnapshotRecord] = []
        for fight in fights:
            result = self._process_fight(fight)
            if result is not None:
                snapshots.extend(result)
        return snapshots

    def get_rating(self, fighter_id: int, division: str) -> float:
        """Return current rating for a fighter in a division.

        Dispatches through ``_lookup_initial_rating`` so seeded debutants get
        their Phase 43 seed value at first encounter (per DEBUT-V25-03), and
        unseeded fighters fall back to ``config.initial_rating`` (1500).
        """
        return self._lookup_initial_rating(fighter_id, division)

    def _lookup_initial_rating(self, fighter_id: int, division: str) -> float:
        """Return the rating to use for this fighter in this division.

        Dispatch priority (per DEBUT-V25-03):
            1) Current rating if already-rated in this division (state from
               a prior fight; ``self._ratings`` lookup).
            2) Seed from ``self.seeds`` if this is first encounter for the
               ``(fighter_id, division)`` key.
            3) ``config.initial_rating`` (1500) as the final fallback when
               no seed is registered for this fighter.

        DomainEloComputer is intentionally NOT routed through this helper
        (Phase 43 scope is overall Elo only; domain striking/grappling Elo
        continues to start at 1500 unconditionally — see domain.py).
        """
        key = (fighter_id, division)
        if key in self._ratings:
            return self._ratings[key]
        return self.seeds.get(fighter_id, self.config.initial_rating)

    def expected_win_probability(self, rating_a: float, rating_b: float) -> float:
        """Compute expected win probability for fighter A (per D-11).

        E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)))

    def _process_fight(self, fight: FightRecord) -> list[SnapshotRecord] | None:
        """Process a single fight and return two SnapshotRecords (or None if skipped)."""
        # Skip logic (per D-02): no-contest detection
        if fight.winner_id is None and fight.method in _SKIP_METHODS:
            return None

        is_draw = fight.winner_id is None
        division = fight.weight_class

        # Pre-fight adjustments for both fighters
        for fighter_id in (fight.fighter_a_id, fight.fighter_b_id):
            self._apply_inactivity_regression(fighter_id, fight.event_date)
            self._check_division_transfer(fighter_id, division)

        # Get current ratings (default to initial_rating per D-10; seed
        # dispatched per DEBUT-V25-03 for first-encounter of seeded fighters)
        cfg = self.config
        key_a = (fight.fighter_a_id, division)
        key_b = (fight.fighter_b_id, division)
        rating_a = self._lookup_initial_rating(fight.fighter_a_id, division)
        rating_b = self._lookup_initial_rating(fight.fighter_b_id, division)

        # Expected win probabilities (per D-11)
        e_a = self.expected_win_probability(rating_a, rating_b)
        e_b = 1.0 - e_a

        # K-factors (per D-03, D-04: per-fighter count, not per-division)
        k_a = self._get_k_factor(fight.fighter_a_id)
        k_b = self._get_k_factor(fight.fighter_b_id)

        # MOV multiplier (per D-01)
        mov = self._get_mov_multiplier(fight.method, fight.method_detail)

        # Compute Elo deltas
        if is_draw:
            # Draw: both fighters score 0.5 (per D-01 draw multiplier)
            mov_draw = cfg.mov_draw
            delta_a = k_a * mov_draw * (0.5 - e_a)
            delta_b = k_b * mov_draw * (0.5 - e_b)
            effective_k_a = k_a * mov_draw
            effective_k_b = k_b * mov_draw
        elif fight.winner_id == fight.fighter_a_id:
            # Fighter A wins
            delta_a = k_a * mov * (1.0 - e_a)
            delta_b = k_b * mov * (0.0 - e_b)
            effective_k_a = k_a * mov
            effective_k_b = k_b * mov
        else:
            # Fighter B wins
            delta_a = k_a * mov * (0.0 - e_a)
            delta_b = k_b * mov * (1.0 - e_b)
            effective_k_a = k_a * mov
            effective_k_b = k_b * mov

        # Update ratings
        new_rating_a = rating_a + delta_a
        new_rating_b = rating_b + delta_b
        self._ratings[key_a] = new_rating_a
        self._ratings[key_b] = new_rating_b

        # Increment fight counts AFTER using count for K-factor lookup
        self._fight_counts[fight.fighter_a_id] = (
            self._fight_counts.get(fight.fighter_a_id, 0) + 1
        )
        self._fight_counts[fight.fighter_b_id] = (
            self._fight_counts.get(fight.fighter_b_id, 0) + 1
        )

        count_a_after = self._fight_counts[fight.fighter_a_id]
        count_b_after = self._fight_counts[fight.fighter_b_id]

        # Update last fight date and last division
        self._last_fight_date[fight.fighter_a_id] = fight.event_date
        self._last_fight_date[fight.fighter_b_id] = fight.event_date

        # Only update _last_division for non-catchweight/open-weight (per D-06)
        if division not in _NON_TRANSFER_DIVISIONS:
            self._last_division[fight.fighter_a_id] = division
            self._last_division[fight.fighter_b_id] = division

        # Compute shrinkage (per D-07)
        shrinkage_a = self._compute_shrinkage(new_rating_a, count_a_after)
        shrinkage_b = self._compute_shrinkage(new_rating_b, count_b_after)

        # Build snapshot records
        snap_a = SnapshotRecord(
            fighter_id=fight.fighter_a_id,
            fight_id=fight.fight_id,
            division=division,
            elo_type="overall",
            elo_before=rating_a,
            elo_after=new_rating_a,
            elo_after_shrinkage=shrinkage_a,
            k_factor_used=effective_k_a,
            fight_date=fight.event_date,
        )
        snap_b = SnapshotRecord(
            fighter_id=fight.fighter_b_id,
            fight_id=fight.fight_id,
            division=division,
            elo_type="overall",
            elo_before=rating_b,
            elo_after=new_rating_b,
            elo_after_shrinkage=shrinkage_b,
            k_factor_used=effective_k_b,
            fight_date=fight.event_date,
        )

        return [snap_a, snap_b]

    def _get_k_factor(self, fighter_id: int) -> float:
        """Get K-factor for a fighter based on their total fight count (per D-03)."""
        count = self._fight_counts.get(fighter_id, 0)
        if count < self.config.k_transition_fights:
            return self.config.k_initial
        return self.config.k_experienced

    def _get_mov_multiplier(self, method: str | None, method_detail: str | None) -> float:
        """Resolve MOV K-factor multiplier from fight method and detail (per D-01).

        Handles both Kaggle-format ("KO/TKO", "Submission", "Decision")
        and scraper-format ("U-DEC", "S-DEC", "M-DEC", "SUB", "TKO") method strings.
        Defensive fallback to mov_unanimous for unrecognized method values (T-03-01).
        """
        if method is None:
            return self.config.mov_unanimous  # fallback to baseline

        # KO/TKO variants (Kaggle: "KO/TKO", Scraper: "TKO")
        if method in ("KO/TKO", "TKO"):
            return self.config.mov_ko_tko
        # Submission variants (Kaggle: "Submission", Scraper: "SUB")
        if method in ("Submission", "SUB"):
            return self.config.mov_submission
        # Decision with detail (Kaggle format)
        if method == "Decision":
            detail = (method_detail or "").strip()
            if detail == "Split":
                return self.config.mov_split
            # Unanimous, Majority, or unknown detail -> baseline
            return self.config.mov_unanimous
        # Scraper decision formats
        if method in ("U-DEC", "M-DEC"):
            return self.config.mov_unanimous
        if method == "S-DEC":
            return self.config.mov_split
        if method == "DQ":
            return self.config.mov_dq
        # "Draw", "Other", or anything else -> baseline fallback
        return self.config.mov_unanimous

    def _apply_inactivity_regression(
        self, fighter_id: int, current_date: date,
    ) -> None:
        """Regress rating toward initial_rating if fighter inactive >12 months (per D-08).

        Applied to ALL divisions the fighter has ratings in, before division
        transfer check (per RESEARCH.md Pitfall 4).
        """
        last_date = self._last_fight_date.get(fighter_id)
        if last_date is None:
            return  # first fight, no regression

        days_inactive = (current_date - last_date).days
        if days_inactive <= self.config.inactivity_threshold_days:
            return  # within threshold, no regression

        months_past = (days_inactive - self.config.inactivity_threshold_days) / 30.0
        regression_pct = min(
            months_past * self.config.inactivity_regression_rate,
            self.config.inactivity_regression_cap,
        )

        # Apply to ALL divisions the fighter has ratings in
        initial = self.config.initial_rating
        for key in list(self._ratings):
            if key[0] == fighter_id:
                current_elo = self._ratings[key]
                delta = current_elo - initial
                self._ratings[key] = current_elo - delta * regression_pct

    def _check_division_transfer(self, fighter_id: int, current_division: str) -> None:
        """If fighter changed divisions, carry 75% of Elo delta to new division (per D-05).

        Catchweight and Open Weight fights never trigger transfers (per D-06).
        """
        if current_division in _NON_TRANSFER_DIVISIONS:
            return

        last_div = self._last_division.get(fighter_id)
        if last_div is None or last_div == current_division:
            return  # first fight or same division

        if last_div in _NON_TRANSFER_DIVISIONS:
            return  # previous was catchweight/open weight, no transfer

        new_key = (fighter_id, current_division)
        if new_key not in self._ratings:
            old_key = (fighter_id, last_div)
            old_elo = self._ratings.get(old_key, self.config.initial_rating)
            old_delta = old_elo - self.config.initial_rating
            new_elo = self.config.initial_rating + self.config.division_transfer_pct * old_delta
            self._ratings[new_key] = new_elo

    def _compute_shrinkage(self, raw_elo: float, fight_count: int) -> float:
        """Compute Bayesian-shrunk display rating (per D-07).

        shrinkage_factor = min(fight_count / shrinkage_min_fights, 1.0)
        elo_after_shrinkage = initial + (raw - initial) * shrinkage_factor
        """
        shrinkage_factor = min(fight_count / self.config.shrinkage_min_fights, 1.0)
        initial = self.config.initial_rating
        return initial + (raw_elo - initial) * shrinkage_factor
