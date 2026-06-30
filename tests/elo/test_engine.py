"""Unit tests for EloEngine covering all Elo behaviors.

Tests cover: basic updates, K-factor decay, MOV weighting (KO, Sub, Split, Draw, DQ),
no-contest skip, division transfer, catchweight handling, Bayesian shrinkage,
inactivity regression (applied, capped, not applied), fight count per-fighter,
expected win probability formula, and chronological snapshot ordering.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine

from .conftest import make_fight


class TestBasicEloUpdate:
    """Test fundamental Elo update mechanics."""

    def test_basic_elo_update(self, elo_engine: EloEngine) -> None:
        """Two fighters at 1500, fighter A wins by UD. Symmetric K=40.

        E(A) = 0.5, delta = 40 * 1.0 * (1 - 0.5) = 20.
        A -> 1520, B -> 1480.
        """
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            method="Decision",
            method_detail="Unanimous",
        )
        snapshots = elo_engine.compute_all([fight])

        assert len(snapshots) == 2

        snap_a = next(s for s in snapshots if s.fighter_id == 1)
        snap_b = next(s for s in snapshots if s.fighter_id == 2)

        assert snap_a.elo_before == 1500.0
        assert snap_a.elo_after == pytest.approx(1520.0)
        assert snap_b.elo_before == 1500.0
        assert snap_b.elo_after == pytest.approx(1480.0)

    def test_expected_win_probability(self, elo_engine: EloEngine) -> None:
        """E(A) = 1/(1+10^((1400-1600)/400)) for R_A=1600, R_B=1400 should be ~0.7597."""
        prob = elo_engine.expected_win_probability(1600.0, 1400.0)
        assert prob == pytest.approx(0.7597, abs=0.001)


class TestKFactorDecay:
    """Test K-factor transitions from 40 to 20 based on fight count."""

    def test_k_factor_decay(self, elo_engine: EloEngine) -> None:
        """Fighter with 4 prior fights uses K=40. With 5 uses K=20."""
        # Give fighter 1 four prior fights (against different opponents each time)
        fights = []
        for i in range(4):
            fights.append(
                make_fight(
                    event_date=date(2020, 1, i + 1),
                    fighter_a_id=1,
                    fighter_b_id=100 + i,
                    winner_id=1,
                )
            )

        # 5th fight: fighter 1 still at K=40 (has 4 prior fights, < 5)
        fights.append(
            make_fight(
                event_date=date(2020, 2, 1),
                fighter_a_id=1,
                fighter_b_id=200,
                winner_id=1,
            )
        )

        snapshots = elo_engine.compute_all(fights)

        # Find fighter 1's snapshot from fight #5 (4 prior fights -> K=40 still)
        fight5_snaps = [s for s in snapshots if s.fight_id == fights[4].fight_id]
        snap_a_5 = next(s for s in fight5_snaps if s.fighter_id == 1)
        assert snap_a_5.k_factor_used == 40.0

        # 6th fight: fighter 1 now has 5 prior fights -> K=20
        fights.append(
            make_fight(
                event_date=date(2020, 3, 1),
                fighter_a_id=1,
                fighter_b_id=201,
                winner_id=1,
            )
        )

        # Re-create engine for clean state
        engine2 = EloEngine(EloConfig())
        snapshots2 = engine2.compute_all(fights)

        fight6_snaps = [s for s in snapshots2 if s.fight_id == fights[5].fight_id]
        snap_a_6 = next(s for s in fight6_snaps if s.fighter_id == 1)
        assert snap_a_6.k_factor_used == 20.0

    def test_fight_counts_are_per_fighter_not_per_division(
        self,
        elo_engine: EloEngine,
    ) -> None:
        """Fighter with 3 LW + 2 WW fights = 5 total -> K=20 (per D-04)."""
        fights = []
        # 3 Lightweight fights
        for i in range(3):
            fights.append(
                make_fight(
                    event_date=date(2020, 1, i + 1),
                    fighter_a_id=1,
                    fighter_b_id=100 + i,
                    winner_id=1,
                    weight_class="Lightweight",
                )
            )
        # 2 Welterweight fights
        for i in range(2):
            fights.append(
                make_fight(
                    event_date=date(2020, 2, i + 1),
                    fighter_a_id=1,
                    fighter_b_id=200 + i,
                    winner_id=1,
                    weight_class="Welterweight",
                )
            )

        # 6th fight (total fights = 5 -> K=20)
        fights.append(
            make_fight(
                event_date=date(2020, 3, 1),
                fighter_a_id=1,
                fighter_b_id=300,
                winner_id=1,
                weight_class="Welterweight",
            )
        )

        snapshots = elo_engine.compute_all(fights)

        fight6_snaps = [s for s in snapshots if s.fight_id == fights[5].fight_id]
        snap_a_6 = next(s for s in fight6_snaps if s.fighter_id == 1)
        assert snap_a_6.k_factor_used == 20.0


class TestMovWeighting:
    """Test margin-of-victory K-factor multipliers.

    These tests lock explicit config values to document the MOV mechanic
    independently of the current default values (approach (b) per plan).
    Defaults were updated to backtest-optimal values in Phase 9.1:
      mov_ko_tko: 1.5 -> 1.2, mov_submission: 1.4 -> 1.1, mov_split: 0.8 -> 0.5
    """

    def test_mov_ko_tko(self) -> None:
        """KO/TKO win with K=40, mov_ko_tko=1.5 -> effective K=40*1.5=60. Larger delta than UD."""
        engine = EloEngine(EloConfig(mov_ko_tko=1.5))
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            method="KO/TKO",
            method_detail=None,
        )
        snapshots = engine.compute_all([fight])
        snap_a = next(s for s in snapshots if s.fighter_id == 1)

        # K=40, MOV=1.5, E=0.5 -> delta = 40*1.5*(1-0.5) = 30
        assert snap_a.elo_after == pytest.approx(1530.0)
        assert snap_a.k_factor_used == pytest.approx(60.0)

    def test_mov_submission(self) -> None:
        """Submission win with K=40, mov_submission=1.4 -> effective K=40*1.4=56."""
        engine = EloEngine(EloConfig(mov_submission=1.4))
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            method="Submission",
            method_detail=None,
        )
        snapshots = engine.compute_all([fight])
        snap_a = next(s for s in snapshots if s.fighter_id == 1)

        # K=40, MOV=1.4, E=0.5 -> delta = 40*1.4*(1-0.5) = 28
        assert snap_a.elo_after == pytest.approx(1528.0)
        assert snap_a.k_factor_used == pytest.approx(56.0)

    def test_mov_split_decision(self) -> None:
        """Split decision with K=40, mov_split=0.8 -> effective K=40*0.8=32. Smaller delta than UD."""
        engine = EloEngine(EloConfig(mov_split=0.8))
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            method="Decision",
            method_detail="Split",
        )
        snapshots = engine.compute_all([fight])
        snap_a = next(s for s in snapshots if s.fighter_id == 1)

        # K=40, MOV=0.8, E=0.5 -> delta = 40*0.8*(1-0.5) = 16
        assert snap_a.elo_after == pytest.approx(1516.0)
        assert snap_a.k_factor_used == pytest.approx(32.0)

    def test_mov_draw(self, elo_engine: EloEngine) -> None:
        """Draw with K=40 -> effective K=40*0.5=20. Both move toward expected=0.5."""
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=None,
            method="Draw",
            method_detail=None,
        )
        snapshots = elo_engine.compute_all([fight])

        assert len(snapshots) == 2
        snap_a = next(s for s in snapshots if s.fighter_id == 1)
        snap_b = next(s for s in snapshots if s.fighter_id == 2)

        # Both at 1500, E=0.5, draw score=0.5, so delta = K*MOV*(0.5-0.5) = 0
        assert snap_a.elo_after == pytest.approx(1500.0)
        assert snap_b.elo_after == pytest.approx(1500.0)
        assert snap_a.k_factor_used == pytest.approx(20.0)
        assert snap_b.k_factor_used == pytest.approx(20.0)

    def test_dq_treated_as_minimal_win(self, elo_engine: EloEngine) -> None:
        """Fight with method='DQ' uses 0.8x multiplier."""
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            method="DQ",
            method_detail=None,
        )
        snapshots = elo_engine.compute_all([fight])
        snap_a = next(s for s in snapshots if s.fighter_id == 1)

        # K=40, MOV=0.8, E=0.5 -> delta = 40*0.8*(1-0.5) = 16
        assert snap_a.elo_after == pytest.approx(1516.0)
        assert snap_a.k_factor_used == pytest.approx(32.0)


class TestSkipLogic:
    """Test that no-contests and method='Other' fights are skipped."""

    def test_no_contest_skipped(self, elo_engine: EloEngine) -> None:
        """Fight with winner_id=None and method=None produces no snapshots."""
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=None,
            method=None,
            method_detail=None,
        )
        snapshots = elo_engine.compute_all([fight])
        assert len(snapshots) == 0

    def test_method_other_skipped(self, elo_engine: EloEngine) -> None:
        """Fight with method='Other' produces no snapshots (per D-02)."""
        fight = make_fight(
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=None,
            method="Other",
            method_detail=None,
        )
        snapshots = elo_engine.compute_all([fight])
        assert len(snapshots) == 0


class TestDivisionTransfer:
    """Test division transfer mechanics.

    Tests lock explicit division_transfer_pct=0.75 to document the mechanic
    independently of the current default (0.50 after Phase 9.1 backtest).
    """

    def test_division_transfer(self) -> None:
        """Fighter at LW (elo=1600), then fights at WW.

        New WW rating = 1500 + 0.75*(1600-1500) = 1575 (per D-05).
        Uses explicit division_transfer_pct=0.75 to lock mechanic test.
        """
        engine = EloEngine(EloConfig(division_transfer_pct=0.75))
        # First fight at Lightweight: fighter 1 wins, goes to ~1520
        fight1 = make_fight(
            event_date=date(2020, 1, 1),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            weight_class="Lightweight",
        )

        # Give fighter 1 enough wins at LW to reach ~1600
        # Each win against 1500-rated opponent: delta = 40*1.0*(1-E)
        # After fight 1: 1520
        fights = [fight1]
        # Add more fights to push rating higher
        for i in range(4):
            fights.append(
                make_fight(
                    event_date=date(2020, 1, 2 + i),
                    fighter_a_id=1,
                    fighter_b_id=10 + i,
                    winner_id=1,
                    weight_class="Lightweight",
                )
            )

        # Now fight at Welterweight
        fights.append(
            make_fight(
                event_date=date(2020, 6, 1),
                fighter_a_id=1,
                fighter_b_id=50,
                winner_id=1,
                weight_class="Welterweight",
            )
        )

        snapshots = engine.compute_all(fights)

        # Find the WW fight snapshot for fighter 1
        ww_snap = next(s for s in snapshots if s.fighter_id == 1 and s.division == "Welterweight")

        # Get fighter 1's LW elo after 5 fights
        lw_snaps = [s for s in snapshots if s.fighter_id == 1 and s.division == "Lightweight"]
        last_lw_elo = lw_snaps[-1].elo_after

        # WW elo_before should be 1500 + 0.75*(last_lw_elo - 1500)
        expected_ww_start = 1500.0 + 0.75 * (last_lw_elo - 1500.0)
        assert ww_snap.elo_before == pytest.approx(expected_ww_start, abs=0.01)

    def test_catchweight_no_transfer(self, elo_engine: EloEngine) -> None:
        """Fighter's previous fight at 'Catch Weight' does not trigger transfer."""
        # Fight 1 at Catch Weight
        fight1 = make_fight(
            event_date=date(2020, 1, 1),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            weight_class="Catch Weight",
        )
        # Fight 2 at Lightweight
        fight2 = make_fight(
            event_date=date(2020, 6, 1),
            fighter_a_id=1,
            fighter_b_id=3,
            winner_id=1,
            weight_class="Lightweight",
        )

        snapshots = elo_engine.compute_all([fight1, fight2])

        lw_snap = next(s for s in snapshots if s.fighter_id == 1 and s.division == "Lightweight")
        # Should start at 1500 (no transfer from Catch Weight)
        assert lw_snap.elo_before == 1500.0

    def test_current_catchweight_no_transfer(self, elo_engine: EloEngine) -> None:
        """Fighter fighting at 'Catch Weight' does not trigger transfer from prior."""
        # Fight 1 at Lightweight (fighter wins -> elo > 1500)
        fight1 = make_fight(
            event_date=date(2020, 1, 1),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            weight_class="Lightweight",
        )
        # Fight 2 at Catch Weight -- should NOT transfer from LW
        fight2 = make_fight(
            event_date=date(2020, 6, 1),
            fighter_a_id=1,
            fighter_b_id=3,
            winner_id=1,
            weight_class="Catch Weight",
        )

        snapshots = elo_engine.compute_all([fight1, fight2])

        cw_snap = next(s for s in snapshots if s.fighter_id == 1 and s.division == "Catch Weight")
        # Should start at 1500 (no transfer into Catch Weight)
        assert cw_snap.elo_before == 1500.0


class TestBayesianShrinkage:
    """Test Bayesian shrinkage for fighters with few fights."""

    def test_bayesian_shrinkage(self, elo_engine: EloEngine) -> None:
        """Fighter with 2 fights -> shrinkage_factor=2/5=0.4.

        elo_after_shrinkage = 1500 + (raw-1500)*0.4.
        Fighter with 5+ fights -> elo_after_shrinkage = elo_after.
        """
        # Fighter 1 wins first fight -> raw elo ~1520
        fight1 = make_fight(
            event_date=date(2020, 1, 1),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
        )
        snapshots = elo_engine.compute_all([fight1])

        snap_a = next(s for s in snapshots if s.fighter_id == 1)
        # After 1 fight: shrinkage_factor = 1/5 = 0.2
        raw_delta = snap_a.elo_after - 1500.0
        expected_shrunk = 1500.0 + raw_delta * (1.0 / 5.0)
        assert snap_a.elo_after_shrinkage == pytest.approx(expected_shrunk, abs=0.01)

        # Now test with 5+ fights -> no shrinkage
        engine2 = EloEngine(EloConfig())
        fights = []
        for i in range(6):
            fights.append(
                make_fight(
                    event_date=date(2020, 1, i + 1),
                    fighter_a_id=1,
                    fighter_b_id=100 + i,
                    winner_id=1,
                )
            )

        snapshots2 = engine2.compute_all(fights)
        # After fight 6 (5 prior fights), fight count after = 6 -> shrinkage=1.0
        last_snap = [s for s in snapshots2 if s.fighter_id == 1][-1]
        assert last_snap.elo_after_shrinkage == pytest.approx(last_snap.elo_after)


class TestInactivityRegression:
    """Test inactivity regression mechanics.

    Tests lock explicit inactivity_threshold_days=365 to document the mechanic
    independently of the current default (270 days after Phase 9.1 backtest).
    """

    def test_inactivity_regression_applied(self) -> None:
        """Fighter inactive 14 months (2 months past 365-day threshold).

        10%*2 = 20% regression. elo=1600 -> delta=100, regressed=1600-100*0.20=1580.
        Uses explicit inactivity_threshold_days=365 to lock mechanic test.
        """
        engine = EloEngine(EloConfig(inactivity_threshold_days=365))
        # Give fighter 1 a rating > 1500
        fight1 = make_fight(
            event_date=date(2020, 1, 1),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
        )
        snapshots1 = engine.compute_all([fight1])
        snap_after_1 = next(s for s in snapshots1 if s.fighter_id == 1)
        elo_after_fight1 = snap_after_1.elo_after

        # Fight 2: 14 months later (425 days = 365 + 60)
        fight2 = make_fight(
            event_date=date(2021, 3, 1),  # ~14 months after Jan 2020
            fighter_a_id=1,
            fighter_b_id=3,
            winner_id=1,
        )

        # Reset engine and process both fights
        engine2 = EloEngine(EloConfig(inactivity_threshold_days=365))
        snapshots2 = engine2.compute_all([fight1, fight2])

        fight2_snaps = [s for s in snapshots2 if s.fight_id == fight2.fight_id]
        snap_a_2 = next(s for s in fight2_snaps if s.fighter_id == 1)

        # Inactivity: days = (2021-03-01 - 2020-01-01) = 425 days
        days_inactive = (date(2021, 3, 1) - date(2020, 1, 1)).days
        months_past = (days_inactive - 365) / 30.0
        regression_pct = min(months_past * 0.10, 0.50)

        # elo_before should reflect regression from elo_after_fight1
        delta = elo_after_fight1 - 1500.0
        expected_regressed = elo_after_fight1 - delta * regression_pct
        assert snap_a_2.elo_before == pytest.approx(expected_regressed, abs=0.1)

    def test_inactivity_regression_capped(self, elo_engine: EloEngine) -> None:
        """Fighter inactive 24 months -> 10%*12=120%, capped at 50%.

        elo=1600 -> 1600-100*0.50=1550.
        With default threshold=270d, 24 months (730d) is well past threshold,
        so cap is still reached regardless of threshold value.
        """
        # Fight 1: win to get above 1500
        fight1 = make_fight(
            event_date=date(2020, 1, 1),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
        )

        # Fight 2: 24 months later
        fight2 = make_fight(
            event_date=date(2022, 1, 1),
            fighter_a_id=1,
            fighter_b_id=3,
            winner_id=1,
        )

        snapshots = elo_engine.compute_all([fight1, fight2])

        # Get fighter 1's elo after fight 1
        fight1_snaps = [s for s in snapshots if s.fight_id == fight1.fight_id]
        snap_a_1 = next(s for s in fight1_snaps if s.fighter_id == 1)
        elo_after_1 = snap_a_1.elo_after

        # Get fighter 1's elo_before for fight 2
        fight2_snaps = [s for s in snapshots if s.fight_id == fight2.fight_id]
        snap_a_2 = next(s for s in fight2_snaps if s.fighter_id == 1)

        # Regression capped at 50%
        delta = elo_after_1 - 1500.0
        expected_regressed = elo_after_1 - delta * 0.50
        assert snap_a_2.elo_before == pytest.approx(expected_regressed, abs=0.1)

    def test_inactivity_no_regression_within_threshold(self) -> None:
        """Fighter inactive 8 months (~240 days < 270-day threshold) -> no regression applied.

        Uses explicit inactivity_threshold_days=270 (current default after Phase 9.1).
        """
        engine = EloEngine(EloConfig(inactivity_threshold_days=270))
        fight1 = make_fight(
            event_date=date(2020, 1, 1),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
        )
        # Fight 2: ~8 months later (~240 days < 270-day threshold)
        fight2 = make_fight(
            event_date=date(2020, 9, 1),
            fighter_a_id=1,
            fighter_b_id=3,
            winner_id=1,
        )

        snapshots = engine.compute_all([fight1, fight2])

        fight1_snaps = [s for s in snapshots if s.fight_id == fight1.fight_id]
        snap_a_1 = next(s for s in fight1_snaps if s.fighter_id == 1)
        elo_after_1 = snap_a_1.elo_after

        fight2_snaps = [s for s in snapshots if s.fight_id == fight2.fight_id]
        snap_a_2 = next(s for s in fight2_snaps if s.fighter_id == 1)

        # No regression: elo_before for fight 2 should equal elo_after from fight 1
        assert snap_a_2.elo_before == pytest.approx(elo_after_1)


class TestComputeAll:
    """Test overall compute_all behavior."""

    def test_compute_all_returns_chronological_snapshots(
        self,
        elo_engine: EloEngine,
    ) -> None:
        """3 fights in order produce 6 snapshots (2 per fight).

        Each fighter's elo_before matches prior elo_after.
        """
        fights = [
            make_fight(
                event_date=date(2020, 1, 1),
                fighter_a_id=1,
                fighter_b_id=2,
                winner_id=1,
            ),
            make_fight(
                event_date=date(2020, 2, 1),
                fighter_a_id=1,
                fighter_b_id=3,
                winner_id=1,
            ),
            make_fight(
                event_date=date(2020, 3, 1),
                fighter_a_id=2,
                fighter_b_id=3,
                winner_id=2,
            ),
        ]

        snapshots = elo_engine.compute_all(fights)
        assert len(snapshots) == 6  # 2 per fight

        # Fighter 1's second fight elo_before should equal first fight elo_after
        f1_snaps = [s for s in snapshots if s.fighter_id == 1]
        assert len(f1_snaps) == 2
        assert f1_snaps[1].elo_before == pytest.approx(f1_snaps[0].elo_after)

        # Fighter 2's second fight elo_before should equal first fight elo_after
        f2_snaps = [s for s in snapshots if s.fighter_id == 2]
        assert len(f2_snaps) == 2
        assert f2_snaps[1].elo_before == pytest.approx(f2_snaps[0].elo_after)


class TestScraperMethodStrings:
    """Test MOV multiplier handling for scraper-format method strings.

    These tests lock explicit config values to document the method-string
    mapping mechanic independently of the current default MOV values.
    Defaults were updated to backtest-optimal values in Phase 9.1.
    """

    def test_mov_multiplier_scraper_tko(self) -> None:
        """Scraper-format 'TKO' maps to KO/TKO multiplier (1.5x). K=40*1.5=60."""
        engine = EloEngine(EloConfig(mov_ko_tko=1.5))
        fight = make_fight(method="TKO", method_detail=None)
        snaps = engine.compute_all([fight])
        snap_a = next(s for s in snaps if s.fighter_id == 1)
        # K=40 * MOV=1.5 = 60.0
        assert snap_a.k_factor_used == pytest.approx(60.0)

    def test_mov_multiplier_scraper_sub(self) -> None:
        """Scraper-format 'SUB' maps to Submission multiplier (1.4x). K=40*1.4=56."""
        engine = EloEngine(EloConfig(mov_submission=1.4))
        fight = make_fight(method="SUB", method_detail=None)
        snaps = engine.compute_all([fight])
        snap_a = next(s for s in snaps if s.fighter_id == 1)
        # K=40 * MOV=1.4 = 56.0
        assert snap_a.k_factor_used == pytest.approx(56.0)

    def test_mov_multiplier_scraper_udec(self, elo_engine: EloEngine) -> None:
        """Scraper-format 'U-DEC' maps to Unanimous Decision multiplier (1.0x). K=40*1.0=40."""
        fight = make_fight(method="U-DEC", method_detail=None)
        snaps = elo_engine.compute_all([fight])
        snap_a = next(s for s in snaps if s.fighter_id == 1)
        # K=40 * MOV=1.0 = 40.0
        assert snap_a.k_factor_used == pytest.approx(40.0)

    def test_mov_multiplier_scraper_sdec(self) -> None:
        """Scraper-format 'S-DEC' maps to Split Decision multiplier (0.8x). K=40*0.8=32."""
        engine = EloEngine(EloConfig(mov_split=0.8))
        fight = make_fight(method="S-DEC", method_detail=None)
        snaps = engine.compute_all([fight])
        snap_a = next(s for s in snaps if s.fighter_id == 1)
        # K=40 * MOV=0.8 = 32.0
        assert snap_a.k_factor_used == pytest.approx(32.0)

    def test_mov_multiplier_scraper_mdec(self, elo_engine: EloEngine) -> None:
        """Scraper-format 'M-DEC' maps to Majority Decision = Unanimous multiplier (1.0x)."""
        fight = make_fight(method="M-DEC", method_detail=None)
        snaps = elo_engine.compute_all([fight])
        snap_a = next(s for s in snaps if s.fighter_id == 1)
        # K=40 * MOV=1.0 = 40.0
        assert snap_a.k_factor_used == pytest.approx(40.0)


class TestSnapshotFields:
    """Test that SnapshotRecord fields are populated correctly."""

    def test_snapshot_has_correct_fields(self, elo_engine: EloEngine) -> None:
        """Verify all SnapshotRecord fields are set correctly."""
        fight = make_fight(
            fight_id=42,
            event_date=date(2020, 6, 15),
            fighter_a_id=1,
            fighter_b_id=2,
            winner_id=1,
            weight_class="Heavyweight",
        )
        snapshots = elo_engine.compute_all([fight])
        snap_a = next(s for s in snapshots if s.fighter_id == 1)

        assert snap_a.fight_id == 42
        assert snap_a.division == "Heavyweight"
        assert snap_a.elo_type == "overall"
        assert snap_a.fight_date == date(2020, 6, 15)
        assert isinstance(snap_a.elo_before, float)
        assert isinstance(snap_a.elo_after, float)
        assert isinstance(snap_a.elo_after_shrinkage, float)
        assert isinstance(snap_a.k_factor_used, float)
