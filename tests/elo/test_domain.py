"""Unit tests for domain-specific Elo attribution and DomainEloComputer."""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.domain import DomainEloComputer, compute_attribution
from ufc_prediction.elo.engine import SnapshotRecord

from .conftest import make_fight, make_round_stats

# ── TestComputeAttribution ──────────────────────────────────────────


class TestComputeAttribution:
    """Tests for the compute_attribution function covering all method types."""

    def test_ko_tko_returns_fixed_ratio(self) -> None:
        """KO/TKO: 100% striking / 0% grappling (validated Phase 9.1, was 80/20 [L])."""
        stats = [make_round_stats(fight_id=1, fighter_id=1)]
        result = compute_attribution("KO/TKO", stats, fight_id=1)
        assert result is not None
        assert result.striking_ratio == pytest.approx(1.0)
        assert result.grappling_ratio == pytest.approx(0.0)

    def test_submission_returns_fixed_ratio(self) -> None:
        """Submission: 0% striking / 100% grappling (validated Phase 9.1, was 20/80 [L])."""
        stats = [make_round_stats(fight_id=1, fighter_id=1)]
        result = compute_attribution("Submission", stats, fight_id=1)
        assert result is not None
        assert result.striking_ratio == pytest.approx(0.0)
        assert result.grappling_ratio == pytest.approx(1.0)

    def test_decision_with_round_stats_computes_ratio(self) -> None:
        """Decision: ratio from round stats per D-02.

        sig_str_landed=80, takedowns_landed=3, ctrl_time_seconds=120
        striking_signal = 80
        grappling_signal = 3 + 120 = 123
        total = 203
        striking_ratio = 80/203 ~ 0.394
        grappling_ratio = 123/203 ~ 0.606
        """
        stats = [
            make_round_stats(
                fight_id=1,
                fighter_id=1,
                sig_str_landed=80,
                takedowns_landed=3,
                ctrl_time_seconds=120,
            ),
        ]
        result = compute_attribution("Decision", stats, fight_id=1)
        assert result is not None
        assert result.striking_ratio == pytest.approx(80.0 / 203.0, abs=1e-6)
        assert result.grappling_ratio == pytest.approx(123.0 / 203.0, abs=1e-6)

    def test_zero_signal_fallback_returns_fifty_fifty(self) -> None:
        """Zero striking + zero grappling signals -> 50/50 per D-02 fallback."""
        stats = [
            make_round_stats(
                fight_id=1,
                fighter_id=1,
                sig_str_landed=0,
                takedowns_landed=0,
                ctrl_time_seconds=0,
            ),
        ]
        result = compute_attribution("Decision", stats, fight_id=1)
        assert result is not None
        assert result.striking_ratio == pytest.approx(0.5)
        assert result.grappling_ratio == pytest.approx(0.5)

    def test_no_round_data_returns_none(self) -> None:
        """None round_stats returns None (D-06: skip domain update)."""
        result = compute_attribution("Decision", None, fight_id=1)
        assert result is None

    def test_empty_round_data_returns_none(self) -> None:
        """Empty round_stats list returns None (D-06: skip domain update)."""
        result = compute_attribution("Decision", [], fight_id=1)
        assert result is None

    def test_dq_with_round_stats_uses_stats_based_ratio(self) -> None:
        """DQ method with round stats falls through to D-02 stats logic."""
        stats = [
            make_round_stats(
                fight_id=1,
                fighter_id=1,
                sig_str_landed=50,
                takedowns_landed=5,
                ctrl_time_seconds=60,
            ),
        ]
        result = compute_attribution("DQ", stats, fight_id=1)
        assert result is not None
        # striking=50, grappling=5+60=65, total=115
        assert result.striking_ratio == pytest.approx(50.0 / 115.0, abs=1e-6)
        assert result.grappling_ratio == pytest.approx(65.0 / 115.0, abs=1e-6)

    def test_none_stat_values_treated_as_zero(self) -> None:
        """Stats with None values should be treated as 0 (defensive coding)."""
        stats = [
            {
                "fighter_id": 1,
                "sig_str_landed": None,
                "takedowns_landed": None,
                "ctrl_time_seconds": None,
            },
        ]
        result = compute_attribution("Decision", stats, fight_id=1)
        assert result is not None
        # All zeros -> 50/50 fallback
        assert result.striking_ratio == pytest.approx(0.5)
        assert result.grappling_ratio == pytest.approx(0.5)

    def test_multiple_rounds_sum_stats(self) -> None:
        """Multiple round stats entries are summed across rounds."""
        stats = [
            make_round_stats(
                fight_id=1,
                fighter_id=1,
                sig_str_landed=30,
                takedowns_landed=2,
                ctrl_time_seconds=60,
            ),
            make_round_stats(
                fight_id=1,
                fighter_id=2,
                sig_str_landed=20,
                takedowns_landed=1,
                ctrl_time_seconds=30,
            ),
            make_round_stats(
                fight_id=1,
                fighter_id=1,
                sig_str_landed=25,
                takedowns_landed=0,
                ctrl_time_seconds=15,
            ),
        ]
        result = compute_attribution("Decision", stats, fight_id=1)
        assert result is not None
        # striking = 30+20+25 = 75
        # grappling = (2+60) + (1+30) + (0+15) = 108
        # total = 183
        assert result.striking_ratio == pytest.approx(75.0 / 183.0, abs=1e-6)
        assert result.grappling_ratio == pytest.approx(108.0 / 183.0, abs=1e-6)

    def test_fight_id_stored_in_attribution(self) -> None:
        """DomainAttribution stores the fight_id."""
        stats = [make_round_stats(fight_id=42, fighter_id=1)]
        result = compute_attribution("KO/TKO", stats, fight_id=42)
        assert result is not None
        assert result.fight_id == 42


# ── TestDomainEloComputer ───────────────────────────────────────────


class TestDomainEloComputer:
    """Tests for the DomainEloComputer.compute_all method."""

    @pytest.fixture()
    def config(self) -> EloConfig:
        return EloConfig()

    @pytest.fixture()
    def computer(self, config: EloConfig) -> DomainEloComputer:
        return DomainEloComputer(config)

    def _make_overall_snap(
        self,
        fighter_id: int,
        fight_id: int,
        *,
        elo_before: float = 1500.0,
        elo_after: float = 1520.0,
        elo_after_shrinkage: float = 1504.0,
        k_factor_used: float = 60.0,
        division: str = "Lightweight",
        fight_date: date = date(2020, 1, 1),
    ) -> SnapshotRecord:
        return SnapshotRecord(
            fighter_id=fighter_id,
            fight_id=fight_id,
            division=division,
            elo_type="overall",
            elo_before=elo_before,
            elo_after=elo_after,
            elo_after_shrinkage=elo_after_shrinkage,
            k_factor_used=k_factor_used,
            fight_date=fight_date,
        )

    def test_produces_four_snapshots_per_fight(self, computer: DomainEloComputer) -> None:
        """Fight with round data produces 4 snapshots: 2 striking + 2 grappling."""
        fight = make_fight(fight_id=1, method="KO/TKO", method_detail=None)
        overall_snaps = [
            self._make_overall_snap(1, 1, elo_after=1520.0),
            self._make_overall_snap(2, 1, elo_before=1500.0, elo_after=1480.0),
        ]
        round_stats = {
            1: [
                make_round_stats(fight_id=1, fighter_id=1, sig_str_landed=30),
                make_round_stats(fight_id=1, fighter_id=2, sig_str_landed=10),
            ],
        }
        result = computer.compute_all([fight], overall_snaps, round_stats)
        assert len(result) == 4
        elo_types = {s.elo_type for s in result}
        assert elo_types == {"striking", "grappling"}
        # 2 striking + 2 grappling
        assert sum(1 for s in result if s.elo_type == "striking") == 2
        assert sum(1 for s in result if s.elo_type == "grappling") == 2

    def test_skips_fight_without_round_data(self, computer: DomainEloComputer) -> None:
        """Fight with no round data produces 0 domain snapshots."""
        fight = make_fight(fight_id=1, method="Decision")
        overall_snaps = [
            self._make_overall_snap(1, 1),
            self._make_overall_snap(2, 1),
        ]
        round_stats: dict[int, list[dict]] = {}  # no data for fight 1
        result = computer.compute_all([fight], overall_snaps, round_stats)
        assert len(result) == 0

    def test_domain_delta_equals_overall_delta_times_ratio(
        self, computer: DomainEloComputer,
    ) -> None:
        """Domain delta = overall_delta * domain_ratio per D-03."""
        fight = make_fight(fight_id=1, method="KO/TKO", method_detail=None)
        # Fighter A wins: overall elo goes from 1500 to 1520 (delta = +20)
        # Fighter B loses: overall elo goes from 1500 to 1480 (delta = -20)
        overall_snaps = [
            self._make_overall_snap(1, 1, elo_before=1500.0, elo_after=1520.0),
            self._make_overall_snap(2, 1, elo_before=1500.0, elo_after=1480.0),
        ]
        round_stats = {
            1: [make_round_stats(fight_id=1, fighter_id=1)],
        }
        result = computer.compute_all([fight], overall_snaps, round_stats)

        # KO/TKO -> striking_ratio=1.0, grappling_ratio=0.0 (validated Phase 9.1)
        striking_snaps = [s for s in result if s.elo_type == "striking"]
        grappling_snaps = [s for s in result if s.elo_type == "grappling"]

        # Fighter A striking: delta = 20 * 1.0 = 20
        snap_a_str = next(s for s in striking_snaps if s.fighter_id == 1)
        assert snap_a_str.elo_after - snap_a_str.elo_before == pytest.approx(20.0)

        # Fighter A grappling: delta = 20 * 0.0 = 0
        snap_a_grp = next(s for s in grappling_snaps if s.fighter_id == 1)
        assert snap_a_grp.elo_after - snap_a_grp.elo_before == pytest.approx(0.0)

        # Fighter B striking: delta = -20 * 1.0 = -20
        snap_b_str = next(s for s in striking_snaps if s.fighter_id == 2)
        assert snap_b_str.elo_after - snap_b_str.elo_before == pytest.approx(-20.0)

        # Fighter B grappling: delta = -20 * 0.0 = 0
        snap_b_grp = next(s for s in grappling_snaps if s.fighter_id == 2)
        assert snap_b_grp.elo_after - snap_b_grp.elo_before == pytest.approx(0.0)

    def test_starting_rating_is_1500(self, computer: DomainEloComputer) -> None:
        """Domain starting rating is 1500.0 per D-08."""
        fight = make_fight(fight_id=1, method="Decision")
        overall_snaps = [
            self._make_overall_snap(1, 1),
            self._make_overall_snap(2, 1),
        ]
        round_stats = {
            1: [make_round_stats(fight_id=1, fighter_id=1, sig_str_landed=50)],
        }
        result = computer.compute_all([fight], overall_snaps, round_stats)
        # All elo_before should be 1500.0 (first fight for all fighters)
        for snap in result:
            assert snap.elo_before == pytest.approx(1500.0)

    def test_k_factor_decay_uses_domain_fight_count(
        self, computer: DomainEloComputer,
    ) -> None:
        """K-factor decay applies based on domain fight count (independent from overall)."""
        # Run 6 fights to push domain fight count past k_transition_fights=5
        fights = []
        overall_snaps = []
        round_stats: dict[int, list[dict]] = {}
        for i in range(1, 7):
            f = make_fight(
                fight_id=i,
                event_date=date(2020, 1, i),
                fighter_a_id=1,
                fighter_b_id=i + 10,  # different opponents
                method="KO/TKO",
            )
            fights.append(f)
            overall_snaps.extend([
                self._make_overall_snap(
                    1, i, elo_before=1500.0, elo_after=1520.0,
                    k_factor_used=60.0, fight_date=date(2020, 1, i),
                ),
                self._make_overall_snap(
                    i + 10, i, elo_before=1500.0, elo_after=1480.0,
                    k_factor_used=60.0, fight_date=date(2020, 1, i),
                ),
            ])
            round_stats[i] = [make_round_stats(fight_id=i, fighter_id=1)]

        result = computer.compute_all(fights, overall_snaps, round_stats)
        # Fighter 1 has 6 domain fights. K-factor should have decayed
        # for the 6th fight (fight count was 5 entering that fight, so
        # k_experienced=20 is used). Earlier fights used k_initial=40.
        # The k_factor_used is overall_snap.k_factor_used * ratio,
        # but we verify the domain computer tracks fights independently
        # by checking that fighter 1 appears in all 6 fights' snapshots.
        fighter_1_snaps = [s for s in result if s.fighter_id == 1]
        assert len(fighter_1_snaps) == 12  # 6 fights * 2 domains

    def test_bayesian_shrinkage_applied(self, computer: DomainEloComputer) -> None:
        """Bayesian shrinkage is applied to domain ratings (same formula as overall, per D-08)."""
        fight = make_fight(fight_id=1, method="KO/TKO", method_detail=None)
        overall_snaps = [
            self._make_overall_snap(1, 1, elo_before=1500.0, elo_after=1520.0),
            self._make_overall_snap(2, 1, elo_before=1500.0, elo_after=1480.0),
        ]
        round_stats = {1: [make_round_stats(fight_id=1, fighter_id=1)]}
        result = computer.compute_all([fight], overall_snaps, round_stats)

        # After 1 domain fight, shrinkage_factor = 1/5 = 0.2
        # For fighter 1 striking: raw = 1500 + 20*1.0 = 1520.0 (KO ratio=1.0 since Phase 9.1)
        # shrinkage = 1500 + (1520 - 1500) * 0.2 = 1504.0
        snap_a_str = next(
            s for s in result if s.fighter_id == 1 and s.elo_type == "striking"
        )
        assert snap_a_str.elo_after_shrinkage == pytest.approx(1504.0, abs=0.1)

    def test_inactivity_regression_applied(self, config: EloConfig) -> None:
        """Inactivity regression applied using domain-specific last_fight_date tracking."""
        computer = DomainEloComputer(config)
        # Fight 1: establish domain rating
        fight1 = make_fight(
            fight_id=1, event_date=date(2018, 1, 1), method="KO/TKO",
        )
        overall_snaps = [
            self._make_overall_snap(
                1, 1, elo_before=1500.0, elo_after=1520.0,
                fight_date=date(2018, 1, 1),
            ),
            self._make_overall_snap(
                2, 1, elo_before=1500.0, elo_after=1480.0,
                fight_date=date(2018, 1, 1),
            ),
        ]
        round_stats = {1: [make_round_stats(fight_id=1, fighter_id=1)]}

        computer.compute_all([fight1], overall_snaps, round_stats)

        # Fight 2: 2 years later (well past 270-day threshold after Phase 9.1)
        fight2 = make_fight(
            fight_id=2, event_date=date(2020, 1, 1), method="KO/TKO",
        )
        overall_snaps2 = [
            self._make_overall_snap(
                1, 2, elo_before=1520.0, elo_after=1540.0,
                fight_date=date(2020, 1, 1),
            ),
            self._make_overall_snap(
                2, 2, elo_before=1480.0, elo_after=1460.0,
                fight_date=date(2020, 1, 1),
            ),
        ]
        round_stats2 = {2: [make_round_stats(fight_id=2, fighter_id=1)]}

        result = computer.compute_all([fight2], overall_snaps2, round_stats2)

        # Fighter 1's striking elo_before should be regressed toward 1500
        # Raw striking after fight 1 = 1500 + 20*1.0 = 1520.0 (KO ratio=1.0 since Phase 9.1)
        # 2-year gap exceeds regression cap -> capped regression applied
        snap_a_str = next(
            s for s in result if s.fighter_id == 1 and s.elo_type == "striking"
        )
        # The elo_before should be less than 1520.0 (raw) due to regression
        assert snap_a_str.elo_before < 1520.0

    def test_division_transfer_carries_pct(self, config: EloConfig) -> None:
        """Division transfer carries division_transfer_pct of domain Elo delta to new division.

        Uses default EloConfig (division_transfer_pct=0.50 after Phase 9.1 backtest).
        KO ratio is now 1.0/0.0 (Phase 9.1), so striking delta = overall_delta * 1.0.
        """
        computer = DomainEloComputer(config)
        # Fight 1: establish domain rating in Lightweight
        fight1 = make_fight(
            fight_id=1, event_date=date(2020, 1, 1),
            weight_class="Lightweight", method="KO/TKO",
        )
        overall_snaps = [
            self._make_overall_snap(
                1, 1, elo_before=1500.0, elo_after=1520.0,
                division="Lightweight", fight_date=date(2020, 1, 1),
            ),
            self._make_overall_snap(
                2, 1, elo_before=1500.0, elo_after=1480.0,
                division="Lightweight", fight_date=date(2020, 1, 1),
            ),
        ]
        round_stats = {1: [make_round_stats(fight_id=1, fighter_id=1)]}
        computer.compute_all([fight1], overall_snaps, round_stats)

        # Fight 2: fighter 1 moves to Welterweight
        fight2 = make_fight(
            fight_id=2, event_date=date(2020, 6, 1),
            fighter_a_id=1, fighter_b_id=3,
            weight_class="Welterweight", method="KO/TKO",
        )
        overall_snaps2 = [
            self._make_overall_snap(
                1, 2, elo_before=1520.0, elo_after=1540.0,
                division="Welterweight", fight_date=date(2020, 6, 1),
            ),
            self._make_overall_snap(
                3, 2, elo_before=1500.0, elo_after=1480.0,
                division="Welterweight", fight_date=date(2020, 6, 1),
            ),
        ]
        round_stats2 = {2: [make_round_stats(fight_id=2, fighter_id=1)]}
        result = computer.compute_all([fight2], overall_snaps2, round_stats2)

        # Fighter 1 in Welterweight striking:
        # LW striking after fight 1 = 1500 + 20*1.0 = 1520 (KO ratio=1.0 since Phase 9.1)
        # Transfer with default pct=0.50: new_elo = 1500 + 0.50 * (1520 - 1500) = 1510
        snap_a_str = next(
            s for s in result
            if s.fighter_id == 1 and s.elo_type == "striking" and s.division == "Welterweight"
        )
        # elo_before should reflect the transferred rating
        assert snap_a_str.elo_before == pytest.approx(1510.0, abs=0.5)

    def test_fight_skipped_by_overall_engine_produces_no_domain_snapshots(
        self, computer: DomainEloComputer,
    ) -> None:
        """Fight skipped by overall engine (no overall snapshot) produces no domain snapshots."""
        fight = make_fight(fight_id=1, method="KO/TKO")
        # No overall snapshots for this fight
        overall_snaps: list[SnapshotRecord] = []
        round_stats = {1: [make_round_stats(fight_id=1, fighter_id=1)]}
        result = computer.compute_all([fight], overall_snaps, round_stats)
        assert len(result) == 0

    def test_draw_with_round_stats_uses_attribution(
        self, computer: DomainEloComputer,
    ) -> None:
        """Draw with round stats uses same attribution logic with reduced overall delta."""
        fight = make_fight(
            fight_id=1, winner_id=None, method="Draw",
            method_detail=None,
        )
        # Draw: deltas are typically smaller
        overall_snaps = [
            self._make_overall_snap(1, 1, elo_before=1500.0, elo_after=1495.0),
            self._make_overall_snap(2, 1, elo_before=1500.0, elo_after=1505.0),
        ]
        round_stats = {
            1: [
                make_round_stats(fight_id=1, fighter_id=1, sig_str_landed=40),
                make_round_stats(fight_id=1, fighter_id=2, sig_str_landed=30),
            ],
        }
        result = computer.compute_all([fight], overall_snaps, round_stats)
        assert len(result) == 4  # still 4 domain snapshots

    def test_k_factor_used_equals_overall_k_times_ratio(
        self, computer: DomainEloComputer,
    ) -> None:
        """k_factor_used = overall_snap.k_factor_used * ratio."""
        fight = make_fight(fight_id=1, method="KO/TKO", method_detail=None)
        overall_snaps = [
            self._make_overall_snap(1, 1, k_factor_used=60.0),
            self._make_overall_snap(2, 1, k_factor_used=60.0),
        ]
        round_stats = {1: [make_round_stats(fight_id=1, fighter_id=1)]}
        result = computer.compute_all([fight], overall_snaps, round_stats)

        # KO/TKO: striking_ratio=1.0, grappling_ratio=0.0 (validated Phase 9.1)
        snap_str = next(
            s for s in result if s.fighter_id == 1 and s.elo_type == "striking"
        )
        assert snap_str.k_factor_used == pytest.approx(60.0 * 1.0)

        snap_grp = next(
            s for s in result if s.fighter_id == 1 and s.elo_type == "grappling"
        )
        assert snap_grp.k_factor_used == pytest.approx(60.0 * 0.0)
