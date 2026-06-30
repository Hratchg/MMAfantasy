"""Unit tests for Elo backtesting module."""

from __future__ import annotations

from datetime import date

from ufc_prediction.elo.backtesting import (
    evaluate_brier_score,
    generate_domain_attribution_grid,
    generate_ewma_grid,
    generate_grid_configs,
    generate_inactivity_grid,
    generate_mov_grid,
    generate_transfer_grid,
    load_backtest_results,
    run_backtest,
    run_joint_optimization,
    run_parameter_backtest,
    save_backtest_results,
)
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import FightRecord


def test_evaluate_brier_score_perfect():
    """Perfect predictions yield Brier score of 0.0."""
    predictions = [1.0, 0.0]
    outcomes = [1, 0]
    score = evaluate_brier_score(predictions, outcomes)
    assert score == 0.0


def test_evaluate_brier_score_random():
    """Random 50/50 predictions yield Brier score of 0.25."""
    predictions = [0.5, 0.5]
    outcomes = [1, 0]
    score = evaluate_brier_score(predictions, outcomes)
    assert abs(score - 0.25) < 1e-9


def test_evaluate_brier_score_bad():
    """Worst-case predictions yield Brier score of 1.0."""
    predictions = [0.0, 1.0]
    outcomes = [1, 0]
    score = evaluate_brier_score(predictions, outcomes)
    assert abs(score - 1.0) < 1e-9


def test_generate_grid_configs():
    """Grid search generates 225 EloConfig objects (9 * 5 * 5)."""
    configs = generate_grid_configs()
    assert len(configs) == 225
    # All should be EloConfig instances
    assert all(isinstance(c, EloConfig) for c in configs)
    # Check boundary values are present
    k_initials = {c.k_initial for c in configs}
    k_experienced = {c.k_experienced for c in configs}
    k_transitions = {c.k_transition_fights for c in configs}
    assert k_initials == {20, 25, 30, 35, 40, 45, 50, 55, 60}
    assert k_experienced == {10, 15, 20, 25, 30}
    assert k_transitions == {3, 4, 5, 6, 7}


def test_run_backtest_returns_sorted_results():
    """Backtest returns results sorted by Brier score ascending (best first)."""
    # Create synthetic fight data
    fights = []
    fid = 0
    # 10 fights before cutoff (training)
    for i in range(10):
        fid += 1
        fights.append(
            FightRecord(
                fight_id=fid,
                event_date=date(2020, 1, 1),
                fighter_a_id=1,
                fighter_b_id=2,
                winner_id=1,
                weight_class="Lightweight",
                method="Decision",
                method_detail="Unanimous",
            )
        )
    # 5 fights after cutoff (test)
    for i in range(5):
        fid += 1
        fights.append(
            FightRecord(
                fight_id=fid,
                event_date=date(2022, 1, 1),
                fighter_a_id=1,
                fighter_b_id=2,
                winner_id=1,
                weight_class="Lightweight",
                method="Decision",
                method_detail="Unanimous",
            )
        )

    configs = [
        EloConfig(k_initial=20.0, k_experienced=10.0, k_transition_fights=3),
        EloConfig(k_initial=40.0, k_experienced=20.0, k_transition_fights=5),
        EloConfig(k_initial=60.0, k_experienced=30.0, k_transition_fights=7),
    ]

    results = run_backtest(fights, configs, cutoff_date=date(2021, 1, 1))

    assert len(results) == 3
    # Results sorted by brier_score ascending (best first)
    for i in range(len(results) - 1):
        assert results[i]["brier_score"] <= results[i + 1]["brier_score"]
    # Each result has expected keys
    for r in results:
        assert "config" in r
        assert "brier_score" in r
        assert "n_test_fights" in r
        assert isinstance(r["config"], EloConfig)
        assert r["n_test_fights"] == 5


def test_generate_mov_grid():
    """MOV grid generates exactly 294 EloConfig variants (7*7*6)."""
    configs = generate_mov_grid()
    assert len(configs) == 294
    assert all(isinstance(c, EloConfig) for c in configs)
    ko_values = {round(c.mov_ko_tko, 1) for c in configs}
    sub_values = {round(c.mov_submission, 1) for c in configs}
    sd_values = {round(c.mov_split, 1) for c in configs}
    assert ko_values == {1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8}
    assert sub_values == {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7}
    assert sd_values == {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    # DQ stays fixed at 0.8 (per D-04)
    assert all(c.mov_dq == 0.8 for c in configs)


def test_generate_transfer_grid():
    """Transfer grid generates exactly 7 EloConfig variants."""
    configs = generate_transfer_grid()
    assert len(configs) == 7
    pcts = {round(c.division_transfer_pct, 2) for c in configs}
    assert pcts == {0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00}


def test_generate_inactivity_grid():
    """Inactivity grid generates at least 14 configs covering all 3 sub-parameters."""
    configs = generate_inactivity_grid()
    assert len(configs) >= 14
    assert all(isinstance(c, EloConfig) for c in configs)
    thresholds = {c.inactivity_threshold_days for c in configs}
    assert {270, 365, 456, 548}.issubset(thresholds)


def test_run_parameter_backtest_sorted():
    """run_parameter_backtest returns results sorted by brier_score ascending."""
    fights = []
    fid = 0
    for i in range(10):
        fid += 1
        fights.append(
            FightRecord(
                fight_id=fid,
                event_date=date(2020, 1, 1),
                fighter_a_id=1,
                fighter_b_id=2,
                winner_id=1,
                weight_class="Lightweight",
                method="Decision",
                method_detail="Unanimous",
            )
        )
    for i in range(5):
        fid += 1
        fights.append(
            FightRecord(
                fight_id=fid,
                event_date=date(2022, 1, 1),
                fighter_a_id=1,
                fighter_b_id=2,
                winner_id=1,
                weight_class="Lightweight",
                method="Decision",
                method_detail="Unanimous",
            )
        )

    configs = [
        EloConfig(mov_ko_tko=1.2, mov_submission=1.1, mov_split=0.5),
        EloConfig(mov_ko_tko=1.8, mov_submission=1.7, mov_split=1.0),
    ]
    results = run_parameter_backtest(fights, configs, cutoff_date=date(2021, 1, 1))
    assert len(results) == 2
    for r in results:
        assert "config" in r
        assert "brier_score" in r
        assert "n_test_fights" in r
    assert results[0]["brier_score"] <= results[1]["brier_score"]


# ── Task 1: EWMA grid, domain attribution grid, and persistence tests ─────────


def test_generate_ewma_grid():
    """EWMA grid generates exactly 4 configs for half-lives 2, 3, 4, 5."""
    configs = generate_ewma_grid()
    assert len(configs) == 4
    half_lives = {c["half_life"] for c in configs}
    assert half_lives == {2, 3, 4, 5}
    assert all("alpha" in c for c in configs)


def test_generate_ewma_grid_alpha_values():
    """Alpha for half_life=3 is approx 0.206 (= 1 - 0.5^(1/3))."""
    configs = generate_ewma_grid()
    hl3 = next(c for c in configs if c["half_life"] == 3)
    expected_alpha = 1.0 - 0.5 ** (1.0 / 3)
    assert abs(hl3["alpha"] - expected_alpha) < 0.001


def test_generate_domain_attribution_grid():
    """Domain attribution grid generates exactly 25 configs with the correct keys."""
    configs = generate_domain_attribution_grid()
    assert len(configs) == 25
    assert all(
        {"ko_striking", "ko_grappling", "sub_striking", "sub_grappling"} == set(c.keys())
        for c in configs
    )


def test_domain_attribution_grid_ratios_sum_to_one():
    """For every config, ko_striking + ko_grappling == 1.0 and sub_striking + sub_grappling == 1.0."""
    configs = generate_domain_attribution_grid()
    for c in configs:
        assert abs(c["ko_striking"] + c["ko_grappling"] - 1.0) < 1e-9
        assert abs(c["sub_striking"] + c["sub_grappling"] - 1.0) < 1e-9


def test_domain_attribution_grid_contains_current():
    """The current defaults (ko_striking=0.8, sub_grappling=0.8) are in the grid."""
    configs = generate_domain_attribution_grid()
    current = [c for c in configs if c["ko_striking"] == 0.8 and c["sub_grappling"] == 0.8]
    assert len(current) == 1


def test_save_and_load_backtest_results(tmp_path):
    """save_backtest_results writes valid JSON; load_backtest_results round-trips correctly."""
    results = [
        {"config": EloConfig(), "brier_score": 0.24, "n_test_fights": 100},
        {"config": EloConfig(k_initial=30.0), "brier_score": 0.25, "n_test_fights": 100},
    ]
    filepath = save_backtest_results(
        "test_group",
        results,
        date(2023, 1, 1),
        {"param": "test"},
        output_dir=tmp_path,
    )
    assert filepath.exists()
    loaded = load_backtest_results(filepath)
    assert loaded["parameter_group"] == "test_group"
    assert len(loaded["results"]) == 2
    assert loaded["results"][0]["brier_score"] == 0.24


# ── Task 2: Joint optimization test ───────────────────────────────────────────


def test_run_joint_optimization_structure():
    """run_joint_optimization returns sorted results with config and brier_score keys."""
    fights = []
    fid = 0
    for _ in range(10):
        fid += 1
        fights.append(
            FightRecord(
                fight_id=fid,
                event_date=date(2020, 1, 1),
                fighter_a_id=1,
                fighter_b_id=2,
                winner_id=1,
                weight_class="Lightweight",
                method="KO/TKO",
                method_detail=None,
            )
        )
    for _ in range(5):
        fid += 1
        fights.append(
            FightRecord(
                fight_id=fid,
                event_date=date(2022, 1, 1),
                fighter_a_id=1,
                fighter_b_id=2,
                winner_id=1,
                weight_class="Lightweight",
                method="KO/TKO",
                method_detail=None,
            )
        )

    independent = {
        "mov": [{"config": EloConfig(mov_ko_tko=1.5), "brier_score": 0.24}],
        "transfer": [{"config": EloConfig(division_transfer_pct=0.75), "brier_score": 0.24}],
        "inactivity": [{"config": EloConfig(inactivity_threshold_days=365), "brier_score": 0.24}],
    }
    results = run_joint_optimization(fights, independent, date(2021, 1, 1), top_n=1)
    assert len(results) == 1
    assert "config" in results[0]
    assert "brier_score" in results[0]
