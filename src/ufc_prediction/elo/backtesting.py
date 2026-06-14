"""Backtesting infrastructure for Elo parameter tuning.

Grid search over EloConfig parameter space with Brier score evaluation
on temporally held-out data (per D-12, D-13). Supports K-factor, MOV
multipliers, division transfer, inactivity regression, EWMA half-life,
and domain attribution parameter groups.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine, FightRecord


def evaluate_brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """Compute Brier score for a set of predictions vs outcomes.

    Lower is better. 0.25 = random. <0.20 = meaningful signal.
    """
    return float(brier_score_loss(np.array(outcomes), np.array(predictions)))


def generate_grid_configs() -> list[EloConfig]:
    """Generate 225 EloConfig objects for K-factor grid search (per D-12).

    Parameter ranges:
        k_initial: 20-60 step 5 (9 values)
        k_experienced: 10-30 step 5 (5 values)
        k_transition_fights: 3-7 step 1 (5 values)
    """
    k_initial_range = range(20, 65, 5)
    k_experienced_range = range(10, 35, 5)
    k_transition_range = range(3, 8)

    configs = []
    for ki, ke, kt in itertools.product(k_initial_range, k_experienced_range, k_transition_range):
        configs.append(
            EloConfig(
                k_initial=float(ki),
                k_experienced=float(ke),
                k_transition_fights=kt,
            )
        )
    return configs


def generate_mov_grid() -> list[EloConfig]:
    """Generate 294 EloConfig variants for MOV multiplier grid search (per D-04).

    Parameter ranges:
        mov_ko_tko: 1.2 to 1.8 step 0.1 (7 values)
        mov_submission: 1.1 to 1.7 step 0.1 (7 values)
        mov_split: 0.5 to 1.0 step 0.1 (6 values)
        mov_dq: held fixed at 0.8 (per D-04)

    Total: 7 * 7 * 6 = 294 configs.
    """
    base = EloConfig()
    ko_range = [x / 10 for x in range(12, 19)]   # 1.2–1.8
    sub_range = [x / 10 for x in range(11, 18)]  # 1.1–1.7
    sd_range = [x / 10 for x in range(5, 11)]    # 0.5–1.0
    return [
        replace(base, mov_ko_tko=ko, mov_submission=sub, mov_split=sd)
        for ko, sub, sd in itertools.product(ko_range, sub_range, sd_range)
    ]


def generate_transfer_grid() -> list[EloConfig]:
    """Generate 7 EloConfig variants for division transfer percentage sweep (per D-06).

    Test values: [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
    """
    base = EloConfig()
    transfer_pcts = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
    return [replace(base, division_transfer_pct=pct) for pct in transfer_pcts]


def generate_inactivity_grid() -> list[EloConfig]:
    """Generate EloConfig variants for inactivity regression parameter sweep (per D-07).

    Phase 1 (threshold sweep, hold rate=0.10, cap=0.50):
        thresholds = [270, 365, 456, 548] days (9, 12, 15, 18 months)
    Phase 2 (rate sweep, hold threshold=365, cap=0.50):
        rates = [0.05, 0.08, 0.10, 0.15, 0.20]
    Phase 3 (cap sweep, hold threshold=365, rate=0.10):
        caps = [0.30, 0.40, 0.50, 0.60, 0.75]

    Total: 4 + 5 + 5 = 14 independent sweep configs.
    """
    base = EloConfig()
    configs: list[EloConfig] = []

    # Phase 1: Sweep threshold (hold rate=0.10, cap=0.50)
    thresholds = [270, 365, 456, 548]
    for t in thresholds:
        configs.append(replace(base, inactivity_threshold_days=t))

    # Phase 2: Sweep rate (hold threshold=365, cap=0.50)
    rates = [0.05, 0.08, 0.10, 0.15, 0.20]
    for r in rates:
        configs.append(replace(base, inactivity_regression_rate=r))

    # Phase 3: Sweep cap (hold threshold=365, rate=0.10)
    caps = [0.30, 0.40, 0.50, 0.60, 0.75]
    for c in caps:
        configs.append(replace(base, inactivity_regression_cap=c))

    return configs


def run_backtest(
    fights: list[FightRecord],
    configs: list[EloConfig],
    cutoff_date: date,
) -> list[dict]:
    """Run grid search over configs, evaluating each against Brier score.

    Works for any EloConfig parameter sweep -- K-factor, MOV multipliers,
    division transfer, inactivity regression, or any combination.

    Splits fights into train (before cutoff) and test (on/after cutoff).
    For each config, builds rating state on training data, then evaluates
    predictions on test data.

    Returns results sorted by Brier score ascending (best first).
    """
    train_fights = [f for f in fights if f.event_date < cutoff_date]
    test_fights = [f for f in fights if f.event_date >= cutoff_date]

    results = []
    for config in configs:
        engine = EloEngine(config)
        # Build rating state on training fights
        engine.compute_all(train_fights)

        # Evaluate on test fights
        predictions = []
        outcomes = []
        for fight in test_fights:
            # Skip draws / no-contest
            if fight.winner_id is None:
                continue

            division = fight.weight_class
            rating_a = engine.get_rating(fight.fighter_a_id, division)
            rating_b = engine.get_rating(fight.fighter_b_id, division)
            expected_prob = engine.expected_win_probability(rating_a, rating_b)
            predictions.append(expected_prob)
            outcomes.append(1 if fight.winner_id == fight.fighter_a_id else 0)

        score = evaluate_brier_score(predictions, outcomes) if predictions else float("inf")

        results.append({
            "config": config,
            "brier_score": score,
            "n_test_fights": len(predictions),
        })

    results.sort(key=lambda r: r["brier_score"])
    return results


# Generic alias: run_parameter_backtest accepts any EloConfig variation grid (per D-10).
run_parameter_backtest = run_backtest


# ── EWMA Grid ─────────────────────────────────────────────────────────────────


def generate_ewma_grid() -> list[dict]:
    """Generate EWMA half-life configs for backtesting (per D-08).

    Returns dicts (not EloConfig) because EWMA is a FeatureConfig parameter.
    Each dict: {"half_life": int, "alpha": float}
    Half-lives swept: 2, 3, 4, 5 fights (integer values only).
    """
    configs = []
    for hl in [2, 3, 4, 5]:
        alpha = 1.0 - 0.5 ** (1.0 / hl)
        configs.append({"half_life": hl, "alpha": round(alpha, 6)})
    return configs


# ── Domain Attribution Grid ───────────────────────────────────────────────────


def generate_domain_attribution_grid() -> list[dict]:
    """Generate domain attribution ratio configs for backtesting (per D-09).

    Returns dicts with striking/grappling ratios for KO and Sub finishes.
    KO striking: 0.6 to 1.0 step 0.1 (5 values)
    Sub grappling: 0.6 to 1.0 step 0.1 (5 values)
    Total: 25 configs.
    """
    configs = []
    for ko_striking in [round(x / 10, 1) for x in range(6, 11)]:
        for sub_grappling in [round(x / 10, 1) for x in range(6, 11)]:
            configs.append({
                "ko_striking": ko_striking,
                "ko_grappling": round(1.0 - ko_striking, 1),
                "sub_striking": round(1.0 - sub_grappling, 1),
                "sub_grappling": sub_grappling,
            })
    return configs


# ── Results Persistence ───────────────────────────────────────────────────────


def save_backtest_results(
    parameter_group: str,
    results: list[dict],
    cutoff_date: date,
    search_space: dict,
    output_dir: Path | None = None,
) -> Path:
    """Save backtest results to JSON file (per D-11).

    Args:
        parameter_group: Name of parameter group (e.g., "mov_multipliers")
        results: Sorted list of result dicts from backtest
        cutoff_date: Temporal split date used
        search_space: Description of the search space
        output_dir: Directory for output. Defaults to project root / "results"

    Returns:
        Path to the saved JSON file.
    """
    if output_dir is None:
        output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"backtest_{parameter_group}.json"

    # Serialize configs -- handle both EloConfig dataclass objects and plain dicts
    serialized_results = []
    for r in results:
        entry: dict = {
            "rank": len(serialized_results) + 1,
            "brier_score": r["brier_score"],
        }
        if "n_test_fights" in r:
            entry["n_test_fights"] = r["n_test_fights"]
        cfg = r.get("config")
        if cfg is not None:
            if hasattr(cfg, "__dataclass_fields__"):
                entry["config"] = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
            else:
                entry["config"] = cfg
        # Include any other keys (half_life, alpha, etc.)
        for key in r:
            if key not in ("config", "brier_score", "n_test_fights"):
                entry[key] = r[key]
        serialized_results.append(entry)

    payload = {
        "parameter_group": parameter_group,
        "timestamp": datetime.now().isoformat(),
        "cutoff_date": str(cutoff_date),
        "search_space": search_space,
        "total_configs": len(results),
        "n_test_fights": results[0].get("n_test_fights", 0) if results else 0,
        "results": serialized_results,
    }

    filepath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return filepath


def load_backtest_results(filepath: Path) -> dict:
    """Load backtest results from JSON file."""
    return json.loads(filepath.read_text(encoding="utf-8"))


# ── EWMA Evaluator ────────────────────────────────────────────────────────────


def evaluate_ewma_brier(
    half_life: int,
    alpha: float,
    fights: list[FightRecord],
    round_stats_by_fight: dict[int, list[dict]],
    domain_elo: list[dict],
    cutoff_date: date,
) -> dict:
    """Evaluate EWMA half-life via feature-based Brier score (per D-02).

    Since EWMA doesn't affect Elo computation, evaluates downstream impact
    by training a logistic model on EWMA feature diffs pre-cutoff and
    measuring Brier score on post-cutoff predictions.

    Returns dict with "half_life", "alpha", "brier_score", "n_test_fights".
    """
    from sklearn.linear_model import LogisticRegression

    from ufc_prediction.features.compute import FeatureComputer
    from ufc_prediction.features.config import FeatureConfig

    config = FeatureConfig(ewma_half_life=half_life, ewma_alpha=alpha)
    computer = FeatureComputer(config)
    # FeatureComputer.compute_all expects list[dict]; convert FightRecord dataclasses if needed
    fights_as_dicts = [
        f if isinstance(f, dict) else {
            "fight_id": f.fight_id,
            "event_date": f.event_date,
            "fighter_a_id": f.fighter_a_id,
            "fighter_b_id": f.fighter_b_id,
            "winner_id": f.winner_id,
            "weight_class": f.weight_class,
            "method": f.method,
            "method_detail": f.method_detail,
            "round_finished": None,
            "time_finished": None,
            "num_rounds": 3,
        }
        for f in fights
    ]
    feature_rows = computer.compute_all(fights_as_dicts, round_stats_by_fight, domain_elo)

    # Index features by (fighter_id, fight_id) for lookup
    feature_index: dict[tuple[int, int], dict] = {}
    for row in feature_rows:
        feature_index[(row["fighter_id"], row["fight_id"])] = row

    # EWMA feature keys to use as predictors
    ewma_keys = [
        "sig_str_per_minute_ewma",
        "total_str_per_minute_ewma",
        "td_rate_ewma",
        "td_accuracy_ewma",
        "td_defense_ewma",
        "strike_defense_ewma",
        "ctrl_time_per_fight_ewma",
        "sub_att_per_fight_ewma",
    ]

    train_X: list[list[float]] = []
    train_y: list[int] = []
    test_X: list[list[float]] = []
    test_y: list[int] = []

    for fight in fights:
        if fight.winner_id is None:
            continue
        fa = feature_index.get((fight.fighter_a_id, fight.fight_id))
        fb = feature_index.get((fight.fighter_b_id, fight.fight_id))
        if fa is None or fb is None:
            continue

        diff = []
        for key in ewma_keys:
            va = fa.get(key, 0.0) or 0.0
            vb = fb.get(key, 0.0) or 0.0
            diff.append(float(va) - float(vb))

        outcome = 1 if fight.winner_id == fight.fighter_a_id else 0

        if fight.event_date < cutoff_date:
            train_X.append(diff)
            train_y.append(outcome)
        else:
            test_X.append(diff)
            test_y.append(outcome)

    if not train_X or not test_X:
        return {
            "half_life": half_life,
            "alpha": alpha,
            "brier_score": float("inf"),
            "n_test_fights": 0,
        }

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(np.array(train_X), np.array(train_y))
    probs = model.predict_proba(np.array(test_X))[:, 1]

    brier = evaluate_brier_score(probs.tolist(), test_y)

    return {
        "half_life": half_life,
        "alpha": alpha,
        "brier_score": brier,
        "n_test_fights": len(test_y),
    }


# ── Domain Attribution Evaluator ──────────────────────────────────────────────


def evaluate_domain_brier(
    attribution_config: dict,
    fights: list[FightRecord],
    overall_snapshots: list,
    round_stats_by_fight: dict[int, list[dict]],
    cutoff_date: date,
    elo_config: EloConfig | None = None,
) -> dict:
    """Evaluate domain attribution ratios via domain-specific Brier score (per D-02).

    For KO fights: uses striking Elo differential to predict outcome.
    For Sub fights: uses grappling Elo differential to predict outcome.
    Measures how well domain Elo calibrates to method-specific outcomes.

    Returns dict with attribution_config, brier_score, n_test_fights.
    """
    from ufc_prediction.elo.domain import DomainEloComputer, _FINISH_RATIOS

    config = elo_config or EloConfig()

    # Save and temporarily override _FINISH_RATIOS for this evaluation (T-9.1-03)
    original_ratios = dict(_FINISH_RATIOS)
    try:
        _FINISH_RATIOS["KO/TKO"] = (
            attribution_config["ko_striking"],
            attribution_config["ko_grappling"],
        )
        _FINISH_RATIOS["Submission"] = (
            attribution_config["sub_striking"],
            attribution_config["sub_grappling"],
        )
        # Handle scraper-format strings too
        _FINISH_RATIOS["TKO"] = _FINISH_RATIOS["KO/TKO"]
        _FINISH_RATIOS["SUB"] = _FINISH_RATIOS["Submission"]

        domain_computer = DomainEloComputer(config)
        domain_snaps = domain_computer.compute_all(
            fights, overall_snapshots, round_stats_by_fight,
        )
    finally:
        # Restore original ratios
        _FINISH_RATIOS.clear()
        _FINISH_RATIOS.update(original_ratios)

    # Build domain snapshot index: (fight_id, fighter_id, elo_type) -> elo_after
    domain_index: dict[tuple[int, int, str], float] = {}
    for snap in domain_snaps:
        domain_index[(snap.fight_id, snap.fighter_id, snap.elo_type)] = snap.elo_after

    # Evaluate: for KO fights use striking Elo; for Sub fights use grappling Elo
    predictions: list[float] = []
    outcomes: list[int] = []

    for fight in fights:
        if fight.event_date < cutoff_date:
            continue
        if fight.winner_id is None:
            continue

        is_ko = fight.method in ("KO/TKO", "TKO")
        is_sub = fight.method in ("Submission", "SUB")
        if not is_ko and not is_sub:
            continue

        domain = "striking" if is_ko else "grappling"

        elo_a = domain_index.get((fight.fight_id, fight.fighter_a_id, domain))
        elo_b = domain_index.get((fight.fight_id, fight.fighter_b_id, domain))

        if elo_a is None or elo_b is None:
            continue

        prob_a = EloEngine(config).expected_win_probability(elo_a, elo_b)
        predictions.append(prob_a)
        outcomes.append(1 if fight.winner_id == fight.fighter_a_id else 0)

    brier = evaluate_brier_score(predictions, outcomes) if predictions else float("inf")

    return {
        "config": attribution_config,
        "brier_score": brier,
        "n_test_fights": len(predictions),
    }


# ── Joint Optimization ────────────────────────────────────────────────────────


def run_joint_optimization(
    fights: list[FightRecord],
    independent_results: dict[str, list[dict]],
    cutoff_date: date,
    top_n: int = 3,
) -> list[dict]:
    """Run focused joint optimization using top-N from independent backtests (per D-01, D-03).

    Takes top_n configs from each independent Elo-parameter backtest
    (MOV, transfer, inactivity), generates cross-product, runs full
    backtest on each combination.

    Args:
        fights: Full fight list.
        independent_results: Dict mapping parameter group name to sorted results.
            Expected keys: "mov", "transfer", "inactivity".
        cutoff_date: Temporal split date.
        top_n: Number of top configs to take from each group.

    Returns:
        Sorted results (best Brier first) for joint configs.
    """
    mov_top = independent_results.get("mov", [])[:top_n]
    transfer_top = independent_results.get("transfer", [])[:top_n]
    inactivity_top = independent_results.get("inactivity", [])[:top_n]

    if not mov_top or not transfer_top or not inactivity_top:
        return []

    # Build cross-product configs
    joint_configs = []
    for m in mov_top:
        mc = m["config"]
        for t in transfer_top:
            tc = t["config"]
            for i in inactivity_top:
                ic = i["config"]
                joint_configs.append(replace(
                    EloConfig(),
                    mov_ko_tko=mc.mov_ko_tko,
                    mov_submission=mc.mov_submission,
                    mov_split=mc.mov_split,
                    division_transfer_pct=tc.division_transfer_pct,
                    inactivity_threshold_days=ic.inactivity_threshold_days,
                    inactivity_regression_rate=ic.inactivity_regression_rate,
                    inactivity_regression_cap=ic.inactivity_regression_cap,
                ))

    return run_backtest(fights, joint_configs, cutoff_date)
