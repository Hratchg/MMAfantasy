"""ML prediction layer configuration.

Defines MLConfig dataclass, feature column ordering, and stance encoding.
All tunable ML parameters are defined here as a frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

from ufc_prediction.features.config import CANONICAL_FEATURE_ORDER


@dataclass(frozen=True)
class MLConfig:
    """Immutable ML configuration.

    Defaults per decisions D-05, D-06, D-07 from CONTEXT.md.
    Phase 15.1 D-05/D-06/D-07 added train_lower_bound_date for the
    coverage-driven training-window lower bound.
    """

    cutoff_date: str = "2023-01-01"
    train_lower_bound_date: str | None = None  # Phase 15.1 D-05/D-06/D-07
    model_dir: str = "models"
    n_optuna_trials: int = 50
    cv_splits: int = 5
    random_seed: int = 42


# 20 numeric features from CANONICAL_FEATURE_ORDER (excluding style_tag).
# Used internally to extract features from ComputedFeature JSON.
PERFORMANCE_FEATURE_KEYS: list[str] = [f for f in CANONICAL_FEATURE_ORDER if f != "style_tag"]

# Feature columns in deterministic order.
# 3 Elo + 20 performance + 4 physical + 1 stance + 16 career
# + 2 cross-domain + 3 context + 4 pace decay + 3 non-linear layoff
# + 8 rolling windows + 2 rematch + 1 pre-UFC record differential
# + 3 betting odds differentials = 70 features
FEATURE_COLUMNS: list[str] = [
    # 3 Elo differentials
    "elo_overall_diff",
    "elo_striking_diff",
    "elo_grappling_diff",
    # 20 performance feature differentials
    *[f"{key}_diff" for key in PERFORMANCE_FEATURE_KEYS],
    # 4 physical differentials
    "height_diff",
    "reach_diff",
    "leg_reach_diff",
    "age_diff",
    # 1 stance binary feature
    "stance_matchup",
    # 16 career stat differentials
    "win_streak_diff",
    "loss_streak_diff",
    "career_win_pct_diff",
    "ufc_fight_count_diff",
    "days_since_last_fight_diff",
    "ko_finish_rate_diff",
    "sub_finish_rate_diff",
    "ko_loss_rate_diff",
    "sub_loss_rate_diff",
    "total_cage_minutes_diff",
    "avg_fight_duration_diff",
    "division_fight_count_diff",
    "is_debut_diff",
    "fights_per_year_diff",
    "avg_opponent_elo_diff",
    "elo_momentum_diff",
    # 2 cross-domain matchup features
    "a_striking_vs_b_grappling",
    "a_grappling_vs_b_striking",
    # 3 fight context flags
    "is_title_fight",
    "num_rounds",
    "weight_class_ordinal",
    # ── Tier 1 new features ──
    # 4 pace decay differentials (round-by-round cardio)
    "pace_decay_strikes_diff",
    "pace_decay_td_diff",
    "pace_output_variance_diff",
    "avg_r1_sig_str_diff",
    # 3 non-linear layoff features
    "log_days_since_last_fight_diff",
    "is_short_turnaround_diff",
    "is_comeback_diff",
    # 8 rolling window differentials (last 3 and last 5 fights)
    "sig_str_per_min_last3_diff",
    "td_rate_last3_diff",
    "strike_defense_last3_diff",
    "ctrl_time_last3_diff",
    "sig_str_per_min_last5_diff",
    "td_rate_last5_diff",
    "strike_defense_last5_diff",
    "ctrl_time_last5_diff",
    # 2 rematch features
    "is_rematch",
    "first_fight_winner_diff",
    # 1 pre-UFC record differential (Sherdog career before UFC debut)
    "pre_ufc_win_pct_diff",
    # 3 betting odds differentials (BestFightOdds consensus, vig-removed per D-02)
    # Order locked by D-04 (opening + closing as separate features) + D-05
    # (line movement = (cl_a - op_a) - (cl_b - op_b)) + D-09 (A minus B).
    "opening_prob_diff",
    "closing_prob_diff",
    "line_movement_diff",
    # 2 engineered odds features (Phase 15.1 gap closure on D-05).
    # sharp_money_signal: |line_movement_diff| — magnitude of asymmetric
    #                     drift, captures sharp-money intervention.
    # odds_elo_divergence: closing market prob for A − Elo-implied prob
    #                     for A — where market and our Elo disagree.
    "sharp_money_signal",
    "odds_elo_divergence",
    # 3 opponent-network differentials (Phase 16-03 NET-01/02; operator-
    # approved config = pan-mma + MOV-weighted edges per gsd-checkpoint).
    # APPEND-ONLY per Gotcha 9 — xgb_v2 byte-identity is the v1.1 rollback
    # path (D-09(P15)); these MUST stay at the END of the list.
    # pagerank_diff:           A's PageRank − B's PageRank, as-of fight date.
    # sos_2hop_diff:           A's 2-hop SoS − B's 2-hop SoS (mean of
    #                          in-neighbors' PageRank).
    # is_debutant_in_graph_diff: A_debutant − B_debutant, each in {0, 1};
    #                          the diff lands in {-1, 0, 1}.
    "pagerank_diff",
    "sos_2hop_diff",
    "is_debutant_in_graph_diff",
]


# NET-V2-01 (Phase 18): 72-col view dropping the trailing 3 NET-* cols.
# Safe-by-construction: NET-* are guaranteed at the END of FEATURE_COLUMNS by
# the APPEND-ONLY discipline (D-09(P15)).
FEATURE_COLUMNS_NO_NET: list[str] = FEATURE_COLUMNS[:-3]


# ── Phase 23 v2.2 feature set (REF + TRAVEL + META) ─────────────────────────
#
# Phase 23 D-08 (APPEND-ONLY discipline per Pitfall #8 / D-09(P15)):
# FEATURE_COLUMNS_NO_NET is the xgb_v2 byte-identity baseline (72 cols).
# FEATURE_COLUMNS_V22 is a NEW constant — never mutate the baseline.
# REF cols added in Plan 23-01; TRAVEL cols in Plan 23-02; META cols in Plan
# 23-03 (NaN-padded until those plans land).
#
# Per RESEARCH §"Open Question Q4 (RESOLVED)" + CONTEXT D-07 dedup discipline:
#   - `layoff_days_diff` is DROPPED (semantic duplicate of
#     FEATURE_COLUMNS_NO_NET[61] `days_since_last_fight_diff`).
#   - Raw per-fighter `layoff_days_red` + `layoff_days_blue` KEPT (the raw
#     per-fighter form is NOT in FEATURE_COLUMNS_NO_NET; only the
#     differential is — and ours uses a 720-clip the existing col does not).
#
# Final count: 72 + 3 + 6 + 10 = 91 cols (revised from 92 after Q4 RESOLVED).
FEATURE_COLUMNS_V22: list[str] = FEATURE_COLUMNS_NO_NET + [
    # ── REF (3 cols; per-event, NOT per-fighter-differential) ───────
    "ref_finish_rate_shrunk",
    "ref_decision_rate_shrunk",
    "ref_no_action_rate_shrunk",
    # ── TRAVEL (6 cols) — added in Plan 23-02 ────────────────────────
    "travel_distance_miles_red",
    "travel_distance_miles_blue",
    "travel_distance_miles_diff",
    "tz_shift_red_signed",
    "tz_shift_blue_signed",
    "tz_shift_diff_signed",
    # ── META rich Level-1 (10 cols) — added in Plan 23-03 ────────────
    # NB: layoff_days_diff DROPPED per RESEARCH Q4 RESOLVED + D-07 dedup.
    "layoff_days_red",
    "layoff_days_blue",
    "age_at_fight_red",
    "age_at_fight_blue",
    "elo_overall_velocity_diff",
    "elo_striking_velocity_diff",
    "elo_grappling_velocity_diff",
    "division_finish_rate_shrunk",
    "reach_diff_normalized",
]


# ── Phase 42 v2.5 TRAVEL close-out (Plan 42-01) ──────────────────────────
#
# Phase 42 D-21 + D-22 (CONTEXT-locked): NEW siblings beyond the v2.2 TRAVEL
# block (FEATURE_COLUMNS_V22 indices 75-80) — kilometer-scale Haversine and
# ±12-clipped UTC tz_shift, both NaN-debut (NOT 0.0-sentinel like v2.2).
#
# Additive-only per APPEND-ONLY discipline (D-09(P15) carry-forward).
# FEATURE_COLUMNS_V22 stays byte-stable (it's xgb_v2 + meta_v2 input space —
# touching it would invalidate AUDIT-01 SHA invariants).
#
# The v2.5 cols are consumed ONLY by the candidate META-V22+CALIB+TRAVEL
# blender (Plan 42-02). xgb_v2 + meta_v2 continue to consume the 90-col
# FEATURE_COLUMNS_V22 substrate verbatim.
#
# Both cols are red - blue differentials (parallel to META-V22 input shape;
# the per-fighter raw form is not exposed at this layer).
#
# Final count: 90 + 2 = 92 cols.
FEATURE_COLUMNS_V25_TRAVEL: list[str] = FEATURE_COLUMNS_V22 + [
    "travel_distance_km",  # Haversine km, fighter_red - fighter_blue differential
    "tz_shift_hours",  # ±12 clipped hours, fighter_red - fighter_blue differential
]


def get_feature_columns(
    *,
    feature_set: str = "v2.1-no-net",
    include_net: bool | None = None,
) -> list[str]:
    """Return the canonical feature-column list for train/predict.

    Per CONTEXT D-09: ``feature_set`` is the new public knob.
        - ``"v1.0"`` returns the full 75-col list (includes 3 NET-* cols).
        - ``"v2.1-no-net"`` returns the 72-col xgb_v2 baseline (default).
        - ``"v2.2"`` returns the 91-col Phase 23 list (REF + TRAVEL + META
          appended).
        - ``"v2.5-travel"`` returns the 92-col Phase 42 list (v2.2 + 2 new
          TRAVEL-V25 siblings ``travel_distance_km`` + ``tz_shift_hours``).

    ``include_net`` is the Phase 18 back-compat shim:
        - ``include_net=True``  → ``feature_set="v1.0"``
        - ``include_net=False`` → ``feature_set="v2.1-no-net"``
        - ``include_net=None``  → respect ``feature_set`` as-is (default).

    Unknown ``feature_set`` raises ``ValueError`` (Pitfall #1 column-drift
    guard — no silent wrong-list returns).
    """
    if include_net is not None:
        feature_set = "v1.0" if include_net else "v2.1-no-net"
    if feature_set == "v1.0":
        return list(FEATURE_COLUMNS)
    if feature_set == "v2.1-no-net":
        return list(FEATURE_COLUMNS_NO_NET)
    if feature_set == "v2.2":
        return list(FEATURE_COLUMNS_V22)
    if feature_set == "v2.5-travel":
        return list(FEATURE_COLUMNS_V25_TRAVEL)
    raise ValueError(f"unknown feature_set={feature_set!r}")


def encode_stance_matchup(stance_a: str | None, stance_b: str | None) -> float:
    """Encode stance matchup as binary: 1.0 = same stance, 0.0 = opposite.

    Per D-02: southpaw vs orthodox is a known MMA factor.
    Switch and None are treated as Orthodox (most common stance, ~70% of fighters).
    """

    def normalize(s: str | None) -> str:
        if s is None or s == "Switch":
            return "Orthodox"
        return s

    return 1.0 if normalize(stance_a) == normalize(stance_b) else 0.0
