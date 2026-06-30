"""Feature matrix assembly for ML prediction.

Joins Elo differentials, 20 computed performance features, and physical
attribute differentials into per-fight feature vectors ready for XGBoost.

Per D-03: All features are Fighter A minus Fighter B differentials.
Per D-12/ML-06: Elo ratings used as input features, not replaced.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date

import numpy as np

from ufc_prediction.ml.config import (
    FEATURE_COLUMNS_V22,
    FEATURE_COLUMNS_V25_TRAVEL,
    PERFORMANCE_FEATURE_KEYS,
    MLConfig,
    encode_stance_matchup,
    get_feature_columns,
)
from ufc_prediction.ml.features_v22.meta import (
    age_at_fight,
    division_finish_rate_shrunk,
    elo_velocity,
    layoff_days,
    reach_diff_normalized,
)
from ufc_prediction.ml.features_v22.ref import (
    classify_outcome,
    compute_ref_rates_shrunk,
)
from ufc_prediction.ml.features_v22.travel import (
    compute_travel_features,
    compute_travel_v25_features,
)


def compute_division_medians(
    fighter_physicals: dict[int, dict],
    fight_records: list[dict],
    cutoff_date: date,
) -> dict[str, dict[str, float]]:
    """Compute median height/reach/leg_reach per weight class.

    Per D-01: Uses division median imputation for missing physical values.
    Per Pitfall 4: Only uses fights with event_date before cutoff_date
    to avoid temporal leakage.

    Args:
        fighter_physicals: Dict keyed by fighter_id with physical attributes.
        fight_records: List of fight dicts with weight_class and event_date.
        cutoff_date: Only fights before this date contribute to medians.

    Returns:
        Dict mapping weight_class -> {"height_inches": median, "reach_inches": median,
        "leg_reach_inches": median}.
    """
    # Collect unique fighters per division from pre-cutoff fights
    division_fighters: dict[str, set[int]] = {}

    for fight in fight_records:
        if fight["event_date"] >= cutoff_date:
            continue

        weight_class = fight["weight_class"]
        if weight_class not in division_fighters:
            division_fighters[weight_class] = set()

        division_fighters[weight_class].add(fight["fighter_a_id"])
        division_fighters[weight_class].add(fight["fighter_b_id"])

    # Collect physical data per division (one entry per unique fighter)
    division_data: dict[str, dict[str, list[float]]] = {}

    for weight_class, fighter_ids in division_fighters.items():
        division_data[weight_class] = {
            "height_inches": [],
            "reach_inches": [],
            "leg_reach_inches": [],
        }
        for fighter_id in fighter_ids:
            phys = fighter_physicals.get(fighter_id)
            if phys is None:
                continue

            for attr in ("height_inches", "reach_inches", "leg_reach_inches"):
                val = phys.get(attr)
                if val is not None:
                    division_data[weight_class][attr].append(val)

    # Compute medians per division
    result: dict[str, dict[str, float]] = {}
    for weight_class, attrs in division_data.items():
        result[weight_class] = {}
        for attr, values in attrs.items():
            if values:
                result[weight_class][attr] = float(np.nanmedian(values))
            else:
                result[weight_class][attr] = 0.0
    return result


def split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    fight_dates: np.ndarray,
    cutoff_date: date,
    train_lower: date | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split feature matrix into train/test sets based on cutoff date.

    Args:
        X: Feature matrix (n_fights, n_features).
        y: Target vector (n_fights,).
        fight_dates: Array of event dates (n_fights,).
        cutoff_date: Fights before this date go to train, on/after to test.
        train_lower: Optional lower bound for train fold (Phase 15.1 D-05/D-06/D-07).
            If set, fights with event_date < train_lower are DROPPED from
            the train fold (and never appear in the test fold). Test fold is
            unaffected: it always uses event_date >= cutoff_date.

    Returns:
        (X_train, X_test, y_train, y_test).

    Invariants:
        - When train_lower is None, behavior is identical to the pre-15.1
          2-bucket split.
        - When train_lower is set,
          len(X_train) + len(X_test) <= len(X) (strict less-than if any
          fight_dates < train_lower exist).
        - Test fold size is invariant to train_lower (Pitfall 5).
    """
    if train_lower is not None:
        train_mask = np.array([train_lower <= d < cutoff_date for d in fight_dates])
    else:
        train_mask = np.array([d < cutoff_date for d in fight_dates])
    # IMPORTANT (Pitfall 2): compute test_mask EXPLICITLY, NOT ~train_mask.
    # With 3 buckets, ~train_mask would lump dropped pre-train_lower rows
    # into the test fold, which is a temporal-leakage bug.
    test_mask = np.array([d >= cutoff_date for d in fight_dates])
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ── Phase 23 v2.2 REF helpers ───────────────────────────────────────────────
#
# Per CONTEXT D-10 + Pitfall #12: both feature_matrix.py and inference_features.py
# import compute_ref_rates_shrunk from features_v22.ref so train/predict math is
# byte-identical. The two helpers below are private to the training path —
# they aggregate the full corpus into the dicts that compute_ref_rates_shrunk
# consumes per-fight. inference_features.py builds the equivalent dicts from
# the live DB via _query_ref_state.
#
# Phase 23 D-10 (LIVE-03 parity / Pitfall #8 column-order drift guard):
# When feature_set='v2.2' is used, the assembler MUST produce exactly
# FEATURE_COLUMNS_V22 cols in exactly that order. The static guard is the
# module-level constant + length-check below; the dynamic guard lives in
# assemble() after row construction.
_EXPECTED_V22_NCOLS = len(FEATURE_COLUMNS_V22)
assert _EXPECTED_V22_NCOLS == 72 + 3 + 6 + 9, (
    f"FEATURE_COLUMNS_V22 length drift: expected 90 (72+3+6+9), "
    f"got {_EXPECTED_V22_NCOLS}. Anti-Pattern #8 violation "
    f"(column-order / count drift). NB: META section is 9 cols, not 10 — "
    f"age_diff is already in FEATURE_COLUMNS_NO_NET (D-07 dedup wins)."
)

# Phase 42 D-21 + D-22 (LIVE-03 / Pitfall #8 column-order drift guard — v2.5
# sibling): the v2.5-travel feature_set MUST produce exactly
# FEATURE_COLUMNS_V25_TRAVEL cols (90 v2.2 + 2 v2.5 = 92) in exactly that
# order. Append-only invariant — v2.5 is APPEND-ONLY (D-09(P15) carry-forward
# + Phase 42 D-21 lock). If FEATURE_COLUMNS_V22 prefix drifts here, the
# v2.2 baseline got mutated (revert immediately to preserve AUDIT-01).
_EXPECTED_V25_TRAVEL_NCOLS = len(FEATURE_COLUMNS_V25_TRAVEL)
assert _EXPECTED_V25_TRAVEL_NCOLS == 92, (
    f"FEATURE_COLUMNS_V25_TRAVEL length drift: expected 92 (90+2), "
    f"got {_EXPECTED_V25_TRAVEL_NCOLS}. Anti-Pattern #8 violation."
)
assert FEATURE_COLUMNS_V25_TRAVEL[:90] == FEATURE_COLUMNS_V22, (
    "FEATURE_COLUMNS_V25_TRAVEL prefix must equal FEATURE_COLUMNS_V22 — "
    "v2.5 is APPEND-ONLY (Phase 42 D-21 + D-09(P15) discipline). "
    "If you see this, the v2.2 baseline got mutated; revert immediately."
)


def _compute_ref_global_rates(sorted_records: list[dict]) -> dict[str, float]:
    """Pool global rates across the full corpus (no per-referee filter).

    Returns ``{"finish": x, "decision": y, "no_action": z}`` aggregated over
    all fight ``method`` strings. Used as the Beta-binomial prior in
    ``compute_ref_rates_shrunk``. Pitfall #6: caller must sort chronologically
    before passing in (we enforce this in ``assemble()``).
    """
    counts = {"finish": 0, "decision": 0, "no_action": 0}
    total = 0
    for f in sorted_records:
        cat = classify_outcome(f.get("method"))
        counts[cat] += 1
        total += 1
    if total == 0:
        return {"finish": 0.0, "decision": 0.0, "no_action": 0.0}
    return {k: v / total for k, v in counts.items()}


def _build_referee_lookup(
    sorted_records: list[dict],
) -> tuple[dict[int, int | None], dict[int, list[dict]]]:
    """Walk sorted fight records once to extract REF compute inputs.

    Returns ``(event_referees, ref_history)``:
        - ``event_referees``: event_id → referee_id (so we can look up the
          referee for a given fight via its ``event_id``).
        - ``ref_history``: referee_id → chronological list of
          ``{"event_date": d, "method": m}`` dicts for all past fights this
          referee has officiated.

    Records without ``referee_id`` are skipped from ``ref_history`` (their
    contribution becomes the Bayesian-fallback to global rates per D-02).
    Pitfall #6: caller must sort chronologically before passing in.
    """
    event_referees: dict[int, int | None] = {}
    ref_history: dict[int, list[dict]] = {}
    for f in sorted_records:
        event_id = f.get("event_id")
        ref_id = f.get("referee_id")
        if event_id is not None:
            event_referees[event_id] = ref_id
        if ref_id is not None:
            ref_history.setdefault(ref_id, []).append(
                {"event_date": f["event_date"], "method": f.get("method")}
            )
    return event_referees, ref_history


# ── Phase 23 v2.2 TRAVEL helpers (Plan 23-02) ────────────────────────────────
#
# Per CONTEXT D-10 + Pitfall #12: both feature_matrix.py and inference_features.py
# import compute_travel_features from features_v22.travel so train/predict math
# is byte-identical. The two helpers below are private to the training path —
# they aggregate the full corpus into the venue lookup dicts that
# compute_travel_features consumes per-fight. inference_features.py builds the
# equivalent state from the live DB via SQLAlchemy queries.


def _build_event_venue_lookup(
    sorted_records: list[dict],
) -> dict[int, dict | None]:
    """event_id → ``{"lat", "lon", "timezone_iana", "event_date"}`` or ``None``.

    Reads ``venue_lat`` / ``venue_lon`` / ``venue_timezone_iana`` keys from
    each fight record (joined from Event ↔ Venue at corpus-load time).
    Missing any of the three columns → ``None`` (event has no usable venue;
    consumer must NaN-pad rather than emit sentinel 0).
    """
    result: dict[int, dict | None] = {}
    for f in sorted_records:
        event_id = f.get("event_id")
        if event_id is None or event_id in result:
            continue
        lat = f.get("venue_lat")
        lon = f.get("venue_lon")
        tz = f.get("venue_timezone_iana")
        if lat is None or lon is None or tz is None:
            result[event_id] = None
        else:
            result[event_id] = {
                "lat": lat,
                "lon": lon,
                "timezone_iana": tz,
                "event_date": f["event_date"],
            }
    return result


def _build_fighter_prior_venues(
    sorted_records: list[dict],
    event_venues: dict[int, dict | None],
) -> dict[tuple[int, int], dict | None]:
    """``(fighter_id, fight_id)`` → most-recent-prior-fight venue or ``None``.

    Pre-fight as-of-date discipline (Q1 RESOLVED in 23-RESEARCH.md): the
    prior venue is the location of this fighter's previous fight, snapshot
    BEFORE updating the per-fighter ``last_seen`` map. ``None`` = first
    UFC fight for this fighter → caller emits sentinel 0 per D-04.

    Pitfall #6 defense: caller must pass ``sorted_records`` sorted by
    ``event_date``; the assembler enforces this in its v2.2 pre-pass.
    """
    result: dict[tuple[int, int], dict | None] = {}
    last_seen: dict[int, dict] = {}  # fighter_id → last venue dict
    for f in sorted_records:
        a_id = f["fighter_a_id"]
        b_id = f["fighter_b_id"]
        fight_id = f["fight_id"]
        # Snapshot BEFORE update (Pitfall #6 / strict pre-fight as-of-date).
        result[(a_id, fight_id)] = last_seen.get(a_id)
        result[(b_id, fight_id)] = last_seen.get(b_id)
        # Update AFTER snapshot — only when this event has a known venue.
        curr_venue = event_venues.get(f.get("event_id"))
        if curr_venue is not None:
            last_seen[a_id] = curr_venue
            last_seen[b_id] = curr_venue
    return result


# ── Phase 23 v2.2 META helpers (Plan 23-03) ──────────────────────────────────
#
# Per CONTEXT D-10 + Pitfall #12: both feature_matrix.py and
# inference_features.py import META helpers from features_v22.meta so
# train/predict math is byte-identical. The pre-pass builders below are
# private to the training path — they aggregate the full corpus into the
# dicts the pure helpers in meta.py consume per-fight.
# inference_features.py builds the equivalent state from the live DB via
# SQLAlchemy queries (`_query_fighter_prior_fight_date`, `_query_elo_history`,
# `_query_division_state`, `_query_division_mean_reach`,
# `_query_fighter_division`).


def _build_fighter_birth_dates(
    fighter_physicals: dict[int, dict],
) -> dict[int, date | None]:
    """fighter_id → date_of_birth (nullable per Fighter ORM column).

    Reads from the existing ``fighter_physicals`` lookup the caller already
    provides — no new DB query.
    """
    return {fid: phys.get("date_of_birth") for fid, phys in fighter_physicals.items()}


def _build_fighter_reaches(
    fighter_physicals: dict[int, dict],
) -> dict[int, float | None]:
    """fighter_id → reach_inches (nullable per Fighter ORM column)."""
    return {fid: phys.get("reach_inches") for fid, phys in fighter_physicals.items()}


def _build_elo_histories(
    sorted_records: list[dict],
    elo_features: dict[tuple[int, int], dict[str, float]],
) -> dict[int, list[dict]]:
    """fighter_id → chronological list of Elo snapshots.

    Each snapshot is ``{"event_date": d, "elo_overall": x,
    "elo_striking": y, "elo_grappling": z}``. Built from the same
    ``elo_features`` table feature_matrix already uses for Elo
    differentials — no new DB query. Caller filters by event_date to
    enforce strict pre-fight cutoff before passing to ``elo_velocity``.

    Pitfall #6: caller must pass ``sorted_records`` chronologically;
    enforced by the v2.2 pre-pass.
    """
    result: dict[int, list[dict]] = {}
    default = {"elo_overall": 1500.0, "elo_striking": 1500.0, "elo_grappling": 1500.0}
    for f in sorted_records:
        for fid_key in ("fighter_a_id", "fighter_b_id"):
            fid = f[fid_key]
            elo_dict = elo_features.get((fid, f["fight_id"]), default)
            result.setdefault(fid, []).append(
                {
                    "event_date": f["event_date"],
                    "elo_overall": elo_dict.get("elo_overall", 1500.0),
                    "elo_striking": elo_dict.get("elo_striking", 1500.0),
                    "elo_grappling": elo_dict.get("elo_grappling", 1500.0),
                }
            )
    return result


def _build_fighter_prior_fight_dates(
    sorted_records: list[dict],
) -> dict[tuple[int, int], date | None]:
    """(fighter_id, fight_id) → prior fight event_date or None (debut).

    Snapshot BEFORE update (Pitfall #6 / strict pre-fight as-of-date).
    None ↔ first UFC fight for this fighter → ``layoff_days`` emits 0
    sentinel per Q4 + D-06.

    Pitfall #6: caller must pass ``sorted_records`` chronologically.
    """
    result: dict[tuple[int, int], date | None] = {}
    last_seen: dict[int, date] = {}
    for f in sorted_records:
        a_id = f["fighter_a_id"]
        b_id = f["fighter_b_id"]
        fight_id = f["fight_id"]
        # Snapshot BEFORE updating last_seen.
        result[(a_id, fight_id)] = last_seen.get(a_id)
        result[(b_id, fight_id)] = last_seen.get(b_id)
        last_seen[a_id] = f["event_date"]
        last_seen[b_id] = f["event_date"]
    return result


def _build_division_history(
    sorted_records: list[dict],
) -> tuple[dict[str, list[dict]], float]:
    """weight_class → chronological list of {event_date, method}; +
    global finish_rate aggregate.

    Returns ``(div_hist, global_finish_rate)`` for direct use by
    ``division_finish_rate_shrunk``. Strict pre-fight discipline is
    enforced inside the helper (filter event_date < as_of_date).

    Pitfall #6: caller must pass ``sorted_records`` chronologically.
    """
    div_hist: dict[str, list[dict]] = {}
    total_finishes = 0
    total = 0
    for f in sorted_records:
        wc = f.get("weight_class")
        if wc is not None:
            div_hist.setdefault(wc, []).append(
                {
                    "event_date": f["event_date"],
                    "method": f.get("method"),
                }
            )
        total += 1
        if classify_outcome(f.get("method")) == "finish":
            total_finishes += 1
    global_finish_rate = total_finishes / total if total > 0 else 0.0
    return div_hist, global_finish_rate


def _build_division_mean_reaches(
    fighter_physicals: dict[int, dict],
    fight_records: list[dict],
) -> dict[str, float]:
    """weight_class → mean reach_inches across fighters who fought in it.

    Used as the denominator in ``reach_diff_normalized``. Falls back to
    a global mean for divisions with no fighters with known reach.
    NaN-skipping average across non-None reaches.
    """
    # Build (weight_class -> set of fighter_ids that fought in this class).
    division_fighters: dict[str, set[int]] = {}
    for f in fight_records:
        wc = f.get("weight_class")
        if wc is None:
            continue
        division_fighters.setdefault(wc, set()).add(f["fighter_a_id"])
        division_fighters[wc].add(f["fighter_b_id"])

    result: dict[str, float] = {}
    for wc, fids in division_fighters.items():
        reaches = [fighter_physicals.get(fid, {}).get("reach_inches") for fid in fids]
        valid = [r for r in reaches if r is not None]
        if valid:
            result[wc] = sum(valid) / len(valid)
        # else: weight_class absent from result → division_mean=0.0 in caller
        # → reach_diff_normalized → NaN (graceful degradation per Pattern D).
    return result


class FeatureMatrixAssembler:
    """Assembles per-fight differential feature vectors from three data sources.

    Per D-03: All features are Fighter A minus Fighter B differentials.
    Per D-12/ML-06: Elo ratings used as input features, not replaced.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()

    @staticmethod
    def _build_career_stats(
        fight_records: list[dict],
        elo_features: dict[tuple[int, int], dict[str, float]],
        computed_features: dict[tuple[int, int], dict[str, float | None]] | None = None,
    ) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int], list[dict]]]:
        """Build per-fighter career history from chronological fight list.

        Computes running stats up to (but not including) each fight.
        All stats use only pre-fight data to prevent leakage.

        Returns (fighter_id, fight_id) -> snapshot dict for O(1) lookup.
        """
        fighter_state: dict[int, dict] = {}
        snapshots: dict[tuple[int, int], dict] = {}

        # Track fight pairs for rematch detection
        fight_pairs: dict[tuple[int, int], list[dict]] = {}

        def _get_or_init(fid: int) -> dict:
            if fid not in fighter_state:
                fighter_state[fid] = {
                    "wins": 0,
                    "losses": 0,
                    "win_streak": 0,
                    "loss_streak": 0,
                    "ko_wins": 0,
                    "sub_wins": 0,
                    "ko_losses": 0,
                    "sub_losses": 0,
                    "last_fight_date": None,
                    "first_fight_date": None,
                    "total_cage_seconds": 0.0,
                    "division_fights": {},
                    "opponent_elos": [],
                    "recent_elos": [],
                    # Rolling window buffers
                    "recent_sig_str": [],
                    "recent_td_rate": [],
                    "recent_strike_def": [],
                    "recent_ctrl_time": [],
                }
            return fighter_state[fid]

        def _estimate_duration_seconds(fight: dict) -> float:
            """Estimate fight duration in seconds from round and time data."""
            rd = fight.get("num_rounds", 3)
            round_finished = fight.get("round_finished")
            time_str = fight.get("time_finished")
            if round_finished and time_str:
                try:
                    parts = time_str.split(":")
                    mins, secs = int(parts[0]), int(parts[1])
                    return (round_finished - 1) * 300 + mins * 60 + secs
                except (ValueError, IndexError):
                    pass
            # Default: assume full fight
            return rd * 300.0

        for fight in fight_records:
            fight_id = fight["fight_id"]
            a_id = fight["fighter_a_id"]
            b_id = fight["fighter_b_id"]
            winner_id = fight["winner_id"]
            event_date = fight["event_date"]
            method = fight.get("method") or ""
            weight_class = fight.get("weight_class", "")

            # Snapshot BEFORE this fight for both fighters
            for fid in (a_id, b_id):
                st = _get_or_init(fid)
                total = st["wins"] + st["losses"]
                days_since = (
                    (event_date - st["last_fight_date"]).days
                    if st["last_fight_date"]
                    else float("nan")
                )
                # Career span in years for fights-per-year
                career_years = (
                    (event_date - st["first_fight_date"]).days / 365.25
                    if st["first_fight_date"]
                    else 0.0
                )
                div_fights = st["division_fights"].get(weight_class, 0)

                # Elo momentum: change over last 3 fights
                elo_momentum = float("nan")
                if len(st["recent_elos"]) >= 2:
                    elo_momentum = st["recent_elos"][-1] - st["recent_elos"][0]

                # Average opponent Elo
                avg_opp_elo = (
                    sum(st["opponent_elos"]) / len(st["opponent_elos"])
                    if st["opponent_elos"]
                    else 1500.0
                )

                # Rolling window averages
                def _window_avg(buf: list[float], n: int) -> float:
                    if len(buf) < n:
                        return float("nan")
                    return sum(buf[-n:]) / n

                # Non-linear layoff
                import math

                log_days = math.log1p(days_since) if days_since == days_since else float("nan")
                is_short = 1.0 if (days_since == days_since and days_since < 60) else 0.0
                is_comeback = 1.0 if (days_since == days_since and days_since > 365) else 0.0

                snapshots[(fid, fight_id)] = {
                    "win_streak": st["win_streak"],
                    "loss_streak": st["loss_streak"],
                    "career_win_pct": st["wins"] / total if total > 0 else 0.5,
                    "fight_count": total,
                    "days_since_last_fight": days_since,
                    "ko_finish_rate": st["ko_wins"] / st["wins"] if st["wins"] > 0 else 0.0,
                    "sub_finish_rate": st["sub_wins"] / st["wins"] if st["wins"] > 0 else 0.0,
                    "ko_loss_rate": st["ko_losses"] / st["losses"] if st["losses"] > 0 else 0.0,
                    "sub_loss_rate": st["sub_losses"] / st["losses"] if st["losses"] > 0 else 0.0,
                    "total_cage_minutes": st["total_cage_seconds"] / 60.0,
                    "avg_fight_duration": (
                        st["total_cage_seconds"] / total if total > 0 else float("nan")
                    ),
                    "division_fight_count": div_fights,
                    "is_debut": 1.0 if total == 0 else 0.0,
                    "fights_per_year": total / career_years if career_years > 0.5 else float("nan"),
                    "avg_opponent_elo": avg_opp_elo,
                    "elo_momentum": elo_momentum,
                    # Non-linear layoff
                    "log_days_since_last_fight": log_days,
                    "is_short_turnaround": is_short,
                    "is_comeback": is_comeback,
                    # Rolling windows
                    "sig_str_per_min_last3": _window_avg(st["recent_sig_str"], 3),
                    "td_rate_last3": _window_avg(st["recent_td_rate"], 3),
                    "strike_defense_last3": _window_avg(st["recent_strike_def"], 3),
                    "ctrl_time_last3": _window_avg(st["recent_ctrl_time"], 3),
                    "sig_str_per_min_last5": _window_avg(st["recent_sig_str"], 5),
                    "td_rate_last5": _window_avg(st["recent_td_rate"], 5),
                    "strike_defense_last5": _window_avg(st["recent_strike_def"], 5),
                    "ctrl_time_last5": _window_avg(st["recent_ctrl_time"], 5),
                }

            # Update accumulators AFTER snapshotting
            is_ko = method in ("KO/TKO", "TKO")
            is_sub = method in ("Submission", "SUB")
            duration = _estimate_duration_seconds(fight)

            # Get opponent Elo for strength-of-schedule tracking
            default_elo = {"elo_overall": 1500.0}
            opp_elo_a = elo_features.get((b_id, fight_id), default_elo).get("elo_overall", 1500.0)
            opp_elo_b = elo_features.get((a_id, fight_id), default_elo).get("elo_overall", 1500.0)

            # Track rematch pairs (canonical order: smaller id first)
            pair_key = (min(a_id, b_id), max(a_id, b_id))
            if pair_key not in fight_pairs:
                fight_pairs[pair_key] = []
            fight_pairs[pair_key].append(
                {
                    "fight_id": fight_id,
                    "winner_id": winner_id,
                    "a_id": a_id,
                    "b_id": b_id,
                }
            )

            for fid, opp_elo in ((a_id, opp_elo_a), (b_id, opp_elo_b)):
                st = _get_or_init(fid)
                st["opponent_elos"].append(opp_elo)

                # Track current fighter's Elo for momentum (last 3)
                cur_elo = elo_features.get((fid, fight_id), default_elo).get("elo_overall", 1500.0)
                st["recent_elos"].append(cur_elo)
                if len(st["recent_elos"]) > 3:
                    st["recent_elos"] = st["recent_elos"][-3:]

                if fid == winner_id:
                    st["wins"] += 1
                    st["win_streak"] += 1
                    st["loss_streak"] = 0
                    if is_ko:
                        st["ko_wins"] += 1
                    elif is_sub:
                        st["sub_wins"] += 1
                else:
                    st["losses"] += 1
                    st["loss_streak"] += 1
                    st["win_streak"] = 0
                    if is_ko:
                        st["ko_losses"] += 1
                    elif is_sub:
                        st["sub_losses"] += 1

                st["total_cage_seconds"] += duration
                st["last_fight_date"] = event_date
                if st["first_fight_date"] is None:
                    st["first_fight_date"] = event_date
                st["division_fights"][weight_class] = st["division_fights"].get(weight_class, 0) + 1

                # Update rolling window buffers from computed features
                if computed_features:
                    cf = computed_features.get((fid, fight_id))
                    if cf:
                        v = cf.get("sig_str_per_minute")
                        if v is not None:
                            st["recent_sig_str"].append(v)
                        v = cf.get("td_rate")
                        if v is not None:
                            st["recent_td_rate"].append(v)
                        v = cf.get("strike_defense")
                        if v is not None:
                            st["recent_strike_def"].append(v)
                        v = cf.get("ctrl_time_per_fight")
                        if v is not None:
                            st["recent_ctrl_time"].append(v)
                    # Keep buffers bounded to last 8 fights
                    for buf_key in (
                        "recent_sig_str",
                        "recent_td_rate",
                        "recent_strike_def",
                        "recent_ctrl_time",
                    ):
                        if len(st[buf_key]) > 8:
                            st[buf_key] = st[buf_key][-8:]

        return snapshots, fight_pairs

    def assemble(
        self,
        fight_records: list[dict],
        elo_features: dict[tuple[int, int], dict[str, float]],
        computed_features: dict[tuple[int, int], dict[str, float | None]],
        fighter_physicals: dict[int, dict],
        division_medians: dict[str, dict[str, float]],
        round_stats: dict[tuple[int, int], list[dict]] | None = None,
        pre_ufc_records: dict[int, dict] | None = None,
        fight_odds: dict[tuple[int, int], dict[str, float | None]] | None = None,
        *,
        include_net: bool | None = None,
        feature_set: str = "v1.0",
        event_referees: dict[int, int | None] | None = None,
        ref_history: dict[int, list[dict]] | None = None,
        ref_global_rates: dict[str, float] | None = None,
        event_venues: dict[int, dict | None] | None = None,
        fighter_prior_venues: dict[tuple[int, int], dict | None] | None = None,
        fighter_birth_dates: dict[int, date | None] | None = None,
        fighter_reaches: dict[int, float | None] | None = None,
        elo_histories: dict[int, list[dict]] | None = None,
        fighter_prior_fight_dates: dict[tuple[int, int], date | None] | None = None,
        division_history: dict[str, list[dict]] | None = None,
        global_finish_rate: float | None = None,
        division_mean_reaches: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix from pre-loaded data sources.

        Per Phase 23 D-09: ``feature_set`` is the new public knob.
            - ``"v1.0"``        → 75 cols (includes 3 NET-* tail).
            - ``"v2.1-no-net"`` → 72 cols (xgb_v2 baseline, no NET).
            - ``"v2.2"``        → 90 cols (72 + 3 REF + 6 TRAVEL + 9 META;
                                 TRAVEL+META NaN-padded until Plans 23-02/03).

        ``include_net`` is the Phase 18 back-compat shim — translated to
        ``feature_set`` at the top of the method:
            - ``include_net=True``  → ``feature_set="v1.0"``
            - ``include_net=False`` → ``feature_set="v2.1-no-net"``
            - ``include_net=None``  → respect ``feature_set`` as-is.

        Returns (X, y, fight_dates):
            X: (n_fights, n_cols) float64 array of features
            y: (n_fights,) int array of outcomes (1 = A wins, 0 = B wins)
            fight_dates: (n_fights,) array of event dates for temporal splitting
        """
        # Phase 18 back-compat shim.
        if include_net is not None:
            feature_set = "v1.0" if include_net else "v2.1-no-net"
        # Internal flag used by the row-build branches below.
        _include_net = feature_set == "v1.0"

        rows: list[list[float]] = []
        targets: list[int] = []
        dates: list[date] = []

        default_elo = {"elo_overall": 1500.0, "elo_striking": 1500.0, "elo_grappling": 1500.0}

        # Phase 23 REF pre-pass — fires for feature_set='v2.2' AND the
        # Phase 42 v2.5-travel sibling (v2.5 = v2.2 + 2 cols, needs the
        # same venue + ref + meta state).
        # Pitfall #6: defensive chronological sort. Callers may pass unsorted
        # records; we sort here so the cumulative-count pre-pass uses real
        # as-of-date ordering for the global rate aggregate AND so per-referee
        # history dicts are chronological.
        if feature_set in ("v2.2", "v2.5-travel"):
            sorted_records = sorted(
                fight_records,
                key=lambda f: f["event_date"],
            )
            if ref_global_rates is None:
                ref_global_rates = _compute_ref_global_rates(sorted_records)
            if event_referees is None or ref_history is None:
                _event_refs_built, _ref_history_built = _build_referee_lookup(
                    sorted_records,
                )
                if event_referees is None:
                    event_referees = _event_refs_built
                if ref_history is None:
                    ref_history = _ref_history_built
            # Plan 23-02 TRAVEL pre-pass — build venue lookups from the same
            # chronological sort (Pitfall #6 defense already applied above).
            if event_venues is None:
                event_venues = _build_event_venue_lookup(sorted_records)
            if fighter_prior_venues is None:
                fighter_prior_venues = _build_fighter_prior_venues(
                    sorted_records,
                    event_venues,
                )
            # Plan 23-03 META pre-pass — derived from existing tables only
            # (no new scrapers / migrations per D-06).
            if fighter_birth_dates is None:
                fighter_birth_dates = _build_fighter_birth_dates(
                    fighter_physicals,
                )
            if fighter_reaches is None:
                fighter_reaches = _build_fighter_reaches(fighter_physicals)
            if elo_histories is None:
                elo_histories = _build_elo_histories(
                    sorted_records,
                    elo_features,
                )
            if fighter_prior_fight_dates is None:
                fighter_prior_fight_dates = _build_fighter_prior_fight_dates(
                    sorted_records,
                )
            if division_history is None or global_finish_rate is None:
                _div_hist_built, _global_rate_built = _build_division_history(
                    sorted_records,
                )
                if division_history is None:
                    division_history = _div_hist_built
                if global_finish_rate is None:
                    global_finish_rate = _global_rate_built
            if division_mean_reaches is None:
                division_mean_reaches = _build_division_mean_reaches(
                    fighter_physicals,
                    fight_records,
                )

        # Pre-compute career stats and rematch pairs from chronological fight list
        career_snapshots, fight_pairs = self._build_career_stats(
            fight_records,
            elo_features,
            computed_features,
        )

        # Build pace decay stats per fighter from round stats
        pace_stats = self._build_pace_stats(fight_records, round_stats or {})

        # Build rematch index: (fight_id) -> {is_rematch, first_fight_winner}
        rematch_index = self._build_rematch_index(fight_pairs)

        for fight in fight_records:
            fight_id = fight["fight_id"]
            event_date = fight["event_date"]
            weight_class = fight["weight_class"]

            # Deterministic random swap to remove positional bias.
            # Fighter A wins ~80% of fights in the DB (systematic assignment),
            # so without this swap the model learns "predict A" instead of
            # learning from actual features. The hash of fight_id produces a
            # deterministic coin flip — same swap every run for reproducibility.
            swap = int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 0
            if swap:
                a_id = fight["fighter_b_id"]
                b_id = fight["fighter_a_id"]
            else:
                a_id = fight["fighter_a_id"]
                b_id = fight["fighter_b_id"]

            row: list[float] = []

            # ── 1. Elo differentials (3 features) ────────────────────────
            elo_a = elo_features.get((a_id, fight_id), default_elo)
            elo_b = elo_features.get((b_id, fight_id), default_elo)

            row.append(elo_a.get("elo_overall", 1500.0) - elo_b.get("elo_overall", 1500.0))
            row.append(elo_a.get("elo_striking", 1500.0) - elo_b.get("elo_striking", 1500.0))
            row.append(elo_a.get("elo_grappling", 1500.0) - elo_b.get("elo_grappling", 1500.0))

            # ── 2. Performance feature differentials (20 features) ───────
            feats_a = computed_features.get((a_id, fight_id))
            feats_b = computed_features.get((b_id, fight_id))

            for feat_key in PERFORMANCE_FEATURE_KEYS:
                val_a = feats_a.get(feat_key) if feats_a else None
                val_b = feats_b.get(feat_key) if feats_b else None
                if val_a is not None and val_b is not None:
                    row.append(val_a - val_b)
                else:
                    row.append(float("nan"))

            # ── 3. Physical differentials (4 features) ───────────────────
            phys_a = fighter_physicals.get(a_id, {})
            phys_b = fighter_physicals.get(b_id, {})

            div_med = division_medians.get(weight_class, {})

            # Apply division median imputation (D-01)
            for attr in ("height_inches", "reach_inches", "leg_reach_inches"):
                val_a = phys_a.get(attr)
                val_b = phys_b.get(attr)
                median_val = div_med.get(attr)

                if val_a is None:
                    val_a = median_val
                if val_b is None:
                    val_b = median_val

                if val_a is not None and val_b is not None:
                    row.append(val_a - val_b)
                else:
                    row.append(float("nan"))

            # Age differential using fight date (not today)
            dob_a = phys_a.get("date_of_birth")
            dob_b = phys_b.get("date_of_birth")
            age_a = (event_date - dob_a).days / 365.25 if dob_a else None
            age_b = (event_date - dob_b).days / 365.25 if dob_b else None

            if age_a is not None and age_b is not None:
                row.append(age_a - age_b)
            else:
                row.append(float("nan"))

            # ── 4. Stance matchup (1 binary feature) ────────────────────
            stance_a = phys_a.get("stance")
            stance_b = phys_b.get("stance")
            row.append(encode_stance_matchup(stance_a, stance_b))

            # ── 5. Career stat differentials (17 features) ─────────────
            career_a = career_snapshots.get((a_id, fight_id), {})
            career_b = career_snapshots.get((b_id, fight_id), {})

            for key in (
                "win_streak",
                "loss_streak",
                "career_win_pct",
                "fight_count",
                "days_since_last_fight",
                "ko_finish_rate",
                "sub_finish_rate",
                "ko_loss_rate",
                "sub_loss_rate",
                "total_cage_minutes",
                "avg_fight_duration",
                "division_fight_count",
                "is_debut",
                "fights_per_year",
                "avg_opponent_elo",
                "elo_momentum",
            ):
                va = career_a.get(key, float("nan"))
                vb = career_b.get(key, float("nan"))
                if va != va or vb != vb:  # NaN check
                    row.append(float("nan"))
                else:
                    row.append(va - vb)

            # ── 6. Cross-domain matchup features (2 features) ──────────
            # Striker Elo vs opponent's grappling Elo (and vice versa)
            elo_a_str = elo_a.get("elo_striking", 1500.0)
            elo_b_gra = elo_b.get("elo_grappling", 1500.0)
            elo_a_gra = elo_a.get("elo_grappling", 1500.0)
            elo_b_str = elo_b.get("elo_striking", 1500.0)
            row.append(elo_a_str - elo_b_gra)  # A's striking vs B's grappling
            row.append(elo_a_gra - elo_b_str)  # A's grappling vs B's striking

            # ── 7. Fight context flags (3 features, not differentials) ──
            row.append(1.0 if fight.get("is_title_fight") else 0.0)
            row.append(float(fight.get("num_rounds", 3)))
            wc_order = {
                "Strawweight": 1,
                "Flyweight": 2,
                "Bantamweight": 3,
                "Featherweight": 4,
                "Lightweight": 5,
                "Welterweight": 6,
                "Middleweight": 7,
                "Light Heavyweight": 8,
                "Heavyweight": 9,
                "Women's Strawweight": 1,
                "Women's Flyweight": 2,
                "Women's Bantamweight": 3,
                "Women's Featherweight": 4,
            }
            row.append(float(wc_order.get(weight_class, 5)))

            # ── 8. Pace decay differentials (4 features) ───────────────
            pace_a = pace_stats.get((a_id, fight_id), {})
            pace_b = pace_stats.get((b_id, fight_id), {})
            for key in (
                "pace_decay_strikes",
                "pace_decay_td",
                "pace_output_variance",
                "avg_r1_sig_str",
            ):
                va = pace_a.get(key, float("nan"))
                vb = pace_b.get(key, float("nan"))
                if va != va or vb != vb:
                    row.append(float("nan"))
                else:
                    row.append(va - vb)

            # ── 9. Non-linear layoff differentials (3 features) ────────
            for key in ("log_days_since_last_fight", "is_short_turnaround", "is_comeback"):
                va = career_a.get(key, float("nan"))
                vb = career_b.get(key, float("nan"))
                if va != va or vb != vb:
                    row.append(float("nan"))
                else:
                    row.append(va - vb)

            # ── 10. Rolling window differentials (8 features) ──────────
            for key in (
                "sig_str_per_min_last3",
                "td_rate_last3",
                "strike_defense_last3",
                "ctrl_time_last3",
                "sig_str_per_min_last5",
                "td_rate_last5",
                "strike_defense_last5",
                "ctrl_time_last5",
            ):
                va = career_a.get(key, float("nan"))
                vb = career_b.get(key, float("nan"))
                if va != va or vb != vb:
                    row.append(float("nan"))
                else:
                    row.append(va - vb)

            # ── 11. Rematch features (2 features) ─────────────────────
            rm = rematch_index.get(fight_id, {})
            row.append(float(rm.get("is_rematch", 0)))
            # first_fight_winner_diff: +1 if A won first fight, -1 if B won, 0 if no rematch
            first_winner = rm.get("first_fight_winner")
            if first_winner is None:
                row.append(0.0)
            elif first_winner == a_id:
                row.append(1.0)
            elif first_winner == b_id:
                row.append(-1.0)
            else:
                row.append(0.0)

            # ── 12. Pre-UFC record differential (1 feature) ──────────
            pre_a = (pre_ufc_records or {}).get(a_id, {})
            pre_b = (pre_ufc_records or {}).get(b_id, {})
            for key in ("win_pct",):
                va = pre_a.get(key, float("nan")) if pre_a else float("nan")
                vb = pre_b.get(key, float("nan")) if pre_b else float("nan")
                if va != va or vb != vb:  # NaN check
                    row.append(float("nan"))
                else:
                    row.append(va - vb)

            # ── 13. Betting odds differentials (3 features) ──────────
            # Per D-02: implied probs already vig-removed at ingest time.
            # Per D-09: all three diffs are Fighter A minus Fighter B
            # (post the deterministic A/B swap above on line ~371).
            # Per D-07 + Pitfall 3: use float("nan") — NEVER 0.0 — for
            # missing odds; XGBoost learns a default branch direction
            # via sparsity-aware split finding. 0.0 is a legal probability
            # value (mathematical pickem) and must remain distinguishable
            # from missing-data NaN.
            odds_a = (fight_odds or {}).get((a_id, fight_id))
            odds_b = (fight_odds or {}).get((b_id, fight_id))

            if odds_a is not None and odds_b is not None:
                # opening_prob_diff
                op_a = odds_a.get("opening_implied_prob")
                op_b = odds_b.get("opening_implied_prob")
                if op_a is not None and op_b is not None:
                    row.append(op_a - op_b)
                else:
                    row.append(float("nan"))

                # closing_prob_diff
                cl_a = odds_a.get("closing_implied_prob")
                cl_b = odds_b.get("closing_implied_prob")
                if cl_a is not None and cl_b is not None:
                    row.append(cl_a - cl_b)
                else:
                    row.append(float("nan"))

                # line_movement_diff = (closing - opening) for A minus the
                # same for B per D-05. Requires ALL FOUR values to be
                # present; otherwise NaN.
                if op_a is not None and op_b is not None and cl_a is not None and cl_b is not None:
                    line_move_diff = (cl_a - op_a) - (cl_b - op_b)
                    row.append(line_move_diff)
                    # sharp_money_signal = |line_movement_diff|. Captures
                    # the MAGNITUDE of the asymmetric line drift between
                    # open and close — a proxy for sharp / professional
                    # money intervention regardless of direction. Direction
                    # is already in line_movement_diff; this gives the
                    # model the non-linear magnitude signal independently.
                    row.append(abs(line_move_diff))
                else:
                    row.append(float("nan"))
                    row.append(float("nan"))

                # odds_elo_divergence = closing market probability for A
                # minus the Elo-implied probability for A. Positive means
                # the market favors A more than our Elo system does;
                # negative means the market is bearish on A vs Elo. Tells
                # the model where market and our Elo disagree — D-05.
                if cl_a is not None:
                    elo_diff = elo_a.get("elo_overall", 1500.0) - elo_b.get("elo_overall", 1500.0)
                    elo_prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
                    row.append(cl_a - elo_prob_a)
                else:
                    row.append(float("nan"))
            else:
                # No odds row at all on at least one side → all 5 NaN.
                row.extend([float("nan")] * 5)

            # ── 14. Opponent-network differentials (3 features) ──────
            # Phase 16-03 NET-01/02 — operator-approved config: pan-mma
            # + MOV-weighted edges per gsd-checkpoint resolution.
            # Per Pitfall #12: feature_matrix.py and inference_features.py
            # use the SAME helper (compute_network_diff_features) so the
            # train-time and predict-time math cannot diverge.
            # Phase 18 NET-V2-01: when feature_set != "v1.0" (e.g. v2.1-no-net
            # or v2.2), this entire block is SKIPPED so the row ends 72 cols
            # before the v2.2 REF append below (ablation variant + v2.2 layout).
            if _include_net:
                from ufc_prediction.features.network import (
                    compute_network_diff_features,
                )

                net_a = {
                    "pagerank": (feats_a or {}).get("pagerank", float("nan")),
                    "sos_2hop": (feats_a or {}).get("sos_2hop", float("nan")),
                    "is_debutant_in_graph": (feats_a or {}).get(
                        "is_debutant_in_graph",
                        float("nan"),
                    ),
                }
                net_b = {
                    "pagerank": (feats_b or {}).get("pagerank", float("nan")),
                    "sos_2hop": (feats_b or {}).get("sos_2hop", float("nan")),
                    "is_debutant_in_graph": (feats_b or {}).get(
                        "is_debutant_in_graph",
                        float("nan"),
                    ),
                }
                net_diffs = compute_network_diff_features(net_a, net_b)
                row.append(net_diffs["pagerank_diff"])
                row.append(net_diffs["sos_2hop_diff"])
                row.append(net_diffs["is_debutant_in_graph_diff"])

            # ── 15. Phase 23 v2.2 REF + TRAVEL + META cols ──────────────
            # Per CONTEXT D-10: feature_matrix.py and inference_features.py
            # both call compute_ref_rates_shrunk from features_v22.ref so the
            # train/predict math is byte-identical (Pitfall #12). REF is the
            # only v2.2 family wired here — TRAVEL (Plan 23-02) and META
            # (Plan 23-03) land later; for now we NaN-pad them so the row
            # length stays == len(FEATURE_COLUMNS_V22).
            #
            # Phase 42 Plan 42-01 (D-21 + D-22): v2.5-travel fires the SAME
            # v2.2 body (sibling-not-replacement — v2.5 reuses v2.2 substrate
            # byte-stable) then appends 2 new TRAVEL-V25 cols after. The
            # v2.2 length assert is replaced with a feature_set-conditional
            # assert at the END of this block.
            if feature_set in ("v2.2", "v2.5-travel"):
                # Use _event_referees + _ref_history captured in the pre-pass
                # above. Both are guaranteed non-None when feature_set='v2.2'
                # (built unconditionally if caller didn't supply).
                fight_event_id = fight.get("event_id")
                ref_id = (
                    event_referees.get(fight_event_id)
                    if event_referees and fight_event_id is not None
                    else None
                )
                ref_rates = compute_ref_rates_shrunk(
                    ref_id,
                    fight["event_date"],
                    ref_history or {},
                    ref_global_rates or {"finish": 0.0, "decision": 0.0, "no_action": 0.0},
                )
                row.append(ref_rates["ref_finish_rate_shrunk"])
                row.append(ref_rates["ref_decision_rate_shrunk"])
                row.append(ref_rates["ref_no_action_rate_shrunk"])

                # ── Plan 23-02 TRAVEL cols (indices 75-80) ──────────────
                # Per CONTEXT D-10: shared helper with inference_features.py
                # (compute_travel_features in features_v22/travel.py).
                # Sentinel discipline (D-04 + Pitfall #3): first-fight
                # fighter → 0; venue-NULL on current event → NaN. Distinct
                # cases; do NOT conflate.
                fight_event_id_t = fight.get("event_id")
                curr_venue = (
                    event_venues.get(fight_event_id_t)
                    if event_venues and fight_event_id_t is not None
                    else None
                )
                if curr_venue is None:
                    # Current event has no usable venue — NaN all 6 TRAVEL
                    # cols (graceful degradation per Pattern D, distinct
                    # from the debut-fighter sentinel of 0).
                    row.extend([float("nan")] * 6)
                else:
                    prior_a = (
                        fighter_prior_venues.get((a_id, fight_id)) if fighter_prior_venues else None
                    )
                    prior_b = (
                        fighter_prior_venues.get((b_id, fight_id)) if fighter_prior_venues else None
                    )
                    travel_red = compute_travel_features(
                        prior_a,
                        curr_venue,
                        fight["event_date"],
                    )
                    travel_blue = compute_travel_features(
                        prior_b,
                        curr_venue,
                        fight["event_date"],
                    )
                    row.append(travel_red["travel_distance_miles"])
                    row.append(travel_blue["travel_distance_miles"])
                    row.append(
                        travel_red["travel_distance_miles"] - travel_blue["travel_distance_miles"]
                    )
                    row.append(travel_red["tz_shift_signed"])
                    row.append(travel_blue["tz_shift_signed"])
                    row.append(travel_red["tz_shift_signed"] - travel_blue["tz_shift_signed"])

                # ── Plan 23-03 META cols (indices 81-89; 9 cols) ────────
                # Per CONTEXT D-10: shared helpers with inference_features.py
                # (layoff_days, age_at_fight, elo_velocity,
                # division_finish_rate_shrunk, reach_diff_normalized in
                # features_v22/meta.py). NB: layoff_days_diff DROPPED per
                # Q4 RESOLVED + D-07 — days_since_last_fight_diff at
                # FEATURE_COLUMNS_NO_NET[61] is the canonical differential.

                # Layoff (indices 81-82) — per-fighter clip at 720 (Pitfall #8).
                prior_a = (
                    fighter_prior_fight_dates.get((a_id, fight_id))
                    if fighter_prior_fight_dates
                    else None
                )
                prior_b = (
                    fighter_prior_fight_dates.get((b_id, fight_id))
                    if fighter_prior_fight_dates
                    else None
                )
                row.append(layoff_days(event_date, prior_a))
                row.append(layoff_days(event_date, prior_b))

                # Age (indices 83-84) — uses event_date, NOT today (Pitfall #5).
                dob_a = fighter_birth_dates.get(a_id) if fighter_birth_dates else None
                dob_b = fighter_birth_dates.get(b_id) if fighter_birth_dates else None
                row.append(age_at_fight(dob_a, event_date))
                row.append(age_at_fight(dob_b, event_date))

                # Elo velocity (indices 85-87) — diff of red velocity minus
                # blue velocity. Strict pre-fight slice: history entries with
                # event_date < this fight's event_date.
                eh_a = elo_histories.get(a_id, []) if elo_histories else []
                eh_b = elo_histories.get(b_id, []) if elo_histories else []
                prior_elo_a = [s for s in eh_a if s["event_date"] < event_date]
                prior_elo_b = [s for s in eh_b if s["event_date"] < event_date]

                def _vel_diff(key: str) -> float:
                    va = elo_velocity(
                        [s[key] for s in prior_elo_a],
                        window=5,
                    )
                    vb = elo_velocity(
                        [s[key] for s in prior_elo_b],
                        window=5,
                    )
                    # NaN propagates if either side has insufficient history.
                    if va != va or vb != vb:
                        return float("nan")
                    return va - vb

                row.append(_vel_diff("elo_overall"))
                row.append(_vel_diff("elo_striking"))
                row.append(_vel_diff("elo_grappling"))

                # Division finish rate (index 88) — 1 col per division
                # (NOT per-fighter-differential per D-06).
                row.append(
                    division_finish_rate_shrunk(
                        weight_class,
                        event_date,
                        division_history or {},
                        global_finish_rate or 0.0,
                        k_shrink=50.0,
                    )
                )

                # Reach normalized (index 89).
                reach_a = fighter_reaches.get(a_id) if fighter_reaches else None
                reach_b = fighter_reaches.get(b_id) if fighter_reaches else None
                mean_reach = (division_mean_reaches or {}).get(weight_class, 0.0)
                row.append(reach_diff_normalized(reach_a, reach_b, mean_reach))

                # Phase 23 D-10 dynamic guard: at this point we MUST have the
                # 90-col v2.2 body assembled (REF + TRAVEL + META). The v2.5
                # append (2 extra cols) lands BELOW for v2.5-travel only.
                assert len(row) == _EXPECTED_V22_NCOLS, (
                    f"v2.2 row length drift: expected {_EXPECTED_V22_NCOLS}, "
                    f"got {len(row)}. Anti-Pattern #8 violation."
                )

                # ── Phase 42 Plan 42-01 v2.5 TRAVEL close-out append ───
                # Per CONTEXT D-21 + D-22: appended AFTER the v2.2 body so
                # the v2.2 emission stays byte-identical (additive-only
                # whitelist invariant — see
                # .planning/phases/42-travel-feature-engineering-closeout/42-FEATURE-MATRIX-WHITELIST.md).
                #
                # Reuses the SAME event_venues + fighter_prior_venues dicts
                # the v2.2 path already built (Pitfall #6 strict pre-fight
                # as-of-date already enforced by the pre-pass).
                #
                # Semantics: ONE column per primitive PER FIGHT, emitted as
                # red - blue differential (parallel to META-V22 input shape).
                # Debut sentinel: NaN (NOT 0.0 — that's the v2.5 difference
                # from v2.2 compute_travel_features). NaN propagates via
                # Python float semantics: NaN - NaN = NaN; NaN - finite = NaN.
                if feature_set == "v2.5-travel":
                    fight_event_id_v25 = fight.get("event_id")
                    curr_venue_v25 = (
                        event_venues.get(fight_event_id_v25)
                        if event_venues and fight_event_id_v25 is not None
                        else None
                    )
                    if curr_venue_v25 is None:
                        # No usable current venue — NaN both v2.5 cols
                        # (consistent with debut; meta blender NaN-drops
                        # the row per Phase 29 EVAL-V23-01 pattern).
                        row.append(math.nan)
                        row.append(math.nan)
                    else:
                        prior_a_v25 = (
                            fighter_prior_venues.get((a_id, fight_id))
                            if fighter_prior_venues
                            else None
                        )
                        prior_b_v25 = (
                            fighter_prior_venues.get((b_id, fight_id))
                            if fighter_prior_venues
                            else None
                        )
                        travel_red_v25 = compute_travel_v25_features(
                            prior_a_v25,
                            curr_venue_v25,
                            fight["event_date"],
                        )
                        travel_blue_v25 = compute_travel_v25_features(
                            prior_b_v25,
                            curr_venue_v25,
                            fight["event_date"],
                        )
                        # Differential: red - blue. NaN semantics flow
                        # correctly via Python float arithmetic (no special
                        # handling needed — the meta blender NaN-drops).
                        d_km = (
                            travel_red_v25["travel_distance_km"]
                            - travel_blue_v25["travel_distance_km"]
                        )
                        d_hrs = travel_red_v25["tz_shift_hours"] - travel_blue_v25["tz_shift_hours"]
                        row.append(d_km)
                        row.append(d_hrs)

                    # Phase 42 D-21 dynamic guard: v2.5-travel row length
                    # must match FEATURE_COLUMNS_V25_TRAVEL exactly (92 cols).
                    assert len(row) == _EXPECTED_V25_TRAVEL_NCOLS, (
                        f"v2.5-travel row length drift: expected "
                        f"{_EXPECTED_V25_TRAVEL_NCOLS}, got {len(row)}. "
                        f"Anti-Pattern #8 violation."
                    )

            rows.append(row)

            # ── Target ──────────────────────────────────────────────────
            targets.append(1 if fight["winner_id"] == a_id else 0)
            dates.append(event_date)

        X = np.array(rows, dtype=np.float64)
        y = np.array(targets, dtype=np.int32)
        fight_dates = np.array(dates, dtype=object)

        return X, y, fight_dates

    @staticmethod
    def _build_pace_stats(
        fight_records: list[dict],
        round_stats: dict[tuple[int, int], list[dict]],
    ) -> dict[tuple[int, int], dict]:
        """Build per-fighter pace decay stats from round-by-round data.

        Computes running averages of pace decay (later rounds vs early rounds),
        output variance, and R1 striking rate — all using only pre-fight data.

        Returns (fighter_id, fight_id) -> pace snapshot dict.
        """
        fighter_pace: dict[int, dict] = {}
        snapshots: dict[tuple[int, int], dict] = {}

        for fight in fight_records:
            fight_id = fight["fight_id"]
            a_id = fight["fighter_a_id"]
            b_id = fight["fighter_b_id"]

            # Snapshot BEFORE this fight
            for fid in (a_id, b_id):
                if fid not in fighter_pace:
                    fighter_pace[fid] = {
                        "decay_strikes": [],
                        "decay_td": [],
                        "output_vars": [],
                        "r1_sig_strs": [],
                    }
                st = fighter_pace[fid]
                n = len(st["decay_strikes"])
                snapshots[(fid, fight_id)] = {
                    "pace_decay_strikes": (sum(st["decay_strikes"]) / n if n > 0 else float("nan")),
                    "pace_decay_td": (sum(st["decay_td"]) / n if n > 0 else float("nan")),
                    "pace_output_variance": (sum(st["output_vars"]) / n if n > 0 else float("nan")),
                    "avg_r1_sig_str": (sum(st["r1_sig_strs"]) / n if n > 0 else float("nan")),
                }

            # Update accumulators AFTER snapshotting
            for fid in (a_id, b_id):
                rounds = round_stats.get((fid, fight_id), [])
                if len(rounds) < 2:
                    continue

                st = fighter_pace[fid]
                r1 = rounds[0]
                r_last = rounds[-1]

                r1_str = r1.get("sig_str_landed", 0)
                rl_str = r_last.get("sig_str_landed", 0)
                r1_td = r1.get("td_landed", 0)
                rl_td = r_last.get("td_landed", 0)

                # Pace decay: later output / earlier output (< 1.0 means fading)
                if r1_str > 0:
                    st["decay_strikes"].append(rl_str / r1_str)
                if r1_td > 0:
                    st["decay_td"].append(rl_td / r1_td)

                # Output variance across rounds
                all_str = [r.get("sig_str_landed", 0) for r in rounds]
                if len(all_str) >= 2:
                    mean_str = sum(all_str) / len(all_str)
                    variance = sum((x - mean_str) ** 2 for x in all_str) / len(all_str)
                    st["output_vars"].append(variance)

                # R1 sig strikes
                st["r1_sig_strs"].append(float(r1_str))

                # Keep bounded
                for buf in (
                    st["decay_strikes"],
                    st["decay_td"],
                    st["output_vars"],
                    st["r1_sig_strs"],
                ):
                    if len(buf) > 10:
                        buf[:] = buf[-10:]

        return snapshots

    @staticmethod
    def _build_rematch_index(
        fight_pairs: dict[tuple[int, int], list[dict]],
    ) -> dict[int, dict]:
        """Build rematch lookup from fight pair history.

        Returns fight_id -> {is_rematch: bool, first_fight_winner: int|None}.
        Only the second (and later) fights in a pair are marked as rematches.
        """
        rematch_index: dict[int, dict] = {}

        for _pair_key, fights in fight_pairs.items():
            if len(fights) < 2:
                continue
            # First fight is not a rematch; subsequent fights are
            first_fight = fights[0]
            for later_fight in fights[1:]:
                rematch_index[later_fight["fight_id"]] = {
                    "is_rematch": 1,
                    "first_fight_winner": first_fight["winner_id"],
                }

        return rematch_index
