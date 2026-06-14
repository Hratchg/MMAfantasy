"""FeatureComputer orchestrator -- chronological accumulator pipeline.

Processes all fights in date order, maintaining per-fighter state to
produce feature vectors.  Follows the same accumulator pattern as
EloEngine.compute_all (Phase 3).

Multi-pass design:
  Pass 1: Compute raw features for every fighter at every fight.
  Pass 2: Compute league-wide means, apply Bayesian shrinkage (D-12/D-13).
  Pass 3: Compute PCA embeddings on latest feature vectors (D-09/D-10).
  Pass 4: Compute opponent-network features — PageRank + 2-hop SoS +
          is_debutant_in_graph (Phase 16-03 NET-01/02; operator-approved
          pan-mma + MOV-weighted config per gsd-checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ufc_prediction.features.config import (
    CANONICAL_FEATURE_ORDER,
    FeatureConfig,
)
from ufc_prediction.features.embeddings import compute_embeddings
from ufc_prediction.features.ewma import compute_ewma
from ufc_prediction.features.network import apply_network_features
from ufc_prediction.features.opponent_adjusted import compute_opponent_adjusted
from ufc_prediction.features.rates import compute_rate_features, estimate_fight_duration_seconds
from ufc_prediction.features.regularize import apply_shrinkage_to_features
from ufc_prediction.features.style import classify_style


def _sum_stat(stats: list[dict[str, Any]], key: str) -> int:
    """Sum a stat across rounds, treating None as 0."""
    return sum(r.get(key) or 0 for r in stats)


@dataclass
class FighterAccumulator:
    """Tracks running stats for a fighter across their career."""

    fight_count: int = 0

    # Career totals for rate computation (from raw round stats)
    total_sig_str_landed: int = 0
    total_total_str_landed: int = 0
    total_td_landed: int = 0
    total_td_attempted: int = 0
    total_opp_sig_str_landed: int = 0
    total_opp_sig_str_attempted: int = 0
    total_opp_td_landed: int = 0
    total_opp_td_attempted: int = 0
    total_ctrl_time: int = 0
    total_sub_attempts: int = 0
    total_fight_time_seconds: int = 0

    # Per-fight history for EWMA (oldest-first)
    career_sig_str_per_min: list[float] = field(default_factory=list)
    career_total_str_per_min: list[float] = field(default_factory=list)
    career_td_rate: list[float] = field(default_factory=list)
    career_td_accuracy: list[float | None] = field(default_factory=list)
    career_td_defense: list[float | None] = field(default_factory=list)
    career_strike_defense: list[float | None] = field(default_factory=list)
    career_ctrl_time: list[float] = field(default_factory=list)
    career_sub_att: list[float] = field(default_factory=list)


class FeatureComputer:
    """Orchestrates feature computation across all fights.

    Constructor:
        config: FeatureConfig -- tunable parameters.

    Main method:
        compute_all(fights, round_stats_by_fight, domain_elo_by_fighter_fight)
        -> list[dict] ready for bulk_insert_features.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def compute_all(
        self,
        fights: list[dict[str, Any]],
        round_stats_by_fight: dict[int, list[dict[str, Any]]],
        domain_elo_by_fighter_fight: dict[tuple[int, int], dict[str, float | None]],
    ) -> list[dict[str, Any]]:
        """Process all fights chronologically and produce feature vectors.

        Args:
            fights: Chronologically ordered fight dicts from load_fights_with_duration.
            round_stats_by_fight: fight_id -> list of per-round stat dicts.
            domain_elo_by_fighter_fight: (fighter_id, fight_id) -> {striking_elo, grappling_elo}.

        Returns:
            List of dicts with: fighter_id, fight_id, as_of_date, feature_set_version, features.
        """
        accumulators: dict[int, FighterAccumulator] = {}
        raw_results: list[dict[str, Any]] = []
        # Track fight_count at time of feature creation for shrinkage pass
        fight_count_at_feature: list[int] = []

        # ── Pass 1: Compute raw features ──────────────────────────────
        for fight in fights:
            fight_id = fight["fight_id"]
            event_date = fight["event_date"]
            fighter_a_id = fight["fighter_a_id"]
            fighter_b_id = fight["fighter_b_id"]

            fight_duration = estimate_fight_duration_seconds(
                fight.get("round_finished"),
                fight.get("time_finished"),
                fight.get("num_rounds", 3),
            )

            # Get round stats for this fight, split by fighter
            all_round_stats = round_stats_by_fight.get(fight_id, [])
            a_round_stats = [rs for rs in all_round_stats if rs["fighter_id"] == fighter_a_id]
            b_round_stats = [rs for rs in all_round_stats if rs["fighter_id"] == fighter_b_id]

            # Process each fighter in the fight
            for fighter_id, opponent_id, fighter_rs, opponent_rs in [
                (fighter_a_id, fighter_b_id, a_round_stats, b_round_stats),
                (fighter_b_id, fighter_a_id, b_round_stats, a_round_stats),
            ]:
                acc = accumulators.setdefault(fighter_id, FighterAccumulator())
                opp_acc = accumulators.setdefault(opponent_id, FighterAccumulator())

                if acc.fight_count >= 1:
                    # Build feature vector from pre-fight accumulator state
                    features = self._build_features(
                        acc, opp_acc, fighter_id, fight_id,
                        domain_elo_by_fighter_fight,
                    )
                    raw_results.append({
                        "fighter_id": fighter_id,
                        "fight_id": fight_id,
                        "as_of_date": event_date,
                        "feature_set_version": self.config.feature_set_version,
                        "features": features,
                    })
                    fight_count_at_feature.append(acc.fight_count)

                # Update accumulator with this fight's raw data
                if fighter_rs:
                    rates = compute_rate_features(fighter_rs, opponent_rs, fight_duration)
                    self._update_accumulator(
                        acc, fighter_rs, opponent_rs, rates, fight_duration,
                    )

                acc.fight_count += 1

        if not raw_results:
            return []

        # ── Pass 2: League means + shrinkage ──────────────────────────
        league_means = self._compute_league_means(raw_results)

        for row, fcount in zip(raw_results, fight_count_at_feature, strict=True):
            row["features"] = apply_shrinkage_to_features(
                row["features"],
                league_means,
                fight_count=fcount,
                min_fights=self.config.shrinkage_min_fights,
            )

        # ── Pass 3: PCA embeddings on latest feature per fighter ──────
        self._apply_embeddings(raw_results)

        # ── Pass 4: Opponent-network features (NET-01/02, Phase 16) ──
        # Operator-approved config: pan-mma + MOV-weighted per
        # 16-03-NET-00-SPIKE-RESULTS.md gsd-checkpoint resolution.
        # The graph is built from `fights`; subgraph is computed per
        # (fighter_id, as_of_date) snapshot inside compute_pagerank_at /
        # compute_2hop_sos_at — that is the Pitfall #1 / Gotcha 4
        # countermeasure (NET-03 enforces it as a CI regression).
        net_fights = [
            {
                "fight_id": f["fight_id"],
                "event_date": f["event_date"],
                "winner_id": f.get("winner_id"),
                "loser_id": (
                    f["fighter_b_id"] if f.get("winner_id") == f["fighter_a_id"]
                    else f["fighter_a_id"] if f.get("winner_id") == f["fighter_b_id"]
                    else None
                ),
                "method": f.get("method"),
                "weight_class": f.get("weight_class"),
            }
            for f in fights
        ]
        apply_network_features(raw_results, net_fights)

        return raw_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(
        self,
        acc: FighterAccumulator,
        opp_acc: FighterAccumulator,
        fighter_id: int,
        fight_id: int,
        domain_elo: dict[tuple[int, int], dict[str, float | None]],
    ) -> dict[str, float | str | None]:
        """Build feature dict from pre-fight accumulator state."""
        cfg = self.config
        fight_minutes = (
            acc.total_fight_time_seconds / 60.0
            if acc.total_fight_time_seconds > 0 else 0.0
        )

        # Career averages from accumulator totals
        sig_str_per_min = (
            acc.total_sig_str_landed / fight_minutes if fight_minutes > 0 else 0.0
        )
        total_str_per_min = (
            acc.total_total_str_landed / fight_minutes if fight_minutes > 0 else 0.0
        )
        td_rate = (
            acc.total_td_landed / acc.fight_count if acc.fight_count > 0 else 0.0
        )
        td_accuracy: float | None = None
        if acc.total_td_attempted > 0:
            td_accuracy = acc.total_td_landed / acc.total_td_attempted
        td_defense: float | None = None
        if acc.total_opp_td_attempted > 0:
            td_defense = 1.0 - (acc.total_opp_td_landed / acc.total_opp_td_attempted)
        strike_defense: float | None = None
        if acc.total_opp_sig_str_attempted > 0:
            strike_defense = 1.0 - (
                acc.total_opp_sig_str_landed / acc.total_opp_sig_str_attempted
            )
        ctrl_time = (
            acc.total_ctrl_time / acc.fight_count if acc.fight_count > 0 else 0.0
        )
        sub_att = (
            acc.total_sub_attempts / acc.fight_count if acc.fight_count > 0 else 0.0
        )

        # EWMA values (filter out None for accuracy/defense)
        sig_str_per_min_ewma = compute_ewma(
            acc.career_sig_str_per_min, cfg.ewma_alpha,
        )
        total_str_per_min_ewma = compute_ewma(
            acc.career_total_str_per_min, cfg.ewma_alpha,
        )
        td_rate_ewma = compute_ewma(acc.career_td_rate, cfg.ewma_alpha)
        td_accuracy_ewma = compute_ewma(
            [v for v in acc.career_td_accuracy if v is not None], cfg.ewma_alpha,
        )
        td_defense_ewma = compute_ewma(
            [v for v in acc.career_td_defense if v is not None], cfg.ewma_alpha,
        )
        strike_defense_ewma = compute_ewma(
            [v for v in acc.career_strike_defense if v is not None], cfg.ewma_alpha,
        )
        ctrl_time_ewma = compute_ewma(acc.career_ctrl_time, cfg.ewma_alpha)
        sub_att_ewma = compute_ewma(acc.career_sub_att, cfg.ewma_alpha)

        # Opponent-adjusted rates (D-07, D-08)
        # Compare fighter's career avg vs opponent's career avg for the same stat
        opp_fight_min = (
            opp_acc.total_fight_time_seconds / 60.0
            if opp_acc.total_fight_time_seconds > 0 else 0.0
        )
        opp_avg_sig_str = (
            opp_acc.total_sig_str_landed / opp_fight_min
            if opp_fight_min > 0 else None
        )
        opp_avg_td = (
            opp_acc.total_td_landed / opp_acc.fight_count
            if opp_acc.fight_count > 0 else None
        )
        opp_avg_strike_def: float | None = None
        if opp_acc.total_opp_sig_str_attempted > 0:
            opp_avg_strike_def = 1.0 - (
                opp_acc.total_opp_sig_str_landed / opp_acc.total_opp_sig_str_attempted
            )
        opp_avg_ctrl_time = (
            opp_acc.total_ctrl_time / opp_acc.fight_count
            if opp_acc.fight_count > 0 else None
        )

        opp_adj_sig_str = compute_opponent_adjusted(sig_str_per_min, opp_avg_sig_str)
        opp_adj_td = compute_opponent_adjusted(td_rate, opp_avg_td)
        opp_adj_strike_def = compute_opponent_adjusted(
            strike_defense if strike_defense is not None else 0.0,
            opp_avg_strike_def,
        )
        opp_adj_ctrl_time = compute_opponent_adjusted(ctrl_time, opp_avg_ctrl_time)

        # Style tag (D-01)
        elo_data = domain_elo.get((fighter_id, fight_id))
        if elo_data:
            style_tag = classify_style(
                elo_data.get("striking_elo"),
                elo_data.get("grappling_elo"),
                cfg.style_elo_threshold,
            )
        else:
            style_tag = classify_style(None, None, cfg.style_elo_threshold)

        return {
            "sig_str_per_minute": sig_str_per_min,
            "sig_str_per_minute_ewma": sig_str_per_min_ewma,
            "total_str_per_minute": total_str_per_min,
            "total_str_per_minute_ewma": total_str_per_min_ewma,
            "td_rate": td_rate,
            "td_rate_ewma": td_rate_ewma,
            "td_accuracy": td_accuracy,
            "td_accuracy_ewma": td_accuracy_ewma,
            "td_defense": td_defense,
            "td_defense_ewma": td_defense_ewma,
            "strike_defense": strike_defense,
            "strike_defense_ewma": strike_defense_ewma,
            "ctrl_time_per_fight": ctrl_time,
            "ctrl_time_per_fight_ewma": ctrl_time_ewma,
            "sub_att_per_fight": sub_att,
            "sub_att_per_fight_ewma": sub_att_ewma,
            "opp_adj_sig_str": opp_adj_sig_str,
            "opp_adj_td": opp_adj_td,
            "opp_adj_strike_def": opp_adj_strike_def,
            "opp_adj_ctrl_time": opp_adj_ctrl_time,
            "style_tag": style_tag,
        }

    @staticmethod
    def _update_accumulator(
        acc: FighterAccumulator,
        fighter_rs: list[dict[str, Any]],
        opponent_rs: list[dict[str, Any]],
        rates: dict[str, float | None],
        fight_duration: int,
    ) -> None:
        """Update accumulator with this fight's raw stats and computed rates."""
        acc.total_fight_time_seconds += fight_duration

        # Raw stat aggregation from round stats
        acc.total_sig_str_landed += _sum_stat(fighter_rs, "sig_strikes_landed")
        total_str = (
            _sum_stat(fighter_rs, "head_strikes_landed")
            + _sum_stat(fighter_rs, "body_strikes_landed")
            + _sum_stat(fighter_rs, "leg_strikes_landed")
        )
        acc.total_total_str_landed += total_str
        acc.total_td_landed += _sum_stat(fighter_rs, "takedowns_landed")
        acc.total_td_attempted += _sum_stat(fighter_rs, "takedowns_attempted")
        acc.total_ctrl_time += _sum_stat(fighter_rs, "control_time_seconds")
        acc.total_sub_attempts += _sum_stat(fighter_rs, "submission_attempts")

        # Opponent raw stats for defensive metrics
        acc.total_opp_sig_str_landed += _sum_stat(opponent_rs, "sig_strikes_landed")
        acc.total_opp_sig_str_attempted += _sum_stat(opponent_rs, "sig_strikes_attempted")
        acc.total_opp_td_landed += _sum_stat(opponent_rs, "takedowns_landed")
        acc.total_opp_td_attempted += _sum_stat(opponent_rs, "takedowns_attempted")

        # Append per-fight rate values to history lists for EWMA
        acc.career_sig_str_per_min.append(
            rates["sig_str_per_minute"] if rates["sig_str_per_minute"] is not None else 0.0
        )
        acc.career_total_str_per_min.append(
            rates["total_str_per_minute"] if rates["total_str_per_minute"] is not None else 0.0
        )
        acc.career_td_rate.append(
            float(rates["td_rate"]) if rates["td_rate"] is not None else 0.0
        )
        acc.career_td_accuracy.append(rates["td_accuracy"])
        acc.career_td_defense.append(rates["td_defense"])
        acc.career_strike_defense.append(rates["strike_defense"])
        acc.career_ctrl_time.append(
            float(rates["ctrl_time_per_fight"]) if rates["ctrl_time_per_fight"] is not None else 0.0
        )
        acc.career_sub_att.append(
            float(rates["sub_att_per_fight"]) if rates["sub_att_per_fight"] is not None else 0.0
        )

    @staticmethod
    def _compute_league_means(results: list[dict[str, Any]]) -> dict[str, float]:
        """Compute league-wide mean for each numeric feature across all results."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        for row in results:
            feats = row["features"]
            for key in CANONICAL_FEATURE_ORDER:
                val = feats.get(key)
                if val is not None and isinstance(val, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(val)
                    counts[key] = counts.get(key, 0) + 1

        means: dict[str, float] = {}
        for key in sums:
            if counts[key] > 0:
                means[key] = sums[key] / counts[key]
        return means

    def _apply_embeddings(self, results: list[dict[str, Any]]) -> None:
        """Compute PCA embeddings on latest feature vector per fighter.

        Writes embedding values back into each fighter's LATEST feature dict.
        Per D-10, embeddings stored in the features JSON.
        """
        if not results:
            return

        # Find latest feature row per fighter (last occurrence)
        latest_by_fighter: dict[int, int] = {}  # fighter_id -> index in results
        for idx, row in enumerate(results):
            latest_by_fighter[row["fighter_id"]] = idx

        if len(latest_by_fighter) < 2:
            # PCA needs at least 2 samples
            return

        # Build numeric feature keys (exclude style_tag string)
        numeric_keys = [k for k in CANONICAL_FEATURE_ORDER if k != "style_tag"]

        # Build matrix from latest features
        fighter_ids_ordered = sorted(latest_by_fighter.keys())
        matrix_rows = []
        for fid in fighter_ids_ordered:
            feats = results[latest_by_fighter[fid]]["features"]
            row_vals = []
            for key in numeric_keys:
                val = feats.get(key)
                if val is None or not isinstance(val, (int, float)):
                    row_vals.append(0.0)
                else:
                    row_vals.append(float(val))
            matrix_rows.append(row_vals)

        feature_matrix = np.array(matrix_rows, dtype=np.float64)
        embeddings = compute_embeddings(feature_matrix, self.config.pca_n_components)

        # Write embeddings back to latest feature dicts
        for i, fid in enumerate(fighter_ids_ordered):
            idx = latest_by_fighter[fid]
            results[idx]["features"]["embedding"] = embeddings[i].tolist()
