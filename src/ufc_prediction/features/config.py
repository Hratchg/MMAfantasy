"""Immutable configuration for feature computation.

All tunable parameters are defined here as a frozen dataclass, enabling
experimentation by instantiating FeatureConfig with different values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable feature computation configuration.

    Defaults per decisions D-01, D-03, D-09, D-12, D-14 from CONTEXT.md.
    """

    # EWMA (D-03, validated Phase 9.1) [E]: all 4 half-life values (2-5 fights)
    # produced identical Brier score 0.172240 -- parameter is structurally
    # indeterminate via this evaluation method. Current default retained.
    ewma_half_life: int = 3
    ewma_alpha: float = 0.206

    # Bayesian shrinkage (D-12): min 5 fights before full weight
    shrinkage_min_fights: int = 5

    # PCA embeddings (D-09): reduce to 8 dimensions
    pca_n_components: int = 8

    # Style classification (D-01): Elo differential threshold
    style_elo_threshold: float = 30.0

    # Feature versioning (D-14)
    feature_set_version: str = "v1"

    def __post_init__(self) -> None:
        """Validate parameter constraints to prevent silent misconfiguration."""
        if not (0 < self.ewma_alpha < 1):
            raise ValueError("ewma_alpha must be in (0, 1)")
        if self.shrinkage_min_fights <= 0:
            raise ValueError("shrinkage_min_fights must be > 0")
        if self.pca_n_components <= 0:
            raise ValueError("pca_n_components must be > 0")


# Canonical feature order per RESEARCH Pitfall 6.
# All downstream code must use this ordering for deterministic feature vectors.
CANONICAL_FEATURE_ORDER: tuple[str, ...] = (
    "sig_str_per_minute",
    "sig_str_per_minute_ewma",
    "total_str_per_minute",
    "total_str_per_minute_ewma",
    "td_rate",
    "td_rate_ewma",
    "td_accuracy",
    "td_accuracy_ewma",
    "td_defense",
    "td_defense_ewma",
    "strike_defense",
    "strike_defense_ewma",
    "ctrl_time_per_fight",
    "ctrl_time_per_fight_ewma",
    "sub_att_per_fight",
    "sub_att_per_fight_ewma",
    "opp_adj_sig_str",
    "opp_adj_td",
    "opp_adj_strike_def",
    "opp_adj_ctrl_time",
    "style_tag",  # string, not numeric -- excluded from PCA matrix
)
