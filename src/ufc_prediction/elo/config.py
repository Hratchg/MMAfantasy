"""Immutable configuration for Elo rating computation.

All tunable parameters are defined here as a frozen dataclass, enabling
backtesting by instantiating EloConfig with different values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloConfig:
    """Immutable Elo configuration with all tunable parameters.

    Defaults per decisions D-01 through D-10 from CONTEXT.md.
    """

    # Base rating (D-10)
    initial_rating: float = 1500.0

    # K-factor decay (D-03) [E] validated Phase 9
    k_initial: float = 40.0
    k_experienced: float = 20.0
    k_transition_fights: int = 5

    # Division transfer (D-05, validated Phase 9.1) [E]
    division_transfer_pct: float = 0.50  # was 0.75 [L]; backtest optimal: 50%

    # Bayesian shrinkage (D-07) [L]
    shrinkage_min_fights: int = 5

    # Inactivity regression (D-08, validated Phase 9.1) [E]
    inactivity_threshold_days: int = 270  # was 365 [L]; backtest optimal: 270d (9mo)
    inactivity_regression_rate: float = 0.10  # per month past threshold; confirmed optimal [E]
    inactivity_regression_cap: float = 0.50   # confirmed optimal [E]

    # Margin-of-victory multipliers (D-01, validated Phase 9.1) [E]
    mov_ko_tko: float = 1.2      # was 1.5 [L]; backtest optimal: 1.2
    mov_submission: float = 1.1  # was 1.4 [L]; backtest optimal: 1.1
    mov_unanimous: float = 1.0   # [S] standard baseline
    mov_split: float = 0.5       # was 0.8 [L]; backtest optimal: 0.5
    mov_draw: float = 0.5        # [S] standard
    mov_dq: float = 0.8          # [L] held fixed during backtest per D-04

    def __post_init__(self) -> None:
        """Validate parameter constraints to prevent silent misconfiguration."""
        if self.shrinkage_min_fights <= 0:
            raise ValueError("shrinkage_min_fights must be > 0")
