"""META-V22 rich Level-1 feature stacker (Phase 26).

Per CONTEXT D-03 + RESEARCH §"Specific Ideas": 13-col Level-1 set replacing the
3-col META-01 set `[xgb_oof_prob, elo_prob, closing_prob_diff]`. The "~14"
target trims to 13 after DEDUP (planner-time check: `age_diff` chosen over
`age_at_fight_*` to avoid duplicate semantics).

Exports
-------
- META_V22_FEATURE_COLUMNS: list[str]  (len == 13)
- build_meta_features_v22(xgb_oof_prob, elo_prob, X_v22) -> np.ndarray (n, 13)

Invariants (enforced at module-import time):
- exactly 13 columns
- all 11 internal columns resolve in FEATURE_COLUMNS_V22 (Pitfall #1 guard:
  no hardcoded indices — uses .index() lookup; fail-fast on rename/reorder)

Used by `scripts/train_meta_v22.py` (Plan 26-02) to assemble the Level-1
matrix for the 5-seed × 3-slice training spike.
"""

from __future__ import annotations

import numpy as np

from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

# 13 Level-1 cols. First two are EXTERNAL (computed in-driver from base xgb_v2
# OOF + EloEngine.expected_win_probability). The other 11 are pulled from
# FEATURE_COLUMNS_V22 at runtime via .index() (Pitfall #1: no hardcoded
# indices — APPEND-ONLY discipline tolerates index drift, but a name rename
# fails fast).
META_V22_FEATURE_COLUMNS: list[str] = [
    "xgb_oof_prob",                # external — supplied by oof.generate_oof_predictions
    "elo_prob",                    # external — EloEngine.expected_win_probability
    "closing_prob_diff",           # FEATURE_COLUMNS_NO_NET (BFO-backfilled post Phase 21)
    "stance_matchup",              # FEATURE_COLUMNS_NO_NET
    "height_diff",                 # FEATURE_COLUMNS_NO_NET
    "reach_diff",                  # FEATURE_COLUMNS_NO_NET (raw — NOT reach_diff_normalized)
    "days_since_last_fight_diff",  # FEATURE_COLUMNS_NO_NET
    "age_diff",                    # FEATURE_COLUMNS_NO_NET (DEDUP: NOT age_at_fight_*)
    "elo_overall_diff",            # FEATURE_COLUMNS_NO_NET
    "elo_striking_diff",           # FEATURE_COLUMNS_NO_NET
    "elo_grappling_diff",          # FEATURE_COLUMNS_NO_NET
    "division_finish_rate_shrunk", # FEATURE_COLUMNS_V22 (Phase 23 META rich addition)
    "sharp_money_signal",          # FEATURE_COLUMNS_NO_NET (BFO-backfilled)
]


# Module-import-time invariants (Pitfall #1 fail-fast):
assert len(META_V22_FEATURE_COLUMNS) == 13, (
    f"META_V22_FEATURE_COLUMNS must have exactly 13 cols "
    f"(got {len(META_V22_FEATURE_COLUMNS)})"
)
# Resolve all 11 internal cols once at import time so a rename fails immediately.
for _name in META_V22_FEATURE_COLUMNS[2:]:
    FEATURE_COLUMNS_V22.index(_name)  # raises ValueError on miss


def build_meta_features_v22(
    xgb_oof_prob: np.ndarray,        # shape (n,) — base xgb_v2 OOF positive-class probabilities
    elo_prob: np.ndarray,            # shape (n,) — as-of fight-date Elo P(A wins)
    X_v22: np.ndarray,               # shape (n, 90) — FEATURE_COLUMNS_V22 matrix
) -> np.ndarray:
    """Stack the 13 META-V22 Level-1 features. NaN-preserving.

    Per CONTEXT D-03 + Pattern 1 (symmetric NaN-drop): preserves any pre-existing NaNs
    so the consumer (train_meta_v22.py) can apply a single symmetric drop on train + eval.
    """
    cols_internal: list[np.ndarray] = []
    for name in META_V22_FEATURE_COLUMNS[2:]:  # skip the 2 external cols
        idx = FEATURE_COLUMNS_V22.index(name)
        cols_internal.append(X_v22[:, idx])
    return np.column_stack([xgb_oof_prob, elo_prob, *cols_internal])
