"""Phase 18 NET-V2-01 Wave-0 RED — include_net flag plumbing through config.

D-09(P15) APPEND-ONLY discipline: NET-* are guaranteed at the END of
FEATURE_COLUMNS, so FEATURE_COLUMNS_NO_NET = FEATURE_COLUMNS[:-3] is
safe-by-construction. Default include_net=True preserves backwards compat
for unrelated callers; include_net=False produces the 72-col view that
matches xgb_v2's exact column space.

These tests RED on import (Wave 0) — `FEATURE_COLUMNS_NO_NET` and
`get_feature_columns` do not exist in `config.py` yet. Goes GREEN at
Wave 1 Task 9.
"""
from ufc_prediction.ml.config import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_NO_NET,
    get_feature_columns,
)


def test_feature_columns_no_net_is_72_cols():
    assert len(FEATURE_COLUMNS_NO_NET) == 72
    assert len(FEATURE_COLUMNS) == 75


def test_feature_columns_no_net_drops_trailing_3_net():
    assert FEATURE_COLUMNS_NO_NET == FEATURE_COLUMNS[:-3]
    assert "pagerank_diff" not in FEATURE_COLUMNS_NO_NET
    assert "sos_2hop_diff" not in FEATURE_COLUMNS_NO_NET
    assert "is_debutant_in_graph_diff" not in FEATURE_COLUMNS_NO_NET


def test_get_feature_columns_default_includes_net():
    assert get_feature_columns() == list(FEATURE_COLUMNS)
    assert get_feature_columns(include_net=True) == list(FEATURE_COLUMNS)


def test_get_feature_columns_include_net_false_drops_net():
    assert get_feature_columns(include_net=False) == list(FEATURE_COLUMNS_NO_NET)
