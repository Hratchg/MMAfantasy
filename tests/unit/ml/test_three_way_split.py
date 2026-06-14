"""Phase 19 META-06 Wave-0 RED — make_three_way_split asserts disjoint fight_ids."""
from __future__ import annotations

from datetime import date

import pytest

oof = pytest.importorskip("ufc_prediction.ml.oof")


def test_disjoint_assertion():
    """3 fights spanning the 3 partitions → 3 disjoint id sets."""
    fights = [
        {"fight_id": 1, "event_date": date(2022, 6, 1)},   # base (pre-cutoff)
        {"fight_id": 2, "event_date": date(2024, 1, 1)},   # meta_train (post-cutoff, not in last 365d)
        {"fight_id": 3, "event_date": date(2025, 11, 1)},  # meta_eval (last 365d)
    ]
    base, meta_train, meta_eval = oof.make_three_way_split(
        fights,
        base_cutoff=date(2023, 1, 1),
        meta_eval_window_days=365,
        today=date(2026, 5, 9),
    )
    assert {f["fight_id"] for f in base} == {1}
    assert {f["fight_id"] for f in meta_train} == {2}
    assert {f["fight_id"] for f in meta_eval} == {3}


def test_disjoint_assertion_fires_on_overlap():
    """Synthetic overlap (duplicate fight_id straddling boundaries) → AssertionError."""
    fights = [
        {"fight_id": 1, "event_date": date(2022, 6, 1)},  # would land in base
        {"fight_id": 1, "event_date": date(2024, 1, 1)},  # SAME id, but in meta_train window
    ]
    with pytest.raises(AssertionError, match="non-empty"):
        oof.make_three_way_split(
            fights, base_cutoff=date(2023, 1, 1),
            meta_eval_window_days=365, today=date(2026, 5, 9),
        )
