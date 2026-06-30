"""Phase 26 META-V22-02 regression tests: TimeSeriesSplit three-way + disjoint-fight-ID.

Per Plan 26-01 Task 3 behaviors A1..A3:
- A1: test_three_way_split_disjoint_fight_ids — duplicate fight_id across partitions
      raises AssertionError mentioning "D-01(P19) violation" (proves guard is wired).
- A2: test_split_event_date_semantics — clean synthetic fights respect the per-partition
      event_date ranges per oof.make_three_way_split contract.
- A3: SENTINEL — test_oof_uses_72col_view_assertion is SKIPPED until Plan 26-02
      activates it by enforcing the 72-col assertion at the OOF call site in
      scripts/train_meta_v22.py.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from ufc_prediction.ml.oof import make_three_way_split


def test_three_way_split_disjoint_fight_ids():
    """A1 — Inject a duplicate fight_id across base/meta_train and assert
    `make_three_way_split` raises AssertionError mentioning "D-01(P19) violation".
    """
    # Two fights share fight_id=999 but have event_dates straddling base_cutoff;
    # one lands in base, the other in meta_train → the disjoint-id ASSERT must fire.
    base_cutoff = date(2023, 1, 1)
    today = date(2026, 5, 17)
    # base partition: event_date < 2023-01-01
    # meta_train partition: 2023-01-01 <= event_date < (today - 365d) = 2025-05-17
    # meta_eval partition:  event_date >= 2025-05-17
    fights = [
        {"fight_id": 1, "event_date": date(2022, 6, 1)},  # base
        {"fight_id": 999, "event_date": date(2022, 7, 1)},  # base (duplicate fight_id)
        {"fight_id": 999, "event_date": date(2024, 3, 1)},  # meta_train (duplicate fight_id)
        {"fight_id": 3, "event_date": date(2025, 12, 1)},  # meta_eval
    ]
    with pytest.raises(AssertionError, match=r"D-01\(P19\) violation"):
        make_three_way_split(
            fights,
            base_cutoff=base_cutoff,
            meta_eval_window_days=365,
            today=today,
        )


def test_split_event_date_semantics():
    """A2 — Clean synthetic fights produce per-partition event_date ranges per contract.

    base_train:  event_date < base_cutoff
    meta_train:  base_cutoff <= event_date < (today - meta_eval_window_days)
    meta_eval:   event_date >= (today - meta_eval_window_days)
    """
    base_cutoff = date(2023, 1, 1)
    today = date(2026, 5, 17)
    eval_start = today - timedelta(days=365)  # 2025-05-17

    fights = []
    # 20 base fights spanning 2020 → 2022-12 (event_date < base_cutoff)
    for i in range(20):
        fights.append(
            {
                "fight_id": 100 + i,
                "event_date": date(2020, 1, 1) + timedelta(days=i * 50),
            }
        )
    # 20 meta_train fights spanning 2023-01 → 2025-04 (post-cutoff; pre-eval)
    for i in range(20):
        fights.append(
            {
                "fight_id": 200 + i,
                "event_date": date(2023, 1, 5) + timedelta(days=i * 40),
            }
        )
    # 10 meta_eval fights spanning 2025-06 → 2026-05 (>=eval_start)
    for i in range(10):
        fights.append(
            {
                "fight_id": 300 + i,
                "event_date": date(2025, 6, 1) + timedelta(days=i * 30),
            }
        )

    base, meta_train, meta_eval = make_three_way_split(
        fights,
        base_cutoff=base_cutoff,
        meta_eval_window_days=365,
        today=today,
    )

    assert all(f["event_date"] < base_cutoff for f in base), (
        "base partition contains event_date >= base_cutoff"
    )
    assert all(base_cutoff <= f["event_date"] < eval_start for f in meta_train), (
        "meta_train event_dates outside [base_cutoff, eval_start)"
    )
    assert all(f["event_date"] >= eval_start for f in meta_eval), (
        "meta_eval partition contains event_date < eval_start"
    )

    # Disjoint sets — make_three_way_split already asserts internally; double-check.
    base_ids = {f["fight_id"] for f in base}
    mt_ids = {f["fight_id"] for f in meta_train}
    me_ids = {f["fight_id"] for f in meta_eval}
    assert base_ids.isdisjoint(mt_ids)
    assert base_ids.isdisjoint(me_ids)
    assert mt_ids.isdisjoint(me_ids)


def test_oof_uses_72col_view_assertion():
    """A3 ACTIVATED by Plan 26-02 — scripts/train_meta_v22.py exposes `_enforce_72col_view`
    which asserts X_oof.shape[1] == 72 at the OOF call site (Pitfall #2 / Pitfall B).

    Pass a 90-col matrix to the guard; assert it raises AssertionError with a message
    matching `n_features=72` or `72-col view` or `Pitfall B`.
    """
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_meta_v22",
        Path("scripts/train_meta_v22.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_meta_v22_test_import"] = module
    spec.loader.exec_module(module)
    enforce = module._enforce_72col_view
    import numpy as np

    X_90 = np.zeros((10, 90))
    with pytest.raises(AssertionError, match=r"(72-col view|n_features=72|Pitfall B)"):
        enforce(X_90)
    # Also verify the positive case (72 cols) does not raise.
    X_72 = np.zeros((10, 72))
    enforce(X_72)  # should not raise
