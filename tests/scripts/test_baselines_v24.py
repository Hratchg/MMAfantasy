"""TRUST-V24-01 smoke tests — locks the contract of results/baselines_v24.json.

Plan 34-03 Task 3 — tests the JSON-shape invariants of the 4-baseline benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "results" / "baselines_v24.json"
CANONICAL_XGB_V2_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"


@pytest.fixture(scope="module")
def baselines_doc() -> dict:
    if not RESULTS_PATH.exists():
        pytest.skip("results/baselines_v24.json missing — run scripts/baselines_v24.py first")
    return json.loads(RESULTS_PATH.read_text())


def test_15_rows_present(baselines_doc: dict) -> None:
    """5 baselines × 3 slices = 15 rows (meta_v22 anchor + 4 baselines × 3 slices)."""
    rows = baselines_doc["rows"]
    assert len(rows) == 15, f"expected 15 rows, got {len(rows)}"


def test_required_keys_per_row(baselines_doc: dict) -> None:
    """Each row must carry slice, model, brier, accuracy, n."""
    required = {"slice", "model", "brier", "accuracy", "n"}
    for r in baselines_doc["rows"]:
        missing = required - set(r.keys())
        assert not missing, f"row {r} missing keys: {missing}"


def test_market_baseline_has_n_with_market(baselines_doc: dict) -> None:
    """market_directional_implied_baseline rows must carry n_with_market."""
    market_rows = [
        r for r in baselines_doc["rows"] if r["model"] == "market_directional_implied_baseline"
    ]
    assert len(market_rows) == 3
    for r in market_rows:
        assert "n_with_market" in r, f"row {r} missing n_with_market"


def test_delta_columns_set_for_non_anchor_rows(baselines_doc: dict) -> None:
    """delta_brier_vs_meta_v22 + delta_acc_vs_meta_v22 set on all non-meta_v22 rows."""
    for r in baselines_doc["rows"]:
        if r["model"] == "meta_v22":
            continue
        assert "delta_brier_vs_meta_v22" in r
        assert "delta_acc_vs_meta_v22" in r


def test_top_level_sha_anchors_match_canonical(baselines_doc: dict) -> None:
    """xgb_v2 SHA in JSON top-level == canonical anchor."""
    assert baselines_doc["xgb_v2_sha256"] == CANONICAL_XGB_V2_SHA
    assert len(baselines_doc["meta_v2_sha256"]) == 64
    assert "xgb_v2_no_odds_sha256" in baselines_doc
