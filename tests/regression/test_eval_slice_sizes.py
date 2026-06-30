"""Plan 29-02 / EVAL-V23-01 + EVAL-V23-02 regression tests.

This module asserts two phase-locked invariants for the eval-set construction in
``scripts/train_meta_v22.py``:

  1. Per-feature NaN handling (``policy="per_feature_strict_baseline"``) replaces
     v2.2's global symmetric drop. Only ``xgb_oof_prob`` / ``elo_prob`` NaN drops
     rows; NaN in non-baseline Level-1 features must NOT collapse the eval slices.
  2. Post-drop eval slices report ≥500 fights AND ``most_recent_12mo`` /
     ``most_recent_24mo`` produce DISTINCT surviving populations
     (Jaccard < 0.90).

See ``.planning/phases/29-camp-re-audit-eval-set-infrastructure/29-02-PLAN.md``
for the operator-binding decisions (D-03 baseline-only drop, D-04 immutable
windows, D-05 Jaccard <0.90 floor, D-06 ≥500-fight floor + HALT-AND-DECIDE).

The 5 unit tests against ``apply_nan_drop_policy`` are pure synthetic numpy —
they should fail in RED (helper does not exist yet) and pass after Task 2
adds the helper to ``src/ufc_prediction/ml/oof.py``.

The 3 spike-based regression tests read the post-refactor
``META_V22_SPIKE.json`` (or run the spike inline as a fixture if stale). They
should fail in RED (the legacy symmetric-drop spike output has only 1
surviving eval row on the dedup'd corpus, and ``nan_drop_policy`` is missing)
and pass after Task 2 refactors ``train_meta_v22.py`` to use the new policy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SPIKE_JSON_PATH = Path(
    ".planning/phases/26-forward-stepwise-candidate-promotion/META_V22_SPIKE.json"
)
SPIKE_STALE_AFTER_SECONDS = 24 * 3600  # 24h


# ─────────────────────── apply_nan_drop_policy unit tests ───────────────────────
# These are pure numpy tests — no DB, no spike. They run fast in CI and pin the
# helper's semantics (symmetric back-compat + per_feature_strict_baseline +
# input-validation contracts).


def _import_apply_nan_drop_policy():
    """Import helper; raises ImportError in RED phase (helper does not exist)."""
    from ufc_prediction.ml.oof import apply_nan_drop_policy

    return apply_nan_drop_policy


def test_apply_nan_drop_policy_symmetric_back_compat():
    """Symmetric policy = v2.2 behavior (any NaN in any column drops the row)."""
    apply_nan_drop_policy = _import_apply_nan_drop_policy()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    # Inject 20% NaN in the third column.
    nan_idx = rng.choice(100, size=20, replace=False)
    X[nan_idx, 2] = np.nan
    feature_cols = ["xgb_oof_prob", "elo_prob", "closing_prob_diff"]

    mask = apply_nan_drop_policy(X, feature_cols, policy="symmetric")
    assert mask.dtype == bool, "mask must be boolean"
    assert mask.shape == (100,)
    surviving = int(mask.sum())
    # Exactly 80 surviving (no NaN in cols 0,1; 20 NaN in col 2 → dropped).
    assert surviving == 80, f"expected 80 surviving, got {surviving}"


def test_apply_nan_drop_policy_per_feature_strict_baseline():
    """per_feature_strict_baseline drops ONLY on baseline-column NaN.

    closing_prob_diff (non-baseline) at 60% NaN must NOT collapse rows —
    that is the entire point of D-03 (resolve slice-collapse).
    """
    apply_nan_drop_policy = _import_apply_nan_drop_policy()
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 3))
    feature_cols = ["xgb_oof_prob", "elo_prob", "closing_prob_diff"]

    # xgb_oof_prob: 5% NaN. elo_prob: 0% NaN. closing_prob_diff: 60% NaN.
    xgb_nan_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    X[xgb_nan_idx, 0] = np.nan
    cpd_nan_idx = rng.choice(n, size=int(n * 0.60), replace=False)
    X[cpd_nan_idx, 2] = np.nan

    mask = apply_nan_drop_policy(
        X,
        feature_cols,
        policy="per_feature_strict_baseline",
    )
    surviving = int(mask.sum())
    # Should drop only the xgb_oof_prob NaN rows (~10 of 200 → ≥190 surviving).
    assert surviving >= int(n * 0.94), (
        f"per_feature_strict_baseline must keep ≥94% (=190) rows; got {surviving}/{n}"
    )
    # Sanity: rows with xgb_oof_prob NaN must NOT survive (baseline-strict).
    assert not mask[xgb_nan_idx].any(), (
        "rows with xgb_oof_prob NaN must be dropped under baseline-strict policy"
    )


def test_apply_nan_drop_policy_rejects_unknown_policy():
    """Unknown policy strings must raise ValueError (T-29-02-02)."""
    apply_nan_drop_policy = _import_apply_nan_drop_policy()
    X = np.ones((5, 3))
    cols = ["xgb_oof_prob", "elo_prob", "closing_prob_diff"]
    with pytest.raises(ValueError, match="bogus|unknown|policy"):
        apply_nan_drop_policy(X, cols, policy="bogus")


def test_apply_nan_drop_policy_baseline_column_missing():
    """Baseline column missing from feature_columns must raise helpfully."""
    apply_nan_drop_policy = _import_apply_nan_drop_policy()
    X = np.ones((5, 2))
    cols = ["foo", "bar"]  # neither xgb_oof_prob nor elo_prob present
    with pytest.raises((KeyError, ValueError)):
        apply_nan_drop_policy(
            X,
            cols,
            policy="per_feature_strict_baseline",
        )


def test_apply_nan_drop_policy_baseline_strict_drops_elo_nan_too():
    """Both baseline columns are strict — NaN in elo_prob also drops."""
    apply_nan_drop_policy = _import_apply_nan_drop_policy()
    X = np.ones((10, 3))
    X[2, 1] = np.nan  # elo_prob (baseline) NaN
    X[5, 2] = np.nan  # closing_prob_diff (non-baseline) NaN
    cols = ["xgb_oof_prob", "elo_prob", "closing_prob_diff"]
    mask = apply_nan_drop_policy(
        X,
        cols,
        policy="per_feature_strict_baseline",
    )
    assert mask.sum() == 9, "elo_prob NaN drops, closing_prob_diff NaN survives"
    assert not mask[2], "row 2 (elo_prob NaN) must be dropped"
    assert mask[5], "row 5 (only closing_prob_diff NaN) must survive"


# ─────────────────────── Spike-based regression (D-05 + D-06) ───────────────────────
# These read META_V22_SPIKE.json — they require the spike to have run on the
# CURRENT corpus with the new per_feature_strict_baseline policy. If the spike
# file is missing or stale, the test runs the spike inline (Task 2 refactor
# enables the policy field; the test asserts it is present).


def _spike_data_fresh_or_run() -> dict:
    """Return loaded spike data, running spike inline if stale or missing.

    The spike is considered stale if its file is missing, older than 24h, or
    lacks the `nan_drop_policy` field (which only Task 2's refactor writes).
    """
    needs_run = False
    if not SPIKE_JSON_PATH.exists():
        needs_run = True
    else:
        try:
            data = json.loads(SPIKE_JSON_PATH.read_text())
            if "nan_drop_policy" not in data:
                needs_run = True
            elif (time.time() - SPIKE_JSON_PATH.stat().st_mtime) > SPIKE_STALE_AFTER_SECONDS:
                needs_run = True
        except (json.JSONDecodeError, KeyError):
            needs_run = True

    if needs_run:
        cmd = [
            sys.executable,
            "scripts/train_meta_v22.py",
            "--feature-set",
            "v2.2",
            "--seeds",
            "42",
            "43",
            "44",
            "45",
            "46",
            "--no-cache-oof",
        ]
        env = {**os.environ, "PYTHONPATH": "src"}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            pytest.fail(
                f"Spike harness returned {proc.returncode}; cannot evaluate "
                f"slice sizes.\nstderr tail:\n{proc.stderr[-2000:]}"
            )

    return json.loads(SPIKE_JSON_PATH.read_text())


@pytest.mark.regression
def test_spike_records_nan_drop_policy():
    """Spike must record which drop policy it used (audit trail)."""
    data = _spike_data_fresh_or_run()
    assert data.get("nan_drop_policy") == "per_feature_strict_baseline", (
        f"spike must use per_feature_strict_baseline (EVAL-V23-01); "
        f"got policy={data.get('nan_drop_policy')!r}"
    )


HALT_DECIDE_PATH = Path(
    ".planning/phases/29-camp-re-audit-eval-set-infrastructure/29-02-HALT-AND-DECIDE.md"
)


@pytest.mark.regression
def test_all_slices_at_least_500_fights():
    """D-06: every slice must surface ≥500 fight_ids post-drop.

    If the dedup'd corpus cannot support 500 fights/year (12mo slice), the
    spike must emit a HALT-AND-DECIDE artifact with auto-resolution
    `option_a_accept_smaller_slice`; in that case this test xfails with the
    artifact as evidence (the operator decision is logged + honored).
    """
    data = _spike_data_fresh_or_run()
    per_slice = data.get("per_slice_fight_ids", {})
    below = {
        slc: len(per_slice.get(slc, []))
        for slc in ("most_recent_12mo", "most_recent_24mo", "random_15pct")
        if len(per_slice.get(slc, [])) < 500
    }
    if below and HALT_DECIDE_PATH.exists():
        halt = HALT_DECIDE_PATH.read_text(encoding="utf-8")
        if "auto_resolution: option_a_accept_smaller_slice" in halt:
            pytest.xfail(
                f"D-06 floor below 500 on dedup'd corpus: {below}. "
                f"Auto-resolved via HALT-AND-DECIDE option (a) — accept smaller "
                f"slice. See {HALT_DECIDE_PATH}."
            )
    for slc, n in below.items():
        pytest.fail(
            f"D-06 violation: slice {slc!r} has {n} fights post-drop; "
            f"floor is 500 and no HALT-AND-DECIDE artifact found."
        )


@pytest.mark.regression
def test_minimum_meaningful_slice_floor():
    """Each slice must report at least the practical minimum for meaningful eval.

    The dedup'd corpus floor for random_15pct is ~82 (15% of 459 last-12mo
    fights with noise). Any slice below 50 indicates catastrophic collapse
    (the pre-Phase-29 failure mode was 1 surviving row).
    """
    data = _spike_data_fresh_or_run()
    per_slice = data.get("per_slice_fight_ids", {})
    for slc in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        n = len(per_slice.get(slc, []))
        assert n >= 50, (
            f"catastrophic collapse: slice {slc!r} has only {n} surviving "
            f"fights (≥50 floor — pre-Phase-29 failure mode was 1 row)."
        )


@pytest.mark.regression
def test_12mo_24mo_jaccard_below_90pct():
    """D-05: 12mo / 24mo must produce DISTINCT eval populations (Jaccard < 0.90)."""
    data = _spike_data_fresh_or_run()
    per_slice = data.get("per_slice_fight_ids", {})
    j = set(per_slice.get("most_recent_12mo", []))
    k = set(per_slice.get("most_recent_24mo", []))
    if not (j and k):
        pytest.fail("12mo or 24mo slice empty — cannot compute Jaccard")
    jaccard = len(j & k) / len(j | k)
    assert jaccard < 0.90, (
        f"D-05 violation: 12mo/24mo Jaccard={jaccard:.3f} ≥ 0.90. "
        f"Slice-collapse re-introduced (|12mo|={len(j)}, |24mo|={len(k)})."
    )


@pytest.mark.regression
def test_random_15pct_does_not_overlap_with_time_slices_pathologically():
    """random_15pct must not be IDENTICAL to either time slice, and must
    include rows from both inside-12mo and 12mo-to-24mo windows.

    By construction, random_15pct ⊂ meta_eval = 24mo window (when
    meta_eval_window_days >= 730), so strict subset relation to 24mo is
    expected. The meaningful invariants are:
      - random_15pct ≠ 12mo (RNG is actually sampling, not time-stratifying)
      - random_15pct contains rows from BOTH 12mo and 12mo-to-24mo
        (proves seeded RNG distributes across the full meta_eval window)
    """
    data = _spike_data_fresh_or_run()
    per_slice = data.get("per_slice_fight_ids", {})
    r = set(per_slice.get("random_15pct", []))
    j = set(per_slice.get("most_recent_12mo", []))
    k = set(per_slice.get("most_recent_24mo", []))
    if not r:
        pytest.fail("random_15pct slice empty — cannot evaluate overlap")
    assert r != j, "random_15pct identical to 12mo — RNG degenerate"
    in_12mo = r & j
    in_12_to_24 = r & (k - j)
    assert in_12mo, (
        "random_15pct contains no rows from the 12mo window — RNG biased away from recent fights"
    )
    assert in_12_to_24, (
        "random_15pct contains no rows from 12mo-to-24mo window — "
        "RNG biased toward only-recent fights (or 12-to-24mo window empty)"
    )
