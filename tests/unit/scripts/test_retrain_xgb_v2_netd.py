"""Phase 66 Plan 66-01 (FEAT-V261-03) — unit tests for the xgb_v2_netd retrain script.

Targets ``scripts/retrain_xgb_v2_netd.py`` (Phase 66 D-03 deliverable).

Mirrors ``tests/unit/scripts/test_retrain_xgb_v2_refv2.py`` (Phase 65) — same
two-tier shape (cheap + gated heavy) so CI cost stays in the unit-tier
budget while operators can opt into the full integration smoke via
``RUN_HEAVY_TESTS=1``.

Cheap-tier tests (always run; <5s budget):
  1. ``test_script_imports`` — module exposes ``main``, ``OUT_JOBLIB``,
     ``OUT_META``, ``OUT_OOF``, ``PROTECTED_OUTPUTS``,
     ``EXPECTED_XGB_V2_SHA256``, ``EXPECTED_META_V2_SHA256``,
     ``NET_V2_COLS``, ``XGB_NETD_FROZEN_DATE``,
     ``assert_audit01_invariants`` as importable symbols.
  2. ``test_argparse_help_exits_zero`` — ``main(["--help"])`` raises
     ``SystemExit(0)`` (argparse default behaviour).
  3. ``test_expected_xgb_v2_sha256_matches_audit01`` — the locked AUDIT-01
     SHA constant matches the canonical hex.
  4. ``test_expected_meta_v2_sha256_matches_audit01`` — the meta_v2 anchor
     SHA matches the locked AUDIT-01 value.
  5. ``test_net_v2_cols_ordering_locked`` — ``NET_V2_COLS == (
     "net_v2_pagerank_at", "net_v2_2hop_sos_at")`` AND length == 2.
  6. ``test_frozen_date_is_phase_66_phase_start`` — ``XGB_NETD_FROZEN_DATE
     == date(2026, 6, 6)`` and distinct from Phase 65's ``date(2026, 6, 4)``.
  7. ``test_protected_outputs_contains_canonical_joblib_and_meta`` —
     ``PROTECTED_OUTPUTS`` contains the resolved paths of ``xgb_v2.joblib``
     and ``xgb_v2_meta.json`` so the anti-overwrite guard cannot be
     bypassed by a stray ``--output`` argv.
  8. ``test_anti_overwrite_guard_fires_on_canonical_joblib`` — invoking
     ``main`` with ``--output models/xgb_v2.joblib`` exits with rc != 0
     and writes a clean stderr message containing "refusing to overwrite"
     (mirrors Phase 64 CR-01 + Phase 65 carry-forward).
  9. ``test_anti_overwrite_guard_fires_on_resolved_canonical_meta_via_output_meta`` —
     when ``--output-meta models/xgb_v2_meta.json`` is supplied the guard
     also fires before any side effects.
 10. ``test_audit01_invariant_assertion_message`` — calling
     ``assert_audit01_invariants`` with a monkeypatched wrong-SHA constant
     raises ``AssertionError`` whose message contains the literal string
     ``AUDIT-01`` so future canonical drift is not silently absorbed.
 11. ``test_canonical_joblib_sha_unchanged_at_test_load_time`` — sanity
     check the canonical artifact is byte-identical at TEST collection
     time. If this fails, the whole Phase 66 test suite is moot.
 12. ``test_compute_net_v2_columns_returns_pair_aligned_with_rows`` —
     direct unit test of ``_compute_net_v2_columns``: a small in-memory
     2-row corpus produces a length-2 ``(pagerank, sos)`` tuple where
     debutant rows map to ``NaN``.

Heavy-tier integration tests (GATED by ``RUN_HEAVY_TESTS=1``):
 13. ``test_dry_run_emits_sibling_artifacts`` — full ``--dry-run`` produces
     the joblib + JSON + OOF parquet under ``tmp_path``.
 14. ``test_dry_run_meta_json_schema`` — sidecar JSON has
     ``canonical_status == "candidate_sibling_NOT_canonical"``,
     ``n_features == 92``, ``decay_base == 0.98``, and the 2 NET v2 cols
     at indices 90, 91.
 15. ``test_dry_run_oof_parquet_schema`` — OOF parquet has the locked
     3-column schema ``{fight_id, oof_prob, event_date}``.
 16. ``test_audit01_invariants_unchanged_after_dry_run`` — canonical SHAs
     remain byte-identical after the dry-run completes.
 17. ``test_emitted_model_accepts_92_col_input`` — joblib.load(...).
     ``n_features_in_ == 92``.

Heavy tests run when the environment exposes ``RUN_HEAVY_TESTS=1``; otherwise
they are SKIPPED. Default CI runs only the cheap tier (12 tests).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import pytest

# ── Module-import helpers ─────────────────────────────────────────────────
#
# ``scripts/`` is not on ``sys.path`` by default for the tests/unit/ tree;
# we make a one-shot ``sys.path`` injection at module import time so the
# direct import below resolves. Mirrors the Phase 64 / 65 test pattern.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from retrain_xgb_v2_netd import (
    EXPECTED_META_V2_SHA256,
    EXPECTED_XGB_V2_SHA256,
    NET_V2_COLS,
    OUT_JOBLIB,
    OUT_META,
    OUT_OOF,
    PROTECTED_OUTPUTS,
    XGB_NETD_FROZEN_DATE,
    assert_audit01_invariants,
    main,
)

# ── Cheap tier (always runs) ──────────────────────────────────────────────


def test_script_imports() -> None:
    """All locked module-level symbols are importable.

    The downstream Plan 66-02 + 66-03 scripts depend on these names; a
    rename would surface here first.
    """
    assert callable(main)
    assert callable(assert_audit01_invariants)
    assert isinstance(EXPECTED_XGB_V2_SHA256, str)
    assert isinstance(EXPECTED_META_V2_SHA256, str)
    assert isinstance(OUT_JOBLIB, Path)
    assert isinstance(OUT_META, Path)
    assert isinstance(OUT_OOF, Path)
    assert isinstance(PROTECTED_OUTPUTS, (set, frozenset))
    assert isinstance(NET_V2_COLS, tuple)
    assert isinstance(XGB_NETD_FROZEN_DATE, date)


def test_argparse_help_exits_zero() -> None:
    """``main(["--help"])`` raises ``SystemExit(0)`` (argparse default)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    code = excinfo.value.code
    assert code in (0, None), f"argparse --help exit code = {code!r}"


def test_expected_xgb_v2_sha256_matches_audit01() -> None:
    """The locked AUDIT-01 SHA constant matches the canonical hex (D-10)."""
    assert (
        EXPECTED_XGB_V2_SHA256 == "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"
    )


def test_expected_meta_v2_sha256_matches_audit01() -> None:
    """The meta_v2 AUDIT-01 anchor SHA matches the locked value (D-10)."""
    assert (
        EXPECTED_META_V2_SHA256
        == "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
    )


def test_net_v2_cols_ordering_locked() -> None:
    """NET_V2_COLS ordering is part of the contract — Plan 66-02 / 66-03
    substrate readers assume col[0] == pagerank, col[1] == 2hop-SoS.

    A silent re-ordering would scramble the substrate parquet AND the
    sidecar JSON's feature_columns[90:92] entries.
    """
    assert len(NET_V2_COLS) == 2, f"NET_V2_COLS must have exactly 2 cols, got {len(NET_V2_COLS)}"
    assert NET_V2_COLS == (
        "net_v2_pagerank_at",
        "net_v2_2hop_sos_at",
    ), f"NET_V2_COLS ordering drifted: {NET_V2_COLS!r}"


def test_frozen_date_is_phase_66_phase_start() -> None:
    """XGB_NETD_FROZEN_DATE == date(2026, 6, 6) — Phase 66 phase-start.

    Distinct from Phase 65's date(2026, 6, 4) so a Phase 66 synthetic
    substrate does NOT collide with Phase 65's synthetic substrate by
    frozen date (the cross-phase determinism check the v2.6.1 audit
    chain relies on).
    """
    assert date(2026, 6, 6) == XGB_NETD_FROZEN_DATE, (
        f"XGB_NETD_FROZEN_DATE drifted: {XGB_NETD_FROZEN_DATE!r}; "
        f"expected date(2026, 6, 6) per Phase 66 phase-start"
    )
    # Phase 65 used date(2026, 6, 4) — confirm the distinct-date discipline.
    assert date(2026, 6, 4) != XGB_NETD_FROZEN_DATE, (
        "XGB_NETD_FROZEN_DATE collides with Phase 65's date(2026, 6, 4); "
        "distinct frozen dates required so synthetic substrates are "
        "phase-distinguishable"
    )


def test_protected_outputs_contains_canonical_joblib_and_meta() -> None:
    """Anti-overwrite set guards BOTH canonical joblib AND canonical meta JSON.

    A guard that only protects the joblib (and not the meta JSON) would
    still let an operator clobber the AUDIT-01 hyperparameter record.
    """
    canonical_joblib = (REPO_ROOT / "models" / "xgb_v2.joblib").resolve()
    canonical_meta = (REPO_ROOT / "models" / "xgb_v2_meta.json").resolve()
    assert canonical_joblib in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing canonical joblib: {canonical_joblib}"
    )
    assert canonical_meta in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing canonical meta JSON: {canonical_meta}"
    )


def test_anti_overwrite_guard_fires_on_canonical_joblib(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Phase 66 T-66-01 / Phase 65 T-65-05 / Phase 64 CR-01: ``--output
    models/xgb_v2.joblib`` exits non-zero with a clean stderr message and
    DOES NOT overwrite the canonical artifact.

    We snapshot the canonical SHA before + after to prove byte-identity
    even in the failure path.
    """
    canonical = REPO_ROOT / "models" / "xgb_v2.joblib"
    sha_before = hashlib.sha256(canonical.read_bytes()).hexdigest()

    rc = main(
        [
            "--dry-run",
            "--output",
            str(canonical),
        ]
    )
    assert rc != 0, "anti-overwrite guard should return non-zero exit code"
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "refusing to overwrite" in combined, (
        f"expected 'refusing to overwrite' in stderr/stdout; got err={captured.err!r}"
    )

    sha_after = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert sha_before == sha_after, (
        f"AUDIT-01 violation: canonical xgb_v2.joblib SHA drifted after "
        f"anti-overwrite guard test (before={sha_before[:12]}..., "
        f"after={sha_after[:12]}...)"
    )


def test_anti_overwrite_guard_fires_on_resolved_canonical_meta_via_output_meta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T-66-01 extended: ``--output-meta models/xgb_v2_meta.json`` is also
    blocked. The canonical meta JSON encodes the hyperparameters and is
    AUDIT-01 protected.
    """
    canonical_meta = REPO_ROOT / "models" / "xgb_v2_meta.json"
    sha_before = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()

    # Use a tmp_path joblib output so the joblib guard doesn't fire first.
    rc = main(
        [
            "--dry-run",
            "--output",
            str(tmp_path / "tmp_netd.joblib"),
            "--output-meta",
            str(canonical_meta),
        ]
    )
    assert rc != 0, "anti-overwrite guard should return non-zero for canonical meta"
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "refusing to overwrite" in combined, (
        f"expected 'refusing to overwrite' in output; got err={captured.err!r}"
    )

    sha_after = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()
    assert sha_before == sha_after


def test_audit01_invariant_assertion_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """``assert_audit01_invariants`` raises ``AssertionError`` containing the
    string ``AUDIT-01`` when the locked SHA constant is wrong.

    We monkeypatch the module constant to a sentinel; the asserter MUST
    fail noisily so a future canonical drift is not silently absorbed.
    """
    import retrain_xgb_v2_netd as mod

    monkeypatch.setattr(mod, "EXPECTED_XGB_V2_SHA256", "deadbeef" * 8)
    with pytest.raises(AssertionError, match="AUDIT-01"):
        mod.assert_audit01_invariants()


def test_canonical_joblib_sha_unchanged_at_test_load_time() -> None:
    """Sanity-check the canonical artifact is byte-identical at TEST collection
    time. If this fails, the whole Phase 66 test suite is moot (canonical
    drift already happened and downstream tests cannot trust anything).
    """
    canonical = REPO_ROOT / "models" / "xgb_v2.joblib"
    sha_actual = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert sha_actual == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 drift detected: xgb_v2.joblib SHA={sha_actual[:12]}... "
        f"expected={EXPECTED_XGB_V2_SHA256[:12]}..."
    )


def test_compute_net_v2_columns_returns_pair_aligned_with_rows() -> None:
    """Direct unit test of ``_compute_net_v2_columns``:

    Construct a 3-row corpus with a known graph structure:
      - row 0 (2020-01-01): fighter_a=1 vs fighter_b=2, y=1 → 1 beats 2.
        At as_of=2020-01-01 the graph is empty (no edges with
        earliest_date < 2020-01-01), so fighter 1 is a debutant → both
        cols NaN.
      - row 1 (2020-06-01): fighter_a=2 vs fighter_b=1, y=0 → 1 beats 2
        again. At as_of=2020-06-01 the graph carries one edge 2→1
        (loser_id → winner_id direction per build_fight_graph_v2 line
        ~70). fighter_a=2 IS in the graph; pagerank should be finite;
        2hop-SoS: fighter 2 has zero in-neighbors (only appears as a
        loser/source) → returns None → NaN.
      - row 2 (2020-12-01): fighter_a=1 vs fighter_b=3, y=1 → 1 beats 3.
        At as_of=2020-12-01 the graph carries two edges (2→1 and 2→1).
        fighter_a=1 IS in the graph; pagerank is finite; 2hop-SoS for
        node 1: in-neighbors = {2}; mean PageRank of {2} is finite.

    We only assert structural shape (length, dtype, debutant-NaN
    semantics), NOT specific numerical values — the pagerank computation
    is already covered by tests/unit/features/test_network_v2.py.
    """
    import math

    import retrain_xgb_v2_netd as mod

    event_dates = [
        date(2020, 1, 1),
        date(2020, 6, 1),
        date(2020, 12, 1),
    ]
    fight_records = [
        {
            "fight_id": 100,
            "event_date": event_dates[0],
            "fighter_a_id": 1,
            "fighter_b_id": 2,
            "winner_id": 1,
            "loser_id": 2,
        },
        {
            "fight_id": 101,
            "event_date": event_dates[1],
            "fighter_a_id": 2,
            "fighter_b_id": 1,
            "winner_id": 1,
            "loser_id": 2,
        },
        {
            "fight_id": 102,
            "event_date": event_dates[2],
            "fighter_a_id": 1,
            "fighter_b_id": 3,
            "winner_id": 1,
            "loser_id": 3,
        },
    ]
    y = [1, 0, 1]

    pagerank_col, sos_col = mod._compute_net_v2_columns(
        event_dates=event_dates,
        fight_records=fight_records,
        y=y,
    )

    assert len(pagerank_col) == 3
    assert len(sos_col) == 3
    # Row 0: debutant at as_of=2020-01-01 (graph empty for that as_of_date).
    assert math.isnan(pagerank_col[0]), f"row 0 should be debutant (NaN); got {pagerank_col[0]!r}"
    assert math.isnan(sos_col[0]), f"row 0 sos should be NaN (debutant); got {sos_col[0]!r}"


# ── Heavy tier (GATED by RUN_HEAVY_TESTS=1) ───────────────────────────────


_HEAVY_REASON = (
    "Heavy xgb_v2_netd retrain integration smoke — set RUN_HEAVY_TESTS=1 to "
    "run. Default CI executes only the cheap argparse/import/guard tier."
)


def _heavy_enabled() -> bool:
    return os.environ.get("RUN_HEAVY_TESTS") == "1"


@pytest.fixture(scope="module")
def dry_run_built(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Module-scope fixture: run a single ``--dry-run`` once and reuse its
    artifacts across heavy-tier tests. Returns ``None`` when the heavy
    tier is gated off (so dependent tests skip cleanly).
    """
    if not _heavy_enabled():
        return None
    tmp_dir = tmp_path_factory.mktemp("xgb_v2_netd_dry_run")
    out_joblib = tmp_dir / "test_netd.joblib"
    out_meta = tmp_dir / "test_netd_meta.json"
    out_oof = tmp_dir / "test_netd_oof.parquet"
    rc = main(
        [
            "--dry-run",
            "--output",
            str(out_joblib),
            "--output-meta",
            str(out_meta),
            "--output-oof",
            str(out_oof),
        ]
    )
    assert rc == 0, f"dry-run exited rc={rc}"
    return tmp_dir


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_emits_sibling_artifacts(dry_run_built: Path) -> None:
    """All three sibling artifacts land on disk under ``tmp_path``."""
    assert dry_run_built is not None
    assert (dry_run_built / "test_netd.joblib").exists()
    assert (dry_run_built / "test_netd_meta.json").exists()
    assert (dry_run_built / "test_netd_oof.parquet").exists()


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_meta_json_schema(dry_run_built: Path) -> None:
    """Sidecar JSON has the locked canonical_status + n_features == 92,
    with the 2 NET v2 cols at indices 90, 91, AND decay_base == 0.98.
    """
    assert dry_run_built is not None
    meta = json.loads((dry_run_built / "test_netd_meta.json").read_text())
    assert meta["canonical_status"] == "candidate_sibling_NOT_canonical"
    assert meta["n_features"] == 92
    assert meta["base_model_version"] == "v2"
    assert meta["base_model_sha256"] == EXPECTED_XGB_V2_SHA256
    feature_columns = meta["feature_columns"]
    assert len(feature_columns) == 92
    assert feature_columns[90] == "net_v2_pagerank_at"
    assert feature_columns[91] == "net_v2_2hop_sos_at"
    # D-01 locked DECAY_BASE — audit field.
    assert meta["decay_base"] == 0.98, (
        f"decay_base drifted from D-01 0.98 lock: {meta['decay_base']!r}"
    )
    # nan_handling audit field — recorded as xgb-native for debutants.
    assert meta["nan_handling"] == "xgb_native_missing"
    # Phase identity.
    assert "66-net-feat-v261-03" in meta["phase"]
    assert "D-01" in meta["decision_ids"]


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_oof_parquet_schema(dry_run_built: Path) -> None:
    """OOF parquet has the locked ``{fight_id, oof_prob, event_date}`` schema.

    Plan 66-02 (meta candidate) reads col[0] from this file; the column
    name is part of the contract. Plan 66-03 substrate builder also
    consumes col[0].
    """
    assert dry_run_built is not None
    import pandas as pd

    df = pd.read_parquet(dry_run_built / "test_netd_oof.parquet")
    assert set(df.columns) == {"fight_id", "oof_prob", "event_date"}
    assert len(df) > 0
    # OOF prob is float64 with no nulls (every fold writes via predict_proba).
    assert df["oof_prob"].dtype == "float64"
    assert df["oof_prob"].isna().sum() == 0


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_audit01_invariants_unchanged_after_dry_run(dry_run_built: Path) -> None:
    """After running ``--dry-run``, the canonical SHAs are byte-identical."""
    assert dry_run_built is not None
    sha_xgb = hashlib.sha256((REPO_ROOT / "models" / "xgb_v2.joblib").read_bytes()).hexdigest()
    sha_meta = hashlib.sha256(
        (REPO_ROOT / "models" / "meta" / "meta_v2.joblib").read_bytes()
    ).hexdigest()
    assert sha_xgb == EXPECTED_XGB_V2_SHA256
    assert sha_meta == EXPECTED_META_V2_SHA256


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_emitted_model_accepts_92_col_input(dry_run_built: Path) -> None:
    """``joblib.load(emitted_model).n_features_in_ == 92``.

    The trailing 2 cols carry NET v2 features; if the model was fit on a
    different column count the substrate-drift gate Plan 66-04 would
    misfire.
    """
    assert dry_run_built is not None
    import joblib

    model = joblib.load(dry_run_built / "test_netd.joblib")
    assert hasattr(model, "n_features_in_"), (
        "loaded model is missing n_features_in_ — not a fitted xgb sklearn-wrap?"
    )
    assert model.n_features_in_ == 92, (
        f"loaded model n_features_in_ = {model.n_features_in_}; expected 92"
    )
