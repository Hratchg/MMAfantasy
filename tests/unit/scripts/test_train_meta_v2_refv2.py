"""Phase 65 Plan 65-03 (FEAT-V261-02) — unit tests for the meta_v2_refv2 train script.

Targets ``scripts/train_meta_v2_refv2.py`` (Phase 65 D-03a deliverable).

Cheap-tier tests (always run; <2s budget):
  1. ``test_script_imports`` — module exposes ``main``,
     ``META_V2_REFV2_FEATURE_COLUMNS``, ``PROTECTED_OUTPUTS``,
     ``EXPECTED_XGB_V2_SHA256``, ``EXPECTED_META_V2_SHA256``,
     ``OUT_JOBLIB``, ``OUT_META``, ``CANONICAL_META_JSON``.
  2. ``test_argparse_help_exits_zero`` — ``main(["--help"])`` raises
     ``SystemExit(0)`` (argparse default behaviour).
  3. ``test_audit01_sha_constants_match_canonical`` — BOTH locked
     AUDIT-01 SHA constants equal the locked hex values.
  4. ``test_feature_columns_layout_locked`` — exactly 13 cols, col[0] is
     ``xgb_v2_refv2_oof``, cols[1..12] byte-equal canonical
     ``meta_v2_meta.json::meta_feature_columns[1:]``.
  5. ``test_protected_outputs_contains_canonical_artifacts`` — anti-overwrite
     set guards canonical ``meta_v2.joblib`` + ``meta_v2_meta.json`` +
     ``xgb_v2.joblib`` (resolved paths).
  6. ``test_anti_overwrite_guard_fires_on_canonical_meta_joblib`` — invoking
     ``main`` with ``--output models/meta/meta_v2.joblib`` exits with rc != 0
     and writes a clean stderr message containing "refusing to overwrite".
  7. ``test_anti_overwrite_guard_fires_on_canonical_meta_json`` — same for
     ``--output-meta models/meta/meta_v2_meta.json``.

Heavy-tier integration tests (GATED by ``RUN_HEAVY_TESTS=1``):
  8. ``test_dry_run_emits_13wide_pipeline`` — ``--mode synthetic`` produces a
     joblib whose inner pipeline ``n_features_in_`` is 13 (via the same
     ``_introspect_pipeline_width`` pattern as ``gate_verifier.py``).
  9. ``test_dry_run_sidecar_schema_locked`` — sidecar JSON has
     ``canonical_status == "candidate_sibling_NOT_canonical"``,
     ``sibling_of == "models/meta/meta_v2.joblib"``,
     ``n_features == 13``, ``meta_feature_columns[0] == "xgb_v2_refv2_oof"``.
 10. ``test_audit01_invariants_unchanged_after_dry_run`` — canonical SHAs
     remain byte-identical after the dry-run completes.
 11. ``test_dry_run_is_deterministic_across_reruns`` — two consecutive
     ``main`` calls with the same seed produce byte-identical joblibs
     (mirrors Phase 64 test pattern; only joblib bytes need match — JSON
     drifts on ``trained_at`` timestamp).

Heavy tests run when the environment exposes ``RUN_HEAVY_TESTS=1``; otherwise
they are SKIPPED.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# ── Module-import helpers ─────────────────────────────────────────────────
#
# ``scripts/`` is not on ``sys.path`` by default for the tests/unit/ tree;
# we make a one-shot ``sys.path`` injection at module import time so the
# direct import below resolves. Mirrors the Phase 64 + Plan 65-02 test
# pattern.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from train_meta_v2_refv2 import (
    CANONICAL_META_JSON,
    EXPECTED_META_V2_SHA256,
    EXPECTED_XGB_V2_SHA256,
    META_V2_REFV2_FEATURE_COLUMNS,
    OUT_JOBLIB,
    OUT_META,
    PROTECTED_OUTPUTS,
    assert_audit01_invariants,
    main,
)

# ── Cheap tier (always runs) ──────────────────────────────────────────────


def test_script_imports() -> None:
    """All locked module-level symbols are importable.

    Plan 65-04 + 65-05 depend on these names; a rename would surface here first.
    """
    assert callable(main)
    assert callable(assert_audit01_invariants)
    assert isinstance(EXPECTED_XGB_V2_SHA256, str)
    assert isinstance(EXPECTED_META_V2_SHA256, str)
    assert isinstance(OUT_JOBLIB, Path)
    assert isinstance(OUT_META, Path)
    assert isinstance(CANONICAL_META_JSON, Path)
    assert isinstance(PROTECTED_OUTPUTS, (set, frozenset))
    assert isinstance(META_V2_REFV2_FEATURE_COLUMNS, tuple)


def test_argparse_help_exits_zero() -> None:
    """``main(["--help"])`` raises ``SystemExit(0)`` (argparse default)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    code = excinfo.value.code
    assert code in (0, None), f"argparse --help exit code = {code!r}"


def test_audit01_sha_constants_match_canonical() -> None:
    """Locked AUDIT-01 SHA constants equal the canonical hex values (D-10)."""
    assert (
        EXPECTED_XGB_V2_SHA256 == "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"
    )
    assert (
        EXPECTED_META_V2_SHA256
        == "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
    )


def test_feature_columns_layout_locked() -> None:
    """13-wide layout: col[0] = xgb_v2_refv2_oof; cols[1..12] mirror canonical
    META-V22 byte-for-byte (D-03a).
    """
    assert len(META_V2_REFV2_FEATURE_COLUMNS) == 13, (
        f"meta candidate must be 13-wide (Phase 64 width-guard avoidance); "
        f"got {len(META_V2_REFV2_FEATURE_COLUMNS)}"
    )
    assert META_V2_REFV2_FEATURE_COLUMNS[0] == "xgb_v2_refv2_oof", (
        f"col[0] must be the candidate OOF source name; got {META_V2_REFV2_FEATURE_COLUMNS[0]!r}"
    )
    # cols[1..12] byte-equal canonical meta_v2_meta.json::meta_feature_columns[1:]
    canonical = json.loads(CANONICAL_META_JSON.read_text(encoding="utf-8"))
    canonical_cols = list(canonical["meta_feature_columns"])
    assert len(canonical_cols) == 13
    assert tuple(canonical_cols[1:]) == META_V2_REFV2_FEATURE_COLUMNS[1:], (
        f"cols[1..12] must mirror canonical META-V22 ordering byte-for-byte; "
        f"refv2={META_V2_REFV2_FEATURE_COLUMNS[1:]} "
        f"canonical={canonical_cols[1:]}"
    )


def test_protected_outputs_contains_canonical_artifacts() -> None:
    """Anti-overwrite set guards canonical meta joblib + meta JSON + canonical
    xgb joblib (T-65-10 mitigation).
    """
    canonical_meta_joblib = (REPO_ROOT / "models" / "meta" / "meta_v2.joblib").resolve()
    canonical_meta_json = (REPO_ROOT / "models" / "meta" / "meta_v2_meta.json").resolve()
    canonical_xgb_joblib = (REPO_ROOT / "models" / "xgb_v2.joblib").resolve()
    assert canonical_meta_joblib in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing canonical meta joblib: {canonical_meta_joblib}"
    )
    assert canonical_meta_json in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing canonical meta JSON: {canonical_meta_json}"
    )
    assert canonical_xgb_joblib in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing canonical xgb joblib: {canonical_xgb_joblib}"
    )


def test_anti_overwrite_guard_fires_on_canonical_meta_joblib(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T-65-10 + Phase 64 CR-01 + Plan 65-02 carry-forward:
    ``--output models/meta/meta_v2.joblib`` exits non-zero with a clean stderr
    message and DOES NOT overwrite the canonical artifact.
    """
    canonical = REPO_ROOT / "models" / "meta" / "meta_v2.joblib"
    sha_before = hashlib.sha256(canonical.read_bytes()).hexdigest()

    rc = main(
        [
            "--mode",
            "synthetic",
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
        f"AUDIT-01 violation: canonical meta_v2.joblib SHA drifted after "
        f"anti-overwrite guard test (before={sha_before[:12]}..., "
        f"after={sha_after[:12]}...)"
    )


def test_anti_overwrite_guard_fires_on_canonical_meta_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T-65-10 extended: ``--output-meta models/meta/meta_v2_meta.json`` is
    also blocked; the canonical meta JSON encodes the 13-col layout that
    cols[1..12] mirror.
    """
    canonical_meta = REPO_ROOT / "models" / "meta" / "meta_v2_meta.json"
    sha_before = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()

    rc = main(
        [
            "--mode",
            "synthetic",
            "--output",
            str(tmp_path / "tmp_refv2.joblib"),
            "--output-meta",
            str(canonical_meta),
        ]
    )
    assert rc != 0, "anti-overwrite guard should return non-zero for canonical meta JSON"
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "refusing to overwrite" in combined, (
        f"expected 'refusing to overwrite' in output; got err={captured.err!r}"
    )

    sha_after = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()
    assert sha_before == sha_after


# ── CR-02 defensive: NaN guard + sentinel collision ───────────────────────


def test_build_live_13col_matrix_nan_guard_fires_when_col0_has_nan() -> None:
    """Phase 65 CR-02 fix: ``_build_live_13col_matrix`` must raise
    RuntimeError when col[0] (xgb_v2_refv2_oof) contains any NaN AFTER the
    keep_mask filter is applied.

    Construction: monkeypatch the function-scope dependencies so we drive a
    deliberate mask/nan misalignment — keep_mask is True for one row whose
    corresponding fight_id is NOT in oof_by_fid (so the column has np.nan
    at that position). The guard should fire with the documented message.

    Without the fix the NaN would propagate silently into the downstream
    LogisticRegression fit and produce degenerate coefficients.
    """
    import sys as _sys
    import types

    import numpy as np
    import pandas as pd
    import train_meta_v2_refv2 as mod

    # Stub out the heavy dependencies that _build_live_13col_matrix imports
    # at runtime: train_meta_v22 (DB loader + elo helper) + ufc_prediction
    # session + queries. We replace them with minimal mocks before invoking
    # the function under test.

    class _FakeTM:
        @staticmethod
        def _load_assembled_data_v22() -> tuple[Any, Any, Any, list[dict]]:
            # 2-row v2.2 90-col substrate; both rows carry a fight_id.
            x = np.zeros((2, 90), dtype=np.float64)
            y = np.array([0, 1], dtype=np.int64)
            from datetime import date as _date

            return (
                x,
                y,
                [_date(2024, 1, 1), _date(2024, 2, 1)],
                [
                    {"fight_id": 1001},
                    {"fight_id": 1002},
                ],
            )

        @staticmethod
        def _compute_elo_prob_for_fight(rec: dict, elo_features: Any) -> float:
            return 0.5

    # Inject the fake module under the same name the function imports.
    _sys.modules["train_meta_v22"] = _FakeTM  # type: ignore[assignment]

    # Stub queries/session — _build_live_13col_matrix imports them at
    # function scope. We monkeypatch via direct sys.modules injection.
    fake_queries = types.ModuleType("ufc_prediction.ml.queries")
    fake_queries.load_elo_features = lambda session: None  # type: ignore[attr-defined]
    _sys.modules["ufc_prediction.ml.queries"] = fake_queries

    fake_session = types.ModuleType("ufc_prediction.db.session")

    class _FakeSessionLocal:
        def __call__(self):
            class _S:
                def close(self):
                    pass

            return _S()

    fake_session.SessionLocal = _FakeSessionLocal()  # type: ignore[attr-defined]
    _sys.modules["ufc_prediction.db.session"] = fake_session

    # Drive the mask/nan misalignment: monkeypatch keep_mask construction
    # so both rows pass the mask but only ONE is actually in oof_by_fid.
    # We do this by monkeypatching np.array to force the keep_mask True
    # for a row missing from oof_by_fid. Simpler: provide an OOF parquet
    # with fight_id 1001 present but 1002 missing, then monkeypatch the
    # function's local keep_mask computation by replacing the function
    # with a thin wrapper that intercepts.
    #
    # Cleanest: feed an OOF parquet containing only fight_id 1001 and
    # patch _build_live_13col_matrix's `keep_mask = np.array(...)` line.
    # Since editing inside the function isn't possible, we patch np.array
    # to override the keep_mask construction by detecting its signature.
    oof_df = pd.DataFrame(
        {
            "fight_id": pd.array([1001], dtype="int64"),
            "oof_prob": pd.array([0.42], dtype="float64"),
            "event_date": pd.array(["2024-01-01"], dtype="object"),
        }
    )

    # Patch keep_mask via a stand-in: replace np.array within the module's
    # namespace just for the duration of the call.
    orig_np_array = np.array
    call_counter = {"n": 0}

    def _patched_array(obj: Any, *args: Any, **kwargs: Any) -> Any:
        # The keep_mask line passes dtype=bool. Replace the first such call
        # with an all-True mask to force a misaligned keep_mask.
        if kwargs.get("dtype") is bool and call_counter["n"] == 0:
            call_counter["n"] += 1
            return orig_np_array([True, True], dtype=bool)
        return orig_np_array(obj, *args, **kwargs)

    try:
        # Monkeypatch np.array within the function-import scope.
        import numpy

        numpy.array = _patched_array  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="NaN values in col"):
                mod._build_live_13col_matrix(xgb_refv2_oof_df=oof_df)
        finally:
            numpy.array = orig_np_array  # type: ignore[assignment]
    finally:
        # Clean up the stubs to avoid leaking into other tests.
        for k in (
            "train_meta_v22",
            "ufc_prediction.ml.queries",
            "ufc_prediction.db.session",
        ):
            _sys.modules.pop(k, None)


def test_build_live_13col_matrix_uses_unique_sentinel_for_missing_fight_id() -> None:
    """Phase 65 CR-02 fix: when ``fight_id`` is missing from a record, the
    function must NOT fall back to a constant -1 (which could silently
    collide with a real fight_id == -1 in the OOF parquet). Instead each
    missing-id row gets a unique negative sentinel (-(10**9 + i)) so:

    - the same sentinel never collides with a real fight_id
    - different missing-id rows never collide with each other.

    We exercise the helper by inspecting the source lines (a true
    behavioural test would require a full DB stack). This is a structural
    regression guard: if a future maintainer reverts to the constant -1
    fallback the assertion will fail.
    """
    import inspect

    import train_meta_v2_refv2 as mod

    src = inspect.getsource(mod._build_live_13col_matrix)
    # The fix introduces the literal expression "-(10**9 + i)".
    assert "10**9" in src or "10 ** 9" in src, (
        "CR-02 regression: per-row unique sentinel pattern ('-(10**9 + i)') "
        "removed from _build_live_13col_matrix. The constant -1 fallback "
        "is forbidden — it could collide with a real fight_id."
    )
    # And the constant -1 fallback in the rec.get pattern is gone.
    assert ", -1)" not in src, (
        "CR-02 regression: constant -1 fallback re-introduced in "
        "_build_live_13col_matrix. Use per-row negative sentinels instead."
    )


# ── Heavy tier (GATED by RUN_HEAVY_TESTS=1) ───────────────────────────────


_HEAVY_REASON = (
    "Heavy meta train integration smoke — set RUN_HEAVY_TESTS=1 to run. "
    "Default CI executes only the cheap argparse/import/guard tier."
)


def _heavy_enabled() -> bool:
    return os.environ.get("RUN_HEAVY_TESTS") == "1"


@pytest.fixture(scope="module")
def dry_run_built(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Module-scope fixture: run a single ``--mode synthetic`` once and reuse
    its artifacts across heavy-tier tests. Returns ``None`` when the heavy
    tier is gated off (so dependent tests skip cleanly).
    """
    if not _heavy_enabled():
        return None
    tmp_dir = tmp_path_factory.mktemp("meta_v2_refv2_dry_run")
    out_joblib = tmp_dir / "test_refv2.joblib"
    out_meta = tmp_dir / "test_refv2_meta.json"
    rc = main(
        [
            "--mode",
            "synthetic",
            "--seed",
            "42",
            "--output",
            str(out_joblib),
            "--output-meta",
            str(out_meta),
        ]
    )
    assert rc == 0, f"dry-run exited rc={rc}"
    return tmp_dir


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_emits_13wide_pipeline(dry_run_built: Path) -> None:
    """Inner pipeline ``n_features_in_`` is 13 (mirrors canonical meta_v2's
    13-wide MetaLearnerLogistic-wrapped Pipeline).

    Introspection uses the same two-level pattern as
    ``ufc_prediction.ml.gate_verifier._introspect_pipeline_width`` — direct
    attribute on the loaded object first, then one-level ``.pipeline``
    indirection for the MetaLearnerLogistic case.
    """
    assert dry_run_built is not None
    import joblib

    m = joblib.load(dry_run_built / "test_refv2.joblib")
    # Two-level introspection (gate_verifier pattern).
    width = getattr(m, "n_features_in_", None)
    if width is None:
        inner = getattr(m, "pipeline", None)
        if inner is not None:
            width = getattr(inner, "n_features_in_", None)
    assert width == 13, (
        f"meta candidate width must be 13 (Phase 64 width-guard avoidance); got {width}"
    )


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_sidecar_schema_locked(dry_run_built: Path) -> None:
    """Sidecar JSON has all locked fields per D-03a + Phase 42 sibling template."""
    assert dry_run_built is not None
    meta = json.loads((dry_run_built / "test_refv2_meta.json").read_text())
    assert meta["canonical_status"] == "candidate_sibling_NOT_canonical"
    assert meta["sibling_of"] == "models/meta/meta_v2.joblib"
    assert meta["n_features"] == 13
    feature_columns = meta["meta_feature_columns"]
    assert len(feature_columns) == 13
    assert feature_columns[0] == "xgb_v2_refv2_oof"
    # cols[1..12] mirror canonical
    canonical = json.loads(CANONICAL_META_JSON.read_text(encoding="utf-8"))
    assert feature_columns[1:] == canonical["meta_feature_columns"][1:]


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_audit01_invariants_unchanged_after_dry_run(dry_run_built: Path) -> None:
    """After running ``--mode synthetic``, the canonical SHAs are byte-identical."""
    assert dry_run_built is not None
    sha_xgb = hashlib.sha256((REPO_ROOT / "models" / "xgb_v2.joblib").read_bytes()).hexdigest()
    sha_meta = hashlib.sha256(
        (REPO_ROOT / "models" / "meta" / "meta_v2.joblib").read_bytes()
    ).hexdigest()
    assert sha_xgb == EXPECTED_XGB_V2_SHA256
    assert sha_meta == EXPECTED_META_V2_SHA256


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_is_deterministic_across_reruns(tmp_path: Path) -> None:
    """Two consecutive ``main`` calls with the same seed produce byte-identical
    joblib outputs (mirrors Phase 64 ``test_builder_is_deterministic_across_reruns``).

    Only the joblib bytes need to match — the sidecar JSON has a
    ``trained_at`` timestamp that drifts on each run.
    """
    out1_joblib = tmp_path / "run1.joblib"
    out1_meta = tmp_path / "run1_meta.json"
    out2_joblib = tmp_path / "run2.joblib"
    out2_meta = tmp_path / "run2_meta.json"

    rc1 = main(
        [
            "--mode",
            "synthetic",
            "--seed",
            "42",
            "--output",
            str(out1_joblib),
            "--output-meta",
            str(out1_meta),
        ]
    )
    assert rc1 == 0
    rc2 = main(
        [
            "--mode",
            "synthetic",
            "--seed",
            "42",
            "--output",
            str(out2_joblib),
            "--output-meta",
            str(out2_meta),
        ]
    )
    assert rc2 == 0

    sha1 = hashlib.sha256(out1_joblib.read_bytes()).hexdigest()
    sha2 = hashlib.sha256(out2_joblib.read_bytes()).hexdigest()
    assert sha1 == sha2, (
        f"meta candidate joblib must be deterministic across re-runs at the "
        f"same seed; sha1={sha1[:12]}... sha2={sha2[:12]}..."
    )
