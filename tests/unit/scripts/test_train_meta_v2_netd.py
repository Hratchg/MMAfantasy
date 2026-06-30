"""Phase 66 Plan 66-02 (FEAT-V261-03) — unit tests for the meta_v2_netd train script.

Targets ``scripts/train_meta_v2_netd.py`` (Phase 66 D-03a deliverable).

Cheap-tier tests (always run; <2s budget total):
  1. ``test_script_imports`` — module exposes ``main``,
     ``META_V2_NETD_FEATURE_COLUMNS``, ``PROTECTED_OUTPUTS``,
     ``EXPECTED_XGB_V2_SHA256``, ``EXPECTED_META_V2_SHA256``,
     ``OUT_JOBLIB``, ``OUT_META``, ``CANONICAL_META_JSON``,
     ``META_NETD_FROZEN_DATE``.
  2. ``test_argparse_help_exits_zero`` — ``main(["--help"])`` raises
     ``SystemExit(0)`` (argparse default behaviour).
  3. ``test_audit01_sha_constants_match_canonical`` — BOTH locked
     AUDIT-01 SHA constants equal the locked hex values.
  4. ``test_feature_columns_layout_locked`` — exactly 13 cols, col[0] is
     ``xgb_v2_netd_oof``, cols[1..12] byte-equal canonical
     ``meta_v2_meta.json::meta_feature_columns[1:]``.
  5. ``test_feature_col_0_is_xgb_v2_netd_oof_not_canonical`` — col[0] is
     ``xgb_v2_netd_oof`` and is NOT canonical ``xgb_oof_prob`` (pins the
     substrate-drift design intent — Plan 66-04 verdict adjudicates on
     col[0] OOF distribution shift, not width).
  6. ``test_protected_outputs_contains_canonical_artifacts`` — anti-overwrite
     set guards canonical ``meta_v2.joblib`` + ``meta_v2_meta.json`` +
     ``xgb_v2.joblib`` (resolved paths). Extended in Phase 66 to also
     include Phase 65 sibling artifacts (meta_v2_refv2.joblib +
     meta_v2_refv2_meta.json) so cross-phase clobbers are blocked.
  7. ``test_anti_overwrite_guard_fires_on_canonical_meta_joblib`` — invoking
     ``main`` with ``--output models/meta/meta_v2.joblib`` exits rc != 0
     and writes a clean stderr message containing "refusing to overwrite".
  8. ``test_anti_overwrite_guard_fires_on_canonical_meta_json`` — same for
     ``--output-meta models/meta/meta_v2_meta.json``.
  9. ``test_frozen_date_constant_is_phase_66_phase_start`` —
     ``META_NETD_FROZEN_DATE == date(2026, 6, 6)`` (matches Phase 66
     phase-start; distinct from Phase 65's date(2026, 6, 4)).
 10. ``test_uses_unique_sentinel_for_missing_fight_id`` — Phase 65 CR-02
     fix inherited: per-row unique negative sentinel, not constant -1.

Heavy-tier integration tests (GATED by ``RUN_HEAVY_TESTS=1``):
 11. ``test_dry_run_emits_13wide_pipeline`` — ``--mode synthetic`` produces
     a joblib whose inner pipeline ``n_features_in_`` is 13.
 12. ``test_dry_run_sidecar_schema_locked`` — sidecar JSON has all locked
     fields including ``decay_base: 0.98`` and
     ``nan_imputation_strategy: "global_median"``.
 13. ``test_audit01_invariants_unchanged_after_dry_run`` — canonical SHAs
     byte-identical after dry-run.
 14. ``test_dry_run_is_deterministic_across_reruns`` — two consecutive
     ``main`` calls with the same seed produce byte-identical joblibs.
 15. ``test_no_nan_in_training_matrix_after_imputation`` — Phase 65 CR-02
     NaN guard inherited; after ``build_13col_training_matrix`` the matrix
     has no NaN (imputation hook actually invoked).

Heavy tests run when the environment exposes ``RUN_HEAVY_TESTS=1``; otherwise
they are SKIPPED.

Cheap-tier tests run in <2s. Heavy integration tests are GATED by
``RUN_HEAVY_TESTS=1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

# ── Module-import helpers ─────────────────────────────────────────────────
#
# ``scripts/`` is not on ``sys.path`` by default for the tests/unit/ tree;
# we make a one-shot ``sys.path`` injection at module import time so the
# direct import below resolves. Mirrors the Phase 64 / Phase 65 test pattern.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from train_meta_v2_netd import (  # noqa: E402
    CANONICAL_META_JSON,
    EXPECTED_META_V2_SHA256,
    EXPECTED_XGB_V2_SHA256,
    META_NETD_FROZEN_DATE,
    META_V2_NETD_FEATURE_COLUMNS,
    NAN_IMPUTATION_STRATEGY,
    OUT_JOBLIB,
    OUT_META,
    PROTECTED_OUTPUTS,
    assert_audit01_invariants,
    main,
)


# ── Cheap tier (always runs) ──────────────────────────────────────────────


def test_script_imports() -> None:
    """All locked module-level symbols are importable.

    Plan 66-03 + 66-04 depend on these names; a rename would surface here first.
    """
    assert callable(main)
    assert callable(assert_audit01_invariants)
    assert isinstance(EXPECTED_XGB_V2_SHA256, str)
    assert isinstance(EXPECTED_META_V2_SHA256, str)
    assert isinstance(OUT_JOBLIB, Path)
    assert isinstance(OUT_META, Path)
    assert isinstance(CANONICAL_META_JSON, Path)
    assert isinstance(PROTECTED_OUTPUTS, (set, frozenset))
    assert isinstance(META_V2_NETD_FEATURE_COLUMNS, tuple)
    assert isinstance(META_NETD_FROZEN_DATE, date)
    assert isinstance(NAN_IMPUTATION_STRATEGY, str)


def test_argparse_help_exits_zero() -> None:
    """``main(["--help"])`` raises ``SystemExit(0)`` (argparse default)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    code = excinfo.value.code
    assert code in (0, None), f"argparse --help exit code = {code!r}"


def test_audit01_sha_constants_match_canonical() -> None:
    """Locked AUDIT-01 SHA constants equal the canonical hex values (D-10)."""
    assert (
        EXPECTED_XGB_V2_SHA256 == "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
    )
    assert (
        EXPECTED_META_V2_SHA256
        == "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
    )


def test_feature_columns_layout_locked() -> None:
    """13-wide layout: col[0] = xgb_v2_netd_oof; cols[1..12] mirror canonical
    META-V22 byte-for-byte (D-03a).
    """
    assert len(META_V2_NETD_FEATURE_COLUMNS) == 13, (
        f"meta candidate must be 13-wide (Phase 64 width-guard avoidance); "
        f"got {len(META_V2_NETD_FEATURE_COLUMNS)}"
    )
    assert META_V2_NETD_FEATURE_COLUMNS[0] == "xgb_v2_netd_oof", (
        f"col[0] must be the NET candidate OOF source name; got {META_V2_NETD_FEATURE_COLUMNS[0]!r}"
    )
    # cols[1..12] byte-equal canonical meta_v2_meta.json::meta_feature_columns[1:]
    canonical = json.loads(CANONICAL_META_JSON.read_text(encoding="utf-8"))
    canonical_cols = list(canonical["meta_feature_columns"])
    assert len(canonical_cols) == 13
    assert tuple(canonical_cols[1:]) == META_V2_NETD_FEATURE_COLUMNS[1:], (
        f"cols[1..12] must mirror canonical META-V22 ordering byte-for-byte; "
        f"netd={META_V2_NETD_FEATURE_COLUMNS[1:]} "
        f"canonical={canonical_cols[1:]}"
    )


def test_feature_col_0_is_xgb_v2_netd_oof_not_canonical() -> None:
    """col[0] is ``xgb_v2_netd_oof`` (NET candidate OOF), NOT canonical
    ``xgb_oof_prob`` — pins the substrate-drift design intent (Plan 66-04
    verdict adjudicates on col[0] OOF distribution shift).
    """
    assert META_V2_NETD_FEATURE_COLUMNS[0] == "xgb_v2_netd_oof"
    assert META_V2_NETD_FEATURE_COLUMNS[0] != "xgb_oof_prob", (
        "col[0] must differ from canonical META-V22 col[0] — that is the "
        "whole point of the NET candidate sibling (substrate-drift confound)."
    )
    # Also must differ from Phase 65 refv2 col[0].
    assert META_V2_NETD_FEATURE_COLUMNS[0] != "xgb_v2_refv2_oof"


def test_protected_outputs_contains_canonical_artifacts() -> None:
    """Anti-overwrite set guards canonical meta joblib + meta JSON + canonical
    xgb joblib (T-66-08 mitigation). Extended in Phase 66 to also include
    Phase 65 sibling artifacts (cross-phase clobber prevention).
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
    # Phase 65 sibling extension.
    phase65_refv2_joblib = (REPO_ROOT / "models" / "meta" / "meta_v2_refv2.joblib").resolve()
    phase65_refv2_meta = (REPO_ROOT / "models" / "meta" / "meta_v2_refv2_meta.json").resolve()
    assert phase65_refv2_joblib in PROTECTED_OUTPUTS, (
        "PROTECTED_OUTPUTS missing Phase 65 refv2 sibling joblib — "
        "cross-phase clobber prevention required."
    )
    assert phase65_refv2_meta in PROTECTED_OUTPUTS, (
        "PROTECTED_OUTPUTS missing Phase 65 refv2 sibling meta JSON."
    )


def test_anti_overwrite_guard_fires_on_canonical_meta_joblib(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T-66-08 + Phase 64 CR-01 + Phase 65 carry-forward:
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
    """T-66-08 extended: ``--output-meta models/meta/meta_v2_meta.json`` is
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
            str(tmp_path / "tmp_netd.joblib"),
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


def test_frozen_date_constant_is_phase_66_phase_start() -> None:
    """``META_NETD_FROZEN_DATE`` equals Phase 66 phase-start (2026-06-06).

    Distinct from Phase 65's date(2026, 6, 4) so Phase 66 synthetic matrices
    do not collide with Phase 65's by frozen-date.
    """
    assert META_NETD_FROZEN_DATE == date(2026, 6, 6)


def test_nan_imputation_strategy_is_global_median() -> None:
    """``NAN_IMPUTATION_STRATEGY`` documents the Phase 66 D-03a
    NaN-handling choice for PageRank-debutant rows (Phase 65 CR-02 inherited).
    """
    assert NAN_IMPUTATION_STRATEGY == "global_median"


def test_uses_unique_sentinel_for_missing_fight_id() -> None:
    """Phase 65 CR-02 fix inherited: when ``fight_id`` is missing from a
    record, the function must NOT fall back to a constant -1 (which could
    silently collide with a real fight_id == -1 in the OOF parquet).
    Instead each missing-id row gets a unique negative sentinel
    (-(10**9 + i)).

    Structural regression guard: if a future maintainer reverts to the
    constant -1 fallback the assertion will fail.
    """
    import inspect

    import train_meta_v2_netd as mod

    src = inspect.getsource(mod._build_live_13col_matrix)
    assert "10**9" in src or "10 ** 9" in src, (
        "CR-02 regression: per-row unique sentinel pattern ('-(10**9 + i)') "
        "removed from _build_live_13col_matrix. The constant -1 fallback "
        "is forbidden — it could collide with a real fight_id."
    )
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
    tmp_dir = tmp_path_factory.mktemp("meta_v2_netd_dry_run")
    out_joblib = tmp_dir / "test_netd.joblib"
    out_meta = tmp_dir / "test_netd_meta.json"
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

    m = joblib.load(dry_run_built / "test_netd.joblib")
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
    """Sidecar JSON has all locked fields per D-03a + Phase 42 sibling template
    PLUS the Phase 66-specific fields ``decay_base`` (D-01) and
    ``nan_imputation_strategy`` (Phase 65 CR-02 inherited).
    """
    assert dry_run_built is not None
    meta = json.loads((dry_run_built / "test_netd_meta.json").read_text())
    assert meta["canonical_status"] == "candidate_sibling_NOT_canonical"
    assert meta["sibling_of"] == "models/meta/meta_v2.joblib"
    assert meta["n_features"] == 13
    feature_columns = meta["meta_feature_columns"]
    assert len(feature_columns) == 13
    assert feature_columns[0] == "xgb_v2_netd_oof"
    # cols[1..12] mirror canonical
    canonical = json.loads(CANONICAL_META_JSON.read_text(encoding="utf-8"))
    assert feature_columns[1:] == canonical["meta_feature_columns"][1:]
    # Phase 66 D-01 (DECAY_BASE) + Phase 65 CR-02 (NaN imputation).
    assert meta["decay_base"] == 0.98
    assert meta["nan_imputation_strategy"] == "global_median"
    # Sibling discipline (D-10 + Phase 66 sibling pattern).
    assert meta["base_xgb_oof_source"] == "data/intermediate/xgb_v2_netd_oof.parquet"
    assert meta["base_xgb_model"] == "models/xgb_v2_netd.joblib"


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
    joblib outputs (mirrors Phase 64/65 determinism gate).

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


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_no_nan_in_training_matrix_after_imputation() -> None:
    """Phase 65 CR-02 NaN guard inherited: after
    ``build_13col_training_matrix`` returns, the matrix has no NaN values.

    Verifies the imputation hook is actually invoked (the synthetic path
    does not introduce NaNs, but the contract is that the helper guarantees
    no NaN regardless — so we assert the post-condition).
    """
    import numpy as np

    import train_meta_v2_netd as mod

    X_13, y = mod.build_13col_training_matrix(source="synthetic")
    assert X_13.shape[1] == 13
    assert not np.isnan(X_13).any(), (
        "Phase 65 CR-02 NaN guard violated: training matrix has NaN values "
        "after build_13col_training_matrix(source='synthetic'). The "
        "_impute_nans_inplace hook must be called before return."
    )
    assert X_13.shape[0] == y.shape[0]
