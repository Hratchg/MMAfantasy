"""Phase 72 METH-V261-01 — unit tests for the refit-baseline driver.

Targets ``scripts/refit_meta_v22_v2.6.py`` (Phase 72 D-01..D-02 deliverable
per `gate_methodology_v2.6.md` §7.2).

Cheap-tier tests (always run; <2s budget):
  1. ``test_script_imports`` — module exposes ``main``,
     ``META_V2_REFIT_FEATURE_COLUMNS``, ``PROTECTED_OUTPUTS``,
     ``EXPECTED_XGB_V2_SHA256``, ``EXPECTED_META_V2_SHA256``,
     ``OUT_JOBLIB``, ``OUT_META``, ``CANONICAL_META_JSON``,
     ``METHODOLOGY_SPEC_REF``.
  2. ``test_argparse_help_exits_zero`` — ``main(["--help"])`` raises
     ``SystemExit(0)``.
  3. ``test_audit01_sha_constants_match_canonical`` — locked SHA constants.
  4. ``test_feature_columns_layout_byte_equals_canonical`` — 13-wide layout
     byte-equals canonical ``meta_v2_meta.json::meta_feature_columns``.
  5. ``test_protected_outputs_contains_canonical_artifacts`` — anti-overwrite
     set guards canonical anchors.
  6. ``test_anti_overwrite_guard_fires_on_canonical_meta_joblib`` — invoking
     with ``--output models/meta/meta_v2.joblib`` exits non-zero with a
     "refusing to overwrite" message; canonical SHA unchanged.
  7. ``test_anti_overwrite_guard_fires_on_canonical_meta_json`` — same for
     ``--output-meta models/meta/meta_v2_meta.json``.
  8. ``test_methodology_spec_reference_locked`` — sidecar JSON exposes the
     ``gate_methodology_v2.6.md §7.2`` spec reference (D-02 traceability).
  9. ``test_anti_overwrite_guard_fires_on_canonical_xgb_joblib`` — same for
     ``--output models/xgb_v2.joblib``.

Heavy-tier integration tests (GATED by ``RUN_HEAVY_TESTS=1``):
 10. ``test_synthetic_mode_emits_13wide_pipeline`` — produces a joblib whose
     inner pipeline ``n_features_in_`` is 13.
 11. ``test_synthetic_mode_sidecar_schema_locked`` — sidecar JSON has
     ``canonical_status == "candidate_sibling_NOT_canonical"``,
     ``sibling_of == "models/meta/meta_v2.joblib"``,
     ``methodology_spec == "gate_methodology_v2.6.md §7.2"``,
     ``n_features == 13``.
 12. ``test_audit01_unchanged_after_synthetic_emit`` — canonical SHAs
     remain byte-identical after the synthetic emission completes.
 13. ``test_synthetic_mode_is_deterministic_across_reruns`` — two consecutive
     ``main`` calls with the same seed produce byte-identical joblibs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The driver script's filename contains a dot ("refit_meta_v22_v2.6.py") so
# we cannot use a plain ``import`` statement. Load it via importlib for the
# duration of the test session. This pattern is also documented in Phase 65
# test fixtures.
DRIVER_PATH: Path = SCRIPTS_DIR / "refit_meta_v22_v2.6.py"
_spec = importlib.util.spec_from_file_location("refit_meta_v22_v2_6", DRIVER_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"could not build importlib spec for {DRIVER_PATH}"
)
refit_mod = importlib.util.module_from_spec(_spec)
# Register the loaded module under a clean name so any internal imports
# resolve correctly and so subsequent test files can pick up the same module.
sys.modules["refit_meta_v22_v2_6"] = refit_mod
_spec.loader.exec_module(refit_mod)


# ── Cheap tier (always runs) ──────────────────────────────────────────────


def test_script_imports() -> None:
    """All locked module-level symbols are importable."""
    assert callable(refit_mod.main)
    assert callable(refit_mod.assert_audit01_invariants)
    assert callable(refit_mod.assert_meta_v2_layout)
    assert isinstance(refit_mod.EXPECTED_XGB_V2_SHA256, str)
    assert isinstance(refit_mod.EXPECTED_META_V2_SHA256, str)
    assert isinstance(refit_mod.OUT_JOBLIB, Path)
    assert isinstance(refit_mod.OUT_META, Path)
    assert isinstance(refit_mod.CANONICAL_META_JSON, Path)
    assert isinstance(refit_mod.PROTECTED_OUTPUTS, (set, frozenset))
    assert isinstance(refit_mod.META_V2_REFIT_FEATURE_COLUMNS, tuple)
    assert isinstance(refit_mod.METHODOLOGY_SPEC_REF, str)


def test_argparse_help_exits_zero() -> None:
    """``main(["--help"])`` raises ``SystemExit(0)`` (argparse default)."""
    with pytest.raises(SystemExit) as excinfo:
        refit_mod.main(["--help"])
    code = excinfo.value.code
    assert code in (0, None), f"argparse --help exit code = {code!r}"


def test_audit01_sha_constants_match_canonical() -> None:
    """Locked AUDIT-01 SHA constants equal the canonical hex values."""
    assert (
        refit_mod.EXPECTED_XGB_V2_SHA256
        == "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"
    )
    assert (
        refit_mod.EXPECTED_META_V2_SHA256
        == "e04454267b0bb781709e518b033db223cabd58f61dbb3ffdad3c07cbe12502a8"
    )


def test_feature_columns_layout_byte_equals_canonical() -> None:
    """13-wide layout byte-equals canonical META-V22 (methodology §7.2)."""
    cols = refit_mod.META_V2_REFIT_FEATURE_COLUMNS
    assert len(cols) == 13, (
        f"refit baseline must be 13-wide (Phase 64 width-guard + §7.2 "
        f"byte-equality); got {len(cols)}"
    )
    assert cols[0] == "xgb_oof_prob", (
        f"col[0] must be canonical xgb OOF source name (refit ≠ Plan 65-02 "
        f"candidate); got {cols[0]!r}"
    )
    canonical = json.loads(refit_mod.CANONICAL_META_JSON.read_text(encoding="utf-8"))
    canonical_cols = tuple(canonical["meta_feature_columns"])
    assert len(canonical_cols) == 13
    assert canonical_cols == cols, (
        f"refit layout must byte-equal canonical META-V22; refit={cols} canonical={canonical_cols}"
    )


def test_protected_outputs_contains_canonical_artifacts() -> None:
    """Anti-overwrite set guards canonical meta joblib + meta JSON + xgb joblib."""
    canonical_meta_joblib = (REPO_ROOT / "models" / "meta" / "meta_v2.joblib").resolve()
    canonical_meta_json = (REPO_ROOT / "models" / "meta" / "meta_v2_meta.json").resolve()
    canonical_xgb_joblib = (REPO_ROOT / "models" / "xgb_v2.joblib").resolve()
    assert canonical_meta_joblib in refit_mod.PROTECTED_OUTPUTS
    assert canonical_meta_json in refit_mod.PROTECTED_OUTPUTS
    assert canonical_xgb_joblib in refit_mod.PROTECTED_OUTPUTS


def test_anti_overwrite_guard_fires_on_canonical_meta_joblib(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--output models/meta/meta_v2.joblib`` exits non-zero; canonical unchanged."""
    canonical = REPO_ROOT / "models" / "meta" / "meta_v2.joblib"
    sha_before = hashlib.sha256(canonical.read_bytes()).hexdigest()

    rc = refit_mod.main(
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
        f"AUDIT-01 violation: canonical meta_v2.joblib SHA drifted "
        f"(before={sha_before[:12]}..., after={sha_after[:12]}...)"
    )


def test_anti_overwrite_guard_fires_on_canonical_meta_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--output-meta models/meta/meta_v2_meta.json`` is also blocked."""
    canonical_meta = REPO_ROOT / "models" / "meta" / "meta_v2_meta.json"
    sha_before = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()

    rc = refit_mod.main(
        [
            "--mode",
            "synthetic",
            "--output",
            str(tmp_path / "tmp_refit.joblib"),
            "--output-meta",
            str(canonical_meta),
        ]
    )
    assert rc != 0
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "refusing to overwrite" in combined

    sha_after = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()
    assert sha_before == sha_after


def test_anti_overwrite_guard_fires_on_canonical_xgb_joblib(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--output models/xgb_v2.joblib`` is also blocked (typo defense)."""
    canonical_xgb = REPO_ROOT / "models" / "xgb_v2.joblib"
    sha_before = hashlib.sha256(canonical_xgb.read_bytes()).hexdigest()

    rc = refit_mod.main(
        [
            "--mode",
            "synthetic",
            "--output",
            str(canonical_xgb),
        ]
    )
    assert rc != 0
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "refusing to overwrite" in combined

    sha_after = hashlib.sha256(canonical_xgb.read_bytes()).hexdigest()
    assert sha_before == sha_after


def test_methodology_spec_reference_locked() -> None:
    """``METHODOLOGY_SPEC_REF`` is the locked literal per D-02."""
    assert refit_mod.METHODOLOGY_SPEC_REF == "gate_methodology_v2.6.md §7.2"
    assert refit_mod.METHODOLOGY_VERSION == "v2.6"
    assert refit_mod.MILESTONE_LABEL == "v2.6"


# ── Heavy tier (RUN_HEAVY_TESTS=1) ────────────────────────────────────────


HEAVY = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TESTS") != "1",
    reason="heavy test; set RUN_HEAVY_TESTS=1 to enable",
)


@HEAVY
def test_synthetic_mode_emits_13wide_pipeline(tmp_path: Path) -> None:
    """``--mode synthetic`` emits a joblib whose inner pipeline is 13-wide."""
    import joblib

    out_joblib = tmp_path / "meta_v2_refit_test.joblib"
    out_meta = tmp_path / "meta_v2_refit_test_meta.json"

    rc = refit_mod.main(
        [
            "--mode",
            "synthetic",
            "--output",
            str(out_joblib),
            "--output-meta",
            str(out_meta),
        ]
    )
    assert rc == 0, f"refit driver exited non-zero rc={rc}"
    assert out_joblib.exists(), f"joblib not emitted at {out_joblib}"

    obj = joblib.load(out_joblib)
    # MetaLearnerLogistic wraps a Pipeline as ``self.pipeline`` (Phase 64
    # introspection contract).
    inner_pipeline = obj.pipeline
    n_features = inner_pipeline.n_features_in_
    assert n_features == 13, f"inner pipeline n_features_in_ = {n_features}, expected 13"


@HEAVY
def test_synthetic_mode_sidecar_schema_locked(tmp_path: Path) -> None:
    """Sidecar JSON has the locked canonical_status + methodology_spec + n_features."""
    out_joblib = tmp_path / "meta_v2_refit_test.joblib"
    out_meta = tmp_path / "meta_v2_refit_test_meta.json"

    rc = refit_mod.main(
        [
            "--mode",
            "synthetic",
            "--output",
            str(out_joblib),
            "--output-meta",
            str(out_meta),
        ]
    )
    assert rc == 0
    assert out_meta.exists(), f"sidecar not emitted at {out_meta}"

    sidecar = json.loads(out_meta.read_text(encoding="utf-8"))
    assert sidecar["canonical_status"] == "candidate_sibling_NOT_canonical"
    assert sidecar["sibling_of"] == "models/meta/meta_v2.joblib"
    assert sidecar["methodology_spec"] == "gate_methodology_v2.6.md §7.2"
    assert sidecar["methodology_version"] == "v2.6"
    assert sidecar["n_features"] == 13
    assert sidecar["meta_feature_columns"][0] == "xgb_oof_prob"
    # Layout byte-equal canonical assertion exposed on the sidecar.
    canonical = json.loads(refit_mod.CANONICAL_META_JSON.read_text(encoding="utf-8"))
    assert sidecar["meta_feature_columns"] == canonical["meta_feature_columns"]
    # AUDIT-01 anchor record present + status UNCHANGED.
    assert sidecar["audit_01_invariant"]["status"] == "UNCHANGED"
    assert sidecar["audit_01_invariant"]["xgb_v2_sha"] == refit_mod.EXPECTED_XGB_V2_SHA256
    assert sidecar["audit_01_invariant"]["meta_v2_sha"] == refit_mod.EXPECTED_META_V2_SHA256
    # Provenance fields per D-02 + methodology §7.2 (substrate SHA via
    # training-script SHA stands in for the substrate SHA on synthetic mode).
    assert "training_script_sha256" in sidecar
    assert "refit_joblib_sha256" in sidecar
    assert "trained_at" in sidecar
    assert sidecar["rng_seed"] == refit_mod.DEFAULT_SEED


@HEAVY
def test_audit01_unchanged_after_synthetic_emit(tmp_path: Path) -> None:
    """Canonical xgb + meta SHAs remain byte-identical after synthetic emit."""
    canonical_xgb = REPO_ROOT / "models" / "xgb_v2.joblib"
    canonical_meta = REPO_ROOT / "models" / "meta" / "meta_v2.joblib"
    sha_xgb_before = hashlib.sha256(canonical_xgb.read_bytes()).hexdigest()
    sha_meta_before = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()

    rc = refit_mod.main(
        [
            "--mode",
            "synthetic",
            "--output",
            str(tmp_path / "out.joblib"),
            "--output-meta",
            str(tmp_path / "out_meta.json"),
        ]
    )
    assert rc == 0

    sha_xgb_after = hashlib.sha256(canonical_xgb.read_bytes()).hexdigest()
    sha_meta_after = hashlib.sha256(canonical_meta.read_bytes()).hexdigest()
    assert sha_xgb_before == sha_xgb_after
    assert sha_meta_before == sha_meta_after


@HEAVY
def test_synthetic_mode_is_deterministic_across_reruns(tmp_path: Path) -> None:
    """Two ``main`` calls with the same seed produce byte-identical joblibs."""
    a_joblib = tmp_path / "a.joblib"
    a_meta = tmp_path / "a_meta.json"
    b_joblib = tmp_path / "b.joblib"
    b_meta = tmp_path / "b_meta.json"

    rc_a = refit_mod.main(
        [
            "--mode",
            "synthetic",
            "--seed",
            "42",
            "--output",
            str(a_joblib),
            "--output-meta",
            str(a_meta),
        ]
    )
    rc_b = refit_mod.main(
        [
            "--mode",
            "synthetic",
            "--seed",
            "42",
            "--output",
            str(b_joblib),
            "--output-meta",
            str(b_meta),
        ]
    )
    assert rc_a == 0 and rc_b == 0
    sha_a = hashlib.sha256(a_joblib.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(b_joblib.read_bytes()).hexdigest()
    assert sha_a == sha_b, (
        f"synthetic-mode joblib not deterministic across reruns "
        f"(a={sha_a[:12]}..., b={sha_b[:12]}...)"
    )
