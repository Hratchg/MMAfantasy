"""Phase 65 Plan 65-02 (FEAT-V261-02) — unit tests for the xgb_v2_refv2 retrain script.

Targets ``scripts/retrain_xgb_v2_refv2.py`` (Phase 65 D-03 deliverable).

Cheap-tier tests (always run; <5s budget):
  1. ``test_script_imports`` — module exposes ``main``, ``OUT_JOBLIB``,
     ``OUT_META``, ``OUT_OOF``, ``PROTECTED_OUTPUTS``, and
     ``EXPECTED_XGB_V2_SHA256`` as importable symbols.
  2. ``test_argparse_help_exits_zero`` — ``main(["--help"])`` raises
     ``SystemExit(0)`` (argparse default behaviour).
  3. ``test_expected_xgb_v2_sha256_matches_audit01`` — the locked AUDIT-01
     SHA constant matches the canonical hex.
  4. ``test_protected_outputs_contains_canonical_joblib_and_meta`` —
     ``PROTECTED_OUTPUTS`` contains the resolved paths of ``xgb_v2.joblib``
     and ``xgb_v2_meta.json`` so the anti-overwrite guard cannot be
     bypassed by a stray ``--output`` argv.
  5. ``test_anti_overwrite_guard_fires_on_canonical_joblib`` — invoking
     ``main`` with ``--output models/xgb_v2.joblib`` exits with rc != 0 and
     writes a clean stderr message containing "refusing to overwrite"
     (mirrors Phase 64 CR-01 test pattern).
  6. ``test_anti_overwrite_guard_fires_on_resolved_canonical_meta_via_output_meta`` —
     when ``--output-meta models/xgb_v2_meta.json`` is supplied the guard
     also fires before any side effects.
  7. ``test_audit01_invariant_assertion_message`` — calling
     ``assert_audit01_invariants`` with a monkeypatched wrong-SHA constant
     raises with the AUDIT-01 marker in the message.
  8. ``test_canonical_joblib_sha_unchanged_at_test_load_time`` — checks
     ``models/xgb_v2.joblib`` SHA equals the locked AUDIT-01 constant.
     If this fails, no test should proceed (canonical already corrupted).

Heavy-tier integration tests (GATED by ``RUN_HEAVY_TESTS=1``):
  9. ``test_dry_run_emits_sibling_artifacts`` — full ``--dry-run`` produces
     the joblib + JSON + OOF parquet under ``tmp_path``.
 10. ``test_dry_run_meta_json_schema`` — sidecar JSON has
     ``canonical_status == "candidate_sibling_NOT_canonical"`` and
     ``n_features == 92`` with the 2 REF v2 cols at indices 90, 91.
 11. ``test_dry_run_oof_parquet_schema`` — OOF parquet has the locked
     3-column schema ``{fight_id, oof_prob, event_date}``.
 12. ``test_audit01_invariants_unchanged_after_dry_run`` — canonical
     SHAs remain byte-identical after the dry-run completes.

Heavy tests run when the environment exposes ``RUN_HEAVY_TESTS=1``; otherwise
they are SKIPPED. Default CI runs only the cheap tier (8 tests).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

# ── Module-import helpers ─────────────────────────────────────────────────
#
# ``scripts/`` is not on ``sys.path`` by default for the tests/unit/ tree;
# we make a one-shot ``sys.path`` injection at module import time so the
# direct import below resolves. Mirrors the Phase 64 test pattern in
# ``tests/unit/scripts/test_build_travel_substrate_v261.py``.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from retrain_xgb_v2_refv2 import (
    EXPECTED_XGB_V2_SHA256,
    OUT_JOBLIB,
    OUT_META,
    OUT_OOF,
    PROTECTED_OUTPUTS,
    assert_audit01_invariants,
    main,
)

# ── Cheap tier (always runs) ──────────────────────────────────────────────


def test_script_imports() -> None:
    """All locked module-level symbols are importable.

    The downstream Plan 65-03 + 65-04 scripts depend on these names; a
    rename would surface here first.
    """
    assert callable(main)
    assert callable(assert_audit01_invariants)
    assert isinstance(EXPECTED_XGB_V2_SHA256, str)
    assert isinstance(OUT_JOBLIB, Path)
    assert isinstance(OUT_META, Path)
    assert isinstance(OUT_OOF, Path)
    assert isinstance(PROTECTED_OUTPUTS, (set, frozenset))


def test_argparse_help_exits_zero() -> None:
    """``main(["--help"])`` raises ``SystemExit(0)`` (argparse default)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    # argparse exits with int 0; tolerate both 0 and SystemExit(None)
    code = excinfo.value.code
    assert code in (0, None), f"argparse --help exit code = {code!r}"


def test_expected_xgb_v2_sha256_matches_audit01() -> None:
    """The locked AUDIT-01 SHA constant matches the canonical hex (D-10)."""
    assert (
        EXPECTED_XGB_V2_SHA256 == "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 65 T-65-05 + Phase 64 CR-01 pattern: ``--output models/xgb_v2.joblib``
    exits non-zero with a clean stderr message and DOES NOT overwrite the
    canonical artifact.

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
    assert "refusing to overwrite" in captured.err.lower() or (
        "refusing to overwrite" in captured.out.lower()
    ), f"expected 'refusing to overwrite' in stderr/stdout; got err={captured.err!r}"

    sha_after = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert sha_before == sha_after, (
        f"AUDIT-01 violation: canonical xgb_v2.joblib SHA drifted after "
        f"anti-overwrite guard test (before={sha_before[:12]}..., "
        f"after={sha_after[:12]}...)"
    )


def test_anti_overwrite_guard_fires_on_resolved_canonical_meta_via_output_meta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T-65-05 extended: ``--output-meta models/xgb_v2_meta.json`` is also
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
            str(tmp_path / "tmp_refv2.joblib"),
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
    import retrain_xgb_v2_refv2 as mod

    monkeypatch.setattr(mod, "EXPECTED_XGB_V2_SHA256", "deadbeef" * 8)
    with pytest.raises(AssertionError, match="AUDIT-01"):
        mod.assert_audit01_invariants()


def test_canonical_joblib_sha_unchanged_at_test_load_time() -> None:
    """Sanity-check the canonical artifact is byte-identical at TEST collection
    time. If this fails, the whole Phase 65 test suite is moot (canonical
    drift already happened and downstream tests cannot trust anything).
    """
    canonical = REPO_ROOT / "models" / "xgb_v2.joblib"
    sha_actual = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert sha_actual == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 drift detected: xgb_v2.joblib SHA={sha_actual[:12]}... "
        f"expected={EXPECTED_XGB_V2_SHA256[:12]}..."
    )


# ── CR-01 defensive: pre-fight global_finish_rate discipline ──────────────


def test_compute_ref_v2_columns_global_finish_rate_uses_pre_fight_discipline() -> None:
    """Phase 65 CR-01 fix: ``_compute_ref_v2_columns`` must compute
    ``global_finish_rate`` from a strict pre-fight rolling window — never
    from the full corpus including future fights.

    Construction: 4 chronological rows. Row 0 is at t=2020-01-01 with no
    prior history → its prior must be the uninformative 0.5 fallback
    (n_total_running == 0 at that point); the cohort_history is also empty
    so ``compute_ref_v2_features_shrunk`` returns exactly the prior.
    Therefore ``finish_col[0] == 0.5`` if the fix is correct. Under the
    pre-fix code path (which pre-computed the global rate from ALL 4 rows
    before the loop), ``finish_col[0]`` would have absorbed the full-corpus
    rate of 0.75 (3 finishes / 4 fights) — a future leak.

    The test also confirms that mid-corpus rows see a running rate strictly
    derived from chronologically-prior outcomes.
    """
    from datetime import date as _date

    import retrain_xgb_v2_refv2 as mod

    # 4 fights, all in same cohort (US × unified_post_2017), chronological.
    # Methods: KO/TKO, KO/TKO, Decision - Unanimous, KO/TKO → 3 finish + 1
    # decision. Full-corpus global rate = 0.75; per-row running rate must
    # NOT equal 0.75 for the earliest rows.
    event_dates = [
        _date(2020, 1, 1),
        _date(2020, 2, 1),
        _date(2020, 3, 1),
        _date(2020, 4, 1),
    ]
    event_countries = ["US"] * 4
    scoring_regimes = ["unified_post_2017"] * 4
    y = [1, 1, 0, 1]
    fight_records = [
        {"method": "KO/TKO"},
        {"method": "KO/TKO"},
        {"method": "Decision - Unanimous"},
        {"method": "KO/TKO"},
    ]

    finish_col, decision_col = mod._compute_ref_v2_columns(
        event_dates=event_dates,
        event_countries=event_countries,
        scoring_regimes=scoring_regimes,
        y=y,
        fight_records=fight_records,
    )

    # Row 0: no prior history → prior is 0.5 fallback; cohort_history is
    # also empty so the shrunk rate collapses exactly to the prior.
    assert abs(finish_col[0] - 0.5) < 1e-9, (
        f"CR-01 regression: finish_col[0] = {finish_col[0]!r} != 0.5. "
        f"The pre-fix code computed global_finish_rate from the FULL corpus "
        f"(0.75 here) which leaked future-fight info into the earliest row's "
        f"Bayesian prior."
    )
    assert abs(decision_col[0] - 0.5) < 1e-9, (
        f"CR-01 regression: decision_col[0] = {decision_col[0]!r} != 0.5. "
        f"Same leakage as finish — decision prior is 1 - global_finish_rate."
    )

    # Row 0's outcome (KO/TKO) is a finish → at row 1 the running rate is
    # 1/1 = 1.0. cohort_history at row 1 contains one prior "KO/TKO" so the
    # shrunk rate is (1 + 50*1.0)/(1 + 50) = 51/51 = 1.0.
    assert abs(finish_col[1] - 1.0) < 1e-9, (
        f"CR-01 regression: finish_col[1] expected 1.0 (running rate after "
        f"one prior finish), got {finish_col[1]!r}"
    )


def test_compute_ref_v2_columns_synthetic_matches_live_formula() -> None:
    """Phase 65 CR-01 fix: the synthetic-mode global_finish_rate formula
    must align with the live-mode one. The fix synthesizes a method string
    ("KO/TKO" or "Decision - Unanimous") from the binary y proxy and then
    feeds BOTH modes through ``classify_outcome`` for byte-identical
    accounting.

    Construction: a 4-row corpus where synthetic y == [1, 1, 0, 1] and
    live records carry the equivalent method strings. Both paths must
    produce IDENTICAL (finish_col, decision_col) sequences.
    """
    from datetime import date as _date

    import retrain_xgb_v2_refv2 as mod

    event_dates = [
        _date(2020, 1, 1),
        _date(2020, 2, 1),
        _date(2020, 3, 1),
        _date(2020, 4, 1),
    ]
    event_countries = ["US"] * 4
    scoring_regimes = ["unified_post_2017"] * 4
    y = [1, 1, 0, 1]

    # Live mode: explicit method strings matching the synthesized proxy.
    live_records = [
        {"method": "KO/TKO"},
        {"method": "KO/TKO"},
        {"method": "Decision - Unanimous"},
        {"method": "KO/TKO"},
    ]
    # Synthetic mode: records carry no "method" key — the loop synthesizes
    # the same "KO/TKO" / "Decision - Unanimous" strings from y.
    synth_records: list[dict] = [{} for _ in range(4)]

    live_finish, live_decision = mod._compute_ref_v2_columns(
        event_dates=event_dates,
        event_countries=event_countries,
        scoring_regimes=scoring_regimes,
        y=y,
        fight_records=live_records,
    )
    synth_finish, synth_decision = mod._compute_ref_v2_columns(
        event_dates=event_dates,
        event_countries=event_countries,
        scoring_regimes=scoring_regimes,
        y=y,
        fight_records=synth_records,
    )

    for i, (lv, sv) in enumerate(zip(live_finish, synth_finish, strict=True)):
        assert abs(lv - sv) < 1e-12, (
            f"CR-01 regression: live vs synthetic finish_col diverge at "
            f"row {i}: live={lv!r} synth={sv!r}. The two paths must share "
            f"the same classify_outcome-based accounting."
        )
    for i, (lv, sv) in enumerate(zip(live_decision, synth_decision, strict=True)):
        assert abs(lv - sv) < 1e-12, (
            f"CR-01 regression: live vs synthetic decision_col diverge at "
            f"row {i}: live={lv!r} synth={sv!r}."
        )


# ── Heavy tier (GATED by RUN_HEAVY_TESTS=1) ───────────────────────────────


_HEAVY_REASON = (
    "Heavy xgb retrain integration smoke — set RUN_HEAVY_TESTS=1 to run. "
    "Default CI executes only the cheap argparse/import/guard tier."
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
    tmp_dir = tmp_path_factory.mktemp("xgb_v2_refv2_dry_run")
    out_joblib = tmp_dir / "test_refv2.joblib"
    out_meta = tmp_dir / "test_refv2_meta.json"
    out_oof = tmp_dir / "test_refv2_oof.parquet"
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
    assert (dry_run_built / "test_refv2.joblib").exists()
    assert (dry_run_built / "test_refv2_meta.json").exists()
    assert (dry_run_built / "test_refv2_oof.parquet").exists()


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_meta_json_schema(dry_run_built: Path) -> None:
    """Sidecar JSON has the locked canonical_status + n_features == 92,
    with the 2 REF v2 cols at indices 90, 91.
    """
    assert dry_run_built is not None
    meta = json.loads((dry_run_built / "test_refv2_meta.json").read_text())
    assert meta["canonical_status"] == "candidate_sibling_NOT_canonical"
    assert meta["n_features"] == 92
    assert meta["base_model_version"] == "v2"
    assert meta["base_model_sha256"] == EXPECTED_XGB_V2_SHA256
    feature_columns = meta["feature_columns"]
    assert len(feature_columns) == 92
    assert feature_columns[90] == "ref_v2_finish_rate_shrunk"
    assert feature_columns[91] == "ref_v2_decision_rate_shrunk"


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_dry_run_oof_parquet_schema(dry_run_built: Path) -> None:
    """OOF parquet has the locked ``{fight_id, oof_prob, event_date}`` schema.

    Plan 65-03 (meta candidate) reads col[0] from this file; the column
    name is part of the contract.
    """
    assert dry_run_built is not None
    import pandas as pd

    df = pd.read_parquet(dry_run_built / "test_refv2_oof.parquet")
    assert set(df.columns) == {"fight_id", "oof_prob", "event_date"}
    assert len(df) > 0


@pytest.mark.skipif(not _heavy_enabled(), reason=_HEAVY_REASON)
def test_audit01_invariants_unchanged_after_dry_run(dry_run_built: Path) -> None:
    """After running ``--dry-run``, the canonical SHAs are byte-identical."""
    assert dry_run_built is not None
    sha_xgb = hashlib.sha256((REPO_ROOT / "models" / "xgb_v2.joblib").read_bytes()).hexdigest()
    sha_meta = hashlib.sha256(
        (REPO_ROOT / "models" / "meta" / "meta_v2.joblib").read_bytes()
    ).hexdigest()
    assert sha_xgb == EXPECTED_XGB_V2_SHA256
    assert sha_meta == "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
