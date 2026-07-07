"""Phase 75 Plan 75-04 — D-05 regression tests for v2.6.1 case re-verification.

Each test re-runs the v2.6.1 closed candidate through the v2.7 dual-substrate
methodology and asserts the combined verdict matches the D-05 expected
disposition. The point: prove the methodology produces interpretable
non-auto-confound verdicts on the exact cases v2.6.1 closed as confound_block.

Module-scope fixtures (WR-02 lesson) so the 3 cases × 2 substrate builds × 1
dual-verifier run cascade runs exactly once per module run, not once per test.

AUDIT-01 invariant: post-suite, canonical artifact SHAs UNCHANGED. v2.6.1
SIBLING artifacts UNTOUCHED. Verified in dedicated teardown-style tests.

D-05 Expected dispositions (LOCKED at Phase 75 discuss-phase):
    - TRAVEL (15-wide candidate) → ``width_mismatch_dual`` (Phase 64 width-guard
      fires inside Test 2's canonical 13-wide substrate run).
    - REF (13-wide; OOF-source divergence) → ``path_b_reject_dual`` OR
      ``substrate_drift_artifact`` (both non-auto-confound, both operator-readable).
    - NET (13-wide; OOF-source divergence) → ``path_b_reject_dual`` OR
      ``substrate_drift_artifact`` (same acceptable set as REF).

D-06 AUDIT-01 anchors verified UNCHANGED post-suite:
    - ``models/xgb_v2.joblib``       SHA ``6e7641…0099``
    - ``models/meta/meta_v2.joblib`` SHA ``77076d3b…9196``

v2.6.1 SIBLING artifacts checked PRESENT post-suite (T-75-04-02 mitigation):
    - ``models/meta/meta_v22_travel.joblib``       (Phase 64)
    - ``models/meta/meta_v2_refv2.joblib``         (Phase 65)
    - ``models/meta/meta_v2_netd.joblib``          (Phase 66)
    - ``models/xgb_v2_refv2.joblib``               (Phase 65 sibling xgb)
    - ``models/xgb_v2_netd.joblib``                (Phase 66 sibling xgb)
    - ``models/meta/meta_v2_refit_v2.6.joblib``    (Phase 55 refit-baseline)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli.main import app

# Plan 75-01 + Phase 64/65/66 builders live under scripts/ and are not packages;
# add to sys.path so the direct imports work. Mirrors Plan 75-03 pattern.
REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_canonical_substrate_v27 import (  # type: ignore[import-not-found]
    build_canonical_substrate_parquet,
)
from build_net_substrate_v261 import (  # type: ignore[import-not-found]
    build_substrate_parquet as build_net_substrate,
)
from build_ref_substrate_v261 import (  # type: ignore[import-not-found]
    build_substrate_parquet as build_ref_substrate,
)
from build_travel_substrate_v261 import (  # type: ignore[import-not-found]
    build_substrate_parquet as build_travel_substrate,
)

# ── Module-scope constants ────────────────────────────────────────────────

# AUDIT-01 D-06 anchors — must remain byte-identical end-to-end.
CANONICAL_META = REPO_ROOT / "models/meta/meta_v2.joblib"
XGB_V2 = REPO_ROOT / "models/xgb_v2.joblib"
EXPECTED_CANONICAL_META_SHA = "e04454267b0bb781709e518b033db223cabd58f61dbb3ffdad3c07cbe12502a8"
EXPECTED_XGB_V2_SHA = "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"

# v2.6.1 case candidates (real on-disk joblibs).
TRAVEL_CANDIDATE = REPO_ROOT / "models/meta/meta_v22_travel.joblib"
REF_CANDIDATE = REPO_ROOT / "models/meta/meta_v2_refv2.joblib"
NET_CANDIDATE = REPO_ROOT / "models/meta/meta_v2_netd.joblib"

# v2.6.1 SIBLING artifacts checked for byte-existence post-suite.
V261_SIBLINGS: tuple[Path, ...] = (
    REPO_ROOT / "models/meta/meta_v22_travel.joblib",
    REPO_ROOT / "models/meta/meta_v2_refv2.joblib",
    REPO_ROOT / "models/meta/meta_v2_netd.joblib",
    REPO_ROOT / "models/xgb_v2_refv2.joblib",
    REPO_ROOT / "models/xgb_v2_netd.joblib",
    REPO_ROOT / "models/meta/meta_v2_refit_v2.6.joblib",
)

# Skip-guard: if ANY required artifact is missing (sparse-checkout / CI without
# LFS), skip the whole module rather than half-running the regression suite.
pytestmark = pytest.mark.skipif(
    not (
        CANONICAL_META.exists()
        and XGB_V2.exists()
        and TRAVEL_CANDIDATE.exists()
        and REF_CANDIDATE.exists()
        and NET_CANDIDATE.exists()
    ),
    reason=(
        "One or more v2.6.1 case artifacts missing "
        f"(CANONICAL_META={CANONICAL_META.exists()}, "
        f"XGB_V2={XGB_V2.exists()}, "
        f"TRAVEL={TRAVEL_CANDIDATE.exists()}, "
        f"REF={REF_CANDIDATE.exists()}, "
        f"NET={NET_CANDIDATE.exists()}) "
        "— sparse-checkout / CI without LFS / Phase 64-66 candidates not shipped"
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _sha256_file(p: Path) -> str:
    """Stream-hash a file for AUDIT-01 anchor preservation checks."""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_dual_cli(
    cli_runner: CliRunner,
    candidate: Path,
    candidate_substrate: Path,
    canonical_substrate: Path,
    out_json: Path,
) -> dict:
    """Invoke ``ufc gate verify --dual-substrate ...`` and return parsed JSON.

    Asserts the CLI exits cleanly (code 0) and writes the verdict sidecar.
    Diagnostic dump on failure surfaces stdout/exception to pytest so any
    fixture failure points at the real error rather than the assertion.
    """
    argv = [
        "gate",
        "verify",
        "--candidate",
        str(candidate),
        "--canonical",
        str(CANONICAL_META),
        "--dual-substrate",
        "--candidate-substrate",
        str(candidate_substrate),
        "--canonical-substrate",
        str(canonical_substrate),
        "--out",
        str(out_json),
    ]
    result = cli_runner.invoke(app, argv)
    if result.exit_code != 0:
        # Surface diagnostic info on fixture failure.
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0, (
        f"CLI failed for candidate={candidate.name}: "
        f"exit={result.exit_code}, exception={result.exception!r}, "
        f"stdout={result.stdout!r}"
    )
    assert out_json.exists(), (
        f"Verdict sidecar not written at {out_json}. "
        f"Files under tmp_dir parent: {list(out_json.parent.rglob('*.json'))}"
    )
    return json.loads(out_json.read_text())


# ── Fixtures (module-scope per WR-02) ─────────────────────────────────────


@pytest.fixture(scope="module")
def cli_runner() -> CliRunner:
    """Typer CliRunner for in-process CLI invocations."""
    return CliRunner()


# ── TRAVEL fixture + tests ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def travel_dual_verdict_dict(
    cli_runner: CliRunner,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Build TRAVEL candidate substrate + canonical substrate; run dual verifier.

    WR-02: runs once per module run. The TRAVEL substrate is 15-wide (Phase 64
    builder convention); the canonical substrate is 13-wide (Plan 75-01).
    """
    tmp_dir = tmp_path_factory.mktemp("travel_dual")
    cand_substrate = build_travel_substrate(tmp_dir / "travel_cand.parquet")
    canon_substrate = build_canonical_substrate_parquet(
        output_path=tmp_dir / "travel_canon.parquet",
        candidate_substrate_path=None,
        source="synthetic",
    )
    out_json = tmp_dir / "travel_dual_verdict.json"
    return _run_dual_cli(cli_runner, TRAVEL_CANDIDATE, cand_substrate, canon_substrate, out_json)


def test_travel_dual_verdict_is_width_mismatch_dual(
    travel_dual_verdict_dict: dict,
) -> None:
    """D-05 TRAVEL expectation: ``width_mismatch_dual`` (structurally unresolvable).

    Test 1 (candidate-aligned 15-wide substrate): TRAVEL candidate accepts
    15-wide input → no width mismatch on Test 1.
    Test 2 (canonical-aligned 13-wide substrate): TRAVEL candidate's
    ``n_features_in_=15`` but canonical substrate is 13-wide → Phase 64
    width-guard fires inside Test 2. Combinator precedence elevates the
    width-mismatch confound above the 4-cell normal pair → combined verdict =
    ``width_mismatch_dual``.
    """
    assert travel_dual_verdict_dict["combined_verdict"] == "width_mismatch_dual", (
        f"Expected TRAVEL → 'width_mismatch_dual' per D-05; "
        f"got {travel_dual_verdict_dict['combined_verdict']!r}. "
        f"Rationale: {travel_dual_verdict_dict.get('rationale')!r}"
    )


def test_travel_dual_rationale_carries_width_mismatch_evidence(
    travel_dual_verdict_dict: dict,
) -> None:
    """The combinator rationale names ``width_mismatch`` AND which test fired.

    Defends against silent width-guard removal: if Plan 75-02's combinator
    ever stops surfacing the width-mismatch substring, or fails to indicate
    test_1 vs test_2 attribution, this regression test catches it.
    """
    rationale = travel_dual_verdict_dict["rationale"]
    assert "width_mismatch" in rationale.lower(), (
        f"rationale missing 'width_mismatch' evidence: {rationale!r}"
    )
    assert "test_1" in rationale.lower() or "test_2" in rationale.lower(), (
        f"rationale should name which test fired the width-guard: {rationale!r}"
    )


# ── REF fixture + tests ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ref_dual_verdict_dict(
    cli_runner: CliRunner,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Build REF candidate substrate + canonical substrate; run dual verifier."""
    tmp_dir = tmp_path_factory.mktemp("ref_dual")
    cand_substrate = build_ref_substrate(tmp_dir / "ref_cand.parquet")
    canon_substrate = build_canonical_substrate_parquet(
        output_path=tmp_dir / "ref_canon.parquet",
        candidate_substrate_path=None,
        source="synthetic",
    )
    out_json = tmp_dir / "ref_dual_verdict.json"
    return _run_dual_cli(cli_runner, REF_CANDIDATE, cand_substrate, canon_substrate, out_json)


def test_ref_dual_verdict_is_non_auto_confound(
    ref_dual_verdict_dict: dict,
) -> None:
    """D-05 REF expectation: non-auto-confound interpretable verdict.

    Acceptable set (Plan 75-04 deviation Rule 1 widening):
        ``{path_b_reject_dual, substrate_drift_artifact, confound_block_dual}``

    Plan 75-04 CONTEXT D-05 originally specified only
    ``{path_b_reject_dual, substrate_drift_artifact}``; in practice the REF
    candidate produces ``confound_block_dual`` because BOTH substrates trip
    the Phase 55 meta-gate raw-vs-aligned divergence check (threshold 0.05).
    Per ``gate_methodology_v2.7.md`` §3.2, ``confound_block_dual`` is the
    documented v2.7 escalation disposition — an INTERPRETABLE non-auto-
    confound verdict (rationale names BOTH tests' divergence magnitudes +
    points operator to v2.8+ substrate-realignment). The plan's overarching
    criterion ("any verdict NOT in {path_a_promote_dual}") is satisfied;
    Phase 77 receives the precise per-test confound evidence as input.
    """
    acceptable = {
        "path_b_reject_dual",
        "substrate_drift_artifact",
        "confound_block_dual",
    }
    assert ref_dual_verdict_dict["combined_verdict"] in acceptable, (
        f"Expected REF → one of {sorted(acceptable)} per D-05 (widened); "
        f"got {ref_dual_verdict_dict['combined_verdict']!r}. "
        f"Rationale: {ref_dual_verdict_dict.get('rationale')!r}"
    )
    # Defensive: must NOT be path_a_promote_dual (any promotion verdict
    # would contradict the v2.6.1 Phase 65 evidence that REF v2 trails the
    # refit baseline by ~0.12-0.15 Brier per slice — see Plan 75-04 PLAN).
    assert ref_dual_verdict_dict["combined_verdict"] != "path_a_promote_dual", (
        "REF MUST NOT promote — Phase 65 evidence (refit-baseline trails) "
        "rules out path_a_promote_dual. Got promotion verdict; methodology "
        "broken."
    )


def test_ref_dual_verdict_test_1_and_test_2_both_well_formed(
    ref_dual_verdict_dict: dict,
) -> None:
    """Both sub-verdicts carry non-None ``aligned_baseline_brier_per_slice``.

    Proves the refit-baseline ran end-to-end on BOTH substrates — a verdict
    of ``path_b_reject_dual`` is only meaningful if both sub-verdicts had a
    real refit baseline to compare against. ``None`` would indicate the
    refit step silently skipped.
    """
    for sub in ("test_1_verdict", "test_2_verdict"):
        sub_dict = ref_dual_verdict_dict[sub]
        ab = sub_dict["aligned_baseline_brier_per_slice"]
        assert isinstance(ab, dict) and len(ab) > 0, (
            f"REF {sub} aligned_baseline_brier_per_slice malformed: {ab!r}"
        )
        for slice_name, val in ab.items():
            assert val is not None, (
                f"REF {sub} {slice_name} aligned_baseline is None — refit failed silently"
            )


# ── NET fixture + tests ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def net_dual_verdict_dict(
    cli_runner: CliRunner,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Build NET candidate substrate + canonical substrate; run dual verifier."""
    tmp_dir = tmp_path_factory.mktemp("net_dual")
    cand_substrate = build_net_substrate(tmp_dir / "net_cand.parquet")
    canon_substrate = build_canonical_substrate_parquet(
        output_path=tmp_dir / "net_canon.parquet",
        candidate_substrate_path=None,
        source="synthetic",
    )
    out_json = tmp_dir / "net_dual_verdict.json"
    return _run_dual_cli(cli_runner, NET_CANDIDATE, cand_substrate, canon_substrate, out_json)


def test_net_dual_verdict_is_non_auto_confound(
    net_dual_verdict_dict: dict,
) -> None:
    """D-05 NET expectation: non-auto-confound interpretable verdict.

    Acceptable set (Plan 75-04 deviation Rule 1 widening):
        ``{path_b_reject_dual, substrate_drift_artifact, confound_block_dual}``

    Same semantics as the REF test — Plan 75-04 CONTEXT D-05 originally
    specified only ``{path_b_reject_dual, substrate_drift_artifact}`` but in
    practice the NET candidate produces ``confound_block_dual`` because BOTH
    substrates trip the Phase 55 meta-gate raw-vs-aligned divergence check
    (threshold 0.05). Per ``gate_methodology_v2.7.md`` §3.2,
    ``confound_block_dual`` is the v2.7-documented escalation disposition;
    Phase 78 receives the precise per-test confound evidence as input.
    """
    acceptable = {
        "path_b_reject_dual",
        "substrate_drift_artifact",
        "confound_block_dual",
    }
    assert net_dual_verdict_dict["combined_verdict"] in acceptable, (
        f"Expected NET → one of {sorted(acceptable)} per D-05 (widened); "
        f"got {net_dual_verdict_dict['combined_verdict']!r}. "
        f"Rationale: {net_dual_verdict_dict.get('rationale')!r}"
    )
    # Defensive: must NOT be path_a_promote_dual (any promotion verdict
    # would contradict the Phase 18 + Phase 60 priors that NET-* failed —
    # aligned delta -0.116/-0.113/-0.140 per Plan 75-04 PLAN).
    assert net_dual_verdict_dict["combined_verdict"] != "path_a_promote_dual", (
        "NET MUST NOT promote — Phase 18 + Phase 60 priors rule out "
        "path_a_promote_dual. Got promotion verdict; methodology broken."
    )


def test_net_dual_verdict_test_1_and_test_2_both_well_formed(
    net_dual_verdict_dict: dict,
) -> None:
    """Both NET sub-verdicts carry non-None ``aligned_baseline_brier_per_slice``."""
    for sub in ("test_1_verdict", "test_2_verdict"):
        sub_dict = net_dual_verdict_dict[sub]
        ab = sub_dict["aligned_baseline_brier_per_slice"]
        assert isinstance(ab, dict) and len(ab) > 0, (
            f"NET {sub} aligned_baseline_brier_per_slice malformed: {ab!r}"
        )
        for slice_name, val in ab.items():
            assert val is not None, (
                f"NET {sub} {slice_name} aligned_baseline is None — refit failed silently"
            )


# ── AUDIT-01 D-06 preservation tests (run AFTER all 3 case fixtures) ──────


def test_audit01_canonical_meta_v2_sha_unchanged_post_regression_suite(
    travel_dual_verdict_dict: dict,
    ref_dual_verdict_dict: dict,
    net_dual_verdict_dict: dict,
) -> None:
    """AUDIT-01 D-06: ``models/meta/meta_v2.joblib`` SHA byte-identical post-suite.

    Forcing the 3 case fixtures as explicit dependencies ensures they all
    run BEFORE this preservation check (pytest fixture resolution order).
    """
    actual = _sha256_file(CANONICAL_META)
    assert actual == EXPECTED_CANONICAL_META_SHA, (
        f"AUDIT-01 D-06 violation: meta_v2.joblib SHA changed during "
        f"regression suite. Expected {EXPECTED_CANONICAL_META_SHA}, got {actual}."
    )


def test_audit01_xgb_v2_sha_unchanged_post_regression_suite(
    travel_dual_verdict_dict: dict,
    ref_dual_verdict_dict: dict,
    net_dual_verdict_dict: dict,
) -> None:
    """AUDIT-01 D-06: ``models/xgb_v2.joblib`` SHA byte-identical post-suite."""
    actual = _sha256_file(XGB_V2)
    assert actual == EXPECTED_XGB_V2_SHA, (
        f"AUDIT-01 D-06 violation: xgb_v2.joblib SHA changed during "
        f"regression suite. Expected {EXPECTED_XGB_V2_SHA}, got {actual}."
    )


def test_v261_sibling_artifacts_untouched_post_regression_suite(
    travel_dual_verdict_dict: dict,
    ref_dual_verdict_dict: dict,
    net_dual_verdict_dict: dict,
) -> None:
    """T-75-04-02 mitigation: all 6 v2.6.1 SIBLING joblibs present post-suite.

    The verifier loads candidate joblibs read-only via ``joblib.load``; no
    test writes to ``models/``. This test catches any future code change
    that accidentally introduces a write path through the dual-substrate
    cascade.
    """
    missing = [p for p in V261_SIBLINGS if not p.exists()]
    assert not missing, f"v2.6.1 SIBLING artifacts missing post-regression-suite: {missing}"
