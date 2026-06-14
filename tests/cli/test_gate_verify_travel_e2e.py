"""Phase 64 Plan 64-03 (FEAT-V261-01) — end-to-end CLI integration test.

End-to-end integration test for ``ufc gate verify`` against the REAL
TRAVEL candidate (``models/meta/meta_v22_travel.joblib``, 15-wide) and
the REAL canonical (``models/meta/meta_v2.joblib``, 13-wide) using a
freshly-built substrate parquet from Plan 64-02's
``build_substrate_parquet`` materialized into ``tmp_path``.

This is the integration validator that proves the three Wave 1+2 shipped
components interlock correctly:

  - Plan 64-01: width-mismatch guard inside ``verify_candidate_vs_canonical``
    (canonical 13-wide vs candidate 15-wide → ``confound_block`` verdict)
  - Plan 64-02: ``build_substrate_parquet`` produces a Phase 63 D-01-compliant
    15-wide TRAVEL parquet that round-trips through ``load_substrate_snapshot``
  - Phase 63: ``ufc gate verify`` Typer command wires loader → verifier → JSON

The test invokes via Typer's ``CliRunner`` (not subprocess) so failures
surface as Python exceptions instead of opaque exit-code returns.

AUDIT-01 invariant: ``models/meta/meta_v2.joblib`` SHA stays byte-stable
at ``77076d3b…9196`` end-to-end. The verifier reads + ``predict_proba`` only;
the test does NOT write to ``results/`` (all outputs land in ``tmp_path``)
and does NOT modify any model artifact.

Per Plan 64-03 acceptance criteria the three tests share a single verdict
JSON (one ``ufc gate verify`` invocation) — building the substrate matrix
and running the verifier is the costly step; we amortize it across the
three assertions instead of re-invoking three times.

Skip guard: if either real artifact is missing (sparse-checkout / CI without
LFS), the whole module is skipped — both artifacts are tracked in git in
the dev env so 3 PASSED is expected locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli.main import app

# Plan 64-02's builder lives under scripts/ and is not a package; add to
# sys.path so the direct import works. Mirrors Plan 64-02's test pattern
# (tests/unit/scripts/test_build_travel_substrate_v261.py) — the same
# scripts/ sys.path insertion is done there too.
REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_travel_substrate_v261 import (  # type: ignore[import-not-found]  # noqa: E402
    build_substrate_parquet,
)

# ── Module-scope constants (locked artifact paths + AUDIT-01 SHA) ────────

CANDIDATE = REPO_ROOT / "models/meta/meta_v22_travel.joblib"
CANONICAL = REPO_ROOT / "models/meta/meta_v2.joblib"

# AUDIT-01 invariant — canonical meta_v2.joblib SHA. Locked since v2.5.
# Asserting this in the test propagated-through-verifier path proves the
# verifier read the right canonical file and the file was not tampered
# with between this test's start and the verdict JSON emission.
EXPECTED_CANONICAL_SHA = (
    "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
)

# Phase 64 Plan 64-02 locked slice names.
EXPECTED_SLICE_NAMES = {"most_recent_12mo", "most_recent_24mo", "random_15pct"}


# Skip the whole module if either real artifact is missing — defensive
# for CI / sparse-checkout envs. In dev with full checkout both files
# are tracked in git so the skip never triggers locally.
pytestmark = pytest.mark.skipif(
    not (CANDIDATE.exists() and CANONICAL.exists()),
    reason=(
        f"Real TRAVEL artifacts not present "
        f"(CANDIDATE={CANDIDATE}, CANONICAL={CANONICAL}) — "
        f"sparse-checkout / CI without LFS"
    ),
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli_runner() -> CliRunner:
    """Typer CliRunner for invoking ``ufc gate verify`` in-process.

    Module-scope: one instance suffices for all tests in this module.
    """
    return CliRunner()


@pytest.fixture(scope="module")
def built_substrate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize a fresh TRAVEL substrate parquet under a per-module tmp dir.

    Uses Plan 64-02's ``build_substrate_parquet`` with the default
    ``synthetic`` source (DB-free; byte-stable across re-runs via fixed
    RNG seed). The returned path is consumed by the verifier in the
    ``verdict_dict`` fixture downstream.

    WR-02 fix: module-scope so the heavy ``build_substrate_parquet``
    call (~the slowest step in the cascade) runs exactly once per test
    module run, not once per test. Uses ``tmp_path_factory`` (which is
    session-scope-friendly) instead of the function-scope ``tmp_path``
    fixture so the parquet survives across tests sharing this fixture.
    Inputs are deterministic-by-construction (fixed seed +
    ``TRAVEL_SUBSTRATE_REFERENCE_DATE``), so sharing the build across
    tests is safe — no test mutates the parquet.
    """
    tmp_dir = tmp_path_factory.mktemp("built_substrate")
    out = tmp_dir / "travel_substrate.parquet"
    return build_substrate_parquet(out)


@pytest.fixture(scope="module")
def verdict_dict(
    cli_runner: CliRunner,
    built_substrate: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Run ``ufc gate verify`` once; return the parsed verdict JSON.

    WR-02 fix: module-scope so all three tests in this module share the
    same (substrate-build + verifier-run) cycle — exactly once per
    module run, not once per test. The prior comment claimed
    "test session" amortization but the fixture was function-scope, so
    each of the three tests was paying the full cascade cost. This now
    matches the stated intent. The fixture uses ``tmp_path_factory``
    (module/session-safe) instead of ``tmp_path`` (function-only).

    Asserts the basic CLI contract (exit code 0, sidecar file exists)
    so any downstream test failure points at the assertion under test
    rather than at the fixture setup.
    """
    tmp_dir = tmp_path_factory.mktemp("verdict_dict")
    out_json = tmp_dir / "verdict.json"
    argv = [
        "gate",
        "verify",
        "--candidate", str(CANDIDATE),
        "--substrate-parquet", str(built_substrate),
        "--canonical", str(CANONICAL),
        "--out", str(out_json),
    ]
    result = cli_runner.invoke(app, argv)
    if result.exit_code != 0:
        # Diagnostic dump on fixture failure — pytest surfaces this to
        # the user so they can see exactly what the CLI said.
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0, (
        f"ufc gate verify failed: exit={result.exit_code}, "
        f"exception={result.exception!r}"
    )
    assert out_json.exists(), (
        f"Verdict sidecar not written at {out_json}. "
        f"Files under tmp_dir: {list(tmp_dir.rglob('*.json'))}"
    )
    return json.loads(out_json.read_text())


# ── Tests ─────────────────────────────────────────────────────────────────


def test_gate_verify_travel_e2e_emits_confound_block_via_width_guard(
    verdict_dict: dict,
) -> None:
    """The Plan 64-01 width-mismatch guard fires on a TRAVEL vs canonical run.

    Canonical ``meta_v2.joblib`` is 13-wide; candidate
    ``meta_v22_travel.joblib`` is 15-wide. The Plan 64-01 guard
    detects this mismatch via ``n_features_in_`` introspection and
    short-circuits to a ``confound_block`` verdict with:

    - ``verdict == "confound_block"``
    - ``confound_detected is True``
    - ``rationale`` contains literal ``"width_mismatch"``
    - ``raw_baseline_brier_per_slice`` values all ``None`` (raw measurement
      is structurally impossible when widths differ — sentinel for
      "skipped due to width mismatch")
    - all three Phase 64 Plan 64-02 slice names present in the per-slice
      dicts
    """
    # Verdict-literal assertion: width-mismatch guard MUST yield confound_block.
    assert verdict_dict["verdict"] == "confound_block", (
        f"Expected confound_block from width-mismatch guard, "
        f"got {verdict_dict['verdict']!r}. "
        f"rationale={verdict_dict.get('rationale')!r}"
    )

    # Boolean confound flag.
    assert verdict_dict["confound_detected"] is True, (
        f"Expected confound_detected=True, got {verdict_dict['confound_detected']!r}"
    )

    # Rationale carries the width_mismatch marker for operator-debuggability
    # when downstream tooling parses only the top-level rationale field.
    assert "width_mismatch" in verdict_dict["rationale"], (
        f"Expected 'width_mismatch' in rationale, got {verdict_dict['rationale']!r}"
    )

    # Raw baseline Brier per slice — None sentinel for "skipped due to
    # structural width incompatibility". Confirms Plan 64-01 D-01 contract.
    raw = verdict_dict["raw_baseline_brier_per_slice"]
    assert all(v is None for v in raw.values()), (
        f"Expected all raw_baseline_brier_per_slice values to be None "
        f"(structurally impossible to measure raw under width mismatch); "
        f"got {raw!r}"
    )

    # Slice set matches Plan 64-02's locked SLICE_NAMES — confirms the
    # builder's slices propagated through the verifier 1:1.
    assert set(raw.keys()) == EXPECTED_SLICE_NAMES, (
        f"Expected slice set {EXPECTED_SLICE_NAMES}, got {set(raw.keys())}"
    )


def test_gate_verify_travel_e2e_substrate_sha_propagates(
    verdict_dict: dict,
) -> None:
    """The substrate's aggregate SHA propagates into the verdict JSON.

    Phase 55 §6.2: ``substrate_sha`` is a SHA256 hex digest aggregating
    the per-slice substrate SHAs (via ``_aggregate_substrate_sha`` in
    ``gate_verifier.py``). Asserts the field is well-formed: 64-char
    lowercase hex string.
    """
    substrate_sha = verdict_dict["substrate_sha"]
    assert isinstance(substrate_sha, str), (
        f"Expected substrate_sha to be str, got {type(substrate_sha)}"
    )
    assert len(substrate_sha) == 64, (
        f"Expected 64-char SHA256 hex digest, got {len(substrate_sha)} chars: "
        f"{substrate_sha!r}"
    )
    assert all(c in "0123456789abcdef" for c in substrate_sha), (
        f"Expected lowercase hex chars only, got {substrate_sha!r}"
    )


def test_gate_verify_travel_e2e_canonical_sha_matches_audit01(
    verdict_dict: dict,
) -> None:
    """The verdict's ``canonical_sha`` matches the AUDIT-01 anchor.

    AUDIT-01 invariant — ``models/meta/meta_v2.joblib`` SHA256 is locked
    at the ``EXPECTED_CANONICAL_SHA`` constant (defined at module top).
    The verifier computes this hash internally via ``_sha256_hex(canonical)``
    and surfaces it as ``canonical_sha`` in the verdict JSON; asserting
    equality here confirms (a) the verifier read the right file, and (b)
    the file was not tampered with between test start and verdict emission.
    """
    assert verdict_dict["canonical_sha"] == EXPECTED_CANONICAL_SHA, (
        f"AUDIT-01 violation: expected canonical_sha={EXPECTED_CANONICAL_SHA}, "
        f"got {verdict_dict['canonical_sha']!r}. "
        f"Either meta_v2.joblib was modified or the verifier read a "
        f"different file than --canonical pointed to."
    )
