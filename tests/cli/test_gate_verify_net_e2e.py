"""Phase 66 Plan 66-03 (FEAT-V261-03) — end-to-end CLI integration test.

End-to-end integration test for ``ufc gate verify`` against the REAL
Plan 66-02 candidate (``models/meta/meta_v2_netd.joblib``, 13-wide) and
the REAL canonical (``models/meta/meta_v2.joblib``, 13-wide) using a
freshly-built NET substrate parquet from Plan 66-03's
``build_substrate_parquet`` materialized into ``tmp_path``.

This is the integration validator that proves the three Phase 66 components
interlock correctly with the Phase 63 CLI + Phase 55 verifier:

  - Plan 66-01: ``scripts/retrain_xgb_v2_netd.py`` produced
    ``data/intermediate/xgb_v2_netd_oof.parquet`` (substrate col[0] source)
  - Plan 66-02: ``scripts/train_meta_v2_netd.py`` produced
    ``models/meta/meta_v2_netd.joblib`` (the candidate the verifier scores)
  - Plan 66-03 (this plan): ``scripts/build_net_substrate_v261.py`` produces
    a Phase 63 D-01-compliant 13-wide NET substrate parquet that round-trips
    through ``load_substrate_snapshot``

The test invokes via Typer's ``CliRunner`` (not subprocess) so failures
surface as Python exceptions instead of opaque exit-code returns.

Per Plan 66-03 acceptance: this file pins only the SHAPE of the verdict
JSON, NOT the verdict literal. Plan 66-04 owns the verdict-literal
adjudication (the operator-driven path-conditional close). The high-prior
expectation here is ``confound_block`` because the substrate's col[0] is
``xgb_v2_netd_oof`` (a different distribution from the canonical
``xgb_v2_oof_prob`` that the canonical meta was trained on), but the test
does NOT hardcode that expectation — it only asserts the verdict is one of
the 3 valid enum values, the canonical SHA matches AUDIT-01, and the slice
set matches Plan 66-03's locked slice names.

AUDIT-01 invariant: ``models/meta/meta_v2.joblib`` SHA stays byte-stable
at ``77076d3b…9196`` end-to-end. The verifier reads + ``predict_proba`` only;
the test does NOT write to ``results/`` (all outputs land in ``tmp_path``)
and does NOT modify any model artifact.

WR-02 inheritance (Phase 64 Plan 64-03 + Phase 65 Plan 65-04 review-fix):
the three tests share a single module-scope ``verdict_dict`` fixture so the
heavy build + verifier cascade runs exactly once per module run, not once
per test.

Skip guard: if either real artifact is missing (sparse-checkout / CI without
LFS, or Plan 66-02 not yet shipped), the whole module is skipped — both
artifacts are tracked in git in the dev env so 3 PASSED is expected locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli.main import app

# Plan 66-03's builder lives under scripts/ and is not a package; add to
# sys.path so the direct import works. Mirrors Plan 65-04's pattern.
REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_net_substrate_v261 import (  # type: ignore[import-not-found]
    build_substrate_parquet,
)

# ── Module-scope constants (locked artifact paths + AUDIT-01 SHA) ────────

# Plan 66-02 candidate — the NET v2 META retrain (13-wide, candidate-aligned).
CANDIDATE = REPO_ROOT / "models/meta/meta_v2_netd.joblib"

# Canonical META-V22 — AUDIT-01 anchor.
CANONICAL = REPO_ROOT / "models/meta/meta_v2.joblib"

# AUDIT-01 invariant — canonical meta_v2.joblib SHA. Locked since v2.5.
# Asserting this in the test propagated-through-verifier path proves the
# verifier read the right canonical file and the file was not tampered
# with between this test's start and the verdict JSON emission.
EXPECTED_CANONICAL_SHA = "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"

# Phase 66 Plan 66-03 locked slice names (= Phase 64 Plan 64-02 + Phase 65
# Plan 65-04 locked slice names; all three substrates use the same 3-slice
# convention so per-slice metrics line up across the v2.6.1 verifier audit
# trail).
EXPECTED_SLICE_NAMES = {"most_recent_12mo", "most_recent_24mo", "random_15pct"}

# Phase 55 verdict enum — the three valid verdict literals the verifier
# can emit. Plan 66-04 adjudicates which one is the actual outcome; this
# file only pins that the verdict is one of these three.
VALID_VERDICTS = {"path_a_promote", "path_b_reject", "confound_block"}


# Skip the whole module if either real artifact is missing — defensive for
# CI / sparse-checkout envs (or if Plan 66-02 has not yet shipped). In dev
# with both artifacts present this skip never triggers.
pytestmark = pytest.mark.skipif(
    not (CANDIDATE.exists() and CANONICAL.exists()),
    reason=(
        f"Real NET artifacts not present "
        f"(CANDIDATE={CANDIDATE}, CANONICAL={CANONICAL}) — "
        f"sparse-checkout / CI without LFS / Plan 66-02 not yet shipped"
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
    """Materialize a fresh NET substrate parquet under a per-module tmp dir.

    Uses Plan 66-03's ``build_substrate_parquet`` with the default
    ``synthetic`` source (DB-free; byte-stable across re-runs via fixed
    RNG seed + NET_SUBSTRATE_REFERENCE_DATE). The returned path is consumed
    by the verifier in the ``verdict_dict`` fixture downstream.

    WR-02 fix (Phase 64 + Phase 65 review inheritance): module-scope so the
    heavy ``build_substrate_parquet`` call runs exactly once per test module
    run, not once per test. Uses ``tmp_path_factory`` (session-friendly)
    instead of the function-scope ``tmp_path`` fixture so the parquet
    survives across tests sharing this fixture. Inputs are deterministic-
    by-construction (fixed seed + NET_SUBSTRATE_REFERENCE_DATE), so sharing
    the build across tests is safe — no test mutates the parquet.

    Note (Plan 66-03 Task 3 line 389): we pass ``allow_low_coverage=True``
    explicitly. The e2e test is shape-validation only; the coverage gate's
    correctness is tested at unit-tier in Task 2. The synthetic build does
    NOT trip the gate by construction (debutant proportion is 0% on the
    happy-path), but passing the flag makes the e2e test robust against any
    future change to the synthetic-fixture / OOF-map interaction.
    """
    tmp_dir = tmp_path_factory.mktemp("net_substrate_built")
    out = tmp_dir / "net_substrate.parquet"
    return build_substrate_parquet(out, source="synthetic", allow_low_coverage=True)


@pytest.fixture(scope="module")
def verdict_dict(
    cli_runner: CliRunner,
    built_substrate: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Run ``ufc gate verify`` once; return the parsed verdict JSON.

    WR-02 fix (Phase 64 + Phase 65 review inheritance): module-scope so all
    three tests in this module share the same (substrate-build + verifier-
    run) cycle — exactly once per module run, not once per test. Uses
    ``tmp_path_factory`` (module/session-safe) instead of ``tmp_path``
    (function-only).

    Asserts the basic CLI contract (exit code 0, sidecar file exists) so any
    downstream test failure points at the assertion under test rather than
    at the fixture setup.
    """
    tmp_dir = tmp_path_factory.mktemp("net_verdict_dict")
    out_json = tmp_dir / "verdict.json"
    argv = [
        "gate",
        "verify",
        "--candidate",
        str(CANDIDATE),
        "--substrate-parquet",
        str(built_substrate),
        "--canonical",
        str(CANONICAL),
        "--out",
        str(out_json),
    ]
    result = cli_runner.invoke(app, argv)
    if result.exit_code != 0:
        # Diagnostic dump on fixture failure — pytest surfaces this to
        # the user so they can see exactly what the CLI said.
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0, (
        f"ufc gate verify failed: exit={result.exit_code}, exception={result.exception!r}"
    )
    assert out_json.exists(), (
        f"Verdict sidecar not written at {out_json}. "
        f"Files under tmp_dir: {list(tmp_dir.rglob('*.json'))}"
    )
    return json.loads(out_json.read_text())


# ── Tests ─────────────────────────────────────────────────────────────────


def test_gate_verify_net_e2e_emits_valid_verdict_shape(
    verdict_dict: dict,
) -> None:
    """The verdict JSON carries the Phase 55 verdict-record contract fields.

    Phase 55 §6.1 verdict-record shape (verified against actual emitted
    keys on a real verifier run — Plan 66-03 substrate vs Plan 66-02
    candidate):
      - ``verdict``: one of ``{path_a_promote, path_b_reject, confound_block}``
      - ``confound_detected``: bool
      - ``rationale``: str (non-empty)
      - ``canonical_sha``: 64-char hex string (AUDIT-01 anchor)
      - ``candidate_sha``: 64-char hex string (Plan 66-02 anchor)
      - ``substrate_sha``: aggregate per-slice SHA aggregation
      - ``raw_baseline_brier_per_slice``: dict keyed by slice name
      - ``aligned_baseline_brier_per_slice``: dict keyed by slice name
      - ``aligned_candidate_brier_per_slice``: dict keyed by slice name

    NB: Plan 66-03 pins SHAPE only. Plan 66-04 owns the verdict-literal
    adjudication. High-prior expectation is ``confound_block`` (substrate
    col[0] is xgb_v2_netd_oof, distribution-shifted vs canonical xgb_v2
    OOF that canonical meta_v2 was trained on), but this test does NOT
    hardcode that — verdict ∈ VALID_VERDICTS is the only literal check.
    """
    # Verdict literal — one of the three Phase 55 enum values.
    assert verdict_dict["verdict"] in VALID_VERDICTS, (
        f"Verdict {verdict_dict['verdict']!r} not in {VALID_VERDICTS}. "
        f"rationale={verdict_dict.get('rationale')!r}"
    )

    # Boolean confound flag — populated regardless of verdict literal.
    assert isinstance(verdict_dict["confound_detected"], bool), (
        f"Expected confound_detected to be bool, got {type(verdict_dict['confound_detected'])}"
    )

    # Rationale is a non-empty string for operator-readable audit context.
    assert isinstance(verdict_dict["rationale"], str), (
        f"Expected rationale to be str, got {type(verdict_dict['rationale'])}"
    )
    assert len(verdict_dict["rationale"]) > 0, (
        "rationale is empty — should carry operator-readable context"
    )

    # All three per-slice Brier dicts present (the verdict-determining
    # signal lives in these tables; per Phase 55 §6.2 they may carry None
    # sentinels when measurement is structurally impossible).
    for field in (
        "raw_baseline_brier_per_slice",
        "aligned_baseline_brier_per_slice",
        "aligned_candidate_brier_per_slice",
    ):
        assert field in verdict_dict, (
            f"Verdict JSON missing required field {field!r}; keys: {sorted(verdict_dict.keys())}"
        )
        assert isinstance(verdict_dict[field], dict), (
            f"Expected {field!r} to be dict, got {type(verdict_dict[field])}"
        )

    # Substrate SHA aggregate present + well-formed (64-char lowercase hex).
    substrate_sha = verdict_dict["substrate_sha"]
    assert isinstance(substrate_sha, str)
    assert len(substrate_sha) == 64
    assert all(c in "0123456789abcdef" for c in substrate_sha)


def test_gate_verify_net_e2e_canonical_sha_matches_audit01(
    verdict_dict: dict,
) -> None:
    """The verdict's ``canonical_sha`` matches the AUDIT-01 anchor.

    AUDIT-01 invariant — ``models/meta/meta_v2.joblib`` SHA256 is locked at
    ``EXPECTED_CANONICAL_SHA``. The verifier computes this hash internally
    via ``_sha256_hex(canonical)`` and surfaces it as ``canonical_sha`` in
    the verdict JSON; asserting equality here confirms (a) the verifier
    read the right canonical file, and (b) the file was not tampered with
    between test start and verdict emission.
    """
    assert verdict_dict["canonical_sha"] == EXPECTED_CANONICAL_SHA, (
        f"AUDIT-01 violation: expected canonical_sha={EXPECTED_CANONICAL_SHA}, "
        f"got {verdict_dict['canonical_sha']!r}. "
        f"Either meta_v2.joblib was modified or the verifier read a "
        f"different file than --canonical pointed to."
    )


def test_gate_verify_net_e2e_slice_set_matches_substrate(
    verdict_dict: dict,
) -> None:
    """The verifier's per-slice metric tables cover all three Plan 66-03 slices.

    Asserts ``set(raw_baseline_brier_per_slice.keys()) == EXPECTED_SLICE_NAMES``
    (and equivalents for the aligned baseline + aligned candidate dicts).
    Confirms the substrate's slice set propagates through the verifier 1:1
    — a regression where the verifier silently drops a slice would manifest
    as a key-set mismatch here.
    """
    for field in (
        "raw_baseline_brier_per_slice",
        "aligned_baseline_brier_per_slice",
        "aligned_candidate_brier_per_slice",
    ):
        slice_keys = set(verdict_dict[field].keys())
        assert slice_keys == EXPECTED_SLICE_NAMES, (
            f"Verdict field {field!r} has slice set {slice_keys}, "
            f"expected {EXPECTED_SLICE_NAMES}. The Plan 66-03 substrate "
            f"slices did NOT propagate through the verifier — investigate "
            f"build_substrate_parquet + load_substrate_snapshot interlock."
        )
