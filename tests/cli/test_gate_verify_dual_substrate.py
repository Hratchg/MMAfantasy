"""Phase 75 METH-V27-02 Plan 75-03 — end-to-end CLI integration test.

End-to-end test for ``ufc gate verify --dual-substrate`` against:

  - REAL Plan 65-03 candidate (``models/meta/meta_v2_refv2.joblib``, 13-wide)
  - REAL canonical (``models/meta/meta_v2.joblib``, 13-wide; AUDIT-01 anchor)
  - Plan 65-04 ``build_substrate_parquet`` materialized REF substrate (Test 1
    candidate-aligned)
  - Plan 75-01 ``build_canonical_substrate_parquet`` materialized canonical
    substrate (Test 2 canonical-aligned)

This is the integration validator that proves Plan 75-02's
``verify_candidate_vs_canonical_dual_substrate`` + Plan 75-01's
``build_canonical_substrate_v27.py`` + Phase 63 CLI + Phase 65 builder all
interlock through the new ``--dual-substrate`` CLI surface (D-04).

Per Plan 75-03 acceptance: this file pins the SHAPE of the dual-verdict JSON
+ asserts the combined verdict is one of the 6 D-03 LOCKED literals + asserts
backward-compat of the legacy ``--substrate-parquet`` single-substrate path.
Plan 75-04 owns the verdict-literal adjudication (TRAVEL/REF/NET against the
v2.6.1 cases).

WR-02 inheritance (Phase 64 Plan 64-03 review-fix): the seven tests share a
single module-scope ``dual_verdict_dict`` fixture so the heavy build + dual-
verifier cascade runs exactly once per module run, not once per test.

AUDIT-01 invariant: ``models/meta/meta_v2.joblib`` SHA stays byte-stable at
``77076d3b…9196`` end-to-end. The verifier reads + ``predict_proba`` only;
the test does NOT write to ``results/`` (all outputs land in ``tmp_path``,
except ``test_default_out_path_uses_v27_prefix`` which uses ``monkeypatch.
chdir`` to redirect the ``Path("results")`` default) and does NOT modify any
model artifact.

Skip guard: if either real artifact is missing (sparse-checkout / CI without
LFS, or Plan 65-03 not yet shipped), the whole module is skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli.main import app

# Plan 65-04 + Plan 75-01 builders live under scripts/ and are not packages;
# add to sys.path so the direct imports work. Mirrors Plan 65-04 pattern.
REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_canonical_substrate_v27 import (  # type: ignore[import-not-found]  # noqa: E402
    build_canonical_substrate_parquet,
)
from build_ref_substrate_v261 import (  # type: ignore[import-not-found]  # noqa: E402
    build_substrate_parquet as build_ref_substrate_parquet,
)

# ── Module-private helpers ────────────────────────────────────────────────


def _normalize_cli_output(text: str) -> str:
    """Strip Rich box chars + collapse whitespace for substring assertions.

    Typer/Rich renders ``BadParameter`` errors inside a unicode box that
    wraps text at terminal width — splitting 2-word phrases like
    "mutually exclusive" or "candidate-substrate" across line boundaries
    with ``│`` separators. This helper makes substring matching robust to
    that wrapping by removing the box glyphs and collapsing any run of
    whitespace into a single space.
    """
    cleaned = text.lower()
    for box_char in ("│", "╭", "╮", "╰", "╯", "─"):
        cleaned = cleaned.replace(box_char, " ")
    return " ".join(cleaned.split())


# ── Module-scope constants (locked artifact paths + AUDIT-01 SHA) ────────

# Plan 65-03 candidate — REF v2 META retrain (13-wide, candidate-aligned).
CANDIDATE = REPO_ROOT / "models/meta/meta_v2_refv2.joblib"

# Canonical META-V22 — AUDIT-01 D-06 anchor.
CANONICAL = REPO_ROOT / "models/meta/meta_v2.joblib"

# AUDIT-01 invariant — canonical meta_v2.joblib SHA. Locked since v2.5.
EXPECTED_CANONICAL_SHA = (
    "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
)

# Phase 75 D-03 LOCKED dual-substrate combined-verdict literals.
VALID_DUAL_VERDICTS = {
    "path_a_promote_dual",
    "substrate_drift_artifact",
    "highly_suspect",
    "path_b_reject_dual",
    "confound_block_dual",
    "width_mismatch_dual",
}

# Phase 55 single-substrate verdict literals — used by the backward-compat
# regression test which exercises the legacy ``--substrate-parquet`` path.
VALID_SINGLE_VERDICTS = {"path_a_promote", "path_b_reject", "confound_block"}


# Skip the whole module if either real artifact is missing.
pytestmark = pytest.mark.skipif(
    not (CANDIDATE.exists() and CANONICAL.exists()),
    reason=(
        f"Real REF/canonical artifacts not present "
        f"(CANDIDATE={CANDIDATE}, CANONICAL={CANONICAL}) — "
        f"sparse-checkout / CI without LFS / Plan 65-03 not yet shipped"
    ),
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli_runner() -> CliRunner:
    """Typer CliRunner for invoking ``ufc gate verify`` in-process."""
    return CliRunner()


@pytest.fixture(scope="module")
def built_candidate_substrate(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """REF substrate (candidate-aligned, Test 1). Built via Plan 65-04 helper.

    WR-02: module-scope so the build runs exactly once per module run.
    """
    tmp_dir = tmp_path_factory.mktemp("dual_cand_substrate")
    return build_ref_substrate_parquet(tmp_dir / "ref_substrate.parquet")


@pytest.fixture(scope="module")
def built_canonical_substrate(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Canonical substrate (Test 2). Built via Plan 75-01 helper.

    Uses the synthetic source (``source="synthetic"``) with no paired
    candidate fight-id sidecar — the dual CLI smoke test does not require
    paired fight-ids (Plan 75-04 exercises the paired path against real
    TRAVEL/REF/NET candidates).

    WR-02: module-scope so the build runs exactly once per module run.
    """
    tmp_dir = tmp_path_factory.mktemp("dual_canon_substrate")
    return build_canonical_substrate_parquet(
        output_path=tmp_dir / "canonical_substrate.parquet",
        candidate_substrate_path=None,
        source="synthetic",
    )


@pytest.fixture(scope="module")
def dual_verdict_dict(
    cli_runner: CliRunner,
    built_candidate_substrate: Path,
    built_canonical_substrate: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Run ``ufc gate verify --dual-substrate`` once; return parsed verdict JSON.

    WR-02: module-scope so the heavy dual-verify cascade runs exactly once per
    module run. Asserts the basic CLI contract (exit 0, sidecar exists) so any
    downstream test failure points at the assertion under test rather than at
    fixture setup.
    """
    tmp_dir = tmp_path_factory.mktemp("dual_verdict")
    out_json = tmp_dir / "dual_verdict.json"
    argv = [
        "gate", "verify",
        "--candidate", str(CANDIDATE),
        "--canonical", str(CANONICAL),
        "--dual-substrate",
        "--candidate-substrate", str(built_candidate_substrate),
        "--canonical-substrate", str(built_canonical_substrate),
        "--out", str(out_json),
    ]
    result = cli_runner.invoke(app, argv)
    if result.exit_code != 0:
        # Diagnostic dump on fixture failure — pytest surfaces this to the
        # user so they can see exactly what the CLI said.
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0, (
        f"ufc gate verify --dual-substrate failed: exit={result.exit_code}, "
        f"exception={result.exception!r}, stdout={result.stdout!r}"
    )
    assert out_json.exists(), (
        f"Dual verdict sidecar not written at {out_json}. "
        f"Files under tmp_dir: {list(tmp_dir.rglob('*.json'))}"
    )
    return json.loads(out_json.read_text())


# ── Tests ─────────────────────────────────────────────────────────────────


def test_gate_verify_dual_substrate_emits_v27_verdict_json(
    dual_verdict_dict: dict,
) -> None:
    """The dual-verdict JSON carries the D-03 LOCKED top-level shape.

    Verifies the byte-stable JSON contract from Plan 75-02's
    ``emit_dual_verdict_json`` + ``DualSubstrateGateVerdict.to_dict()``:

      - ``combined_verdict``: one of the 6 D-03 literals
      - ``rationale``: non-empty str
      - ``methodology``: ``"dual_substrate"``
      - ``verifier_version``: ``"v2.7.0"``
      - ``test_1_verdict``: nested Phase 55 sub-verdict dict
      - ``test_2_verdict``: nested Phase 55 sub-verdict dict
    """
    for required in (
        "combined_verdict",
        "rationale",
        "methodology",
        "verifier_version",
        "test_1_verdict",
        "test_2_verdict",
    ):
        assert required in dual_verdict_dict, (
            f"Dual-verdict JSON missing required field {required!r}; "
            f"keys={sorted(dual_verdict_dict)}"
        )

    assert dual_verdict_dict["methodology"] == "dual_substrate", (
        f"methodology={dual_verdict_dict['methodology']!r} != 'dual_substrate'"
    )
    assert dual_verdict_dict["verifier_version"] == "v2.7.0", (
        f"verifier_version={dual_verdict_dict['verifier_version']!r} != 'v2.7.0'"
    )
    assert isinstance(dual_verdict_dict["rationale"], str)
    assert len(dual_verdict_dict["rationale"]) > 0, (
        "rationale is empty — combinator should produce non-empty rationale"
    )


def test_combined_verdict_is_one_of_six_locked_literals(
    dual_verdict_dict: dict,
) -> None:
    """``combined_verdict`` is exactly one of the 6 D-03 LOCKED literals.

    Plan 75-03 pins MEMBERSHIP only (the structural assertion that the cascade
    + combinator both ran end-to-end). Plan 75-04 adjudicates the specific
    literal for REF/NET/TRAVEL against the v2.6.1 cases.
    """
    assert dual_verdict_dict["combined_verdict"] in VALID_DUAL_VERDICTS, (
        f"unexpected combined_verdict={dual_verdict_dict['combined_verdict']!r}; "
        f"valid={sorted(VALID_DUAL_VERDICTS)}; "
        f"rationale={dual_verdict_dict.get('rationale')!r}"
    )


def test_sub_verdicts_carry_canonical_sha_audit01_anchor(
    dual_verdict_dict: dict,
) -> None:
    """Both sub-verdicts surface the AUDIT-01 D-06 canonical SHA.

    The Plan 75-02 verifier wraps two ``verify_candidate_vs_canonical`` calls
    against the same ``--canonical`` Pipeline. Both sub-verdicts MUST therefore
    carry the same canonical_sha = AUDIT-01 anchor. A mismatch implies either
    the canonical was modified mid-run (impossible by file lock) or the
    verifier read a different file than --canonical pointed to.
    """
    for sub_label in ("test_1_verdict", "test_2_verdict"):
        actual = dual_verdict_dict[sub_label]["canonical_sha"]
        assert actual == EXPECTED_CANONICAL_SHA, (
            f"AUDIT-01 violation in {sub_label}: "
            f"expected canonical_sha={EXPECTED_CANONICAL_SHA}, got {actual!r}"
        )


def test_sub_verdicts_use_distinct_substrate_shas(
    dual_verdict_dict: dict,
) -> None:
    """Test 1 and Test 2 must use DIFFERENT substrate SHAs.

    The dual-test methodology's whole point is to compare candidate vs canonical
    on TWO different substrates (candidate-aligned col[0] vs canonical-aligned
    col[0]). If the two sub-verdicts report identical substrate_sha, the CLI
    accidentally wired the same parquet to both flags — defeating the purpose.
    """
    t1_sha = dual_verdict_dict["test_1_verdict"]["substrate_sha"]
    t2_sha = dual_verdict_dict["test_2_verdict"]["substrate_sha"]
    assert t1_sha != t2_sha, (
        f"Test 1 and Test 2 substrate_sha are identical ({t1_sha!r}) — "
        f"the dual-test methodology requires distinct substrates."
    )


def test_mutually_exclusive_substrate_parquet_and_dual_substrate_fails(
    cli_runner: CliRunner,
    built_candidate_substrate: Path,
    built_canonical_substrate: Path,
    tmp_path: Path,
) -> None:
    """Passing BOTH ``--substrate-parquet`` and ``--dual-substrate`` is an error.

    Per T-75-03-04 mitigation: surface ``typer.BadParameter`` (exit != 0) with
    a clean message instead of allowing ambiguous behavior.
    """
    argv = [
        "gate", "verify",
        "--candidate", str(CANDIDATE),
        "--canonical", str(CANONICAL),
        "--substrate-parquet", str(built_candidate_substrate),
        "--dual-substrate",
        "--candidate-substrate", str(built_candidate_substrate),
        "--canonical-substrate", str(built_canonical_substrate),
        "--out", str(tmp_path / "v.json"),
    ]
    result = cli_runner.invoke(app, argv)
    assert result.exit_code != 0, (
        f"Expected exit != 0 for mutually-exclusive flags; got 0. "
        f"stdout={result.stdout!r}"
    )
    # ``result.output`` mixes stdout + typer's stderr-style BadParameter
    # rendering. Normalize Rich box wrapping before substring matching.
    combined = _normalize_cli_output(
        result.output + str(result.exception or "")
    )
    assert "mutually exclusive" in combined, (
        f"Error message did not mention 'mutually exclusive'. "
        f"Output: {result.output!r}, exc={result.exception!r}"
    )


def test_missing_both_substrate_flags_fails(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Providing NEITHER ``--substrate-parquet`` nor ``--dual-substrate`` fails.

    Per D-04 dispatch logic: one of the two flag sets is required; absence is
    a typer.BadParameter (exit != 0) with a helpful message.
    """
    argv = [
        "gate", "verify",
        "--candidate", str(CANDIDATE),
        "--canonical", str(CANONICAL),
        "--out", str(tmp_path / "v.json"),
    ]
    result = cli_runner.invoke(app, argv)
    assert result.exit_code != 0, (
        f"Expected exit != 0 when no substrate flag provided; got 0. "
        f"stdout={result.stdout!r}"
    )
    # ``result.output`` mixes stdout + typer's stderr-style BadParameter
    # rendering, which is where typer prints "Invalid value: …" boxes.
    combined = _normalize_cli_output(
        result.output + str(result.exception or "")
    )
    assert "required" in combined or "substrate" in combined, (
        f"Error message did not mention 'required'/'substrate'. "
        f"Output: {result.output!r}, exc={result.exception!r}"
    )


def test_dual_substrate_without_candidate_substrate_fails(
    cli_runner: CliRunner,
    built_canonical_substrate: Path,
    tmp_path: Path,
) -> None:
    """``--dual-substrate`` without ``--candidate-substrate`` is rejected.

    Per D-04 dispatch logic: ``--dual-substrate`` REQUIRES both
    ``--candidate-substrate`` AND ``--canonical-substrate``. Missing either
    is a typer.BadParameter (exit != 0).
    """
    argv = [
        "gate", "verify",
        "--candidate", str(CANDIDATE),
        "--canonical", str(CANONICAL),
        "--dual-substrate",
        "--canonical-substrate", str(built_canonical_substrate),
        "--out", str(tmp_path / "v.json"),
    ]
    result = cli_runner.invoke(app, argv)
    assert result.exit_code != 0, (
        f"Expected exit != 0 when --candidate-substrate missing; got 0. "
        f"stdout={result.stdout!r}"
    )
    # ``result.output`` mixes stdout + typer's stderr-style BadParameter
    # rendering, which is where typer prints "Invalid value: …" boxes.
    combined = _normalize_cli_output(
        result.output + str(result.exception or "")
    )
    assert "candidate-substrate" in combined or "required" in combined, (
        f"Error message did not mention 'candidate-substrate'/'required'. "
        f"Output: {result.output!r}, exc={result.exception!r}"
    )


def test_default_out_path_uses_v27_prefix(
    cli_runner: CliRunner,
    built_candidate_substrate: Path,
    built_canonical_substrate: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--out``, default path uses ``v27`` prefix (not ``v26``).

    Verifies the default-path scheme ``results/gate_verdict_v27_<candidate_stem>
    .json`` introduced by the new dual-substrate dispatch. The legacy single-
    substrate default ``results/gate_verdict_v26_<candidate_stem>.json`` is
    UNCHANGED (covered by ``test_existing_single_substrate_path_still_works``).

    ``monkeypatch.chdir(tmp_path)`` redirects the ``Path("results")`` relative
    default into the test's tmp_path so the test does not pollute the repo's
    ``results/`` directory.
    """
    monkeypatch.chdir(tmp_path)
    argv = [
        "gate", "verify",
        "--candidate", str(CANDIDATE),
        "--canonical", str(CANONICAL),
        "--dual-substrate",
        "--candidate-substrate", str(built_candidate_substrate),
        "--canonical-substrate", str(built_canonical_substrate),
    ]
    result = cli_runner.invoke(app, argv)
    if result.exit_code != 0:
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0, (
        f"Default-path invocation failed: exit={result.exit_code}, "
        f"exc={result.exception!r}"
    )
    expected = tmp_path / "results" / f"gate_verdict_v27_{CANDIDATE.stem}.json"
    assert expected.exists(), (
        f"Default v27 sidecar missing at {expected}. "
        f"Files under tmp_path: {list(tmp_path.rglob('*.json'))}"
    )


def test_existing_single_substrate_path_still_works(
    cli_runner: CliRunner,
    built_candidate_substrate: Path,
    tmp_path: Path,
) -> None:
    """Regression: ``--substrate-parquet`` legacy path is BYTE-IDENTICAL.

    D-04 backward-compat invariant: every existing ``ufc gate verify`` invocation
    must produce the same v26-shaped verdict JSON as before Plan 75-03. This test
    invokes the legacy single-substrate path and asserts:

      1. Exit code 0
      2. Verdict JSON has the v26 ``verdict`` field (NOT the v27 ``combined_verdict``)
      3. Verdict is one of the 3 Phase 55 literals
    """
    out_json = tmp_path / "single_verdict.json"
    argv = [
        "gate", "verify",
        "--candidate", str(CANDIDATE),
        "--canonical", str(CANONICAL),
        "--substrate-parquet", str(built_candidate_substrate),
        "--out", str(out_json),
    ]
    result = cli_runner.invoke(app, argv)
    if result.exit_code != 0:
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0, (
        f"Legacy single-substrate path failed: exit={result.exit_code}, "
        f"exc={result.exception!r}"
    )
    parsed = json.loads(out_json.read_text())
    # v26 single-substrate JSON shape — has 'verdict', NOT 'combined_verdict'.
    assert "verdict" in parsed, (
        f"v26 verdict JSON missing 'verdict' field; keys={sorted(parsed)}"
    )
    assert "combined_verdict" not in parsed, (
        f"v26 verdict JSON should NOT have 'combined_verdict' (that's v27). "
        f"keys={sorted(parsed)}"
    )
    assert parsed["verdict"] in VALID_SINGLE_VERDICTS, (
        f"unexpected v26 verdict={parsed['verdict']!r}; "
        f"valid={sorted(VALID_SINGLE_VERDICTS)}"
    )
