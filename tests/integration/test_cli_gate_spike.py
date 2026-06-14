"""Phase 31 / GATE-V23-01,02,03 — CLI gate-spike integration tests (Wave 0).

Three integration tests covering:

1. ``test_cli_dispatch_synthetic`` — full CliRunner dispatch through Path A
   against fully-stubbed variance harness; asserts contract emit + END SHA
   artifact in isolated tmp dir. SKIPPED if Typer CliRunner cannot resolve
   the embedded subcommand decorators on the current install (rare).

2. ``test_cli_halt_on_below_floor`` — injects synthetic below-floor variance
   via ``_compute_v23_variance_dict`` monkeypatch; asserts:
     - exit code 3 (SystemExit on Path D)
     - 31-HALT-AND-DECIDE.md emitted in isolated PHASE_31_DIR
     - frontmatter contains phase + artifact + operator_decision_required
     - body contains the 2 required headers + all 3 Path D outcome-path IDs
     - END SHA .txt artifact NOT written (Pitfall 5)
     - gate_contract_v2.3.json NOT written (Path D halts contract emit)

3. ``test_cli_dispatch_preserves_v21_v22_contracts`` — SHA-256s
   .planning/gate_contract.json + .planning/gate_contract_v2.2.json
   before + after Path A dispatch; asserts byte-identity (GATE-V23-02
   audit-lineage regression).

All tests use ``tmp_path`` + monkeypatch to avoid live DB / live filesystem
writes outside the test scope.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _stubbed_variance_above_floor() -> tuple[dict, list[str]]:
    """Synthetic variance dict; all slices clear 0.70 (Path A)."""
    return (
        {
            "most_recent_12mo": {
                "median_brier": 0.18, "median_acc": 0.745,
                "seed_std_brier": 0.003, "seed_std_acc": 0.004,
                "bootstrap_ci_half_brier": 0.0035, "bootstrap_ci_half_acc": 0.006,
                "std_brier_used": 0.0035, "std_acc_used": 0.006,
            },
            "most_recent_24mo": {
                "median_brier": 0.19, "median_acc": 0.735,
                "seed_std_brier": 0.003, "seed_std_acc": 0.004,
                "bootstrap_ci_half_brier": 0.0035, "bootstrap_ci_half_acc": 0.006,
                "std_brier_used": 0.0035, "std_acc_used": 0.006,
            },
            "random_15pct": {
                "median_brier": 0.17, "median_acc": 0.755,
                "seed_std_brier": 0.003, "seed_std_acc": 0.004,
                "bootstrap_ci_half_brier": 0.0035, "bootstrap_ci_half_acc": 0.006,
                "std_brier_used": 0.0035, "std_acc_used": 0.006,
            },
        },
        [],
    )


def _stubbed_variance_below_floor() -> tuple[dict, list[str]]:
    """Synthetic variance dict; random_15pct formula_output < 0.70 (Path D)."""
    return (
        {
            "most_recent_12mo": {
                "median_brier": 0.21, "median_acc": 0.745,
                "seed_std_brier": 0.003, "seed_std_acc": 0.004,
                "bootstrap_ci_half_brier": 0.004, "bootstrap_ci_half_acc": 0.006,
                "std_brier_used": 0.004, "std_acc_used": 0.006,
            },
            "most_recent_24mo": {
                "median_brier": 0.22, "median_acc": 0.735,
                "seed_std_brier": 0.003, "seed_std_acc": 0.004,
                "bootstrap_ci_half_brier": 0.004, "bootstrap_ci_half_acc": 0.006,
                "std_brier_used": 0.004, "std_acc_used": 0.006,
            },
            "random_15pct": {
                # median_acc 0.65 + std_acc_used 0.01 = 0.66 < 0.70 BREACH
                "median_brier": 0.24, "median_acc": 0.65,
                "seed_std_brier": 0.003, "seed_std_acc": 0.005,
                "bootstrap_ci_half_brier": 0.004, "bootstrap_ci_half_acc": 0.010,
                "std_brier_used": 0.004, "std_acc_used": 0.010,
            },
        },
        [],
    )


def _patch_path_a_environment(monkeypatch, tmp_path, *, below_floor: bool = False):
    """Monkeypatch the CLI dispatch end-to-end against synthetic variance.

    - PROJECT.md: stub _read_project_md to return text containing D-24 row
    - spike_main: stub to no-op return 0
    - _assert_xgb_v2_sha: stub to return baseline SHA
    - _compute_v23_variance_dict: stub with above-floor or below-floor synthetic
    - PHASE_31_DIR: route artifact writes to tmp_path/phase_31
    """
    from ufc_prediction.cli import predict as pred_mod

    phase_31_isolated = tmp_path / "phase_31"
    phase_31_isolated.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        pred_mod,
        "_read_project_md",
        lambda: "| **D-24(v2.3, GATE)** | Phase 31 GATE row |",
        raising=False,
    )
    monkeypatch.setattr(
        pred_mod, "_spike_main", lambda argv: 0, raising=False,
    )
    monkeypatch.setattr(
        pred_mod,
        "_assert_xgb_v2_sha",
        lambda label: (
            "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pred_mod,
        "_compute_v23_variance_dict",
        lambda seed_list, bootstrap=True: (
            _stubbed_variance_below_floor() if below_floor
            else _stubbed_variance_above_floor()
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pred_mod, "PHASE_31_DIR", phase_31_isolated, raising=False,
    )
    return phase_31_isolated


# ---------------------------------------------------------------------------
# Test 1: synthetic Path A dispatch — contract + END SHA emitted
# ---------------------------------------------------------------------------


def test_cli_dispatch_synthetic(monkeypatch, tmp_path):
    """CLI dispatch on synthetic above-floor data emits contract + END SHA."""
    phase_31_iso = _patch_path_a_environment(monkeypatch, tmp_path)
    contract_path = tmp_path / "gate_contract_v2.3.json"

    from ufc_prediction.cli.predict import predict_gate_spike

    predict_gate_spike(
        feature_set="v2.3",
        seeds=10,
        bootstrap=True,
        contract_path=str(contract_path),
    )

    # Contract emitted with v2.3 + formula_hash matches D-18 binding.
    assert contract_path.exists()
    c = json.loads(contract_path.read_text(encoding="utf-8"))
    assert c["version"] == "v2.3"
    assert c["n_seeds_observed"] == 10
    assert c["formula_hash"] == (
        "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
    )

    # END SHA artifact emitted at PHASE_31_DIR/31-XGB-V2-SHA-PHASE-31-END.txt
    end_sha_path = phase_31_iso / "31-XGB-V2-SHA-PHASE-31-END.txt"
    assert end_sha_path.exists()
    assert end_sha_path.read_text(encoding="utf-8").strip() == (
        "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
    )

    # HALT artifact NOT emitted on Path A.
    assert not (phase_31_iso / "31-HALT-AND-DECIDE.md").exists()


# ---------------------------------------------------------------------------
# Test 2: synthetic Path D HALT — HALT-AND-DECIDE.md emitted; SystemExit(3)
# ---------------------------------------------------------------------------


def test_cli_halt_on_below_floor(monkeypatch, tmp_path):
    """CLI exits 3 + emits 31-HALT-AND-DECIDE.md on synthetic below-floor variance."""
    phase_31_iso = _patch_path_a_environment(
        monkeypatch, tmp_path, below_floor=True,
    )
    contract_path = tmp_path / "gate_contract_v2.3.json"

    from ufc_prediction.cli.predict import predict_gate_spike

    with pytest.raises(SystemExit) as exc_info:
        predict_gate_spike(
            feature_set="v2.3",
            seeds=10,
            bootstrap=True,
            contract_path=str(contract_path),
        )

    # Exit code 3 (Path D HALT per Pitfall 5)
    assert exc_info.value.code == 3

    # HALT artifact emitted in isolated PHASE_31_DIR
    halt_path = phase_31_iso / "31-HALT-AND-DECIDE.md"
    assert halt_path.exists()

    halt_body = halt_path.read_text(encoding="utf-8")

    # Frontmatter: phase + artifact + operator_decision_required
    assert "phase: 31-gate-v23-re-derivation-on-populated-substrate" in halt_body
    assert "artifact: HALT-AND-DECIDE" in halt_body
    assert "operator_decision_required: true" in halt_body

    # Required headers (Pattern 7)
    assert "## Per-Slice Gap Diagnostic" in halt_body
    assert "## Operator Action Required" in halt_body

    # All 3 Path D outcome-path IDs (Pattern 7)
    assert (
        "v23_substrate_fills_but_gate_lt_floor_path_d_delay_ship" in halt_body
    )
    assert (
        "v23_substrate_fills_but_gate_lt_floor_path_d_ship_v22_publicly"
        in halt_body
    )
    assert (
        "v23_substrate_fills_but_gate_lt_floor_path_d_accept_lower_floor"
        in halt_body
    )

    # END SHA artifact NOT emitted on Path D (Pitfall 5 — WR-04 carry-forward)
    assert not (phase_31_iso / "31-XGB-V2-SHA-PHASE-31-END.txt").exists()

    # Contract NOT emitted on Path D
    assert not contract_path.exists()


# ---------------------------------------------------------------------------
# WR-03: Path D AUDIT-01 invariant — END SHA assertion runs even though the
#        artifact file is NOT written. Guards against a future refactor that
#        short-circuits the assertion helper on Path D and silently breaks
#        end-of-spike tampering detection.
# ---------------------------------------------------------------------------


def test_cli_halt_path_d_runs_end_sha_assertion_without_writing_artifact(
    monkeypatch, tmp_path,
):
    """Path D invokes _assert_xgb_v2_sha twice (STARTUP + END) but does NOT
    write the END SHA artifact file (Pitfall 5).

    Regression guard for WR-03: the discipline at predict.py:600-601 is to
    CALL the assertion helper (detects tampering at end of spike) while
    SKIPPING the file write. Earlier tests only verified the artifact's
    absence; this test additionally captures the labels passed to
    _assert_xgb_v2_sha to prove the SHA check still runs.
    """
    from ufc_prediction.cli import predict as pred_mod

    phase_31_isolated = tmp_path / "phase_31"
    phase_31_isolated.mkdir(parents=True, exist_ok=True)

    # Tracking stub: records every label _assert_xgb_v2_sha is called with.
    sha_calls: list[str] = []

    def tracking_sha(label: str) -> str:
        sha_calls.append(label)
        return (
            "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
        )

    monkeypatch.setattr(
        pred_mod,
        "_read_project_md",
        lambda: "| **D-24(v2.3, GATE)** | Phase 31 GATE row |",
        raising=False,
    )
    monkeypatch.setattr(pred_mod, "_spike_main", lambda argv: 0, raising=False)
    monkeypatch.setattr(
        pred_mod, "_assert_xgb_v2_sha", tracking_sha, raising=False,
    )
    monkeypatch.setattr(
        pred_mod,
        "_compute_v23_variance_dict",
        lambda seed_list, bootstrap=True: _stubbed_variance_below_floor(),
        raising=False,
    )
    monkeypatch.setattr(
        pred_mod, "PHASE_31_DIR", phase_31_isolated, raising=False,
    )

    contract_path = tmp_path / "gate_contract_v2.3.json"

    with pytest.raises(SystemExit) as exc_info:
        pred_mod.predict_gate_spike(
            feature_set="v2.3",
            seeds=10,
            bootstrap=True,
            contract_path=str(contract_path),
        )
    assert exc_info.value.code == 3

    # Invariant 1: SHA assertion runs at BOTH boundaries even on Path D.
    assert "PHASE-31-STARTUP" in sha_calls, (
        f"STARTUP SHA assertion missing on Path D; calls={sha_calls}"
    )
    assert "PHASE-31-END" in sha_calls, (
        f"END SHA assertion missing on Path D; calls={sha_calls} — "
        "Pitfall 5 enforcement regressed"
    )

    # Invariant 2: END SHA artifact file NOT written (Path D pre-templates).
    assert not (phase_31_isolated / "31-XGB-V2-SHA-PHASE-31-END.txt").exists(), (
        "Path D wrote the END SHA artifact — Pitfall 5 invariant violated"
    )

    # Invariant 3: HALT artifact IS written (the operator-decision blocker).
    assert (phase_31_isolated / "31-HALT-AND-DECIDE.md").exists()


# ---------------------------------------------------------------------------
# Test 3: audit-lineage regression — v2.1 + v2.2 contracts byte-identical
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_dispatch_preserves_v21_v22_contracts(monkeypatch, tmp_path):
    """v2.1 + v2.2 contract JSON SHA-256 byte-identical pre/post Path A run.

    WR-07 fix: this is an audit-lineage regression, not a smoke test. If
    the on-disk v2.1+v2.2 contracts are absent (sandbox without checkout
    artifacts, CWD outside repo root, fresh clone before pull-LFS, etc.),
    skip rather than red-fail — a skipped-but-not-failed run truthfully
    signals "regression coverage not exercised in this environment"; a
    red-fail would mis-signal a contract mutation that never happened.
    """
    v21_path = Path(".planning/gate_contract.json")
    v22_path = Path(".planning/gate_contract_v2.2.json")
    if not v21_path.exists() or not v22_path.exists():
        pytest.skip(
            "v2.1+v2.2 contracts not in checkout; audit-lineage regression n/a"
        )

    sha_v21_before = _sha256(v21_path)
    sha_v22_before = _sha256(v22_path)

    _patch_path_a_environment(monkeypatch, tmp_path)
    contract_path = tmp_path / "gate_contract_v2.3.json"

    from ufc_prediction.cli.predict import predict_gate_spike

    predict_gate_spike(
        feature_set="v2.3",
        seeds=10,
        bootstrap=True,
        contract_path=str(contract_path),
    )

    sha_v21_after = _sha256(v21_path)
    sha_v22_after = _sha256(v22_path)

    assert sha_v21_before == sha_v21_after, (
        "v2.1 contract mutated during Phase 31 CLI dispatch (GATE-V23-02 violation)"
    )
    assert sha_v22_before == sha_v22_after, (
        "v2.2 contract mutated during Phase 31 CLI dispatch (GATE-V23-02 violation)"
    )

    # Phase 31 dir + tmp_path also unpolluted in repo working tree
    assert not Path(
        ".planning/phases/31-gate-v23-re-derivation-on-populated-substrate/"
        "31-HALT-AND-DECIDE.md"
    ).exists(), "Test polluted real Phase 31 dir with HALT artifact"
