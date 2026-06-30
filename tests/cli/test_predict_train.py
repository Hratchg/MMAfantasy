"""Unit tests for ``ufc predict train`` Phase 15 wiring.

Covers the three Phase 15 contract changes:

1. Coverage check — when training-set odds coverage < 50% AND --force
   is not passed, the command exits non-zero before training begins.
2. --force flag — bypasses the coverage check; sub-50% coverage is
   warned but not fatal.
3. Default --version — `ufc predict train` (no flag) tags the model
   as ``v2`` (D-09).
"""

from __future__ import annotations

import hashlib
import inspect
import math
import pathlib
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli.main import app
from ufc_prediction.cli.predict import (
    _derive_relaxed_thresholds,  # NEW in Task 4
    _enforce_accuracy_gate,
    predict_train,
)

runner = CliRunner()


# ── Coverage check ───────────────────────────────────────────────────────────


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_elo_features")
@patch("ufc_prediction.cli.predict.load_computed_features")
@patch("ufc_prediction.cli.predict.load_fighter_physicals")
@patch("ufc_prediction.cli.predict.load_round_stats_for_ml")
@patch("ufc_prediction.cli.predict.load_pre_ufc_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_coverage_aborts_below_50pct(
    mock_odds: MagicMock,
    mock_pre: MagicMock,
    mock_round: MagicMock,
    mock_phys: MagicMock,
    mock_comp: MagicMock,
    mock_elo: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    """1000 training fights, only 100 with odds → 10% coverage → abort."""
    # All training-era fights (well before any sane cutoff)
    mock_fr.return_value = [
        {"fight_id": f"f{i}", "event_date": date(2015, 1, 1)} for i in range(1000)
    ]
    # Only 100 fight_ids have odds — 10% coverage
    mock_odds.return_value = {(f"f{i}", "fighter_x"): {} for i in range(100)}
    mock_elo.return_value = {}
    mock_comp.return_value = {}
    mock_phys.return_value = {}
    mock_round.return_value = {}
    mock_pre.return_value = {}

    result = runner.invoke(app, ["predict", "train", "--trials", "1"])
    assert result.exit_code == 1, result.output
    # Output names the threshold and the escape hatch
    assert "< 50% threshold" in result.output
    assert "--force" in result.output


# ── --force flag bypass ──────────────────────────────────────────────────────


@patch("ufc_prediction.cli.predict.compute_division_medians")
@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_elo_features")
@patch("ufc_prediction.cli.predict.load_computed_features")
@patch("ufc_prediction.cli.predict.load_fighter_physicals")
@patch("ufc_prediction.cli.predict.load_round_stats_for_ml")
@patch("ufc_prediction.cli.predict.load_pre_ufc_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_force_flag_overrides_coverage_check(
    mock_odds: MagicMock,
    mock_pre: MagicMock,
    mock_round: MagicMock,
    mock_phys: MagicMock,
    mock_comp: MagicMock,
    mock_elo: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
    mock_cdm: MagicMock,
) -> None:
    """--force bypasses the coverage check.

    Pattern: re-use the 10%-coverage setup from test_coverage_aborts_below_50pct.
    Patch ``compute_division_medians`` (the function called immediately
    AFTER the coverage block) to raise a sentinel RuntimeError. If the
    coverage check correctly bypasses on --force, control flow reaches
    compute_division_medians and our sentinel fires — proving the
    coverage check did NOT abort.

    If --force is NOT honored, the run aborts with SystemExit(1) at the
    coverage block BEFORE compute_division_medians is reached — sentinel
    never fires — the assertion below fails.
    """
    mock_fr.return_value = [
        {"fight_id": f"f{i}", "event_date": date(2015, 1, 1)} for i in range(1000)
    ]
    mock_odds.return_value = {(f"f{i}", "fighter_x"): {} for i in range(100)}
    mock_elo.return_value = {}
    mock_comp.return_value = {}
    mock_phys.return_value = {}
    mock_round.return_value = {}
    mock_pre.return_value = {}

    sentinel = "FORCE_BYPASSED_COVERAGE_CHECK"
    mock_cdm.side_effect = RuntimeError(sentinel)

    result = runner.invoke(app, ["predict", "train", "--trials", "1", "--force"])

    # Assertion 1: coverage threshold abort message NOT in output
    assert "< 50% threshold" not in result.output, (
        f"--force did not bypass coverage check: {result.output}"
    )
    # Assertion 2: sentinel raised AFTER the coverage block
    assert sentinel in str(result.exception) or sentinel in result.output, (
        f"compute_division_medians not reached — coverage check still "
        f"aborted under --force. exception={result.exception!r}, "
        f"output={result.output!r}"
    )
    # Assertion 3 (defense-in-depth): compute_division_medians was called
    assert mock_cdm.called, "compute_division_medians was never invoked"


# ── Default --version is v2 ──────────────────────────────────────────────────


def test_v2_is_default_version() -> None:
    """The Typer Option default for --version must be 'v2' (D-09)."""
    sig = inspect.signature(predict_train)
    version_param = sig.parameters["version"]
    # typer.Option returns an OptionInfo whose .default holds the literal
    assert version_param.default.default == "v2"


# ── Phase 15.1 Plan-02 (Path B) — Relaxed-gate derivation ───────────────────


class TestRelaxedGateDerivation:
    """Phase 15.1 Plan-02 Path B: relaxed-gate derivation + persistence (D-04(P15.1)).

    Tests pin the contract for `_derive_relaxed_thresholds` (the D-10(P15.1)
    margin formula) and the `predict relax-gate` Typer subcommand that persists
    the derived thresholds to source via `typer.confirm` (D-02(P15.1)).
    """

    # Test 1: D-10(P15.1) formula on Phase 15 baseline
    def test_relaxed_threshold_derivation_formula(self) -> None:
        """D-10(P15.1): (v1_brier - 0.010, v1_acc + 0.020).

        Uses Phase 15 baseline (Brier 0.2302, Acc 0.6190) — the live
        re-evaluation in Plan-02 measured Acc 0.6191 (off by +0.0001 from
        the rounded 0.6190 in the planning artifact). This test pins the
        formula contract, not the production-side derived value.
        """
        v1_metrics = {"brier_score": 0.2302, "accuracy": 0.6190}
        new_brier_max, new_acc_min = _derive_relaxed_thresholds(v1_metrics)
        assert math.isclose(new_brier_max, 0.2202, abs_tol=1e-9)
        assert math.isclose(new_acc_min, 0.6390, abs_tol=1e-9)

    # Test 2: formula is not sign-flipped on arbitrary input
    def test_relaxed_threshold_derivation_uses_d10_margins(self) -> None:
        """D-10(P15.1): margin constants are 0.010 (Brier) and 0.020 (Acc), not swapped."""
        v1_metrics = {"brier_score": 0.30, "accuracy": 0.50}
        new_brier_max, new_acc_min = _derive_relaxed_thresholds(v1_metrics)
        assert math.isclose(new_brier_max, 0.290, abs_tol=1e-9)
        assert math.isclose(new_acc_min, 0.520, abs_tol=1e-9)

    # Test 3: predict_train Typer Option default for brier_max swapped to 0.2202
    def test_predict_train_brier_max_default_after_swap(self) -> None:
        sig = inspect.signature(predict_train)
        # typer.Option returns an OptionInfo; .default holds the literal
        assert sig.parameters["brier_max"].default.default == 0.2202

    # Test 4: predict_train Typer Option default for acc_min swapped to 0.6391
    def test_predict_train_acc_min_default_after_swap(self) -> None:
        sig = inspect.signature(predict_train)
        assert sig.parameters["acc_min"].default.default == 0.6391

    # Test 5: helper default for brier_max swapped to 0.2202 (bare float)
    def test_helper_brier_max_default_after_swap(self) -> None:
        sig = inspect.signature(_enforce_accuracy_gate)
        # Helper defaults are bare floats (no OptionInfo indirection)
        assert sig.parameters["brier_max"].default == 0.2202

    # Test 6: helper default for acc_min swapped to 0.6391 (bare float)
    def test_helper_acc_min_default_after_swap(self) -> None:
        sig = inspect.signature(_enforce_accuracy_gate)
        assert sig.parameters["acc_min"].default == 0.6391

    # Test 7: D-08(P15) failure-message format unchanged with relaxed thresholds
    def test_d08_failure_message_format_unchanged_with_relaxed_thresholds(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            _enforce_accuracy_gate(
                {"brier_score": 0.2300, "accuracy": 0.5000},
                brier_max=0.2202,
                acc_min=0.6391,
            )
        captured = capsys.readouterr()
        # The Console writes to stderr by default in non-tty mode — combine both
        out = captured.out + captured.err
        assert "Phase 15 accuracy gate FAILED" in out
        assert "Brier 0.2300 > target 0.2202" in out
        assert "missed by +0.0098" in out
        assert "Accuracy 0.5000 < target 0.6391" in out
        assert "missed by -0.1391" in out

    # Test 8: old constants absent from helper + Typer Option line ranges
    def test_old_constants_absent_after_swap(self) -> None:
        predict_py = Path("src/ufc_prediction/cli/predict.py").read_text()
        lines = predict_py.splitlines()
        # 0-indexed slices covering the 1-indexed line ranges from the plan
        typer_option_block = "\n".join(lines[95:105])  # lines 96-105
        helper_default_block = "\n".join(lines[41:46])  # lines 42-46
        for label, block in [
            ("Typer Option (96-105)", typer_option_block),
            ("helper defaults (42-46)", helper_default_block),
        ]:
            assert "0.21" not in block, (
                f"Old D-07(P15) Brier constant 0.21 still present in {label} "
                f"after Path B swap:\n{block}"
            )
            assert "0.65" not in block, (
                f"Old D-07(P15) Acc constant 0.65 still present in {label} "
                f"after Path B swap:\n{block}"
            )

    # Test 9: D-04(P15.1) + D-09(P15.1) — operator typing 'n' leaves predict.py byte-identical
    def test_relax_gate_n_aborts_without_source_edit(self) -> None:
        """typer.confirm 'n' must leave predict.py byte-identical (D-09(P15.1) abort-path)."""
        predict_py_path = pathlib.Path("src/ufc_prediction/cli/predict.py")
        pre_sha = hashlib.sha256(predict_py_path.read_bytes()).hexdigest()

        runner = CliRunner()
        # Invoke via the unified `app` (predict sub-app is registered there).
        result = runner.invoke(app, ["predict", "relax-gate"], input="n\n")

        post_sha = hashlib.sha256(predict_py_path.read_bytes()).hexdigest()

        # The contract is BYTE-IDENTICAL source on operator decline, regardless of exit code.
        assert pre_sha == post_sha, (
            f"predict.py modified despite operator typing 'n' "
            f"(D-09(P15.1) abort-path violation). pre={pre_sha} post={post_sha}"
        )
        # Output must indicate abort/preservation
        out = (result.output or "") + (result.stderr or "")
        assert "Aborted" in out or "preserved" in out.lower() or "abort" in out.lower(), (
            f"Expected abort/preservation message; got:\n{out}"
        )

    # Test 10: docstring de-stale (Warning 6 fix)
    def test_predict_train_docstring_no_stale_thresholds(self) -> None:
        """predict.py:120 docstring must drop stale 0.21/0.65 and trace D-04(P15.1)."""
        predict_py = Path("src/ufc_prediction/cli/predict.py").read_text()
        # Stale docstring substring must be GONE
        assert "Brier <= 0.21 AND accuracy >= 0.65" not in predict_py, (
            "Stale D-07(P15) docstring still present in predict.py after Path B swap"
        )
        # New docstring substring must be PRESENT
        assert "Brier <= 0.2202 AND accuracy >= 0.6391" in predict_py, (
            "Relaxed-threshold docstring not written to predict.py"
        )
        # Supersession trace must be PRESENT (bare or namespaced form acceptable)
        assert (
            "D-04 supersedes D-07" in predict_py
            or "D-04(P15.1) supersedes D-07(P15)" in predict_py
            or "D-04(P15.1) supersedes" in predict_py
        ), "D-04 supersession of D-07 not annotated in docstring"
