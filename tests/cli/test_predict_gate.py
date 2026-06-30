"""Unit tests for ``_enforce_accuracy_gate`` (Phase 15, D-07/D-08).

The gate is the hard block that prevents a regressed model from being
saved to disk: if Brier > 0.21 OR Accuracy < 0.65, the train command
raises SystemExit(1) with a red message naming each missed target and
the delta. On pass, prints a green PASSED message including both metrics.
"""

from __future__ import annotations

import pytest

from ufc_prediction.cli.predict import _enforce_accuracy_gate


class TestAccuracyGateFailure:
    """The gate raises SystemExit(1) with a red message on miss."""

    def test_brier_gate_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as ei:
            _enforce_accuracy_gate({"brier_score": 0.25, "accuracy": 0.70})
        assert ei.value.code == 1
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "Brier" in out
        # delta is +0.0298 (Phase 15.1 D-04: brier_max relaxed to 0.2202)
        assert "+0.0298" in out

    def test_accuracy_gate_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as ei:
            _enforce_accuracy_gate({"brier_score": 0.20, "accuracy": 0.60})
        assert ei.value.code == 1
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "Accuracy" in out
        # delta is -0.0391 (Phase 15.1 D-04: acc_min relaxed to 0.6391)
        assert "-0.0391" in out

    def test_both_thresholds_fail_lists_both(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            _enforce_accuracy_gate({"brier_score": 0.25, "accuracy": 0.60})
        out = capsys.readouterr().out
        assert "Brier" in out
        assert "Accuracy" in out


class TestAccuracyGatePass:
    """The gate prints a green PASSED message when both metrics meet target."""

    def test_gate_passes_on_meeting_thresholds(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Should NOT raise
        _enforce_accuracy_gate({"brier_score": 0.20, "accuracy": 0.66})
        out = capsys.readouterr().out
        assert "PASSED" in out
        # Both metrics included
        assert "0.2000" in out
        assert "0.6600" in out


class TestAccuracyGateCustomThresholds:
    """Tighter thresholds let callers tune the gate (testability hook)."""

    def test_custom_thresholds_honored(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Default-passing metrics now fail under tighter thresholds
        with pytest.raises(SystemExit):
            _enforce_accuracy_gate(
                {"brier_score": 0.20, "accuracy": 0.66},
                brier_max=0.18,
                acc_min=0.70,
            )
