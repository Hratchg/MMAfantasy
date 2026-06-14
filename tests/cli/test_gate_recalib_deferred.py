"""Phase 72 METH-V261-02 — locks the deferred-to-v2.7+ behaviour of ``ufc gate recalib``.

Per gate_methodology_v2.6.md §3.1 + §7.3 + Phase 72 CONTEXT D-04, the
threshold re-derivation logic for ``--apply`` is deferred to v2.7+ because
the corpus-growth trigger has NOT fired since v2.6 close (Phase 71 deferred
prevented any data refresh; fight_odds count unchanged at ~25,632).

This test pins the existing Phase 56 scaffold behaviour:

  - ``gate recalib`` (default, no flags) → exit 0, dry-run report
  - ``gate recalib --apply`` → exit 2, deferred-to-v2.7+ message
  - non-v2.6 feature-set → exit 1, methodology rejection

A future PR that wires up the actual re-derivation logic for METH-V27-01
MUST update OR remove these tests intentionally — they are the gate.
"""

from __future__ import annotations

from typer.testing import CliRunner

from ufc_prediction.cli.main import app

runner = CliRunner()


def test_recalib_dry_run_exits_zero() -> None:
    """Default invocation prints the dry-run report and exits 0."""
    result = runner.invoke(app, ["gate", "recalib"])
    assert result.exit_code == 0, (
        f"dry-run should exit 0; got {result.exit_code}\n{result.stdout}"
    )
    # Output must include the scaffold's headline so operators understand
    # this is the dry-run mode.
    combined = (result.stdout or "") + (result.stderr or "")
    assert "GATE-RECALIB-PERIODIC" in combined or "v2.6.1 follow-on" in combined


def test_recalib_apply_exits_two_deferred_to_v27(monkeypatch) -> None:
    """``--apply`` is rejected with exit 2 — actual re-derivation deferred to v2.7+.

    Per Phase 72 D-04 the corpus-growth trigger has NOT fired, so the apply
    path correctly refuses to overwrite the contract. This test locks the
    exit code so a future PR can't silently turn it on without updating the
    test (and therefore the audit trail).
    """
    result = runner.invoke(app, ["gate", "recalib", "--apply"])
    assert result.exit_code == 2, (
        f"--apply should exit 2 (deferred); got {result.exit_code}\n"
        f"stdout={result.stdout!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    # Phase 56 scaffold's literal "v2.6.1 follow-on" message OR Phase 72's
    # METH-V27-01 reference. Accept either so a future commit can reword
    # without breaking the test (it just has to stay an exit-2 reject).
    assert "v2.6.1 follow-on" in combined or "dry-run-only" in combined


def test_recalib_rejects_non_v26_feature_set() -> None:
    """Pre-v2.6 feature-sets are rejected — methodology lineage gate."""
    result = runner.invoke(app, ["gate", "recalib", "--feature-set", "v2.3"])
    assert result.exit_code == 1, (
        f"non-v2.6 feature-set should exit 1; got {result.exit_code}\n"
        f"stdout={result.stdout!r}"
    )


def test_recalib_explicit_v26_dry_run_exits_zero() -> None:
    """Explicit ``--feature-set v2.6`` dry-run also exits 0."""
    result = runner.invoke(app, ["gate", "recalib", "--feature-set", "v2.6"])
    assert result.exit_code == 0
