"""Phase 64 code-review fix (WR-04 + CR-02 + CR-01 regression) — unit tests
for ``scripts/run_travel_gate_v26.py``.

Covers the back-compat shim's two main code paths:

  - ``delegate_to_ufc_gate_verify`` (Phase 64 D-03a): subprocess
    delegation to ``ufc gate verify``. Tests:
      * Correct argv is built (candidate / canonical / substrate /
        strategy / out are all forwarded 1:1 to ``ufc gate verify``).
      * Verifier exit code propagates verbatim (0 → 0, 1 → 1, 2 → 2).
      * ``FileNotFoundError`` from ``subprocess.run`` (``ufc`` not on
        ``PATH``) is caught and surfaces as rc=127 with a clear stderr
        message (the CR-02 guard).
  - ``emit_provisional_path_b`` (Phase 58 audit-trail emitter, CR-01
    guarded in Phase 64): refuses to overwrite the committed Phase 58
    D-04/D-05 protected artifacts. Tests:
      * Writes both files cleanly when neither exists.
      * Raises ``RuntimeError`` with the D-04/D-05 invariant pointer
        message when EITHER file already exists.
      * ``main()`` surfaces the RuntimeError as rc=1 with a clean
        stderr message rather than an unhandled traceback.

The tests use ``unittest.mock.patch`` against ``subprocess.run`` so no
``ufc`` binary needs to be installed in the test environment — exactly
the CR-02 failure mode we're guarding against.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Same sys.path injection pattern as the sibling
# ``test_build_travel_substrate_v261.py`` so ``scripts/`` is importable
# as a top-level module.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from run_travel_gate_v26 import (  # noqa: E402
    DEFAULT_V261_OUT,
    delegate_to_ufc_gate_verify,
    emit_provisional_path_b,
    main,
)


# ── delegate_to_ufc_gate_verify: argv-construction test ───────────────────


def test_delegate_builds_correct_argv(tmp_path: Path) -> None:
    """The shim forwards all 5 user-facing flags to ``ufc gate verify`` 1:1.

    Covers WR-04 (a): the shim's argument mapping is the contract the
    operator-facing CLI surface depends on. Any drift (e.g.,
    ``--substrate-parquet`` vs ``--substrate_parquet``) would silently
    break Phase 58-era shell aliases.
    """
    candidate = tmp_path / "cand.joblib"
    canonical = tmp_path / "canon.joblib"
    substrate = tmp_path / "sub.parquet"
    out = tmp_path / "out.json"

    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        rc = delegate_to_ufc_gate_verify(
            candidate=candidate,
            canonical=canonical,
            substrate_parquet=substrate,
            strategy="refit_baseline",
            out=out,
        )

    assert rc == 0
    # subprocess.run was invoked exactly once.
    assert mock_run.call_count == 1
    # First positional arg is the argv list.
    argv = mock_run.call_args[0][0]
    assert argv[0:3] == ["ufc", "gate", "verify"], (
        f"Expected argv to start with ['ufc', 'gate', 'verify'], got {argv[0:3]!r}"
    )
    # All five flag names + their values are present (order-insensitive
    # over the flag/value pairs — only the head-3 order is contractual).
    for flag, value in [
        ("--candidate", str(candidate)),
        ("--canonical", str(canonical)),
        ("--substrate-parquet", str(substrate)),
        ("--strategy", "refit_baseline"),
        ("--out", str(out)),
    ]:
        assert flag in argv, f"Missing flag {flag!r} in argv: {argv!r}"
        flag_idx = argv.index(flag)
        assert argv[flag_idx + 1] == value, (
            f"Expected {flag} value {value!r}, got {argv[flag_idx + 1]!r}"
        )
    # check=False is required so the verifier's exit code propagates
    # verbatim (rather than raising CalledProcessError on non-zero).
    assert mock_run.call_args.kwargs.get("check") is False


# ── delegate_to_ufc_gate_verify: exit-code propagation test ───────────────


@pytest.mark.parametrize("verifier_rc", [0, 1, 2, 42])
def test_delegate_propagates_verifier_exit_code(
    tmp_path: Path,
    verifier_rc: int,
) -> None:
    """``ufc gate verify`` rc N → shim returns N verbatim.

    Covers WR-04 (b): the Phase 63 CLI uses rc=0 for clean verdicts
    (including ``confound_block``) and rc=1 for load failures. The shim
    must NOT swallow or remap these — operator automation depends on
    1:1 propagation.
    """
    candidate = tmp_path / "cand.joblib"
    canonical = tmp_path / "canon.joblib"
    substrate = tmp_path / "sub.parquet"
    out = tmp_path / "out.json"

    mock_result = MagicMock()
    mock_result.returncode = verifier_rc
    with patch("subprocess.run", return_value=mock_result):
        rc = delegate_to_ufc_gate_verify(
            candidate=candidate,
            canonical=canonical,
            substrate_parquet=substrate,
            strategy="refit_baseline",
            out=out,
        )
    assert rc == verifier_rc, f"Expected shim to propagate verifier rc {verifier_rc}, got {rc}"


# ── CR-02: FileNotFoundError handling ─────────────────────────────────────


def test_delegate_handles_ufc_not_found_returns_127(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CR-02 regression: missing ``ufc`` on PATH → rc=127, NOT a traceback.

    ``subprocess.run`` raises ``FileNotFoundError`` (not
    ``CalledProcessError``) when the entry-point is not on ``PATH``, and
    ``check=False`` does NOT suppress it. The CR-02 guard catches this
    explicitly and exits 127 (the POSIX/bash convention for "command not
    found"), with a clear stderr message pointing the operator at
    ``pip install -e .`` / ``uv pip install -e .``.
    """
    candidate = tmp_path / "cand.joblib"
    canonical = tmp_path / "canon.joblib"
    substrate = tmp_path / "sub.parquet"
    out = tmp_path / "out.json"

    with patch(
        "subprocess.run",
        side_effect=FileNotFoundError(2, "No such file or directory: 'ufc'"),
    ):
        rc = delegate_to_ufc_gate_verify(
            candidate=candidate,
            canonical=canonical,
            substrate_parquet=substrate,
            strategy="refit_baseline",
            out=out,
        )

    assert rc == 127, (
        f"Expected rc=127 (POSIX 'command not found') on FileNotFoundError, got rc={rc}"
    )
    captured = capsys.readouterr()
    assert "ufc" in captured.err and "PATH" in captured.err, (
        f"Expected clean operator-facing stderr mentioning 'ufc' + 'PATH', "
        f"got stderr={captured.err!r}"
    )


# ── CR-01: emit_provisional_path_b guard ──────────────────────────────────


def test_emit_provisional_writes_both_files_when_neither_exists(
    tmp_path: Path,
) -> None:
    """Happy-path baseline: when neither destination file exists, the
    emitter writes both as expected.

    Establishes the pre-CR-01 baseline behavior is preserved on a clean
    slate. The CR-01 guard ONLY fires when at least one destination
    already exists.
    """
    md = tmp_path / "out.md"
    js = tmp_path / "out.json"
    assert not md.exists() and not js.exists()

    md_returned, js_returned = emit_provisional_path_b(md_path=md, json_path=js)

    assert md_returned == md and js_returned == js
    assert md.exists() and js.exists()
    assert md.read_text(encoding="utf-8").startswith("# TRAVEL Promotion Gate")
    # JSON content should be valid JSON with the expected top-level keys.
    import json as _json

    parsed = _json.loads(js.read_text(encoding="utf-8"))
    assert parsed["verdict"] == "path_b_provisional"
    assert parsed["phase"] == "58-feat-v26-02"


def test_emit_provisional_refuses_when_md_exists(tmp_path: Path) -> None:
    """CR-01: emitter raises RuntimeError when the MD destination exists.

    The pre-CR-01 behavior was a silent overwrite. The CR-01 guard
    converts this to a loud ``RuntimeError`` that names the D-04/D-05
    invariant in the message.
    """
    md = tmp_path / "out.md"
    js = tmp_path / "out.json"
    md.write_text("pre-existing committed content", encoding="utf-8")
    assert not js.exists()

    with pytest.raises(RuntimeError, match="D-04/D-05"):
        emit_provisional_path_b(md_path=md, json_path=js)

    # Pre-existing MD must NOT be modified.
    assert md.read_text(encoding="utf-8") == "pre-existing committed content"
    # JSON must NOT have been written either (atomicity).
    assert not js.exists()


def test_emit_provisional_refuses_when_json_exists(tmp_path: Path) -> None:
    """CR-01: emitter raises RuntimeError when ONLY the JSON destination exists.

    Mirror of the MD-only case — guard fires on EITHER file existing.
    """
    md = tmp_path / "out.md"
    js = tmp_path / "out.json"
    js.write_text('{"existing": "json"}', encoding="utf-8")
    assert not md.exists()

    with pytest.raises(RuntimeError, match="D-04/D-05"):
        emit_provisional_path_b(md_path=md, json_path=js)

    assert js.read_text(encoding="utf-8") == '{"existing": "json"}'
    assert not md.exists()


def test_emit_provisional_refuses_when_both_exist(tmp_path: Path) -> None:
    """CR-01: emitter raises RuntimeError when BOTH destinations exist.

    The most common real-world trigger: operator runs the legacy no-arg
    shim in a checked-out repo where Phase 58's committed artifacts are
    already on disk.
    """
    md = tmp_path / "out.md"
    js = tmp_path / "out.json"
    md.write_text("md content", encoding="utf-8")
    js.write_text('{"json": "content"}', encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        emit_provisional_path_b(md_path=md, json_path=js)

    # Error message must point operators at the v2.6.1 substrate path.
    assert "--substrate-parquet" in str(excinfo.value)
    assert "D-04/D-05" in str(excinfo.value)


# ── main() integration: CR-01 guard surfaces as rc=1, not traceback ───────


def test_main_no_substrate_with_existing_outputs_returns_rc1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CR-01 main()-level wrapper: RuntimeError from emit_provisional
    surfaces as a clean rc=1 + stderr, NOT an unhandled traceback.

    Operators invoking the legacy no-arg path (``python
    scripts/run_travel_gate_v26.py``) in a repo with the Phase 58
    artifacts on disk should see a clear error and exit code, not a
    Python stack trace.
    """
    # Redirect the module-level RESULTS_MD / RESULTS_JSON constants to
    # tmp_path so we can simulate "files already exist on disk" without
    # touching the real protected files.
    import run_travel_gate_v26 as _mod

    fake_md = tmp_path / "fake_v26.md"
    fake_json = tmp_path / "fake_v26.json"
    fake_md.write_text("pre-existing", encoding="utf-8")
    fake_json.write_text('{"pre": "existing"}', encoding="utf-8")
    monkeypatch.setattr(_mod, "RESULTS_MD", fake_md)
    monkeypatch.setattr(_mod, "RESULTS_JSON", fake_json)

    # Invoke main() with NO --substrate-parquet → emit_provisional path.
    rc = main(argv=[])

    assert rc == 1, f"Expected rc=1 from CR-01 main()-level guard, got rc={rc}"
    captured = capsys.readouterr()
    assert "ERROR" in captured.err and "D-04/D-05" in captured.err, (
        f"Expected stderr to surface the D-04/D-05 guard message, got stderr={captured.err!r}"
    )
    # Pre-existing files must NOT have been modified.
    assert fake_md.read_text(encoding="utf-8") == "pre-existing"
    assert fake_json.read_text(encoding="utf-8") == '{"pre": "existing"}'


# ── main() integration: delegation path with default --out ────────────────


def test_main_substrate_path_uses_default_v261_out_when_out_omitted(
    tmp_path: Path,
) -> None:
    """When --out is omitted on the delegation path, ``DEFAULT_V261_OUT``
    is forwarded to ``ufc gate verify --out``.

    Covers a subtle Phase 64 D-04 contract: the v2.6.1 default output
    path (``results/travel_promotion_gate_v261.json``) must be applied
    by ``main()``, not punted to ``ufc gate verify`` (which has its own
    default that may differ in future versions).
    """
    substrate = tmp_path / "sub.parquet"
    substrate.write_bytes(b"fake")

    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        rc = main(argv=["--substrate-parquet", str(substrate)])

    assert rc == 0
    argv = mock_run.call_args[0][0]
    out_idx = argv.index("--out")
    assert argv[out_idx + 1] == str(DEFAULT_V261_OUT), (
        f"Expected --out to default to {DEFAULT_V261_OUT}, got {argv[out_idx + 1]!r}"
    )
