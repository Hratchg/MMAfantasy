"""Phase 63 CRIT-V261-01 — CLI integration tests for ``ufc gate verify``.

Covers the end-to-end CLI surface wired up in Plan 63-03:

- Happy path: synthetic candidate + canonical Pipelines + synthetic substrate
  parquet → ``runner.invoke`` produces exit 0, writes the default JSON sidecar
  at ``results/gate_verdict_v26_<candidate_stem>.json``, and the sidecar
  contains a valid ``SubstrateDriftSafeGateVerdict`` (verdict literal in the
  expected set, methodology defaults to ``refit_baseline``, per-slice metric
  keys present, substrate_sha is a non-empty hex string).
- ``--out`` override: an absolute output path is honored byte-identical;
  the default path is NOT written when override is set.
- Malformed substrate negative path: an R1-violating parquet (missing the
  ``outcome`` column) yields exit code 1, names "outcome" in stdout, and
  emits NO Python traceback and NO JSON sidecar.

All tests invoke the CLI via ``typer.testing.CliRunner`` so failures surface
as Python exceptions (not opaque subprocess exit codes). All fixtures are
synthetic (deterministic, no randomness beyond a fixed ``np.random.default_rng``
seed) per 63-CONTEXT §D-02 — real Phase 42/45 substrate parquets are
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from ufc_prediction.cli.main import app

runner = CliRunner()


# ── Module-private helpers ────────────────────────────────────────────────


def _make_synthetic_pipeline(
    tmp_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> Path:
    """Fit a StandardScaler+LogReg Pipeline and joblib-dump under tmp_path.

    Mirrors ``tests/unit/ml/test_gate_verifier_v26.py::_make_pipeline_and_persist``
    (Phase 55 helper pattern) — same hyperparameters
    (C=1.0, max_iter=10_000, random_state=0) so the candidate/canonical
    artifacts here are byte-identical to what the Phase 55 verifier tests
    produce.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=10_000, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    out = tmp_path / f"{name}.joblib"
    joblib.dump(pipe, out)
    return out


def _build_substrate_parquet(
    tmp_path: Path,
    slices: dict[str, tuple[np.ndarray, np.ndarray, str]],
    name: str = "substrate.parquet",
) -> Path:
    """Build a substrate-snapshot parquet for CLI integration tests.

    Args:
        tmp_path: pytest ``tmp_path`` fixture directory.
        slices: Mapping ``slice_name -> (X, y, substrate_sha)``. The same
            ``X`` arrays drive both the Pipeline fit and the substrate
            snapshot to guarantee width compatibility (the verifier
            otherwise rejects mismatched feature widths).
        name: Output parquet filename under ``tmp_path``.

    Returns:
        Path to the written parquet.

    Schema (locked, matches ``substrate_loader._REQUIRED_DTYPES`` byte-identical):
        - ``slice_name``     : ``pa.string()``
        - ``feature_vector`` : ``pa.list_(pa.float64())``
        - ``outcome``        : ``pa.int8()``
        - ``substrate_sha``  : ``pa.string()``
    """
    slice_names_col: list[str] = []
    feature_vectors_col: list[list[float]] = []
    outcomes_col: list[int] = []
    substrate_shas_col: list[str] = []

    for slice_name, (X, y, sha_value) in slices.items():
        for i in range(X.shape[0]):
            slice_names_col.append(slice_name)
            feature_vectors_col.append([float(v) for v in X[i]])
            outcomes_col.append(int(y[i]))
            substrate_shas_col.append(sha_value)

    table = pa.table(
        {
            "slice_name": pa.array(slice_names_col, type=pa.string()),
            "feature_vector": pa.array(feature_vectors_col, type=pa.list_(pa.float64())),
            "outcome": pa.array(outcomes_col, type=pa.int8()),
            "substrate_sha": pa.array(substrate_shas_col, type=pa.string()),
        }
    )

    out_path = tmp_path / name
    pq.write_table(table, out_path)
    return out_path


# ── Task 1: Happy path + --out override ───────────────────────────────────


def test_gate_verify_happy_path_writes_verdict_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: synthetic candidate + canonical + substrate → exit 0,
    default sidecar exists under ``tmp_path/results/`` (via monkeypatch.chdir),
    JSON parses to a valid SubstrateDriftSafeGateVerdict dict, and the
    Rich stdout includes the verdict literal.
    """
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 3))
    y_train = (X_train[:, 0] > 0).astype(int)

    p_cand = _make_synthetic_pipeline(tmp_path, X_train, y_train, "cand")
    p_canon = _make_synthetic_pipeline(tmp_path, X_train, y_train, "canon")
    p_sub = _build_substrate_parquet(
        tmp_path,
        {
            "most_recent_12mo": (X_train[:30], y_train[:30], "sha-12mo"),
            "random_15pct": (X_train[30:60], y_train[30:60], "sha-random"),
        },
    )

    # Default --out resolves to results/gate_verdict_v26_<candidate.stem>.json
    # under the CWD. Pin CWD to tmp_path so the sidecar does NOT pollute
    # the real repo's results/ directory.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "gate",
            "verify",
            "--candidate",
            str(p_cand),
            "--canonical",
            str(p_canon),
            "--substrate-parquet",
            str(p_sub),
        ],
    )

    if result.exit_code != 0:
        # Diagnostic dump — surfaces traceback to pytest on failure.
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0

    verdict_path = tmp_path / "results" / "gate_verdict_v26_cand.json"
    assert verdict_path.exists(), (
        f"Default sidecar not found at {verdict_path}. "
        f"Files under tmp_path: {list(tmp_path.rglob('*.json'))}"
    )

    verdict_data = json.loads(verdict_path.read_text())

    # Verdict literal is one of the three documented values.
    assert verdict_data["verdict"] in {
        "path_a_promote",
        "path_b_reject",
        "confound_block",
    }

    # Per-slice metric dict keys present + match the two slice names.
    assert "aligned_baseline_brier_per_slice" in verdict_data
    assert set(verdict_data["aligned_baseline_brier_per_slice"].keys()) == {
        "most_recent_12mo",
        "random_15pct",
    }

    # substrate_sha is a non-empty hex string (sha256 over the per-slice SHAs).
    assert isinstance(verdict_data["substrate_sha"], str)
    assert len(verdict_data["substrate_sha"]) > 0
    assert all(c in "0123456789abcdef" for c in verdict_data["substrate_sha"])

    # Default methodology is refit_baseline (no --strategy flag passed).
    assert verdict_data["methodology"] == "refit_baseline"

    # Rich stdout includes the verdict literal — operator can scan in 0.1s.
    assert verdict_data["verdict"] in result.stdout


def test_gate_verify_out_override_honored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--out PATH writes the sidecar to the absolute path, and the default
    path is NOT written. monkeypatch.chdir sandboxes the default-path
    resolution so the assertion is not vacuously true and the real
    results/ directory is not polluted.
    """
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 3))
    y_train = (X_train[:, 0] > 0).astype(int)

    p_cand = _make_synthetic_pipeline(tmp_path, X_train, y_train, "cand")
    p_canon = _make_synthetic_pipeline(tmp_path, X_train, y_train, "canon")
    p_sub = _build_substrate_parquet(
        tmp_path,
        {
            "most_recent_12mo": (X_train[:30], y_train[:30], "sha-12mo"),
            "random_15pct": (X_train[30:60], y_train[30:60], "sha-random"),
        },
    )

    custom_out = tmp_path / "custom" / "my_verdict.json"

    # Pin CWD to tmp_path so the default-path computation resolves under
    # tmp_path/results/ — making the not-exists assertion below meaningful.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "gate",
            "verify",
            "--candidate",
            str(p_cand),
            "--canonical",
            str(p_canon),
            "--substrate-parquet",
            str(p_sub),
            "--out",
            str(custom_out),
        ],
    )

    if result.exit_code != 0:
        print("STDOUT:", result.stdout)
        print("EXCEPTION:", result.exception)
    assert result.exit_code == 0

    # Override honored: sidecar lives at the absolute path.
    assert custom_out.exists()

    # Confirm the JSON at the custom path is a valid verdict.
    verdict_data = json.loads(custom_out.read_text())
    assert verdict_data["verdict"] in {
        "path_a_promote",
        "path_b_reject",
        "confound_block",
    }

    # Default path is NOT written when --out overrides. Walk tmp_path for
    # any stray default-sidecar artifacts.
    default_path = tmp_path / "results" / "gate_verdict_v26_cand.json"
    assert not default_path.exists()


# ── Task 2: Negative path — malformed substrate ──────────────────────────


def test_gate_verify_malformed_substrate_exits_1_with_clean_message(
    tmp_path: Path,
) -> None:
    """An R1-violating parquet (missing ``outcome`` column) yields exit 1,
    names "outcome" in stdout, emits NO Python traceback, and writes NO
    sidecar. Confirms Plan 63-03's loader-ValueError→Rich-error contract.
    """
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 3))
    y_train = (X_train[:, 0] > 0).astype(int)

    p_cand = _make_synthetic_pipeline(tmp_path, X_train, y_train, "cand")
    p_canon = _make_synthetic_pipeline(tmp_path, X_train, y_train, "canon")

    # Build an invalid substrate parquet — missing the `outcome` column
    # entirely. Inline construction so this test file does not depend on
    # the unit test module's helpers.
    invalid_table = pa.table(
        {
            "slice_name": pa.array(["s1", "s1", "s1"], type=pa.string()),
            "feature_vector": pa.array(
                [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
                type=pa.list_(pa.float64()),
            ),
            # NOTE: no `outcome` column → triggers R1 in load_substrate_snapshot.
            "substrate_sha": pa.array(
                ["sha-s1", "sha-s1", "sha-s1"],
                type=pa.string(),
            ),
        }
    )
    invalid_substrate = tmp_path / "invalid_substrate.parquet"
    pq.write_table(invalid_table, invalid_substrate)

    result = runner.invoke(
        app,
        [
            "gate",
            "verify",
            "--candidate",
            str(p_cand),
            "--canonical",
            str(p_canon),
            "--substrate-parquet",
            str(invalid_substrate),
        ],
    )

    # Exit code 1 — NOT 2 (Typer arg-rejection) and NOT 0 (success).
    # If we see 2, Plan 63-03's option config has a bug. If we see 0,
    # the loader's R1 check silently passed which would be catastrophic.
    assert result.exit_code == 1, (
        f"Expected exit 1 (clean error), got {result.exit_code}. "
        f"STDOUT: {result.stdout!r} EXCEPTION: {result.exception!r}"
    )

    # The error message names the missing column.
    assert "outcome" in result.stdout

    # No Python traceback bubbled up — operator sees a clean message.
    assert "Traceback" not in result.stdout

    # CLI handled the error gracefully: exception is either None or a
    # SystemExit (Typer wraps the SystemExit raised in gate_verify's
    # except ValueError block).
    assert result.exception is None or isinstance(
        result.exception,
        SystemExit,
    )

    # No sidecar was written — neither the default path nor any stray
    # JSON anywhere under tmp_path.
    assert list(tmp_path.rglob("*.json")) == []


# ── Task 3: Negative path — missing substrate file ───────────────────────


def test_gate_verify_missing_substrate_path_exits_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-existent --substrate-parquet path yields exit 1 with a clean
    operator-facing message — no Python traceback. Pairs with the WR-01
    fix (broadened except tuple to include FileNotFoundError / OSError)
    to lock the behavior against regression.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = (X[:, 0] > 0).astype(int)

    p_cand = _make_synthetic_pipeline(tmp_path, X, y, "cand")
    p_canon = _make_synthetic_pipeline(tmp_path, X, y, "canon")

    # Pin CWD so no default sidecar can land in the real results/ directory.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "gate",
            "verify",
            "--candidate",
            str(p_cand),
            "--canonical",
            str(p_canon),
            "--substrate-parquet",
            str(tmp_path / "nonexistent.parquet"),
        ],
    )

    # Exit code must be 1 — clean operator error, not a crash (0) or
    # Typer arg-rejection (2).
    assert result.exit_code == 1, (
        f"Expected exit 1 (clean error), got {result.exit_code}. "
        f"STDOUT: {result.stdout!r} EXCEPTION: {result.exception!r}"
    )

    # No raw Python traceback surfaced to the operator.
    assert "Traceback" not in result.stdout

    # The error message names either the missing file path or the load-
    # failed sentinel so the operator knows what to fix.
    assert "nonexistent.parquet" in result.stdout or "load failed" in result.stdout, (
        f"Expected file path or 'load failed' in stdout; got: {result.stdout!r}"
    )

    # CLI handled the error gracefully — exception is None or SystemExit.
    assert result.exception is None or isinstance(
        result.exception,
        SystemExit,
    )

    # No sidecar was written anywhere under tmp_path.
    assert list(tmp_path.rglob("*.json")) == []
