"""Phase 63 CRIT-V261-01 — unit tests for ``substrate_loader.load_substrate_snapshot``.

Covers (per 63-02-PLAN.md ``<success_criteria>``):

- Round-trip on a synthetic snapshot (happy path).
- SHA preservation: per-slice ``substrate_sha`` passes through byte-identical.
- Feature-vector tuple shape + per-slice width homogeneity.
- Outcomes returned as Python ``int`` (not numpy scalars).
- R1-R8 rejection rules raise ``ValueError`` with the offending column / row /
  value named in the message.
- A1 (width-agnostic: 3-col and 13-col snapshots both load).
- A2 (NaN inside ``feature_vector`` preserved).

All fixtures are synthetic (hand-built pyarrow tables, deterministic, no
randomness) per 63-CONTEXT §D-02. Real substrate generation is owned by
Phases 64-67 (Bucket B). Tests run in well under 5 seconds — pure parquet
I/O + dict assertions, no machine-learning library imports.
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ufc_prediction.ml.gate_verifier import EvalSlice
from ufc_prediction.ml.substrate_loader import load_substrate_snapshot


# ── Module-private fixture builder ────────────────────────────────────────


def _build_synthetic_snapshot(
    tmp_path: Path,
    slices: dict[str, tuple[int, int]],
    *,
    sha_map: dict[str, str] | None = None,
    name: str = "synthetic.parquet",
) -> Path:
    """Build a synthetic substrate-snapshot parquet for unit tests.

    Args:
        tmp_path: pytest ``tmp_path`` fixture directory.
        slices: Mapping ``slice_name -> (n_predictions, n_features)`` per
            63-CONTEXT §Specific Ideas. One row is emitted per prediction.
        sha_map: Optional override for per-slice ``substrate_sha`` values.
            Slices not present in the map default to ``f"sha-{slice_name}"``
            so each slice gets a distinct SHA by default (R7-safe).
        name: Output parquet filename under ``tmp_path``.

    Returns:
        Path to the written parquet file.

    Schema (locked, matches ``substrate_loader._REQUIRED_DTYPES`` byte-identical):
        - ``slice_name``     : ``pa.string()``
        - ``feature_vector`` : ``pa.list_(pa.float64())``
        - ``outcome``        : ``pa.int8()``
        - ``substrate_sha``  : ``pa.string()``

    Generated values are deterministic (no randomness) so tests are
    reproducible without random-seed plumbing.
    """
    slice_names_col: list[str] = []
    feature_vectors_col: list[list[float]] = []
    outcomes_col: list[int] = []
    substrate_shas_col: list[str] = []

    for slice_name, (n_predictions, n_features) in slices.items():
        sha_value: str = (
            sha_map[slice_name]
            if sha_map is not None and slice_name in sha_map
            else f"sha-{slice_name}"
        )
        for i in range(n_predictions):
            slice_names_col.append(slice_name)
            feature_vectors_col.append(
                [float(i * 0.1 + j) for j in range(n_features)]
            )
            outcomes_col.append(i % 2)
            substrate_shas_col.append(sha_value)

    table = pa.table(
        {
            "slice_name": pa.array(slice_names_col, type=pa.string()),
            "feature_vector": pa.array(
                feature_vectors_col, type=pa.list_(pa.float64())
            ),
            "outcome": pa.array(outcomes_col, type=pa.int8()),
            "substrate_sha": pa.array(substrate_shas_col, type=pa.string()),
        }
    )

    out_path = tmp_path / name
    pq.write_table(table, out_path)
    return out_path


# ── Task 1: Round-trip + SHA preservation + shape sanity ──────────────────


def test_round_trip_happy_path(tmp_path: Path) -> None:
    """Synthetic 2-slice snapshot loads into ``dict[str, EvalSlice]``
    with byte-identical feature vectors, outcomes, and SHAs.
    """
    parquet_path = _build_synthetic_snapshot(
        tmp_path,
        {"most_recent_12mo": (5, 3), "random_15pct": (4, 3)},
    )

    result = load_substrate_snapshot(parquet_path)

    assert set(result.keys()) == {"most_recent_12mo", "random_15pct"}
    for slice_name in result:
        assert isinstance(result[slice_name], EvalSlice)

    most_recent = result["most_recent_12mo"]
    assert len(most_recent.feature_vectors) == 5
    for row in most_recent.feature_vectors:
        assert isinstance(row, tuple)
        assert len(row) == 3

    # Deterministic alternation by row index (0,1,0,1,0 for n=5)
    assert most_recent.outcomes == (0, 1, 0, 1, 0)
    assert most_recent.substrate_sha == "sha-most_recent_12mo"

    # Frozen-dataclass field-by-field equality against the expected build.
    expected_fv = tuple(
        tuple(float(i * 0.1 + j) for j in range(3)) for i in range(5)
    )
    expected = EvalSlice(
        feature_vectors=expected_fv,
        outcomes=(0, 1, 0, 1, 0),
        substrate_sha="sha-most_recent_12mo",
    )
    assert most_recent == expected

    random_15 = result["random_15pct"]
    assert len(random_15.feature_vectors) == 4
    assert random_15.outcomes == (0, 1, 0, 1)
    assert random_15.substrate_sha == "sha-random_15pct"


def test_sha_preservation_byte_identical(tmp_path: Path) -> None:
    """A 64-char hex SHA (sha256 length) round-trips with no truncation,
    whitespace, or case mutation.
    """
    long_sha = "deadbeef" * 8  # 64 hex chars
    parquet_path = _build_synthetic_snapshot(
        tmp_path,
        {"slice_a": (3, 4)},
        sha_map={"slice_a": long_sha},
    )

    result = load_substrate_snapshot(parquet_path)

    assert result["slice_a"].substrate_sha == long_sha
    assert len(result["slice_a"].substrate_sha) == 64


def test_feature_vector_widths_per_slice_are_homogeneous(tmp_path: Path) -> None:
    """All feature-vectors in a slice share the declared width, and the
    loader returns them as Python ``tuple`` (not lists, not numpy arrays).
    """
    parquet_path = _build_synthetic_snapshot(
        tmp_path,
        {"slice_only": (3, 5)},
    )

    result = load_substrate_snapshot(parquet_path)

    slice_result = result["slice_only"]
    assert len(slice_result.feature_vectors) == 3
    for fv in slice_result.feature_vectors:
        assert isinstance(fv, tuple)
        assert len(fv) == 5
        for element in fv:
            assert isinstance(element, float)


def test_outcome_values_are_python_ints_not_numpy_scalars(tmp_path: Path) -> None:
    """The loader coerces ``int(o)`` per Plan 63-01 spec — numpy int8
    scalars would be ``numpy.int8`` (subclass of int but ``type() is`` test
    would fail). Asserts strict Python-int identity.
    """
    parquet_path = _build_synthetic_snapshot(
        tmp_path,
        {"slice_only": (3, 2)},
    )

    result = load_substrate_snapshot(parquet_path)

    outcomes = result["slice_only"].outcomes
    assert outcomes == (0, 1, 0)
    for outcome in outcomes:
        assert type(outcome) is int


# ── Task 2: R1-R8 rejection + A1-A2 acceptance ────────────────────────────


def _write_table(
    table: pa.Table, tmp_path: Path, name: str = "mutated.parquet"
) -> Path:
    """Helper: write a raw pyarrow Table to ``tmp_path`` and return the path.

    Used by reject-rule tests that need to bypass the locked schema in
    ``_build_synthetic_snapshot`` to inject a specific defect.
    """
    out = tmp_path / name
    pq.write_table(table, out)
    return out


def test_r1_missing_required_column_raises(tmp_path: Path) -> None:
    """R1: a parquet missing the ``outcome`` column is rejected with a
    message naming the missing column.
    """
    table = pa.table(
        {
            "slice_name": pa.array(["slice_a", "slice_a"], type=pa.string()),
            "feature_vector": pa.array(
                [[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float64())
            ),
            # outcome column intentionally omitted
            "substrate_sha": pa.array(["sha-a", "sha-a"], type=pa.string()),
        }
    )
    path = _write_table(table, tmp_path, "missing_outcome.parquet")

    with pytest.raises(ValueError, match=r"outcome") as excinfo:
        load_substrate_snapshot(path)

    msg = str(excinfo.value)
    assert "missing required column" in msg
    assert "'outcome'" in msg


def test_r2_wrong_dtype_on_outcome_raises(tmp_path: Path) -> None:
    """R2: ``outcome`` typed as string (instead of int8) is rejected with
    a message naming the column and the dtype mismatch.
    """
    table = pa.table(
        {
            "slice_name": pa.array(["slice_a", "slice_a"], type=pa.string()),
            "feature_vector": pa.array(
                [[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float64())
            ),
            "outcome": pa.array(["0", "1"], type=pa.string()),  # wrong dtype
            "substrate_sha": pa.array(["sha-a", "sha-a"], type=pa.string()),
        }
    )
    path = _write_table(table, tmp_path, "outcome_string.parquet")

    with pytest.raises(ValueError, match=r"outcome") as excinfo:
        load_substrate_snapshot(path)

    msg = str(excinfo.value)
    assert "dtype" in msg
    assert "int8" in msg  # expected dtype named


def test_r2_wrong_dtype_on_feature_vector_raises(tmp_path: Path) -> None:
    """R2: ``feature_vector`` typed as list<int32> instead of list<float64>
    is rejected with a message naming the column.
    """
    table = pa.table(
        {
            "slice_name": pa.array(["slice_a", "slice_a"], type=pa.string()),
            "feature_vector": pa.array(
                [[1, 2], [3, 4]], type=pa.list_(pa.int32())
            ),
            "outcome": pa.array([0, 1], type=pa.int8()),
            "substrate_sha": pa.array(["sha-a", "sha-a"], type=pa.string()),
        }
    )
    path = _write_table(table, tmp_path, "fv_int32.parquet")

    with pytest.raises(ValueError, match=r"feature_vector") as excinfo:
        load_substrate_snapshot(path)

    msg = str(excinfo.value)
    assert "dtype" in msg


def test_r3_outcome_outside_zero_one_raises(tmp_path: Path) -> None:
    """R3: a row with ``outcome=2`` is rejected. Message names the row
    index, the slice_name, and the offending value.
    """
    table = pa.table(
        {
            "slice_name": pa.array(
                ["slice_a", "slice_a", "slice_a", "slice_a"], type=pa.string()
            ),
            "feature_vector": pa.array(
                [[1.0], [2.0], [3.0], [4.0]], type=pa.list_(pa.float64())
            ),
            # Row 2 has outcome=2 (outside {0, 1})
            "outcome": pa.array([0, 1, 2, 0], type=pa.int8()),
            "substrate_sha": pa.array(["sha-a"] * 4, type=pa.string()),
        }
    )
    path = _write_table(table, tmp_path, "outcome_two.parquet")

    with pytest.raises(ValueError, match=r"outside \{0, 1\}") as excinfo:
        load_substrate_snapshot(path)

    msg = str(excinfo.value)
    assert "row 2" in msg
    assert "outcome=2" in msg
    assert "slice_name='slice_a'" in msg


def test_r4_empty_file_raises(tmp_path: Path) -> None:
    """R4: a parquet with the correct schema but zero rows is rejected."""
    schema = pa.schema(
        [
            ("slice_name", pa.string()),
            ("feature_vector", pa.list_(pa.float64())),
            ("outcome", pa.int8()),
            ("substrate_sha", pa.string()),
        ]
    )
    empty_table = pa.table(
        {
            "slice_name": pa.array([], type=pa.string()),
            "feature_vector": pa.array([], type=pa.list_(pa.float64())),
            "outcome": pa.array([], type=pa.int8()),
            "substrate_sha": pa.array([], type=pa.string()),
        },
        schema=schema,
    )
    path = _write_table(empty_table, tmp_path, "empty.parquet")

    with pytest.raises(ValueError, match=r"empty file \(zero rows\)"):
        load_substrate_snapshot(path)


def test_r6_conflicting_sha_within_slice_raises(tmp_path: Path) -> None:
    """R6: two different ``substrate_sha`` values within a single slice
    are rejected. Message names the slice and lists both SHAs.
    """
    table = pa.table(
        {
            "slice_name": pa.array(
                ["most_recent_12mo"] * 3, type=pa.string()
            ),
            "feature_vector": pa.array(
                [[1.0], [2.0], [3.0]], type=pa.list_(pa.float64())
            ),
            "outcome": pa.array([0, 1, 0], type=pa.int8()),
            # Rows 0-1 use sha-A; row 2 uses sha-B (R6 violation)
            "substrate_sha": pa.array(
                ["sha-A", "sha-A", "sha-B"], type=pa.string()
            ),
        }
    )
    path = _write_table(table, tmp_path, "conflicting_sha.parquet")

    with pytest.raises(ValueError, match=r"most_recent_12mo") as excinfo:
        load_substrate_snapshot(path)

    msg = str(excinfo.value)
    assert "conflicting substrate_sha" in msg
    assert "sha-A" in msg
    assert "sha-B" in msg


def test_r7_duplicate_sha_across_slices_raises(tmp_path: Path) -> None:
    """R7: two different slices sharing the same ``substrate_sha`` are
    rejected. Message names the duplicated SHA and both slice names.
    """
    parquet_path = _build_synthetic_snapshot(
        tmp_path,
        {"slice_a": (2, 2), "slice_b": (2, 2)},
        sha_map={"slice_a": "sha-collision", "slice_b": "sha-collision"},
    )

    with pytest.raises(ValueError, match=r"sha-collision") as excinfo:
        load_substrate_snapshot(parquet_path)

    msg = str(excinfo.value)
    assert "appears in multiple slices" in msg
    assert "slice_a" in msg
    assert "slice_b" in msg


def test_r8_nan_in_outcome_raises(tmp_path: Path) -> None:
    """R8: a NaN value in the ``outcome`` column is rejected.

    Because NaN is only representable in float dtypes, the underlying
    parquet must use ``pa.float64()`` for the outcome column. Plan 63-01's
    R2 check rejects float64 outcomes (expected dtype is int8), so this
    test accepts EITHER an R2 message ("dtype" / "outcome") OR an R8
    message ("NaN ... outcome") — the operator-facing intent is
    "NaN-in-outcome is rejected", regardless of which check fires first.
    """
    table = pa.table(
        {
            "slice_name": pa.array(
                ["slice_a", "slice_a", "slice_a"], type=pa.string()
            ),
            "feature_vector": pa.array(
                [[1.0], [2.0], [3.0]], type=pa.list_(pa.float64())
            ),
            # float64 outcome lets us store NaN; R2 will likely fire first.
            "outcome": pa.array(
                [0.0, float("nan"), 1.0], type=pa.float64()
            ),
            "substrate_sha": pa.array(["sha-a"] * 3, type=pa.string()),
        }
    )
    path = _write_table(table, tmp_path, "nan_outcome.parquet")

    with pytest.raises(ValueError, match=r"NaN|outcome"):
        load_substrate_snapshot(path)


def test_a1_width_agnostic_3_col_and_13_col_both_load(tmp_path: Path) -> None:
    """A1: the loader is width-agnostic. Both a 3-col snapshot (Phase 26
    META-V22 era) and a 13-col snapshot (current META-V22) load
    successfully. Proves the loader is NOT pinned to a meta-architecture.
    """
    path_3col = _build_synthetic_snapshot(
        tmp_path,
        {"slice_3col": (4, 3)},
        name="snap_3col.parquet",
    )
    path_13col = _build_synthetic_snapshot(
        tmp_path,
        {"slice_13col": (4, 13)},
        name="snap_13col.parquet",
    )

    result_3 = load_substrate_snapshot(path_3col)
    result_13 = load_substrate_snapshot(path_13col)

    assert all(len(fv) == 3 for fv in result_3["slice_3col"].feature_vectors)
    assert all(len(fv) == 13 for fv in result_13["slice_13col"].feature_vectors)


def test_a2_nan_in_feature_vector_preserved(tmp_path: Path) -> None:
    """A2: NaN values inside ``feature_vector`` elements are preserved by
    the loader. The gate verifier's training-time scaler / imputer
    handles NaN tolerance downstream — out of scope for the loader.
    """
    table = pa.table(
        {
            "slice_name": pa.array(["slice_a", "slice_a"], type=pa.string()),
            "feature_vector": pa.array(
                [[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]],
                type=pa.list_(pa.float64()),
            ),
            "outcome": pa.array([0, 1], type=pa.int8()),
            "substrate_sha": pa.array(["sha-a", "sha-a"], type=pa.string()),
        }
    )
    path = _write_table(table, tmp_path, "fv_with_nan.parquet")

    # Must NOT raise — NaN in feature_vector is permitted per A2.
    result = load_substrate_snapshot(path)

    fv_row_0 = result["slice_a"].feature_vectors[0]
    assert fv_row_0[0] == 1.0
    assert math.isnan(fv_row_0[1])  # NaN != NaN — use math.isnan
    assert fv_row_0[2] == 3.0

    fv_row_1 = result["slice_a"].feature_vectors[1]
    assert fv_row_1 == (4.0, 5.0, 6.0)
