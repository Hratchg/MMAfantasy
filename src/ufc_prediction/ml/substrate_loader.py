"""Phase 63 CRIT-V261-01 — substrate-snapshot parquet loader.

Public surface: a single function ``load_substrate_snapshot`` that
materializes a flat-row substrate-snapshot parquet into the
``dict[str, EvalSlice]`` shape consumed by
``verify_candidate_vs_canonical(...)`` (Phase 55
`src/ufc_prediction/ml/gate_verifier.py:222`).

Contract reference: ``.planning/gate_methodology_v2.6.md`` §6.1 specifies
the verifier's ``dict[str, EvalSlice]`` input contract. This loader is
the only sanctioned I/O surface that builds that mapping from a parquet
file.

Width-agnostic by design (CONTEXT §D-03): the loader does NOT pin
``feature_vector`` to META-V22's current 13-col contract. META-V22 itself
grew from 3 cols (META-01) to 13 cols (Phase 23); pinning here would
couple the file format to a specific meta architecture. Width mismatch
surfaces downstream when ``predict_proba`` rejects the array — at the
root cause, with a clean sklearn ``ValueError``.

Module-import cost is intentionally cheap: pandas and pyarrow are NOT
imported at module scope. They are deferred to function scope inside
``load_substrate_snapshot`` (mirroring the pattern at
``gate_verifier.py:170-179`` for numpy / sklearn). Importing this module
costs only ``pathlib.Path`` and the ``EvalSlice`` dataclass — safe for
CLI startup paths that may never touch a substrate.

mypy-strict clean (Phase 51 admit-list, DX-V26-02 / DX-V26-03 expansion).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ufc_prediction.ml.gate_verifier import EvalSlice


# ── Module-private schema constants ───────────────────────────────────────
#
# Locked parquet schema per CONTEXT §D-01 (flat per-prediction rows).
# Kept module-private so the schema contract is self-documenting in the
# module but not part of the public API (Plan 63-02's fixture builder
# may import these via ``substrate_loader._REQUIRED_*`` if it wants to
# mirror them byte-identical; that import is allowed but unsupported).

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "slice_name",
    "feature_vector",
    "outcome",
    "substrate_sha",
)

# Dtype strings verified against ``pyarrow.parquet.read_schema(...)``
# rendering (pyarrow 24.0.0 — UFC project pinned interpreter). NOTE the
# in-memory pyarrow type ``str(pa.list_(pa.float64()))`` renders as
# ``"list<item: double>"`` but after a round-trip through parquet the
# read-back schema renames the child field to the parquet-LIST canonical
# label ``element`` — so ``str(schema.field('feature_vector').type)``
# returns ``"list<element: double>"``. This loader compares against the
# read-back schema (the only thing it can see on disk), so the dtype-pin
# is ``"list<element: double>"``. Fixture builders that write parquet via
# ``pq.write_table(..., type=pa.list_(pa.float64()))`` will round-trip to
# this label automatically. If a future pyarrow release renames the
# child label, R2 ValueError will surface immediately and name the
# actual vs expected dtype.
_REQUIRED_DTYPES: dict[str, str] = {
    "slice_name": "string",
    "feature_vector": "list<element: double>",
    "outcome": "int8",
    "substrate_sha": "string",
}


# ── Public API ────────────────────────────────────────────────────────────


def load_substrate_snapshot(path: Path) -> dict[str, EvalSlice]:
    """Load a substrate-snapshot parquet into the verifier-input shape.

    Reads the flat per-prediction-row parquet format defined in
    ``.planning/phases/63-substrate-snapshot-loader-crit-v261-01/63-CONTEXT.md``
    §D-01 and groups rows by ``slice_name`` into one ``EvalSlice`` per
    group, producing the ``dict[str, EvalSlice]`` consumed by
    ``ufc_prediction.ml.gate_verifier.verify_candidate_vs_canonical``.

    Locked schema (D-01):
        - ``slice_name``     (string)           partition key
        - ``feature_vector`` (list<float64>)    per-row variable-length
        - ``outcome``        (int8)             values in {0, 1}
        - ``substrate_sha``  (string)           repeated within a slice

    Reject rules (D-03 — every error names the offending column / row /
    value concretely):
        R1. Missing required column.
        R2. Wrong pyarrow dtype on any required column.
        R3. ``outcome`` value outside {0, 1}.
        R4. Empty file (zero rows).
        R5. Slice with zero rows after group-by (vacuously prevented by
            R4 + pandas groupby semantics; asserted defensively).
        R6. Conflicting ``substrate_sha`` values within a single
            ``slice_name`` group.
        R7. Duplicate ``substrate_sha`` values across different
            ``slice_name`` groups (each slice needs its own SHA so the
            audit trail in ``EvalSlice.substrate_sha`` stays unique).
        R8. NaN in ``outcome``.

    Accept rules (D-03):
        A1. Any ``len(feature_vector)`` consistent within a slice — the
            loader does NOT pin width to META-V22's 13-col contract.
            Width mismatch surfaces downstream when
            ``predict_proba`` rejects the array.
        A2. NaN allowed inside ``feature_vector`` elements. The gate
            verifier handles NaN tolerance via the canonical Pipeline's
            training-time scaler / imputer.

    Per-slice ``substrate_sha`` strings pass through byte-identical to
    the input parquet value (preserved on ``EvalSlice.substrate_sha``)
    so the Phase 55 verdict-artifact audit trail stays intact.

    Args:
        path: Filesystem path to the substrate-snapshot parquet. Accepts
            ``str`` defensively and coerces to ``Path``; CLI plumbing may
            pass either.

    Returns:
        Mapping from ``slice_name`` to populated ``EvalSlice``. Iteration
        order follows pandas groupby (lexicographic slice_name in current
        pandas; do not rely on this).

    Raises:
        ValueError: If any of reject rules R1-R8 fire. The message names
            the offending path + column / row / value concretely.

    Example:
        >>> from pathlib import Path
        >>> from ufc_prediction.ml.substrate_loader import (
        ...     load_substrate_snapshot,
        ... )
        >>> from ufc_prediction.ml.gate_verifier import (
        ...     verify_candidate_vs_canonical,
        ... )
        >>> slices = load_substrate_snapshot(Path("substrate.parquet"))
        >>> verdict = verify_candidate_vs_canonical(
        ...     candidate=Path("models/meta/meta_v3_candidate.joblib"),
        ...     canonical=Path("models/meta/meta_v2.joblib"),
        ...     eval_slices=slices,
        ... )
    """
    # Function-scope imports — keep module import cheap. Pattern mirrors
    # `gate_verifier.py:170-179` (numpy / sklearn deferred to function
    # scope) and `predictor.py` style (pandas with engine='pyarrow').
    import math

    import pandas as pd
    import pyarrow.parquet as _pq_module

    # ``pyarrow.parquet`` ships partial stubs whose ``read_schema`` is
    # annotated as untyped, triggering mypy --strict's
    # ``no-untyped-call`` rule. Treating the module as ``Any`` mirrors
    # the ``pipeline: Any`` pattern in ``gate_verifier.py`` and
    # contains the untyped-3p-API surface to this single binding.
    pq: Any = _pq_module

    # Defensive coercion: signature is ``Path`` but CLI plumbing may
    # pass ``str``. ``Path(Path(x)) == Path(x)`` so this is idempotent.
    path = Path(path)

    # ── R1 + R2: schema validation BEFORE materializing rows ──────────
    # ``pq.read_schema`` reads only the parquet footer (cheap, no row
    # decoding) so we catch column / dtype errors before paying the
    # ``pd.read_parquet`` cost on a malformed file.
    schema: Any = pq.read_schema(path)
    schema_names: list[str] = list(schema.names)

    for col in _REQUIRED_COLUMNS:
        if col not in schema_names:
            raise ValueError(
                f"substrate snapshot {path}: missing required column "
                f"{col!r} — required columns are {_REQUIRED_COLUMNS}"
            )

    for col, expected_dtype in _REQUIRED_DTYPES.items():
        actual_dtype = str(schema.field(col).type)
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"substrate snapshot {path}: column {col!r} has dtype "
                f"{actual_dtype!r}, expected {expected_dtype!r}"
            )

    # ── Materialize via pandas (pyarrow engine, matches predictor.py) ──
    df: pd.DataFrame = pd.read_parquet(path, engine="pyarrow")

    # ── R4: empty file ────────────────────────────────────────────────
    if len(df) == 0:
        raise ValueError(
            f"substrate snapshot {path}: empty file (zero rows)"
        )

    # ── R8: NaN in outcome — MUST run BEFORE R3 ─────────────────────
    # ``int(NaN)`` would silently coerce to a non-{0,1} sentinel in
    # some platforms; check the NaN mask first so the error message
    # names "NaN" explicitly instead of bottoming out in R3.
    outcome_nan_mask: pd.Series = df["outcome"].isna()
    if bool(outcome_nan_mask.any()):
        nan_row_indices: list[int] = [
            int(i) for i in df.index[outcome_nan_mask].tolist()
        ]
        raise ValueError(
            f"substrate snapshot {path}: NaN in outcome column at "
            f"row(s) {nan_row_indices}"
        )

    # ── R3: outcome value outside {0, 1} ──────────────────────────────
    # Iterate by positional index so the error message reports a
    # row index the operator can locate in the source parquet.
    for row_idx in range(len(df)):
        outcome_value: Any = df["outcome"].iloc[row_idx]
        # Defensive: re-check NaN here in case R8 missed a corner case
        # (e.g., numpy NaN vs pandas NA distinction). ``math.isnan``
        # raises TypeError on non-floats, so guard with isinstance.
        if isinstance(outcome_value, float) and math.isnan(outcome_value):
            raise ValueError(
                f"substrate snapshot {path}: NaN in outcome column at "
                f"row {row_idx}"
            )
        if int(outcome_value) not in (0, 1):
            slice_name_for_msg: Any = df["slice_name"].iloc[row_idx]
            raise ValueError(
                f"substrate snapshot {path}: slice_name="
                f"{slice_name_for_msg!r} row {row_idx}: outcome="
                f"{outcome_value} outside {{0, 1}}"
            )

    # ── Group by slice_name and build EvalSlice per group ─────────────
    result: dict[str, EvalSlice] = {}
    slice_shas: dict[str, str] = {}

    for slice_name, group in df.groupby("slice_name", sort=True):
        # ``slice_name`` from groupby is the partition key value
        # (string per the locked schema). Cast defensively in case
        # pandas hands back a non-str scalar.
        slice_key: str = str(slice_name)

        # R5 sanity — vacuously satisfied because pandas groupby never
        # yields empty groups for keys present in the index, but
        # asserted explicitly so a future pandas change is caught.
        if len(group) == 0:
            raise ValueError(
                f"substrate snapshot {path}: slice {slice_key!r} has "
                f"zero rows after group-by (R5 violation)"
            )

        # R6: per-slice substrate_sha must be a single unique value
        unique_shas_arr: Any = group["substrate_sha"].unique()
        unique_shas: list[str] = [str(s) for s in unique_shas_arr.tolist()]
        if len(unique_shas) != 1:
            raise ValueError(
                f"substrate snapshot {path}: slice {slice_key!r} has "
                f"conflicting substrate_sha values: {sorted(unique_shas)}"
            )
        slice_sha: str = unique_shas[0]

        # Build feature_vectors as immutable nested tuples. Each fv
        # arrives as a numpy array or list (depending on parquet driver
        # version) — both iterable. NaN elements are preserved per A2.
        feature_vectors: tuple[tuple[float, ...], ...] = tuple(
            tuple(float(v) for v in fv) for fv in group["feature_vector"]
        )

        # Build outcomes as immutable int tuple
        outcomes: tuple[int, ...] = tuple(
            int(o) for o in group["outcome"]
        )

        result[slice_key] = EvalSlice(
            feature_vectors=feature_vectors,
            outcomes=outcomes,
            substrate_sha=slice_sha,
        )
        slice_shas[slice_key] = slice_sha

    # ── R7: duplicate substrate_sha across different slices ──────────
    sha_to_slices: dict[str, list[str]] = {}
    for slice_name, sha in slice_shas.items():
        sha_to_slices.setdefault(sha, []).append(slice_name)
    for dup_sha, colliding_slice_names in sha_to_slices.items():
        if len(colliding_slice_names) > 1:
            raise ValueError(
                f"substrate snapshot {path}: substrate_sha "
                f"{dup_sha!r} appears in multiple slices: "
                f"{sorted(colliding_slice_names)}"
            )

    return result
