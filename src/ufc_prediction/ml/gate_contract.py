"""Versioned promotion-gate contract loaded from .planning/gate_contract.json.

Per CONTEXT.md D-04(P17): per-slice (brier_max, accuracy_min) tuples;
per D-05(P17): thresholds derived from `median ± 1·max(seed_std, bootstrap_ci_half)`;
per D-08(P17): schema includes formula_hash + cutoff_date + supersedes list.

The contract is process-lifetime stable (operator owns the JSON); we use
@lru_cache on load_gate_contract() so callers can hot-path
gate_verdict without re-parsing JSON on every call.

lru_cache discipline (Pitfall C):
  Callers that mutate `.planning/gate_contract.json` at runtime (testing,
  spike, hot-reload scenarios) MUST call `load_gate_contract.cache_clear()`
  after the file mutation. The cache key is (path, version); maxsize=2
  accommodates both v2.1 and v2.2 contracts simultaneously.

v2.2 dispatch (Phase 24):
  `load_gate_contract(version="v2.2")` resolves to V22_CONTRACT_PATH.
  v2.1 remains the back-compat default. SUPPORTED_VERSIONS expanded
  to include v2.2; GateContract dataclass gains feature_columns_hash
  and bfo_backfill_committed_at fields (required for v2.2; empty
  strings preserved for v2.1 back-compat).

v2.3 dispatch (Phase 31; per CONTEXT D-01..D-07):
  `load_gate_contract(version="v2.3")` resolves to V23_CONTRACT_PATH
  (`.planning/gate_contract_v2.3.json`). v2.1 + v2.2 contracts are
  PRESERVED untouched per GATE-V23-02 audit-lineage binding. The
  GateContract dataclass gains the v2.3-only required field
  `ingest_completed_at` (Phase 28 INGEST_COVERAGE.json commit ISO
  timestamp); empty on v2.1+v2.2 back-compat. SUPPORTED_VERSIONS
  expanded to include v2.3; lru_cache bumped to maxsize=3 so all three
  versions can coexist (Pitfall 6). DEFAULT_VERSION stays "v2.1" per
  CONTEXT deferred ideas (Phase 32 owns the bump).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / ".planning" / "gate_contract.json"
)
V22_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / ".planning" / "gate_contract_v2.2.json"
)
V23_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / ".planning" / "gate_contract_v2.3.json"
)
# v2.6 dispatch (Phase 55 GATE-V26-04). Adds methodology fields
# (methodology_version, substrate_alignment_strategy, gate_verifier_module,
# confound_threshold) on top of v2.3 schema. Per_slice numbers preserved
# verbatim from v2.3 at Phase 55 ship; Phase 56 GATE-RECALIB CLI may
# re-derive them on a corpus-growth trigger.
V26_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / ".planning" / "gate_contract_v2.6.json"
)
SUPPORTED_VERSIONS: frozenset[str] = frozenset({"v2.1", "v2.2", "v2.3", "v2.6"})
EXPECTED_SLICES: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)


class GateContractError(ValueError):
    """Raised when .planning/gate_contract.json is missing or malformed."""


@dataclass(frozen=True)
class PerSliceThresholds:
    brier_max: float
    accuracy_min: float
    median_brier_xgb_v2: float
    median_acc_xgb_v2: float
    seed_std_brier: float
    seed_std_acc: float
    bootstrap_ci_half_brier: float
    bootstrap_ci_half_acc: float
    std_brier_used: float
    std_acc_used: float
    # v2.2-only operator-decision metadata (CR-01 / D-06 / D-21).
    # Present on v2.2 contracts to record whether the operator empirical
    # floor was applied at spike time and which post-HALT branch was chosen.
    # Defaults preserve v2.1 back-compat (loader accepts contracts without
    # these keys; they remain None on v2.1).
    operator_floor_applied: bool | None = None
    operator_floor_value: float | None = None
    operator_decision: str | None = None


@dataclass(frozen=True)
class GateContract:
    version: str
    derived_at: str
    n_seeds_observed: int
    base_features_set: str
    n_features: int
    k_value: int
    formula_hash: str
    cutoff_date: str
    per_slice: dict[str, PerSliceThresholds]
    secondary_metrics_observed: dict[str, dict[str, float]] = field(default_factory=dict)
    supersedes: list[str] = field(default_factory=list)
    notes: str = ""
    feature_columns_hash: str = ""
    bfo_backfill_committed_at: str = ""
    # v2.3-only required field (Phase 31; CONTEXT D-02 + Pitfall 7).
    # Phase 28 INGEST_COVERAGE.json commit ISO timestamp marking when
    # referee/venue ingestion CLOSED. Empty on v2.1+v2.2 back-compat.
    ingest_completed_at: str = ""
    # v2.2-only top-level operator-decision block (CR-01 / D-06 / D-21).
    # Records the post-HALT branch (decision, decided_at, rationale,
    # phase_26_implication). Empty on v2.1 contracts.
    operator_decision: dict[str, str] | None = None
    # v2.6-only methodology block (Phase 55 GATE-V26-04). Empty on
    # v2.1/v2.2/v2.3 back-compat. References .planning/gate_methodology_v2.6.md
    # spec as authority for the v2.6 substrate-drift-safe verifier.
    methodology_version: str = ""
    substrate_alignment_strategy: str = ""
    gate_verifier_module: str = ""
    confound_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.version not in SUPPORTED_VERSIONS:
            msg = (
                f"Unsupported contract version: {self.version!r}; "
                f"expected one of {sorted(SUPPORTED_VERSIONS)}"
            )
            raise GateContractError(msg)
        missing = set(EXPECTED_SLICES) - set(self.per_slice.keys())
        if missing:
            msg = f"Missing slice thresholds in gate_contract.json: {sorted(missing)}"
            raise GateContractError(msg)
        if self.k_value <= 0:
            msg = f"k_value must be positive (got {self.k_value})"
            raise GateContractError(msg)
        # Per-slice sanity: brier_max in [0, 1]; accuracy_min in [0.5, 1].
        for slice_name in EXPECTED_SLICES:
            ts = self.per_slice[slice_name]
            if not 0.0 <= ts.brier_max <= 1.0:
                msg = f"{slice_name}: brier_max {ts.brier_max} not in [0, 1]"
                raise GateContractError(msg)
            if not 0.5 <= ts.accuracy_min <= 1.0:
                msg = f"{slice_name}: accuracy_min {ts.accuracy_min} not in [0.5, 1]"
                raise GateContractError(msg)
        # v2.2 + v2.3 required fields (CONTEXT D-07 v2.2 / CONTEXT D-02 v2.3).
        if self.version in ("v2.2", "v2.3"):
            if not self.feature_columns_hash:
                msg = (
                    f"{self.version} contract missing required field: "
                    "feature_columns_hash"
                )
                raise GateContractError(msg)
            if not self.bfo_backfill_committed_at:
                msg = (
                    f"{self.version} contract missing required field: "
                    "bfo_backfill_committed_at"
                )
                raise GateContractError(msg)
        # v2.3-only required field (Phase 31; CONTEXT D-02 + Pitfall 7).
        if self.version == "v2.3" and not self.ingest_completed_at:
            msg = "v2.3 contract missing required field: ingest_completed_at"
            raise GateContractError(msg)
        # v2.6 required methodology fields (Phase 55 GATE-V26-04).
        if self.version == "v2.6":
            for required_field in (
                "methodology_version",
                "substrate_alignment_strategy",
                "gate_verifier_module",
            ):
                if not getattr(self, required_field):
                    msg = (
                        f"v2.6 contract missing required field: "
                        f"{required_field}"
                    )
                    raise GateContractError(msg)
            if not 0.0 < self.confound_threshold <= 1.0:
                msg = (
                    f"v2.6 contract confound_threshold {self.confound_threshold} "
                    "must be in (0, 1]"
                )
                raise GateContractError(msg)


@lru_cache(maxsize=4)
def load_gate_contract(
    path: Path | None = None,
    *,
    version: str = "v2.1",
) -> GateContract:
    """Lazy-load .planning/gate_contract*.json with cached parse.

    Args:
        path: Override path (testing only). Production callers pass None.
        version: Contract version to load. ``"v2.1"`` (default) loads the
            v2.1 contract at ``DEFAULT_CONTRACT_PATH``. ``"v2.2"`` loads
            ``V22_CONTRACT_PATH``. ``"v2.3"`` loads ``V23_CONTRACT_PATH``.
            Unknown versions raise ``ValueError``.

    Raises:
        ValueError: unknown version string.
        GateContractError: file missing, invalid JSON, schema mismatch.
    """
    if path is None:
        if version == "v2.1":
            contract_path = DEFAULT_CONTRACT_PATH
        elif version == "v2.2":
            contract_path = V22_CONTRACT_PATH
        elif version == "v2.3":
            contract_path = V23_CONTRACT_PATH
        elif version == "v2.6":
            contract_path = V26_CONTRACT_PATH
        else:
            raise ValueError(f"unknown version={version!r}")
    else:
        contract_path = path
    if not contract_path.exists():
        msg = f"gate_contract.json not found at {contract_path}"
        raise GateContractError(msg)
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"gate_contract.json malformed: {exc}"
        raise GateContractError(msg) from exc
    try:
        per_slice = {k: PerSliceThresholds(**v) for k, v in raw["per_slice"].items()}
        return GateContract(
            version=raw["version"],
            derived_at=raw["derived_at"],
            n_seeds_observed=raw["n_seeds_observed"],
            base_features_set=raw["base_features_set"],
            n_features=raw["n_features"],
            k_value=raw["k_value"],
            formula_hash=raw["formula_hash"],
            cutoff_date=raw["cutoff_date"],
            per_slice=per_slice,
            secondary_metrics_observed=raw.get("secondary_metrics_observed", {}),
            supersedes=raw.get("supersedes", []),
            notes=raw.get("notes", ""),
            feature_columns_hash=raw.get("feature_columns_hash", ""),
            bfo_backfill_committed_at=raw.get("bfo_backfill_committed_at", ""),
            ingest_completed_at=raw.get("ingest_completed_at", ""),
            operator_decision=raw.get("operator_decision"),
            methodology_version=raw.get("methodology_version", ""),
            substrate_alignment_strategy=raw.get("substrate_alignment_strategy", ""),
            gate_verifier_module=raw.get("gate_verifier_module", ""),
            confound_threshold=raw.get("confound_threshold", 0.0),
        )
    except KeyError as exc:
        msg = f"gate_contract.json missing required field: {exc}"
        raise GateContractError(msg) from exc
    except TypeError as exc:
        msg = f"gate_contract.json field-type mismatch: {exc}"
        raise GateContractError(msg) from exc
