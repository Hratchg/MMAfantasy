#!/usr/bin/env python3
"""Emit committed partner-contract artifacts. Run ONCE per v1.x.y bump.

Per Phase 25 D-01: PredictorOutputV1 is the source of truth.
Per Phase 25 D-07: FastAPI auto-emits OpenAPI 3.1.0 via app.openapi().
Per Phase 25 D-05: xgb_v2-contract.json is a SIBLING artifact (NOT a sub-object of meta).
Per Phase 32 D-04 (PARTNER-V23-01): emits v1.1.0 sibling files alongside v1.0.0.
Per Phase 69 API-V261-01 (v2.6.1): emits v1.3.0 sibling files alongside v1.2.0;
  v1.2.0 is now BYTE-FROZEN (joins v1.0.0 + v1.1.0 under the forward-compat lock).

Emission targets (all under repo root):
  1. src/ufc_prediction/contracts/predictor.schema.v1.0.0.json   (BYTE-FROZEN — Phase 25 lock)
  2. src/ufc_prediction/contracts/openapi.v1.0.0.json            (BYTE-FROZEN — Phase 25 lock)
  3. src/ufc_prediction/contracts/predictor.schema.v1.1.0.json   (BYTE-FROZEN — Phase 35 lock)
  4. src/ufc_prediction/contracts/openapi.v1.1.0.json            (BYTE-FROZEN — Phase 35 lock)
  5. src/ufc_prediction/contracts/predictor.schema.v1.2.0.json   (BYTE-FROZEN — Phase 69 lock)
  6. src/ufc_prediction/contracts/openapi.v1.2.0.json            (BYTE-FROZEN — Phase 69 lock)
  7. src/ufc_prediction/contracts/predictor.schema.v1.3.0.json   (Phase 69 — live source-of-truth)
  8. src/ufc_prediction/contracts/openapi.v1.3.0.json            (Phase 69 — live source-of-truth)
  9. models/xgb_v2-contract.json                                 (BYTE-FROZEN — Phase 25 lock)
 10. models/xgb_v2-contract.v1.1.0.json                          (BYTE-FROZEN — Phase 69 lock)

AUDIT-01 chain-leaf invariant (xgb_v2.joblib SHA equals canonical) is
asserted at the end of each run; no per-phase MID/END artifact is written
(those existed under `.planning/phases/32-.../` which has since been
archived to `.planning/milestones/v2.3-phases/32-.../`).

V100_FROZEN / V110_FROZEN / V120_FROZEN bindings (Pitfall 5 mitigation):
from Phase 32 / 35 / 69 forward this script does NOT call `.write_text()`
on the v1.0.0, v1.1.0, or v1.2.0 schema or openapi paths. Those files
were emitted ONCE at their owning phase close and are byte-frozen by
the forward-compat lock. Any subsequent Pydantic edit implies a version
bump (v1.0.0 → v1.1.0 → v1.2.0 → v1.3.0 → ...), and the new version gets
its own sibling files. Defensive `assert *.exists()` proves the frozen
files were emitted by a prior run.

WR-03 (Phase 25 review-fix): emission logic is wrapped in main() + the
`if __name__ == "__main__":` guard so import does not silently overwrite
committed JSON files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ufc_prediction.api.app import create_app
from ufc_prediction.api.v1.models import PredictorOutputV1

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "src" / "ufc_prediction" / "contracts"

# v1.0.0 artifacts — Phase 25 forward-compat lock binding (Pitfall 5).
SCHEMA_V100_PATH = CONTRACTS_DIR / "predictor.schema.v1.0.0.json"
OPENAPI_V100_PATH = CONTRACTS_DIR / "openapi.v1.0.0.json"
V100_FROZEN = True  # Phase 32 D-04 binding — do NOT overwrite v1.0.0 files.

# v1.1.0 artifacts — Phase 32 D-04 sibling (BYTE-FROZEN as of Phase 35).
SCHEMA_V110_PATH = CONTRACTS_DIR / "predictor.schema.v1.1.0.json"
OPENAPI_V110_PATH = CONTRACTS_DIR / "openapi.v1.1.0.json"
V110_FROZEN = True  # Phase 35 binding — do NOT overwrite v1.1.0 files.

# v1.2.0 artifacts — Phase 35 CONTRACT-V24-02 sibling (BYTE-FROZEN as of Phase 69).
SCHEMA_V120_PATH = CONTRACTS_DIR / "predictor.schema.v1.2.0.json"
OPENAPI_V120_PATH = CONTRACTS_DIR / "openapi.v1.2.0.json"
V120_FROZEN = True  # Phase 69 API-V261-01 binding — do NOT overwrite v1.2.0 files.

# v1.3.0 artifacts — Phase 69 API-V261-01 sibling (live source-of-truth).
# Success-path bytes IDENTICAL to v1.2.0 by design (PredictorOutputV1 unchanged
# since Phase 52; the v1.3.0 capability is the RFC 7807 error wrapper opt-in
# via `Accept: application/problem+json`, not a success-path schema change).
# The openapi.v1.3.0.json captures the LIVE FastAPI app surface, including
# `accept_schema_version` Literal extension to admit "1.3.0" + any endpoints
# mounted between Phase 35 (v1.2.0 freeze) and Phase 69 close.
SCHEMA_V130_PATH = CONTRACTS_DIR / "predictor.schema.v1.3.0.json"
OPENAPI_V130_PATH = CONTRACTS_DIR / "openapi.v1.3.0.json"

# Canonical xgb_v2.joblib SHA — AUDIT-01 chain leaf invariant.
XGB_V2_CANONICAL_SHA = (
    "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
)


def _check_audit01_xgb_v2_sha(repo_root: Path) -> str:
    """Assert xgb_v2.joblib SHA matches the AUDIT-01 canonical invariant.

    Phase 32 originally wrote a per-plan "MID SHA" artifact file under
    `.planning/phases/32-.../`. That directory has since been archived to
    `.planning/milestones/v2.3-phases/32-.../` (commit 6db877d), so writing
    a fresh sibling at the original path would resurrect a stale phase
    directory each emit-script run. Phase 69 API-V261-01 collapses the
    side effect to a pure check: SHA mismatch raises; no file is written.

    Returns the validated SHA so the caller can echo it to the operator.
    """
    joblib_path = repo_root / "models" / "xgb_v2.joblib"
    sha = hashlib.sha256(joblib_path.read_bytes()).hexdigest()
    if sha != XGB_V2_CANONICAL_SHA:
        msg = (
            f"AUDIT-01 drift! xgb_v2.joblib SHA = {sha[:12]}...; "
            f"expected {XGB_V2_CANONICAL_SHA[:12]}..."
        )
        raise RuntimeError(msg)
    return sha


def main() -> None:
    """Emit partner-contract artifacts.

    Idempotent — running twice with the same inputs produces byte-identical
    output for both v1.0.0 (Phase 25) and v1.1.0 (Phase 32 D-04) files.
    """
    # ── Phase 25 byte-frozen artifacts (Pitfall 5 binding) ────────────────
    # We do NOT re-emit v1.0.0 schema or openapi from the LIVE Pydantic
    # model — post-Phase-32 the model emits a SUPERSET (v1.1.0 shape) and
    # overwriting the v1.0.0 file would corrupt it. Defensive assert that
    # the v1.0.0 files exist (from Phase 25's emit run) but otherwise
    # leave them alone.
    assert V100_FROZEN, "Phase 32 D-04 binding — V100_FROZEN must stay True"
    assert SCHEMA_V100_PATH.exists(), (
        f"Phase 25 v1.0.0 schema missing: {SCHEMA_V100_PATH}. "
        "This script does not re-emit v1.0.0 (forward-compat lock binding); "
        "restore the file from a prior commit before running."
    )
    assert OPENAPI_V100_PATH.exists(), (
        f"Phase 25 v1.0.0 openapi missing: {OPENAPI_V100_PATH}. "
        "This script does not re-emit v1.0.0; restore from a prior commit."
    )
    print(f"v1.0.0 frozen: {SCHEMA_V100_PATH.relative_to(REPO_ROOT)} (unchanged)")
    print(f"v1.0.0 frozen: {OPENAPI_V100_PATH.relative_to(REPO_ROOT)} (unchanged)")

    # ── Phase 35 byte-frozen v1.1.0 artifacts (Phase 35 lock binding) ─────
    # v1.1.0 was emitted live by Phase 32, then byte-frozen at Phase 35
    # when v1.2.0 became the new live source-of-truth. Defensive assert.
    assert V110_FROZEN, "Phase 35 binding — V110_FROZEN must stay True"
    assert SCHEMA_V110_PATH.exists(), (
        f"v1.1.0 schema missing: {SCHEMA_V110_PATH}. "
        "Phase 35+ does not re-emit v1.1.0 (byte-frozen by forward-compat "
        "lock); restore from a prior commit before running."
    )
    assert OPENAPI_V110_PATH.exists(), (
        f"v1.1.0 openapi missing: {OPENAPI_V110_PATH}. "
        "Phase 35+ does not re-emit v1.1.0; restore from a prior commit."
    )
    print(f"v1.1.0 frozen: {SCHEMA_V110_PATH.relative_to(REPO_ROOT)} (unchanged)")
    print(f"v1.1.0 frozen: {OPENAPI_V110_PATH.relative_to(REPO_ROOT)} (unchanged)")

    # ── Phase 69 byte-frozen v1.2.0 artifacts (Phase 69 lock binding) ─────
    # v1.2.0 was emitted live by Phase 35 (CONTRACT-V24-02), then byte-frozen
    # at Phase 69 when v1.3.0 became the new live source-of-truth. The
    # PredictorOutputV1 success-path bytes are IDENTICAL between v1.2.0 and
    # v1.3.0 — the v1.3.0 capability is RFC 7807 on the error path (Phase 52
    # API-V26-01) — so the v1.2.0 freeze is the canonical partner contract
    # for partners not opting into the problem+json error wrapper.
    assert V120_FROZEN, "Phase 69 binding — V120_FROZEN must stay True"
    assert SCHEMA_V120_PATH.exists(), (
        f"v1.2.0 schema missing: {SCHEMA_V120_PATH}. "
        "Phase 69+ does not re-emit v1.2.0 (byte-frozen by forward-compat "
        "lock); restore from a prior commit before running."
    )
    assert OPENAPI_V120_PATH.exists(), (
        f"v1.2.0 openapi missing: {OPENAPI_V120_PATH}. "
        "Phase 69+ does not re-emit v1.2.0; restore from a prior commit."
    )
    print(f"v1.2.0 frozen: {SCHEMA_V120_PATH.relative_to(REPO_ROOT)} (unchanged)")
    print(f"v1.2.0 frozen: {OPENAPI_V120_PATH.relative_to(REPO_ROOT)} (unchanged)")

    # ── Phase 69 API-V261-01 v1.3.0 sibling artifacts ─────────────────────

    # 1. v1.3.0 Pydantic JSON Schema (Draft 2020-12). Success-path bytes
    #    IDENTICAL to v1.2.0 by design — PredictorOutputV1 has not changed
    #    since Phase 52. We emit from the live model (no separate v1.3.0
    #    Pydantic class) and the byte-equality is enforced by the
    #    test_v130_schema_equals_v120_schema regression test.
    schema_v130 = PredictorOutputV1.model_json_schema()
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_V130_PATH.write_text(
        json.dumps(schema_v130, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"emitted {SCHEMA_V130_PATH.relative_to(REPO_ROOT)}")

    # 2. v1.3.0 OpenAPI 3.1 spec (FastAPI auto-emit). Live snapshot of the
    #    full app surface including accept_schema_version Literal expansion
    #    to "1.3.0" + any endpoints mounted between Phase 35 and Phase 69.
    app = create_app()
    openapi_v130 = app.openapi()
    assert openapi_v130["openapi"] == "3.1.0", (
        f"FastAPI emitted OpenAPI {openapi_v130['openapi']!r}, expected '3.1.0'."
    )
    OPENAPI_V130_PATH.write_text(
        json.dumps(openapi_v130, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"emitted {OPENAPI_V130_PATH.relative_to(REPO_ROOT)}")

    # 3. v1.0.0 xgb_v2 sibling contract — BYTE-FROZEN (Phase 25 D-05).
    # save_contract_json sets created_at to today(), which makes re-emission
    # non-deterministic on calendar boundaries. The v1.0.0 partner contract
    # was emitted ONCE at Phase 25 close and is byte-frozen by the
    # forward-compat lock binding. Defensive assert that the file exists.
    v100_contract_path = REPO_ROOT / "models" / "xgb_v2-contract.json"
    assert v100_contract_path.exists(), (
        f"Phase 25 v1.0.0 partner contract missing: {v100_contract_path}. "
        "This script does not re-emit (created_at is non-deterministic); "
        "restore from a prior commit."
    )
    print(f"v1.0.0 frozen: {v100_contract_path.relative_to(REPO_ROOT)} (unchanged)")

    # 4. v1.1.0 xgb_v2 sibling contract — BYTE-FROZEN (Phase 69 lock binding).
    # Same rationale as v1.0.0 contract: created_at is non-deterministic
    # (date.today()) so re-emission drifts on calendar boundaries. The file
    # was emitted ONCE at Phase 32-02 close and is now byte-frozen.
    # Defensive assert that the file exists.
    v110_contract_path = REPO_ROOT / "models" / "xgb_v2-contract.v1.1.0.json"
    assert v110_contract_path.exists(), (
        f"v1.1.0 xgb_v2 contract missing: {v110_contract_path}. "
        "Phase 69+ does not re-emit (created_at is non-deterministic); "
        "restore from a prior commit."
    )
    print(f"v1.1.0 frozen: {v110_contract_path.relative_to(REPO_ROOT)} (unchanged)")

    # 5. AUDIT-01 chain-leaf invariant check (Phase 69 collapse of the
    #    Plan 32-02 MID SHA artifact — see _check_audit01_xgb_v2_sha
    #    docstring for the rationale).
    audit_sha = _check_audit01_xgb_v2_sha(REPO_ROOT)
    print(f"AUDIT-01 ok: xgb_v2.joblib SHA = {audit_sha[:12]}...")


if __name__ == "__main__":
    main()
