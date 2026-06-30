"""Phase 18 Wave-0 RED — assert gate_contract.json is post-Phase-17, not v2.0.

D-06(P18) / Pitfall #16 carry-forward: Phase 17 verifier flagged a 4th-decimal
narrative slip in PROJECT.md D-18 row, so the JSON file is the binding source
for the v2.1 promotion gate — not narrative documentation.

This test reads `.planning/gate_contract.json` DIRECTLY (NOT via
`load_gate_contract()`) so it does not depend on `gate_contract.py` plumbing.
Drift in the JSON's `formula_hash` or per-slice thresholds halts Phase 18 and
any downstream phase that consumes the gate contract.

This is a RED test in the process-discipline sense: it ships passing at commit
time (Phase 17 already produced the contract), but exists as a tripwire so any
future drift is caught immediately.
"""

import json
import pathlib


def test_gate_contract_thresholds_are_v2_1_not_v2_0():
    """D-06(P18): gate_contract.json must contain Phase-17-derived thresholds.

    Reads the JSON DIRECTLY (not via load_gate_contract()) so this test does
    not depend on src plumbing — Phase 17 verifier flagged a 4th-decimal
    narrative slip in PROJECT.md D-18 row; the JSON is the binding source.
    """
    contract_path = pathlib.Path(".planning/gate_contract.json")
    assert contract_path.exists(), "gate_contract.json missing — Phase 17 not closed; halt Phase 18"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    # Sanity: top-level fields wired by Phase 17
    assert contract["version"] == "v2.1", f"contract.version != v2.1; got {contract['version']!r}"
    assert contract["formula_hash"] == (
        "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
    ), "Phase 17 operator-approved formula_hash drifted"

    # Per-slice thresholds — at least ONE of (brier_max < 0.215, accuracy_min > 0.67)
    # must hold per slice, otherwise the threshold equals v2.0's aspirational
    # default and Phase 17 has not actually run.
    for slice_name in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        slice_data = contract["per_slice"][slice_name]
        v2_0_default_brier = 0.215
        v2_0_default_acc = 0.67
        is_phase_17_derived = (
            slice_data["brier_max"] < v2_0_default_brier
            or slice_data["accuracy_min"] > v2_0_default_acc
        )
        assert is_phase_17_derived, (
            f"{slice_name}: brier_max={slice_data['brier_max']}, "
            f"accuracy_min={slice_data['accuracy_min']} — both equal v2.0 "
            f"defaults; Phase 17 did not run; halt Phase 18"
        )
