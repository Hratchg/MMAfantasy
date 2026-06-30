"""TRUST-V24-04 smoke tests — locks calibration_v24.json + 3 PNG contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
JSON_PATH = RESULTS_DIR / "calibration_v24.json"
PNG_12MO = RESULTS_DIR / "calibration_v24_12mo.png"
PNG_24MO = RESULTS_DIR / "calibration_v24_24mo.png"
PNG_RANDOM = RESULTS_DIR / "calibration_v24_random_15pct.png"
SHA_START_PATH = (
    PROJECT_ROOT
    / ".planning/phases/34-trust-baseline-no-odds-fallback"
    / "34-META-V2-SHA-PHASE-34-START-PLAN-01.txt"
)
META_JOBLIB_PATH = PROJECT_ROOT / "models" / "meta" / "meta_v2.joblib"


@pytest.fixture(scope="module")
def doc() -> dict:
    if not JSON_PATH.exists():
        pytest.skip("results/calibration_v24.json missing")
    return json.loads(JSON_PATH.read_text())


def test_json_has_three_slices(doc: dict) -> None:
    assert set(doc["per_slice"].keys()) == {
        "most_recent_12mo",
        "most_recent_24mo",
        "random_15pct",
    }


def test_each_slice_has_required_calib_keys(doc: dict) -> None:
    required = {
        "ece",
        "platt_brier",
        "isotonic_brier_reused_from_phase_32",
        "platt_vs_isotonic_brier_delta",
        "winner",
    }
    for sl, body in doc["per_slice"].items():
        missing = required - set(body.keys())
        assert not missing, f"slice {sl} missing: {missing}"


def test_isotonic_source_documented(doc: dict) -> None:
    src = doc["platt_vs_isotonic_summary"]["isotonic_source"]
    assert src.endswith("COMPOSITION_V23_REPORT.json")


def test_isotonic_source_path_in_json_documented(doc: dict) -> None:
    """Revision M4 — asserts isotonic_source_path_in_json mentions candidate_brier_per_slice."""
    path_str = doc["platt_vs_isotonic_summary"]["isotonic_source_path_in_json"]
    assert isinstance(path_str, str) and len(path_str) > 0
    assert "candidate_brier_per_slice" in path_str


def test_three_pngs_exist_and_nonempty() -> None:
    for p in (PNG_12MO, PNG_24MO, PNG_RANDOM):
        assert p.exists(), f"missing: {p}"
        assert p.stat().st_size > 1024, f"undersized: {p}"


def test_ece_in_unit_interval(doc: dict) -> None:
    for sl, body in doc["per_slice"].items():
        ece = body["ece"]
        if ece is None:
            continue
        assert 0.0 <= ece <= 1.0, f"slice {sl} ECE out of [0,1]: {ece}"


def test_meta_v2_sha_unchanged_vs_start_anchor() -> None:
    if not SHA_START_PATH.exists() or not META_JOBLIB_PATH.exists():
        pytest.skip("missing SHA anchor or meta_v2.joblib")
    expected = SHA_START_PATH.read_text().strip()
    actual = hashlib.sha256(META_JOBLIB_PATH.read_bytes()).hexdigest()
    assert actual == expected


def test_bin_strategy_is_uniform_10(doc: dict) -> None:
    """CONTEXT lock — uniform 10-bin ECE."""
    assert doc["ece_bin_strategy"] == "uniform_10_bin"


def test_bootstrap_n_is_1000(doc: dict) -> None:
    """CONTEXT lock — 1000 bootstrap samples per bin."""
    assert doc["bootstrap_n_per_bin"] == 1000
