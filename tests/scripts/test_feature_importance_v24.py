"""TRUST-V24-03 smoke tests — locks the contract of results/feature_importance_v24.{json,png×2}."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
JSON_PATH = RESULTS_DIR / "feature_importance_v24.json"
PNG_XGB = RESULTS_DIR / "feature_importance_v24_xgb_v2.png"
PNG_META = RESULTS_DIR / "feature_importance_v24_meta_v22.png"
CANONICAL_XGB_V2_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
SHA_START_PATH = (
    PROJECT_ROOT
    / ".planning/phases/34-trust-baseline-no-odds-fallback"
    / "34-META-V2-SHA-PHASE-34-START-PLAN-01.txt"
)


@pytest.fixture(scope="module")
def doc() -> dict:
    if not JSON_PATH.exists():
        pytest.skip("results/feature_importance_v24.json missing")
    return json.loads(JSON_PATH.read_text())


def test_json_top_level_keys_present(doc: dict) -> None:
    for k in [
        "created_at", "xgb_v2_sha256", "meta_v2_sha256",
        "shap_sample_target", "shap_sample_n_actual", "shap_sample_strategy",
        "perm_slice", "perm_n_repeats", "perm_scoring",
        "xgb_v2", "meta_v22", "closing_prob_diff_claim_verdict",
    ]:
        assert k in doc, f"missing top-level key: {k}"


def test_xgb_v2_has_tier_description(doc: dict) -> None:
    td = doc["xgb_v2"]["tier_description"]
    assert isinstance(td, str) and len(td) > 0
    assert "base model" in td.lower()


def test_meta_v22_has_tier_description(doc: dict) -> None:
    td = doc["meta_v22"]["tier_description"]
    assert isinstance(td, str) and len(td) > 0
    assert "level-1" in td.lower() or "polynomial" in td.lower()


def test_both_pngs_exist_and_nonempty() -> None:
    for p in (PNG_XGB, PNG_META):
        assert p.exists(), f"missing: {p}"
        assert p.stat().st_size > 1024, f"undersized PNG: {p} = {p.stat().st_size} bytes"


def test_closing_prob_diff_claim_verdict_present(doc: dict) -> None:
    v = doc["closing_prob_diff_claim_verdict"]
    assert isinstance(v["xgb_v2_rank"], int)
    assert isinstance(v["meta_v22_rank"], int)
    assert isinstance(v["verdict_string"], str)
    assert len(v["verdict_string"]) > 20


def test_shap_sample_strategy_is_stratified(doc: dict) -> None:
    assert doc["shap_sample_strategy"].startswith("stratified")


def test_shap_sample_n_actual_recorded(doc: dict) -> None:
    """Revision m2 — `shap_sample_n_actual` is an int ≤ `shap_sample_target`."""
    n_actual = doc["shap_sample_n_actual"]
    n_target = doc["shap_sample_target"]
    assert isinstance(n_actual, int)
    assert n_actual <= n_target


def test_sha_anchors_match_canonical(doc: dict) -> None:
    assert doc["xgb_v2_sha256"] == CANONICAL_XGB_V2_SHA
    if SHA_START_PATH.exists():
        assert doc["meta_v2_sha256"] == SHA_START_PATH.read_text().strip()
