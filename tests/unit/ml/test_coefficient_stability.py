"""Phase 26 META-V22-04: coefficient_stability_report unit tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ufc_prediction.ml.coefficient_stability import (
    coefficient_stability_report,
    format_coefficient_stability_markdown,
    write_coefficient_stability_json,
)
from ufc_prediction.ml.meta_learner import MetaLearnerLogistic


@pytest.fixture
def per_seed_meta_synthetic():
    """5 fitted MetaLearnerLogistic instances on a small synthetic 3-col fixture.

    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) on 3
    columns produces 3 + C(3,2) = 6 features. So coef_[0].size == 6, and
    `get_feature_names_out(...)` returns a length-6 list.
    """
    n = 200
    rng = np.random.default_rng(0)
    X_meta = rng.uniform(0.0, 1.0, size=(n, 3))
    y = rng.integers(0, 2, size=n)
    per_seed = {}
    for seed in (42, 43, 44, 45, 46):
        meta = MetaLearnerLogistic(random_state=seed).fit(X_meta, y)
        per_seed[seed] = meta
    feature_names = per_seed[42].pipeline.named_steps["poly"].get_feature_names_out(
        ["a", "b", "c"],
    ).tolist()
    return per_seed, feature_names


def test_per_feature_std_shape(per_seed_meta_synthetic):
    per_seed, feature_names = per_seed_meta_synthetic
    report = coefficient_stability_report(per_seed, feature_names)
    assert set(report.keys()) == {
        "feature_names",
        "per_feature_std",
        "per_feature_sign_agreement",
        "median_abs_coef",
        "n_seeds",
    }
    assert report["n_seeds"] == 5
    n_features = len(feature_names)
    assert np.asarray(report["per_feature_std"]).shape == (n_features,)
    assert np.asarray(report["per_feature_sign_agreement"]).shape == (n_features,)
    assert np.asarray(report["median_abs_coef"]).shape == (n_features,)


def test_sign_agreement_in_unit_interval(per_seed_meta_synthetic):
    per_seed, feature_names = per_seed_meta_synthetic
    report = coefficient_stability_report(per_seed, feature_names)
    sign_arr = np.asarray(report["per_feature_sign_agreement"])
    assert (sign_arr >= 0.0).all()
    assert (sign_arr <= 1.0).all()


def test_emits_to_json(tmp_path, per_seed_meta_synthetic):
    per_seed, feature_names = per_seed_meta_synthetic
    report = coefficient_stability_report(per_seed, feature_names)
    out = tmp_path / "stability.json"
    write_coefficient_stability_json(report, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {
        "feature_names",
        "per_feature_std",
        "per_feature_sign_agreement",
        "median_abs_coef",
        "n_seeds",
    }
    # JSON round-trip preserves length.
    assert len(loaded["per_feature_std"]) == len(feature_names)


def test_emits_markdown_rows(per_seed_meta_synthetic):
    per_seed, feature_names = per_seed_meta_synthetic
    report = coefficient_stability_report(per_seed, feature_names)
    md = format_coefficient_stability_markdown(report)
    lines = md.split("\n")
    # one header + one separator + N data rows
    assert len(lines) == 2 + len(feature_names), (
        f"expected {2 + len(feature_names)} lines, got {len(lines)}"
    )
    # Each data row contains feature_name + 3 numeric fields (4 |-separated columns).
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        assert len(cells) == 4
        # The first cell is a feature name we constructed.
        assert cells[0] in feature_names or " " in cells[0]
