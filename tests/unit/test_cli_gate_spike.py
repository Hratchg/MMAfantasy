"""Phase 31 / GATE-V23-01 — CLI gate-spike subcommand unit tests (Wave 0 RED).

Tests the ``ufc predict gate-spike`` Typer subcommand body in
``src/ufc_prediction/cli/predict.py`` — focuses on:
  - Flag translation: ``--seeds N`` (count) -> explicit ``[42..41+N]`` list
    (Pitfall 2 regression)
  - Artifact paths routed to PHASE_31_DIR (Pitfall 8)
  - D-24 precondition check (Pitfall 1; GATE-V23-04 SC4 binding)
  - Module-level constants asserted at import (D-18 formula hash, AUDIT-01 SHA,
    Path A 0.70 operator floor)

Pattern: monkeypatch ``spike_noise_floor_v23.main`` to a stub that captures
argv but does NOT run the heavy harness. Pre/post the spike call we also
stub out the helpers that touch DB / matrices / contract emission so the
unit-level dispatch can be exercised purely.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Module-level constant assertions (no CLI invocation needed)
# ---------------------------------------------------------------------------


def test_cli_formula_hash_constant_matches_d18():
    """predict.EXPECTED_FORMULA_HASH equals the D-18 binding hash."""
    from ufc_prediction.cli.predict import EXPECTED_FORMULA_HASH

    assert (
        EXPECTED_FORMULA_HASH == "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
    )


def test_cli_xgb_v2_sha_constant_matches_audit_01_baseline():
    """predict.EXPECTED_XGB_V2_SHA equals the AUDIT-01 baseline SHA."""
    from ufc_prediction.cli.predict import EXPECTED_XGB_V2_SHA

    assert EXPECTED_XGB_V2_SHA == "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"


def test_cli_operator_floor_constant_is_path_a_0_70():
    """predict.OPERATOR_FLOOR_V23 == 0.70 (Path A pre-commit per CONTEXT D-01)."""
    from ufc_prediction.cli.predict import OPERATOR_FLOOR_V23

    assert OPERATOR_FLOOR_V23 == 0.70


def test_cli_phase_31_dir_constant_points_to_phase_31():
    """predict.PHASE_31_DIR points to the Phase 31 phase directory."""
    from ufc_prediction.cli.predict import PHASE_31_DIR

    assert PHASE_31_DIR.name == "31-gate-v23-re-derivation-on-populated-substrate"


# ---------------------------------------------------------------------------
# Flag translation tests (Pitfall 2)
# ---------------------------------------------------------------------------


def _patch_spike_environment(
    monkeypatch,
    project_md_text: str = "| **D-24(v2.3, GATE)** | Phase 31 GATE row |",
    *,
    isolated_phase_dir=None,
):
    """Monkeypatch helpers so the CLI dispatches without DB / contract writes.

    Returns a list that will be populated with ``(args, kwargs)`` for each
    spike_main invocation so the test can assert argv translation.

    If ``isolated_phase_dir`` is provided, PHASE_31_DIR is monkeypatched to
    that path so Path A side effects (END SHA artifact + cache_clear) write
    to a tmp dir instead of polluting the real Phase 31 directory.
    """
    captured: list[list[str]] = []

    def fake_spike_main(argv):
        captured.append(list(argv) if argv is not None else [])
        return 0  # success path; spike "completes"

    def fake_assert_xgb_v2_sha(label):
        return "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"

    # Patch the spike_noise_floor_v23 module surface (CLI uses module import)
    from ufc_prediction.cli import predict as pred_mod

    monkeypatch.setattr(
        pred_mod,
        "_spike_main",
        fake_spike_main,
        raising=False,
    )
    monkeypatch.setattr(
        pred_mod,
        "_assert_xgb_v2_sha",
        fake_assert_xgb_v2_sha,
        raising=False,
    )
    if isolated_phase_dir is not None:
        monkeypatch.setattr(
            pred_mod,
            "PHASE_31_DIR",
            isolated_phase_dir,
            raising=False,
        )

    # Stub the variance + contract emission so we don't need DB / live state.
    monkeypatch.setattr(
        pred_mod,
        "_compute_v23_variance_dict",
        lambda seed_list, bootstrap=True: (
            {
                "most_recent_12mo": {
                    "seed_std_brier": 0.003,
                    "seed_std_acc": 0.005,
                    "bootstrap_ci_half_brier": 0.004,
                    "bootstrap_ci_half_acc": 0.006,
                    "std_brier_used": 0.004,
                    "std_acc_used": 0.006,
                    "median_brier": 0.18,
                    "median_acc": 0.75,
                },
                "most_recent_24mo": {
                    "seed_std_brier": 0.003,
                    "seed_std_acc": 0.005,
                    "bootstrap_ci_half_brier": 0.004,
                    "bootstrap_ci_half_acc": 0.006,
                    "std_brier_used": 0.004,
                    "std_acc_used": 0.006,
                    "median_brier": 0.19,
                    "median_acc": 0.74,
                },
                "random_15pct": {
                    "seed_std_brier": 0.003,
                    "seed_std_acc": 0.005,
                    "bootstrap_ci_half_brier": 0.004,
                    "bootstrap_ci_half_acc": 0.006,
                    "std_brier_used": 0.004,
                    "std_acc_used": 0.006,
                    "median_brier": 0.17,
                    "median_acc": 0.76,
                },
            },
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pred_mod,
        "_emit_v23_contract",
        lambda *a, **kw: None,
        raising=False,
    )
    monkeypatch.setattr(
        pred_mod,
        "_read_project_md",
        lambda: project_md_text,
        raising=False,
    )

    return captured


def test_cli_flag_translation_seeds_10(monkeypatch, tmp_path):
    """--seeds 10 translates to [42..51] explicit list (Pitfall 2)."""
    captured = _patch_spike_environment(monkeypatch, isolated_phase_dir=tmp_path)
    from ufc_prediction.cli.predict import predict_gate_spike

    predict_gate_spike(
        feature_set="v2.3",
        seeds=10,
        bootstrap=True,
        contract_path=str(tmp_path / "gate_contract_v2.3.json"),
    )

    assert captured, "spike_main was not invoked"
    argv = captured[0]
    assert "--seeds" in argv
    seeds_idx = argv.index("--seeds")
    seed_strings = argv[seeds_idx + 1 : seeds_idx + 11]
    assert seed_strings == [
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
    ]


def test_cli_flag_translation_seeds_3(monkeypatch, tmp_path):
    """--seeds 3 translates to [42, 43, 44] (smallest realistic count)."""
    captured = _patch_spike_environment(monkeypatch, isolated_phase_dir=tmp_path)
    from ufc_prediction.cli.predict import predict_gate_spike

    predict_gate_spike(
        feature_set="v2.3",
        seeds=3,
        bootstrap=True,
        contract_path=str(tmp_path / "gate_contract_v2.3.json"),
    )

    argv = captured[0]
    seeds_idx = argv.index("--seeds")
    seed_strings = argv[seeds_idx + 1 : seeds_idx + 4]
    assert seed_strings == ["42", "43", "44"]


def test_cli_seed_translation_is_explicit_list_not_count(monkeypatch, tmp_path):
    """Regression for Pitfall 2: '10' must NOT appear as a single token after --seeds."""
    captured = _patch_spike_environment(monkeypatch, isolated_phase_dir=tmp_path)
    from ufc_prediction.cli.predict import predict_gate_spike

    predict_gate_spike(
        feature_set="v2.3",
        seeds=10,
        bootstrap=True,
        contract_path=str(tmp_path / "gate_contract_v2.3.json"),
    )

    argv = captured[0]
    seeds_idx = argv.index("--seeds")
    # The token immediately after --seeds is the FIRST seed (42), not the count.
    assert argv[seeds_idx + 1] != "10"
    assert argv[seeds_idx + 1] == "42"


# ---------------------------------------------------------------------------
# Artifact path tests (Pitfall 8)
# ---------------------------------------------------------------------------


def test_cli_artifact_paths_route_to_phase_31(monkeypatch, tmp_path):
    """CLI passes --report-path / --halt-path / --sha-end-path to Phase 31 dir."""
    captured = _patch_spike_environment(monkeypatch, isolated_phase_dir=tmp_path)
    from ufc_prediction.cli.predict import predict_gate_spike

    predict_gate_spike(
        feature_set="v2.3",
        seeds=3,
        bootstrap=True,
        contract_path=str(tmp_path / "gate_contract_v2.3.json"),
    )

    argv = captured[0]
    assert "--report-path" in argv
    report_idx = argv.index("--report-path")
    assert argv[report_idx + 1].endswith("31-VARIANCE-REPORT.md")

    assert "--halt-path" in argv
    halt_idx = argv.index("--halt-path")
    assert argv[halt_idx + 1].endswith("31-VARIANCE-HALT.md")

    assert "--sha-end-path" in argv
    sha_idx = argv.index("--sha-end-path")
    assert argv[sha_idx + 1].endswith("31-XGB-V2-SHA-PHASE-31-END-SPIKE.txt")


# ---------------------------------------------------------------------------
# D-24 precondition test (Pitfall 1)
# ---------------------------------------------------------------------------


def test_cli_d24_precondition_fails_without_row(monkeypatch, tmp_path):
    """CLI exits SystemExit(2) if D-24(v2.3, GATE) row missing from PROJECT.md."""
    # Patch with PROJECT.md text that LACKS the D-24 row.
    _patch_spike_environment(
        monkeypatch,
        project_md_text="# PROJECT.md\n## Key Decisions\n(no D-24 row here)\n",
        isolated_phase_dir=tmp_path,
    )
    from ufc_prediction.cli.predict import predict_gate_spike

    with pytest.raises(SystemExit) as exc_info:
        predict_gate_spike(
            feature_set="v2.3",
            seeds=3,
            bootstrap=True,
            contract_path=str(tmp_path / "gate_contract_v2.3.json"),
        )
    assert exc_info.value.code == 2


def test_cli_d24_precondition_rejects_bare_substring_false_positive(
    monkeypatch,
    tmp_path,
):
    """WR-06: a bare 'D-24(v2.3, GATE)' substring in narrative or a commented-
    out line must NOT satisfy the precondition. The check requires the
    bold-pipe table-row format `| **D-24(v2.3, GATE)** |`.
    """
    # PROJECT.md contains the token only as a narrative reference / quote /
    # commented-out citation — no committed decision row.
    false_positive_md = (
        "# PROJECT.md\n\n"
        "Historical note: the previous D-24(v2.3, GATE) attempt was deferred.\n"
        "<!-- TODO: re-add D-24(v2.3, GATE) row once partner-readiness gate -->\n"
    )
    _patch_spike_environment(
        monkeypatch,
        project_md_text=false_positive_md,
        isolated_phase_dir=tmp_path,
    )
    from ufc_prediction.cli.predict import predict_gate_spike

    with pytest.raises(SystemExit) as exc_info:
        predict_gate_spike(
            feature_set="v2.3",
            seeds=3,
            bootstrap=True,
            contract_path=str(tmp_path / "gate_contract_v2.3.json"),
        )
    assert exc_info.value.code == 2


def test_cli_helpers_exist():
    """The five required helpers are exported from predict.py."""
    from ufc_prediction.cli.predict import (
        _apply_operator_floor_and_detect_breach,
        _build_per_slice_thresholds_v23,
        _compute_per_seed_medians,
        _emit_v23_contract,
        _render_31_halt_and_decide,
    )

    # If imports succeed, helpers are registered.
    assert callable(_apply_operator_floor_and_detect_breach)
    assert callable(_build_per_slice_thresholds_v23)
    assert callable(_compute_per_seed_medians)
    assert callable(_emit_v23_contract)
    assert callable(_render_31_halt_and_decide)


# ---------------------------------------------------------------------------
# WR-02: _render_31_halt_and_decide is defensive against missing slice keys
# ---------------------------------------------------------------------------


def test_render_halt_and_decide_handles_missing_slice():
    """Missing slice key in breach_diagnostic renders '(missing)' row, not KeyError.

    WR-02 regression: the HALT writer is the operator's only safety net on
    Path D. A KeyError here would leave the operator with no HALT artifact
    AND no contract; we now fall back to a sentinel row instead of raising.
    """
    from ufc_prediction.cli.predict import _render_31_halt_and_decide

    # Only two of three expected slices are present.
    partial_diagnostic = {
        "most_recent_12mo": {
            "pre_floor_accuracy_min": 0.68,
            "post_floor_accuracy_min": 0.70,
            "breach": True,
            "gap": 0.02,
        },
        "most_recent_24mo": {
            "pre_floor_accuracy_min": 0.72,
            "post_floor_accuracy_min": 0.72,
            "breach": False,
            "gap": 0.0,
        },
        # random_15pct intentionally omitted
    }
    body = _render_31_halt_and_decide(
        partial_diagnostic,
        triggered_at="2026-05-22T20:30:00+00:00",
    )
    # Present slices render normally.
    assert "| most_recent_12mo | 0.6800 | 0.7000 | 0.0200 | YES |" in body
    assert "| most_recent_24mo | 0.7200 | 0.7200 | 0.0000 | NO |" in body
    # Missing slice still renders, with sentinel marker.
    assert "| random_15pct | (missing) | (missing) | n/a | UNKNOWN |" in body
    # Frontmatter still emitted (no mid-write crash).
    assert "phase: 31-gate-v23-re-derivation-on-populated-substrate" in body
