"""Serve-path debutant seeds: loud on a missing CSV, and never cache an empty result.

``_load_debutant_seeds`` feeds ``_get_latest_elo`` the Sherdog debutant seed for
overall Elo. The stored substrate (``elo_snapshots.elo_before``) was built with
those seeds, so a missing CSV silently skews every debutant's serve-time
``elo_overall_diff`` toward flat 1500. That must be logged, and a CSV that
appears later must be picked up rather than masked by a cached ``{}``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ufc_prediction.ml import inference_features

_CSV = "fighter_id,n_pre_ufc_fights,win_rate,org_tier\n7,10,0.8,major\n"


@pytest.fixture(autouse=True)
def _fresh_seed_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_features, "_debutant_seeds_cache", None, raising=False)
    monkeypatch.setattr(inference_features, "_warned_missing_seeds", False, raising=False)


def test_missing_csv_logs_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "pre_ufc_records.csv"
    monkeypatch.setattr(inference_features, "_SHERDOG_PRE_UFC_CSV", missing)

    with caplog.at_level(logging.ERROR, logger=inference_features.__name__):
        assert inference_features._load_debutant_seeds() == {}

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert str(missing) in errors[0].getMessage()


def test_missing_csv_error_logged_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(inference_features, "_SHERDOG_PRE_UFC_CSV", tmp_path / "missing.csv")

    with caplog.at_level(logging.ERROR, logger=inference_features.__name__):
        for _ in range(5):
            inference_features._load_debutant_seeds()

    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


def test_empty_result_not_cached_csv_added_later_is_picked_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "pre_ufc_records.csv"
    monkeypatch.setattr(inference_features, "_SHERDOG_PRE_UFC_CSV", csv)

    assert inference_features._load_debutant_seeds() == {}

    csv.write_text(_CSV)
    seeds = inference_features._load_debutant_seeds()
    assert set(seeds) == {7}


def test_non_empty_result_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = tmp_path / "pre_ufc_records.csv"
    csv.write_text(_CSV)
    monkeypatch.setattr(inference_features, "_SHERDOG_PRE_UFC_CSV", csv)

    first = inference_features._load_debutant_seeds()
    csv.unlink()
    assert inference_features._load_debutant_seeds() is first
