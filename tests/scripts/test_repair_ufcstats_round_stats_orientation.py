"""Tests for scripts/repair_ufcstats_round_stats_orientation.py.

The script repairs ufcstats ``round_stats`` rows that the pre-fix scraper
stored under the opponent's ``fighter_id`` (fighter-detail page order differs
from the event page's winner-first order). No network: pages come from the
scraper HTML fixtures via a mock client.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy.orm import Session

from ufc_prediction.models.fight import Fight
from ufc_prediction.models.round_stats import RoundStats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "repair_ufcstats_round_stats_orientation.py"
SCRAPER_FIXTURES = PROJECT_ROOT / "tests" / "scraper" / "fixtures"


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "repair_ufcstats_round_stats_orientation", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["repair_ufcstats_round_stats_orientation"] = m
    spec.loader.exec_module(m)
    return m


def _scraper_test_module() -> ModuleType:
    """Reuse the scraper ingest tests' mock client + fixture wiring."""
    from tests.scraper import test_ingest

    return test_ingest


def _ingest_first_event(session: Session) -> ModuleType:
    ti = _scraper_test_module()
    ti._scrape_first_event(session, ti._make_mock_client())
    return ti


def _stats_by_fighter(session: Session, fight_id: int) -> dict[int, dict[int, tuple]]:
    from collections import defaultdict

    out: dict[int, dict[int, tuple]] = defaultdict(dict)
    for rs in session.query(RoundStats).filter(RoundStats.fight_id == fight_id):
        out[rs.fighter_id][rs.round_number] = (rs.sig_strikes_landed, rs.knockdowns)
    return dict(out)


def _cross_round_stats(session: Session, fight: Fight) -> None:
    """Reproduce the pre-fix bug: every row stored under the opponent."""
    a, b = fight.fighter_a_id, fight.fighter_b_id
    for rs in session.query(RoundStats).filter(RoundStats.fight_id == fight.id):
        rs.fighter_id = b if rs.fighter_id == a else a
    session.flush()


def test_classify_distinguishes_correct_swapped_unmatched(mod: ModuleType) -> None:
    a = {0: (10, 1), 1: (10, 1)}
    b = {0: (4, 0), 1: (4, 0)}
    assert mod.classify(a, b, a, b) == "correct"
    assert mod.classify(b, a, a, b) == "swapped"
    assert mod.classify(a, {0: (99, 0)}, a, b) == "unmatched"
    assert mod.classify({}, {}, a, b) == "no_stats"


def test_repair_swaps_only_crossed_fights_and_is_idempotent(
    mod: ModuleType, session: Session, tmp_path: Path
) -> None:
    ti = _ingest_first_event(session)
    fights = {f.source_url: f for f in session.query(Fight).all()}
    crossed = fights[ti._FIGHT_URL_ULBERG]
    untouched = fights[ti._FIGHT_URL_BRADY]

    good_crossed = _stats_by_fighter(session, crossed.id)
    good_untouched = _stats_by_fighter(session, untouched.id)
    _cross_round_stats(session, crossed)
    assert _stats_by_fighter(session, crossed.id) != good_crossed  # bug reproduced

    # Report-only: classifies, writes nothing.
    report = mod.repair(session, ti._make_mock_client(), apply=False, cache_dir=tmp_path)
    assert report.counts["swapped"] == 1
    assert report.counts["correct"] == 2
    assert report.swapped_fight_ids == [crossed.id]
    assert _stats_by_fighter(session, crossed.id) != good_crossed

    # Apply: the crossed fight is restored, the others are untouched.
    report = mod.repair(session, ti._make_mock_client(), apply=True, cache_dir=tmp_path)
    assert report.swapped_fight_ids == [crossed.id]
    session.expire_all()
    assert _stats_by_fighter(session, crossed.id) == good_crossed
    assert _stats_by_fighter(session, untouched.id) == good_untouched

    # Re-run: content-based classification makes it a no-op.
    report = mod.repair(session, ti._make_mock_client(), apply=True, cache_dir=tmp_path)
    assert report.swapped_fight_ids == []
    assert report.counts["correct"] == 3


def test_repair_orients_by_hex_id_when_page_order_is_reversed(
    mod: ModuleType, session: Session, tmp_path: Path
) -> None:
    """A fight ingested by the FIXED scraper from a page that lists the fighters
    in the opposite order to the event page is stored correctly; the repair
    must orient that page by hex id and leave the fight alone."""
    ti = _scraper_test_module()
    original = (SCRAPER_FIXTURES / "fight_detail_1round.html").read_text(encoding="utf-8")
    reversed_page = ti._swap_fighter_identities(original, ti._ULBERG, ti._PROCHAZKA)

    client = ti._make_mock_client()
    client._exact_map[ti._FIGHT_URL_ULBERG] = reversed_page
    ti._scrape_first_event(session, client)

    report = mod.repair(session, client, apply=True, cache_dir=tmp_path)
    assert report.counts["correct"] == 3
    assert report.swapped_fight_ids == []


def test_repair_reports_page_mismatch_and_fetch_failure(
    mod: ModuleType, session: Session, tmp_path: Path
) -> None:
    ti = _ingest_first_event(session)
    client = ti._RaisingMockScraperClient(
        fixture_map=ti._make_mock_client()._fixture_map,
        exact_map={
            **ti._make_mock_client()._exact_map,
            ti._FIGHT_URL_ULBERG: (SCRAPER_FIXTURES / "fight_detail_3round.html").read_text(
                encoding="utf-8"
            ),
        },
        raising_urls={ti._FIGHT_URL_MORALES},
    )
    report = mod.repair(session, client, apply=True, cache_dir=tmp_path)
    assert report.counts["page_mismatch"] == 1
    assert report.counts["fetch_failed"] == 1
    assert report.counts["correct"] == 1
    assert report.swapped_fight_ids == []


def test_repair_serves_pages_from_cache(mod: ModuleType, session: Session, tmp_path: Path) -> None:
    ti = _ingest_first_event(session)
    mod.repair(session, ti._make_mock_client(), apply=False, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.html"))) == 3

    offline = ti.MockScraperClient({})  # any network fetch would fail
    report = mod.repair(session, offline, apply=False, cache_dir=tmp_path)
    assert report.counts["correct"] == 3
    assert offline._call_log == []
