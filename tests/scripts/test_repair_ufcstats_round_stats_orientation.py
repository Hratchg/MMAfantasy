"""Tests for scripts/repair_ufcstats_round_stats_orientation.py.

The script repairs ufcstats ``round_stats`` rows that the pre-fix scraper
stored under the opponent's ``fighter_id`` (fighter-detail page order differs
from the event page's winner-first order). No network: pages come from the
scraper HTML fixtures via a mock client.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy.orm import Session

from tests.scripts.twin_seed import BLUE_STATS, NO_STATS, RED_STATS, add_fight, round0
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


# ── Cache hygiene: never cache a challenge / unparseable page ───────────────

CHALLENGE_HTML = (SCRAPER_FIXTURES / "ufcstats_pow_challenge.html").read_text(encoding="utf-8")


def _cache_file(mod: ModuleType, cache_dir: Path, url: str) -> Path:
    return cache_dir / f"{mod._cache_key(url)}.html"


def test_challenge_page_is_a_fetch_failure_and_is_not_cached(
    mod: ModuleType, session: Session, tmp_path: Path
) -> None:
    ti = _ingest_first_event(session)
    client = ti._make_mock_client()
    client._exact_map[ti._FIGHT_URL_BRADY] = CHALLENGE_HTML

    report = mod.repair(session, client, apply=False, cache_dir=tmp_path)

    assert report.counts["fetch_failed"] == 1
    assert report.counts.get("parse_failed", 0) == 0
    assert report.counts["correct"] == 2
    assert not _cache_file(mod, tmp_path, ti._FIGHT_URL_BRADY).exists()


def test_unparseable_page_is_not_cached(mod: ModuleType, session: Session, tmp_path: Path) -> None:
    ti = _ingest_first_event(session)
    client = ti._make_mock_client()
    client._exact_map[ti._FIGHT_URL_BRADY] = "<html><body>no persons here</body></html>"

    report = mod.repair(session, client, apply=False, cache_dir=tmp_path)

    assert report.counts["parse_failed"] == 1
    assert not _cache_file(mod, tmp_path, ti._FIGHT_URL_BRADY).exists()
    assert len(list(tmp_path.glob("*.html"))) == 2


def test_poisoned_cache_entry_is_ignored_and_replaced(
    mod: ModuleType, session: Session, tmp_path: Path
) -> None:
    """A challenge page cached by an earlier (pre-fix) run must be treated as a
    cache miss: refetched, classified from the real page, and overwritten."""
    ti = _ingest_first_event(session)
    poisoned = _cache_file(mod, tmp_path, ti._FIGHT_URL_BRADY)
    poisoned.write_text(CHALLENGE_HTML, encoding="utf-8")

    client = ti._make_mock_client()
    report = mod.repair(session, client, apply=False, cache_dir=tmp_path)

    assert report.counts["correct"] == 3
    assert ti._FIGHT_URL_BRADY in client._call_log
    assert "Checking your browser" not in poisoned.read_text(encoding="utf-8")


# ── --backend / --fight-ids-from ──────────────────────────────────────────────


def test_build_client_backends(mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    from ufc_prediction.scraper.browser_fetch import BrowserFetcher
    from ufc_prediction.scraper.client import ScraperClient

    httpx_client = mod.build_client("httpx", delay=0.5, workers=2)
    try:
        assert isinstance(httpx_client, ScraperClient)
    finally:
        httpx_client.close()

    browser = mod.build_client("browser", delay=2.5, workers=4)
    assert isinstance(browser, BrowserFetcher)
    assert browser._delay == 2.5  # the fetcher's own polite rate / backoff
    assert browser._page is None  # lazy: no Chromium launched until first get
    browser.close()

    with pytest.raises(ValueError, match="backend"):
        mod.build_client("curl", delay=1.0, workers=1)


def test_backend_flag_is_parsed(mod: ModuleType) -> None:
    args = mod.build_parser().parse_args(["--backend", "browser"])
    assert args.backend == "browser"
    assert mod.build_parser().parse_args([]).backend == "httpx"
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--backend", "curl"])


def test_repair_can_be_restricted_to_a_fight_id_list(
    mod: ModuleType, session: Session, tmp_path: Path
) -> None:
    ti = _ingest_first_event(session)
    brady = session.query(Fight).filter(Fight.source_url == ti._FIGHT_URL_BRADY).one()
    client = ti._make_mock_client()

    report = mod.repair(session, client, apply=False, cache_dir=tmp_path, fight_ids={brady.id})

    assert sum(report.counts.values()) == 1
    assert client.calls_containing("fight-details") == [ti._FIGHT_URL_BRADY]


def test_read_fight_ids_accepts_offline_report(mod: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "offline.json"
    path.write_text(
        json.dumps({"needs_live": [{"fight_id": 7, "reason": "no_twin"}, {"fight_id": 3}]}),
        encoding="utf-8",
    )
    assert mod.read_fight_ids(path) == {3, 7}
    plain = tmp_path / "ids.txt"
    plain.write_text("5\n9\n\n", encoding="utf-8")
    assert mod.read_fight_ids(plain) == {5, 9}


# ── Offline orientation via kaggle twins ──────────────────────────────────────

D = date(2015, 7, 11)


def test_side_equal_ignores_control_time_and_is_inconclusive_on_missing(
    mod: ModuleType,
) -> None:
    other_ctrl = (*RED_STATS[:6], RED_STATS[6] + 3, *RED_STATS[7:])
    assert mod.side_equal(RED_STATS, other_ctrl) is True
    assert mod.side_equal(RED_STATS, BLUE_STATS) is False
    assert mod.side_equal(RED_STATS, None) is None
    assert mod.side_equal(RED_STATS, NO_STATS) is None  # rajeevw D-04 all-None row
    zeros = (0,) * 14
    assert mod.side_equal(zeros, zeros) is None  # no signal


def test_stats_verdict(mod: ModuleType) -> None:
    # stored (a, b) vs kaggle stats of (the same-named a, the same-named b)
    assert mod.stats_verdict(RED_STATS, BLUE_STATS, RED_STATS, BLUE_STATS) == "self"
    assert mod.stats_verdict(BLUE_STATS, RED_STATS, RED_STATS, BLUE_STATS) == "cross"
    assert mod.stats_verdict(RED_STATS, BLUE_STATS, RED_STATS, None) == "self"
    assert mod.stats_verdict(RED_STATS, RED_STATS, RED_STATS, RED_STATS) == "inconclusive"
    assert mod.stats_verdict(RED_STATS, BLUE_STATS, None, None) == "inconclusive"
    other = (1,) * 14
    assert mod.stats_verdict(other, BLUE_STATS, RED_STATS, BLUE_STATS) == "inconclusive"


def _seed_offline_corpus(session: Session) -> dict[str, Fight]:
    """ufcstats fights (fighter_a = winner) + kaggle twins (fighter_a = red).

    The pre-fix scraper stored the fight-detail page's FIRST row (the red
    corner) under the ufcstats fighter_a, so a blue-corner winner has crossed
    stats.
    """
    f: dict[str, Fight] = {}
    # 1. Winner is BLUE; stored stats crossed (winner holds red's stats).
    f["crossed"] = add_fight(
        session,
        "ufcstats",
        D,
        "Blue Winner",
        "Red Loser",
        stats_a=RED_STATS,
        stats_b=BLUE_STATS,
        extra_rounds=2,
    )
    add_fight(
        session,
        "kaggle-rajeevw",
        D,
        "Red Loser",
        "Blue Winner",
        stats_a=RED_STATS,
        stats_b=BLUE_STATS,
    )
    # 2. Winner is RED; stored stats correct.
    f["correct"] = add_fight(
        session,
        "ufcstats",
        D,
        "Red Champ",
        "Blue Challenger",
        stats_a=RED_STATS,
        stats_b=BLUE_STATS,
    )
    add_fight(
        session,
        "kaggle-rajeevw",
        D,
        "Red Champ",
        "Blue Challenger",
        stats_a=RED_STATS,
        stats_b=BLUE_STATS,
    )
    # 3. mdabbert twin only: corner, no stats -> never trusted offline.
    f["corner_only"] = add_fight(
        session, "ufcstats", D, "Md Blue", "Md Red", stats_a=RED_STATS, stats_b=BLUE_STATS
    )
    add_fight(session, "kaggle-mdabbert", D, "Md Red", "Md Blue")
    # 4. No twin at all.
    f["no_twin"] = add_fight(
        session, "ufcstats", D, "Lonely One", "Lonely Two", stats_a=RED_STATS, stats_b=BLUE_STATS
    )
    # 5. Corner says correct (winner red) but stats say crossed -> disagreement.
    f["disagree"] = add_fight(
        session, "ufcstats", D, "Odd Red", "Odd Blue", stats_a=BLUE_STATS, stats_b=RED_STATS
    )
    add_fight(
        session, "kaggle-rajeevw", D, "Odd Red", "Odd Blue", stats_a=RED_STATS, stats_b=BLUE_STATS
    )
    # 6. Kaggle stats missing for both (D-04) -> inconclusive.
    f["inconclusive"] = add_fight(
        session, "ufcstats", D, "Null Blue", "Null Red", stats_a=RED_STATS, stats_b=BLUE_STATS
    )
    add_fight(
        session, "kaggle-rajeevw", D, "Null Red", "Null Blue", stats_a=NO_STATS, stats_b=NO_STATS
    )
    # 7. Two rajeevw twins -> ambiguous.
    f["ambiguous"] = add_fight(
        session, "ufcstats", D, "Twin Blue", "Twin Red", stats_a=RED_STATS, stats_b=BLUE_STATS
    )
    for _ in range(2):
        add_fight(
            session,
            "kaggle-rajeevw",
            D,
            "Twin Red",
            "Twin Blue",
            stats_a=RED_STATS,
            stats_b=BLUE_STATS,
        )
    # 8. ufcstats fight with no stored stats: nothing to orient.
    f["no_stats"] = add_fight(session, "ufcstats", D, "Empty A", "Empty B")
    add_fight(
        session, "kaggle-rajeevw", D, "Empty B", "Empty A", stats_a=RED_STATS, stats_b=BLUE_STATS
    )
    return f


def test_offline_classification(mod: ModuleType, session: Session) -> None:
    f = _seed_offline_corpus(session)

    report = mod.repair_offline(session, apply=False)

    assert report.offline_crossed == [f["crossed"].id]
    assert report.offline_correct == [f["correct"].id]
    reasons = {e["fight_id"]: e["reason"] for e in report.needs_live}
    assert reasons == {
        f["corner_only"].id: "corner_only",
        f["no_twin"].id: "no_twin",
        f["disagree"].id: "stats_corner_disagree",
        f["inconclusive"].id: "stats_inconclusive",
        f["ambiguous"].id: "ambiguous_twin",
    }
    assert all(e["source_url"] for e in report.needs_live)
    assert report.counts["no_stats"] == 1
    # Premise matrix over conclusive stats twins: (corner, stats verdict).
    assert report.premise == {"blue|cross": 1, "red|self": 1, "red|cross": 1}
    # Corner-only projection is reported, never acted on.
    assert report.corner_only_projection == {"crossed": 1}
    # Report-only: nothing written.
    a = f["crossed"]
    assert round0(session, a.id, a.fighter_a_id) == RED_STATS


def test_offline_apply_swaps_only_crossed_and_is_idempotent(
    mod: ModuleType, session: Session
) -> None:
    f = _seed_offline_corpus(session)
    before = {
        k: (round0(session, x.id, x.fighter_a_id), round0(session, x.id, x.fighter_b_id))
        for k, x in f.items()
    }

    report = mod.repair_offline(session, apply=True)
    assert report.swapped_fight_ids == [f["crossed"].id]
    session.expire_all()

    crossed = f["crossed"]
    assert round0(session, crossed.id, crossed.fighter_a_id) == BLUE_STATS
    assert round0(session, crossed.id, crossed.fighter_b_id) == RED_STATS
    # Per-round rows swap with round 0.
    rows = session.query(RoundStats).filter(RoundStats.fight_id == crossed.id).all()
    by = {(r.fighter_id, r.round_number): r.sig_strikes_landed for r in rows}
    assert {by[(crossed.fighter_a_id, n)] for n in range(3)} == {BLUE_STATS[0]}
    for k, x in f.items():
        if k != "crossed":
            after = (round0(session, x.id, x.fighter_a_id), round0(session, x.id, x.fighter_b_id))
            assert after == before[k], k

    again = mod.repair_offline(session, apply=True)
    assert again.swapped_fight_ids == []
    assert again.offline_crossed == []
    assert again.counts["offline_already_repaired"] == 1
    assert crossed.id not in {e["fight_id"] for e in again.needs_live}
