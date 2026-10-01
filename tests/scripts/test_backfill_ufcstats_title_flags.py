"""Tests for scripts/backfill_ufcstats_title_flags.py.

Before S22 the scraper inferred ``is_title_fight`` from the event page's weight
text, which is only the division, so every stored ufcstats fight is a
non-title fight. The backfill sets the flag on existing ufcstats fights from
cached fight-detail pages, kaggle twins, and (``--backend browser``) ufcstats
event pages. No network: pages are scraper HTML fixtures and the browser is a
mock client.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from tests.scripts.twin_seed import add_fight
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "backfill_ufcstats_title_flags.py"
SCRAPER_FIXTURES = PROJECT_ROOT / "tests" / "scraper" / "fixtures"

D = date(2019, 3, 2)

# (fixture, fighter_a hex, fighter_b hex) of the fight-detail fixtures.
BELT_PAGE = ("fight_detail_superfight_belt.html", "63b65af1c5cb02cb", "08ae5cd9aef7ddd3")
TITLE_TEXT_PAGE = ("fight_detail_1round.html", "9014c02eff8b3d62", "1122334455667788")
PLAIN_PAGE = ("fight_detail_3round.html", "aa11bb22cc33dd44", "ee55ff66aa77bb88")

# event_detail.html: the card's three bouts (fight hex, fighter hexes, belt?).
EVENT_URL = "http://ufcstats.com/event-details/f3eb664db7fb1df3"
EVENT_CARD = (
    ("a031f062b7ffefee", "9014c02eff8b3d62", "1122334455667788", True),
    ("bb22cc33dd44ee55", "aa11bb22cc33dd44", "ee55ff66aa77bb88", False),
    ("cc33dd44ee55ff66", "ff88aa99bb00cc11", "dd22ee33ff44aa55", False),
)


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backfill_ufcstats_title_flags", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["backfill_ufcstats_title_flags"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def caches(tmp_path: Path) -> tuple[Path, Path]:
    fights, events = tmp_path / "fights", tmp_path / "events"
    fights.mkdir()
    events.mkdir()
    return fights, events


def _fixture(name: str) -> str:
    return (SCRAPER_FIXTURES / name).read_text(encoding="utf-8")


def _hex(url: str | None) -> str:
    assert url is not None
    return url.rstrip("/").rsplit("/", 1)[-1]


def _cache_page(
    session: Session, fight: Fight, fight_cache: Path, page: tuple[str, str, str]
) -> None:
    """Cache ``page`` as ``fight``'s detail page and give the fight's fighters
    the page's UFCStats hex ids (the page/fight identity check)."""
    name, a_hex, b_hex = page
    session.get(Fighter, fight.fighter_a_id).source_id = a_hex  # type: ignore[union-attr]
    session.get(Fighter, fight.fighter_b_id).source_id = b_hex  # type: ignore[union-attr]
    session.flush()
    (fight_cache / f"{_hex(fight.source_url)}.html").write_text(_fixture(name), encoding="utf-8")


def _flag(session: Session, fight: Fight) -> bool:
    session.expire_all()
    stored = session.get(Fight, fight.id)
    assert stored is not None
    return stored.is_title_fight


def _set_flag(session: Session, fight: Fight, value: bool) -> None:
    fight.is_title_fight = value
    session.flush()


def _run(mod: ModuleType, session: Session, caches: tuple[Path, Path], **kwargs: object) -> object:
    return mod.backfill(session, fight_cache_dir=caches[0], event_cache_dir=caches[1], **kwargs)


def _seed_event_card(session: Session, event_url: str = EVENT_URL) -> list[Fight]:
    """A ufcstats event carrying event_detail.html's three bouts."""
    event = Event(name="UFC 327", date=date(2026, 4, 11), source="ufcstats", source_url=event_url)
    session.add(event)
    session.flush()
    fights = []
    for fight_hex, a_hex, b_hex, _belt in EVENT_CARD:
        fa = Fighter(name=f"A {a_hex}", source="ufcstats", source_id=a_hex)
        fb = Fighter(name=f"B {b_hex}", source="ufcstats", source_id=b_hex)
        session.add_all([fa, fb])
        session.flush()
        fight = Fight(
            event_id=event.id,
            fighter_a_id=fa.id,
            fighter_b_id=fb.id,
            winner_id=fa.id,
            weight_class="Welterweight",
            source="ufcstats",
            source_url=f"http://ufcstats.com/fight-details/{fight_hex}",
        )
        session.add(fight)
        fights.append(fight)
    session.flush()
    return fights


# ── Cached fight-detail pages ──────────────────────────────────────────────


def test_cached_title_pages_set_the_flag_and_dry_run_writes_nothing(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    belt = add_fight(session, "ufcstats", D, "Ken Shamrock", "Kimo Leopoldo")
    titled = add_fight(session, "ufcstats", D, "Carlos Ulberg", "Jiri Prochazka")
    plain = add_fight(session, "ufcstats", D, "Sean Brady", "Kevin Holland")
    _cache_page(session, belt, caches[0], BELT_PAGE)
    _cache_page(session, titled, caches[0], TITLE_TEXT_PAGE)
    _cache_page(session, plain, caches[0], PLAIN_PAGE)

    report = _run(mod, session, caches, apply=False)
    assert report.counts["fight_page:ok"] == 3
    assert report.counts["settled:fight_page"] == 3
    assert report.counts["to_set_true"] == 2
    assert report.counts["unchanged"] == 1
    assert {c["fight_id"] for c in report.changes} == {belt.id, titled.id}
    assert report.counts.get("rows_updated", 0) == 0
    assert not _flag(session, belt) and not _flag(session, titled)  # dry run

    report = _run(mod, session, caches, apply=True)
    assert report.counts["rows_updated"] == 2
    assert _flag(session, belt) and _flag(session, titled)
    assert not _flag(session, plain)

    # Idempotent: a re-run finds nothing to change.
    report = _run(mod, session, caches, apply=True)
    assert report.counts.get("to_set_true", 0) == 0
    assert report.counts["unchanged"] == 3
    assert report.counts["rows_updated"] == 0


def test_page_without_belt_is_an_authoritative_false(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    fight = add_fight(session, "ufcstats", D, "Sean Brady", "Kevin Holland")
    _set_flag(session, fight, True)
    _cache_page(session, fight, caches[0], PLAIN_PAGE)

    report = _run(mod, session, caches, apply=True)
    assert report.counts["to_set_false"] == 1
    assert not _flag(session, fight)


def test_challenge_and_mismatched_pages_are_not_evidence(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    challenged = add_fight(session, "ufcstats", D, "Ken Shamrock", "Kimo Leopoldo")
    (caches[0] / f"{_hex(challenged.source_url)}.html").write_text(
        _fixture("ufcstats_pow_challenge.html"), encoding="utf-8"
    )
    # A cached page for two other fighters (fighter hex ids do not match).
    wrong = add_fight(session, "ufcstats", D, "Carlos Ulberg", "Jiri Prochazka")
    (caches[0] / f"{_hex(wrong.source_url)}.html").write_text(
        _fixture(BELT_PAGE[0]), encoding="utf-8"
    )

    report = _run(mod, session, caches, apply=True)
    assert report.counts["fight_page:challenge"] == 1
    assert report.counts["fight_page:mismatch"] == 1
    assert report.counts["unknown"] == 2
    assert report.counts["rows_updated"] == 0
    assert not _flag(session, challenged) and not _flag(session, wrong)


# ── Kaggle twins ───────────────────────────────────────────────────────────


def test_twin_true_sets_the_flag_and_twin_false_writes_nothing(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    ufc_title = add_fight(session, "ufcstats", D, "Georges St-Pierre", "B.J. Penn")
    kag_title = add_fight(session, "kaggle-rajeevw", D, "BJ Penn", "Georges Saint Pierre")
    _set_flag(session, kag_title, True)
    ufc_plain = add_fight(session, "ufcstats", D, "Sean Brady", "Kevin Holland")
    add_fight(session, "kaggle-mdabbert", D, "Kevin Holland", "Sean Brady")
    # A stored True with only a twin False: a twin is not authoritative for
    # False, so it is left alone.
    ufc_true = add_fight(session, "ufcstats", D, "Jon Jones", "Daniel Cormier")
    _set_flag(session, ufc_true, True)
    add_fight(session, "kaggle-rajeevw", D, "Daniel Cormier", "Jon Jones")

    report = _run(mod, session, caches, apply=True)
    assert report.counts["twin:ok"] == 3
    assert report.counts["settled:twin"] == 3
    assert report.coverage["twin"] == {"settled": 3, "true": 1, "false": 2}
    assert report.counts["to_set_true"] == 1
    assert report.counts.get("to_set_false", 0) == 0
    assert report.counts["rows_updated"] == 1
    assert _flag(session, ufc_title)
    assert not _flag(session, ufc_plain)
    assert _flag(session, ufc_true)


def test_twin_within_one_day_ambiguous_and_conflicting_twins(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    near = add_fight(session, "ufcstats", D, "Fighter One", "Fighter Two")
    near_twin = add_fight(
        session, "kaggle-rajeevw", D + timedelta(days=1), "Fighter Two", "Fighter One"
    )
    _set_flag(session, near_twin, True)

    ambiguous = add_fight(session, "ufcstats", D, "Fighter Three", "Fighter Four")
    for _ in range(2):
        twin = add_fight(session, "kaggle-rajeevw", D, "Fighter Three", "Fighter Four")
        _set_flag(session, twin, True)

    conflicting = add_fight(session, "ufcstats", D, "Fighter Five", "Fighter Six")
    raj = add_fight(session, "kaggle-rajeevw", D, "Fighter Five", "Fighter Six")
    _set_flag(session, raj, True)
    add_fight(session, "kaggle-mdabbert", D, "Fighter Five", "Fighter Six")  # False

    report = _run(mod, session, caches, apply=True)
    assert report.counts["twin:near_date"] == 1
    assert report.counts["twin:ambiguous"] == 1
    assert report.counts["twin:conflict"] == 1
    assert report.counts["unknown"] == 2
    assert _flag(session, near)
    assert not _flag(session, ambiguous)
    assert not _flag(session, conflicting)


def test_page_beats_twin_and_kaggle_rows_are_never_written(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    # Page shows the belt, twin says no title.
    belt = add_fight(session, "ufcstats", D, "Ken Shamrock", "Kimo Leopoldo")
    _cache_page(session, belt, caches[0], BELT_PAGE)
    belt_twin = add_fight(session, "kaggle-rajeevw", D, "Kimo Leopoldo", "Ken Shamrock")
    # Page shows no belt, twin says title.
    plain = add_fight(session, "ufcstats", D, "Sean Brady", "Kevin Holland")
    _cache_page(session, plain, caches[0], PLAIN_PAGE)
    plain_twin = add_fight(session, "kaggle-mdabbert", D, "Sean Brady", "Kevin Holland")
    _set_flag(session, plain_twin, True)
    # Page and twin agree.
    titled = add_fight(session, "ufcstats", D, "Carlos Ulberg", "Jiri Prochazka")
    _cache_page(session, titled, caches[0], TITLE_TEXT_PAGE)
    titled_twin = add_fight(session, "kaggle-rajeevw", D, "Jiri Prochazka", "Carlos Ulberg")
    _set_flag(session, titled_twin, True)

    report = _run(mod, session, caches, apply=True)
    page_vs_twin = report.cross_check["page_vs_twin"]
    assert page_vs_twin == {
        "agree_true": 1,
        "agree_false": 0,
        "page_true_twin_false": 1,
        "page_false_twin_true": 1,
    }
    assert {d["fight_id"] for d in report.cross_check["disagreements"]} == {belt.id, plain.id}
    assert _flag(session, belt) and _flag(session, titled)
    assert not _flag(session, plain)
    # Kaggle rows keep their own values.
    assert not _flag(session, belt_twin)
    assert _flag(session, plain_twin) and _flag(session, titled_twin)


# ── Event pages (--backend browser) ────────────────────────────────────────


def _browser_client() -> object:
    from tests.scraper.test_ingest import MockScraperClient

    return MockScraperClient(fixture_map={}, exact_map={EVENT_URL: _fixture("event_detail.html")})


def test_browser_backend_fetches_only_unsettled_events_and_caches_them(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    card = _seed_event_card(session)
    # A second event whose only fight is settled by its cached fight page.
    settled = add_fight(session, "ufcstats", D, "Ken Shamrock", "Kimo Leopoldo")
    _cache_page(session, settled, caches[0], BELT_PAGE)
    settled_event = session.get(Event, settled.event_id)
    assert settled_event is not None
    settled_event.source_url = "http://ufcstats.com/event-details/0123456789abcdef"
    session.flush()

    offline = _run(mod, session, caches, apply=False)
    assert offline.counts["unknown"] == 3
    assert [e["source_url"] for e in offline.unknown_events] == [EVENT_URL]

    client = _browser_client()
    report = _run(mod, session, caches, apply=True, client=client)
    assert client.calls_containing("event-details") == [EVENT_URL]  # type: ignore[attr-defined]
    assert report.counts["events:fetched"] == 1
    assert report.counts["settled:event_page"] == 3
    assert report.coverage["event_page"] == {"settled": 3, "true": 1, "false": 2}
    assert [_flag(session, f) for f in card] == [belt for *_, belt in EVENT_CARD]
    assert (caches[1] / "f3eb664db7fb1df3.html").exists()

    # Offline re-run reads the cached event page: same decisions, no fetch.
    report = _run(mod, session, caches, apply=True)
    assert report.counts["settled:event_page"] == 3
    assert report.counts["rows_updated"] == 0


def test_browser_challenge_page_is_neither_cached_nor_used(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    from tests.scraper.test_ingest import MockScraperClient

    card = _seed_event_card(session)
    client = MockScraperClient(
        fixture_map={}, exact_map={EVENT_URL: _fixture("ufcstats_pow_challenge.html")}
    )
    report = _run(mod, session, caches, apply=True, client=client)
    assert report.counts["events:fetch_failed"] == 1
    assert report.counts["unknown"] == 3
    assert not any(caches[1].iterdir())
    assert not any(_flag(session, f) for f in card)


def test_max_events_caps_the_browser_pass(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    _seed_event_card(session)
    client = _browser_client()
    report = _run(mod, session, caches, apply=False, client=client, max_events=0)
    assert client.call_count == 0  # type: ignore[attr-defined]
    assert report.counts["events:needed"] == 1
    assert report.counts.get("events:fetched", 0) == 0


# ── CLI ────────────────────────────────────────────────────────────────────


def test_dry_run_session_is_read_only(
    mod: ModuleType, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ufc_prediction.db.session as db_session

    monkeypatch.setattr(db_session, "SessionLocal", sessionmaker(bind=engine))
    session = mod.open_session(apply=False)
    try:
        with pytest.raises(DBAPIError, match="read-only transaction"):
            session.execute(text("UPDATE fights SET is_title_fight = is_title_fight"))
    finally:
        session.rollback()
        session.close()


def test_main_dry_run_writes_a_json_report(
    mod: ModuleType,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    caches: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    import ufc_prediction.db.session as db_session

    monkeypatch.setattr(db_session, "SessionLocal", sessionmaker(bind=engine))
    out = tmp_path / "report.json"
    rc = mod.main(
        [
            "--fight-cache-dir",
            str(caches[0]),
            "--event-cache-dir",
            str(caches[1]),
            "--report",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["applied"] is False
    assert set(report["coverage"]) == {"fight_page", "event_page", "twin", "unknown"}


def test_recheck_twins_lets_the_event_page_override_a_twin(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    """A tournament final the kaggle twin calls non-title: by default the twin
    settles it (no fetch); --recheck-twins fetches the event page, whose belt
    is authoritative."""
    card = _seed_event_card(session)
    title_fight = card[0]
    for fight in card:
        fa = session.get(Fighter, fight.fighter_a_id)
        fb = session.get(Fighter, fight.fighter_b_id)
        assert fa is not None and fb is not None
        add_fight(session, "kaggle-mdabbert", date(2026, 4, 11), fa.name, fb.name)  # all False

    client = _browser_client()
    report = _run(mod, session, caches, apply=True, client=client)
    assert report.counts["events:needed"] == 0
    assert client.call_count == 0  # type: ignore[attr-defined]
    assert report.counts["settled:twin"] == 3
    assert not _flag(session, title_fight)

    report = _run(mod, session, caches, apply=True, client=client, recheck_twins=True)
    assert report.counts["events:fetched"] == 1
    assert report.counts["settled:event_page"] == 3
    assert report.cross_check["page_vs_twin"]["page_true_twin_false"] == 1
    assert _flag(session, title_fight)


def test_page_problems_are_listed(
    mod: ModuleType, session: Session, caches: tuple[Path, Path]
) -> None:
    fight = add_fight(session, "ufcstats", D, "Ken Shamrock", "Kimo Leopoldo")
    (caches[0] / f"{_hex(fight.source_url)}.html").write_text(
        _fixture("ufcstats_pow_challenge.html"), encoding="utf-8"
    )
    report = _run(mod, session, caches, apply=False)
    assert report.page_problems == [{"fight_id": fight.id, "fight_page": "challenge"}]
