"""Wave-0 RED tests for scripts/bfo_backfill.py (Phase 21 Plan 21-01 Task 4).

Per CONTEXT.md D-01..D-11 (REVISION) + 21-RESEARCH.md Recommended Plan
Revision (Option A: reuse existing BFOOddsIngester; per-event CSV staging
in data/bfo/v22-backfill/<batch_id>/; no Alembic migration).

Mocks ScraperClient via MagicMock per the project pattern established in
tests/scraper/test_bfo_scraper.py:21. Uses ``pytest.importorskip`` so
tests SKIP cleanly until ``scripts/bfo_backfill.py`` lands, then turn
GREEN on import.

Plan 21-01 hard boundary: Task 4 ships only the SCAFFOLDING (locked
constants + AF asserts + URL enumeration helpers + gap query +
BFO_V22_GAP.json emit + --resume flag stub). The actual live scrape +
ingest + coverage delta are Plan 21-02. The operator-action checkpoint
(Task 5) gates Plan 21-02 entry.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def driver():
    """Lazy-import the backfill driver as a module.

    Mirrors tests/scripts/test_audit_camp_v22.py:23-30 importorskip pattern
    so Wave-0 RED can run before scripts/bfo_backfill.py lands.
    """
    return pytest.importorskip("scripts.bfo_backfill")


# ── 1. Locked constants + AF startup asserts ──────────────────────────────


class TestLockedConstants:
    """All 7 module-top constants present with the locked values."""

    def test_locked_constants_present_and_correct(self, driver) -> None:
        """Constants match CONTEXT.md D-01..D-11 (REVISION) verbatim."""
        assert driver.BFO_ARCHIVE_START_DATE == date(2007, 6, 1), (
            "Pre-2007 cutoff is locked from Phase 20 reachability finding"
        )
        assert driver.BFO_TIMEOUT_SECONDS == 5
        assert driver.BFO_DELAY_SECONDS == 1.2
        assert driver.BFO_MAX_RETRIES == 3
        assert driver.TARGET_RATE_THRESHOLD == 0.96, (
            "Locked from Phase 20 BFO_ARCHIVE_REACHABILITY.md"
        )
        assert driver.SCRAPE_BATCH_DIR == "data/bfo/v22-backfill"
        assert driver.DISAMBIGUATION_REQUIRE_UFC_PREFIX is True, (
            "CONTEXT.md D-01a — Bellator/Strikeforce collision guard"
        )
        assert "BFO_V22_GAP.json" in driver.GAP_JSON_PATH


class TestAfStartupAsserts:
    """Module-import-time AF asserts fail fast on constant drift."""

    def test_af_startup_asserts_fire_on_drift(
        self, driver, monkeypatch,
    ) -> None:
        """Tampering with BFO_ARCHIVE_START_DATE then re-running asserts trips.

        We can't easily re-import the module after monkeypatching (Python
        caches imports), so we exercise the assertion function directly.
        ``_run_af_startup_asserts()`` is the public hook the asserts
        live behind so they're testable.
        """
        # Baseline: passes with the canonical constants.
        driver._run_af_startup_asserts()

        monkeypatch.setattr(
            driver, "BFO_ARCHIVE_START_DATE", date(2010, 1, 1),
        )
        with pytest.raises(AssertionError):
            driver._run_af_startup_asserts()

    def test_af_startup_asserts_fire_on_target_rate_drift(
        self, driver, monkeypatch,
    ) -> None:
        """TARGET_RATE_THRESHOLD drift halts before any live HTTP work."""
        monkeypatch.setattr(driver, "TARGET_RATE_THRESHOLD", 0.85)
        with pytest.raises(AssertionError):
            driver._run_af_startup_asserts()

    def test_af_startup_asserts_fire_on_disambiguation_disabled(
        self, driver, monkeypatch,
    ) -> None:
        """Disabling the D-01a UFC-slug guard halts before live HTTP."""
        monkeypatch.setattr(
            driver, "DISAMBIGUATION_REQUIRE_UFC_PREFIX", False,
        )
        with pytest.raises(AssertionError):
            driver._run_af_startup_asserts()


# ── 2. _query_gap_events: SQL gap-identification helper ────────────────────


class TestQueryGapEvents:
    """``_query_gap_events`` returns post-cutoff events lacking BFO odds."""

    def test_query_gap_events_excludes_pre_cutoff(self, driver) -> None:
        """Events with date < BFO_ARCHIVE_START_DATE are filtered out.

        Mock the SQLAlchemy session to return a synthetic Event mix; the
        helper must produce ONLY post-cutoff rows in its result.
        """
        from collections import namedtuple

        _EventRow = namedtuple("_EventRow", ["id", "name", "date"])
        synthetic = [
            _EventRow(id=1, name="UFC 64", date=date(2006, 10, 14)),  # pre
            _EventRow(id=2, name="UFC 77", date=date(2007, 10, 20)),  # post
            _EventRow(id=3, name="UFC 200", date=date(2016, 7, 9)),   # post
        ]
        session = MagicMock()
        exec_result = MagicMock()
        # The helper filters in SQL — but our mock returns whatever it wants.
        # We assert the helper builds a query that includes the cutoff
        # constant. The simpler test: the mock returns only the post-cutoff
        # rows, and the helper's contract is to return what SQL gives it.
        post_cutoff = [r for r in synthetic if r.date >= driver.BFO_ARCHIVE_START_DATE]
        exec_result.all.return_value = post_cutoff
        session.execute.return_value = exec_result

        events = driver._query_gap_events(session)
        assert all(
            e.date >= driver.BFO_ARCHIVE_START_DATE for e in events
        ), "_query_gap_events must filter to post-cutoff only"
        assert len(events) == 2

    def test_query_gap_events_query_includes_cutoff_filter(
        self, driver,
    ) -> None:
        """The SQL statement built by ``_query_gap_events`` references the
        cutoff date — proves the WHERE clause exists at all.

        We capture the SQL via session.execute's call_args and inspect the
        compiled statement.
        """
        session = MagicMock()
        exec_result = MagicMock()
        exec_result.all.return_value = []
        session.execute.return_value = exec_result

        driver._query_gap_events(session)

        # session.execute(stmt) was called once; the first positional arg
        # is the SQLAlchemy Select statement.
        assert session.execute.called
        stmt = session.execute.call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        # Cutoff date appears in the compiled WHERE clause.
        assert "2007-06-01" in compiled, (
            "Compiled SQL must reference BFO_ARCHIVE_START_DATE = 2007-06-01"
        )


# ── 3. _enumerate_bfo_url: search → first /events/ match + D-01a guard ────


class TestEnumerateBfoUrl:
    """``_enumerate_bfo_url`` resolves event name → BFO URL via /search."""

    def test_enumerate_bfo_url_ok_match(self, driver) -> None:
        """UFC-prefixed slug → returns (url, "OK")."""
        client = MagicMock()
        # BFO /search response with a single /events/ufc-... match.
        client.get.return_value = (
            '<html><body><a href="/events/ufc-200-1700">UFC 200</a></body></html>'
        )
        url, status = driver._enumerate_bfo_url(client, "UFC 200")
        assert status == "OK"
        assert url is not None
        assert "ufc-200-1700" in url
        assert url.startswith("https://www.bestfightodds.com/events/")

    def test_enumerate_bfo_url_event_not_found(self, driver) -> None:
        """Empty BFO /search result → (None, "EVENT_NOT_FOUND")."""
        client = MagicMock()
        client.get.return_value = "<html><body>No results.</body></html>"
        url, status = driver._enumerate_bfo_url(client, "UFC Nonexistent")
        assert url is None
        assert status == "EVENT_NOT_FOUND"

    def test_enumerate_bfo_url_disambiguation_fail(self, driver) -> None:
        """D-01a guard: non-UFC slug → (None, "DISAMBIGUATION_FAIL").

        Real-world case from Phase 20: BFO /search?query=UFC+272 returns
        /events/bellator-272-2310 first because of name collision.
        """
        client = MagicMock()
        client.get.return_value = (
            '<html><body>'
            '<a href="/events/bellator-272-2310">Bellator 272</a>'
            '</body></html>'
        )
        url, status = driver._enumerate_bfo_url(client, "UFC 272")
        assert url is None, (
            "D-01a: non-ufc-prefixed slug must be rejected as disambiguation_fail"
        )
        assert status == "DISAMBIGUATION_FAIL"

    def test_enumerate_bfo_url_handles_fetch_failure(self, driver) -> None:
        """ScraperClient.get raising RuntimeError → (None, "EVENT_NOT_FOUND")."""
        client = MagicMock()
        client.get.side_effect = RuntimeError("transport_fail after retries")
        url, status = driver._enumerate_bfo_url(client, "UFC 200")
        assert url is None
        assert status == "EVENT_NOT_FOUND"

    def test_enumerate_bfo_url_rejects_trailing_dash_slug(self, driver) -> None:
        """WR-01 regression: regex must NOT accept slugs ending in `-` before id.

        The previous pattern ``[a-z0-9][a-z0-9-]*-\\d+`` ambiguously matched
        ``foo--123`` as slug=``foo-`` + id=``123``. The tightened pattern
        requires the slug's last char to be ``[a-z0-9]`` (or the slug to
        be a single ``[a-z0-9]`` char). When the FIRST href is malformed,
        the regex should SKIP it and find a later well-formed href on the
        same page; if no well-formed href exists, return EVENT_NOT_FOUND.

        The current `_EVENT_HREF_RE.search()` returns the first regex
        match — with the tightened pattern, ``foo--123`` no longer
        matches, so the well-formed ``ufc-200-1700`` later in the HTML
        wins.
        """
        client = MagicMock()
        # Malformed href appears FIRST; well-formed href second. The
        # tightened regex must skip the malformed one and pick the valid one.
        client.get.return_value = (
            '<html><body>'
            '<a href="/events/foo--123">Bogus</a>'
            '<a href="/events/ufc-200-1700">UFC 200</a>'
            '</body></html>'
        )
        url, status = driver._enumerate_bfo_url(client, "UFC 200")
        assert status == "OK", (
            "WR-01: tightened regex must skip trailing-dash slug and "
            "fall through to next well-formed href"
        )
        assert url is not None and "ufc-200-1700" in url

    def test_enumerate_bfo_url_rejects_only_trailing_dash_slug(
        self, driver,
    ) -> None:
        """WR-01 regression: when ONLY a malformed href exists, return NOT_FOUND.

        Documents that the tightened regex does NOT silently match the
        ambiguous slug as a partial match.
        """
        client = MagicMock()
        client.get.return_value = (
            '<html><body>'
            '<a href="/events/foo--123">Bogus</a>'
            '</body></html>'
        )
        url, status = driver._enumerate_bfo_url(client, "Foo")
        assert url is None, (
            "WR-01: malformed slug must not silently match as a partial slug"
        )
        assert status == "EVENT_NOT_FOUND"


# ── 4. _emit_gap_json: BFO_V22_GAP.json schema validation ──────────────────


class TestEmitGapJson:
    """``_emit_gap_json`` produces the operator-review artifact."""

    def test_emit_gap_json_schema(self, driver, tmp_path: Path) -> None:
        """Top-level keys + per_year_breakdown structure per Plan 21-01 spec."""
        from collections import namedtuple

        _EventRow = namedtuple("_EventRow", ["id", "name", "date"])
        gap_events = [
            _EventRow(id=1, name="UFC 200", date=date(2016, 7, 9)),
            _EventRow(id=2, name="UFC 285", date=date(2023, 3, 4)),
        ]
        url_results = [
            (gap_events[0], "https://www.bestfightodds.com/events/ufc-200-1700", "OK"),
            (gap_events[1], None, "EVENT_NOT_FOUND"),
        ]
        out = tmp_path / "BFO_V22_GAP.json"
        driver._emit_gap_json(
            gap_events=gap_events,
            url_results=url_results,
            path=out,
            batch_id="20260513-203500",
        )
        assert out.exists()
        payload = json.loads(out.read_text())

        # Required top-level keys (5 per Plan 21-01 Task 4 spec).
        required_top = {
            "batch_id",
            "gap_query",
            "url_resolution",
            "estimated_scrape_wall_clock_min",
            "per_year_breakdown",
            "next_step",
        }
        missing = required_top - payload.keys()
        assert not missing, f"missing required top-level keys: {missing}"

        # gap_query subkeys.
        assert payload["gap_query"]["cutoff_date"] == "2007-06-01"
        assert payload["gap_query"]["n_in_scope_events"] == 2

        # url_resolution subkeys.
        assert payload["url_resolution"]["url_acquired"] == 1
        assert payload["url_resolution"]["event_not_found"] == 1
        assert payload["url_resolution"]["disambiguation_fail"] == 0

        # per_year_breakdown is keyed by year string.
        assert "2016" in payload["per_year_breakdown"]
        assert "2023" in payload["per_year_breakdown"]

        # batch_id round-trips.
        assert payload["batch_id"] == "20260513-203500"

    def test_emit_gap_json_includes_url_acquired_list(
        self, driver, tmp_path: Path,
    ) -> None:
        """Plan 21-02 forward-compat: per-event URL list persisted.

        ``--scrape`` mode (Plan 21-02 Task 1 prerequisite) reads this
        list to recover the URLs without re-running the BFO /search
        enumeration. The list contains one entry per ``status="OK"``
        event with ``event_id``, ``event_name``, ``event_date``,
        ``url``.
        """
        from collections import namedtuple

        _EventRow = namedtuple("_EventRow", ["id", "name", "date"])
        gap_events = [
            _EventRow(id=10, name="UFC 200", date=date(2016, 7, 9)),
            _EventRow(id=11, name="UFC 285", date=date(2023, 3, 4)),
            _EventRow(id=12, name="UFC 272", date=date(2022, 3, 5)),
        ]
        url_results = [
            (gap_events[0], "https://www.bestfightodds.com/events/ufc-200-1700", "OK"),
            (gap_events[1], "https://www.bestfightodds.com/events/ufc-285-2400", "OK"),
            (gap_events[2], None, "DISAMBIGUATION_FAIL"),
        ]
        out = tmp_path / "BFO_V22_GAP.json"
        driver._emit_gap_json(
            gap_events=gap_events,
            url_results=url_results,
            path=out,
            batch_id="20260514-164300",
        )
        payload = json.loads(out.read_text())
        assert "url_acquired_list" in payload, (
            "Plan 21-02 --scrape mode requires url_acquired_list field"
        )
        ual = payload["url_acquired_list"]
        # Only OK entries persisted (DISAMBIGUATION_FAIL excluded).
        assert len(ual) == 2
        # Entry shape — operator-friendly + machine-readable.
        first = ual[0]
        assert set(first.keys()) == {"event_id", "event_name", "event_date", "url"}
        assert first["event_id"] == 10
        assert first["event_name"] == "UFC 200"
        assert first["event_date"] == "2016-07-09"
        assert first["url"] == "https://www.bestfightodds.com/events/ufc-200-1700"


# ── 5. Anti-pattern guards (Plan 21-01 acceptance criterion) ───────────────


class TestAntiPatternGuards:
    """Source-level checks per Plan 21-01 acceptance criteria."""

    def test_does_not_call_legacy_scrape_all(self) -> None:
        """Anti-pattern #3: Plan 21-01 must NOT call BFOScraper.scrape_all.

        The backfill driver uses scrape_event_urls (Task 3 output);
        scrape_all would re-fetch the entire fighter corpus and miss the
        gap-driven URL list. This is the canonical 21-PATTERNS.md
        anti-pattern #3 guard.
        """
        src = Path("scripts/bfo_backfill.py").read_text(encoding="utf-8")
        assert "BFOScraper.scrape_all" not in src, (
            "Anti-pattern #3: bfo_backfill must NOT invoke scrape_all"
        )
        # Same content, source-pattern check (any module-method call form).
        assert ".scrape_all(" not in src, (
            "Anti-pattern #3: any .scrape_all(...) invocation forbidden"
        )

    def test_module_top_constants_present(self) -> None:
        """Acceptance: module-top constants + AF asserts present in source."""
        src = Path("scripts/bfo_backfill.py").read_text(encoding="utf-8")
        for const in (
            "BFO_ARCHIVE_START_DATE",
            "TARGET_RATE_THRESHOLD",
            "SCRAPE_BATCH_DIR",
            "DISAMBIGUATION_REQUIRE_UFC_PREFIX",
            "GAP_JSON_PATH",
        ):
            assert f"{const}" in src, f"Locked constant {const} missing from source"
        # AF startup assert pattern present.
        assert "assert BFO_ARCHIVE_START_DATE" in src
        assert "assert TARGET_RATE_THRESHOLD" in src


# ── 6. --resume flag stub (Plan 21-02 implements full logic) ───────────────


class TestResumeFlagStub:
    """Plan 21-01 ships --resume as a CLI flag stub; full logic in Plan 21-02."""

    def test_resume_flag_present_in_argparser(self, driver) -> None:
        """argparse parser exposes --resume so Plan 21-02 can flesh out logic."""
        parser = driver._build_arg_parser()
        actions = {a.dest for a in parser._actions}
        assert "resume" in actions, (
            "--resume flag must be exposed by Plan 21-01 argparser stub"
        )
        assert "dry_run" in actions, (
            "--dry-run flag must be exposed (Task 5 operator-action checkpoint)"
        )


# ── 7. --scrape mode (Plan 21-02 Task 1 prerequisite) ──────────────────────


class TestScrapeMode:
    """Plan 21-02 Task 1: --scrape live BFO scrape against url_acquired URLs.

    The --scrape mode reads URLs from BFO_V22_GAP.json's url_acquired_list
    field (added in the prior refactor commit), invokes
    BFOScraper.scrape_event_urls, writes MANIFEST.json + fighters_names.csv,
    and emits the AUDIT-01 mid SHA file before exiting.

    All HTTP work is mocked at the BFOScraper class level — no live network.
    """

    @staticmethod
    def _write_synthetic_gap_json(
        path: Path,
        batch_id: str,
        urls: list[tuple[int, str, str, str]],
    ) -> None:
        """Write a minimal BFO_V22_GAP.json with url_acquired_list populated."""
        payload = {
            "batch_id": batch_id,
            "gap_query": {
                "cutoff_date": "2007-06-01",
                "n_in_scope_events": len(urls),
                "n_events_with_no_odds": len(urls),
                "n_events_with_partial_odds": 0,
            },
            "url_resolution": {
                "url_acquired": len(urls),
                "event_not_found": 0,
                "disambiguation_fail": 0,
            },
            "estimated_scrape_wall_clock_min": 1,
            "per_year_breakdown": {},
            "url_acquired_list": [
                {
                    "event_id": eid,
                    "event_name": ename,
                    "event_date": edate,
                    "url": eurl,
                }
                for eid, ename, edate, eurl in urls
            ],
            "next_step": "test fixture",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def test_scrape_requires_batch_id(self, driver, capsys) -> None:
        """--scrape without --batch-id exits non-zero with a clear error.

        ``main`` returns the exit code (the ``__main__`` entrypoint wraps
        it with ``sys.exit``), so we assert on the return value here.
        """
        rc = driver.main(["--scrape"])
        assert rc != 0, (
            "--scrape without --batch-id must exit non-zero (got 0)"
        )
        # And the operator gets a clear stderr message.
        captured = capsys.readouterr()
        assert "--batch-id" in captured.err

    def test_scrape_rejects_mismatched_batch_id(
        self, driver, tmp_path: Path, capsys,
    ) -> None:
        """--scrape --batch-id wrongid against GAP_JSON with batch_id=rightid exits 1."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(
            gap_json,
            batch_id="rightid",
            urls=[(1, "UFC 200", "2016-07-09",
                   "https://www.bestfightodds.com/events/ufc-200-1700")],
        )
        rc = driver.main([
            "--scrape",
            "--batch-id", "wrongid",
            "--gap-json-path", str(gap_json),
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "mismatch" in captured.err.lower()

    def test_scrape_calls_bfo_scraper_with_url_list(
        self, driver, tmp_path: Path, monkeypatch,
    ) -> None:
        """--scrape invokes BFOScraper.scrape_event_urls with the URL list + output_path."""
        batch_id = "20260514-164300"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        synthetic_urls = [
            (1, "UFC 200", "2016-07-09",
             "https://www.bestfightodds.com/events/ufc-200-1700"),
            (2, "UFC 285", "2023-03-04",
             "https://www.bestfightodds.com/events/ufc-285-2400"),
            (3, "UFC 290", "2023-07-08",
             "https://www.bestfightodds.com/events/ufc-290-2500"),
        ]
        self._write_synthetic_gap_json(gap_json, batch_id, synthetic_urls)

        # Override SCRAPE_BATCH_DIR + AUDIT_01_SHA_MID_PATH so we don't
        # pollute real data/ or the planning-dir SHA file.
        monkeypatch.setattr(driver, "SCRAPE_BATCH_DIR", str(tmp_path / "v22-backfill"))
        monkeypatch.setattr(
            driver, "AUDIT_01_SHA_MID_PATH",
            str(tmp_path / "21-XGB-V2-SHA-MID.txt"),
        )

        # Patch BFOScraper at the import site (lazy-imported inside main).
        scraper_instances: list[Any] = []

        class FakeBFOScraper:
            def __init__(self, *args, **kwargs):
                self.init_args = args
                self.init_kwargs = kwargs
                self.calls: list[tuple[list[str], Path]] = []
                scraper_instances.append(self)

            def scrape_event_urls(
                self, event_urls: list[str], output_path: Path,
            ) -> Path:
                self.calls.append((list(event_urls), Path(output_path)))
                # Simulate emitting an empty CSV (header-only).
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n",
                    encoding="utf-8",
                )
                return output_path

        # Prevent real ScraperClient instantiation (no live HTTP construction).
        class FakeScraperClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        # Inject fakes via sys.modules so the lazy `from ... import` inside
        # main() resolves to our doubles.
        import sys
        import types

        fake_scraper_mod = types.ModuleType("ufc_prediction.scraper.bfo_scraper")
        fake_scraper_mod.BFOScraper = FakeBFOScraper
        fake_client_mod = types.ModuleType("ufc_prediction.scraper.client")
        fake_client_mod.ScraperClient = FakeScraperClient
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.bfo_scraper",
                            fake_scraper_mod)
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.client",
                            fake_client_mod)

        # Run --scrape.
        rc = driver.main([
            "--scrape",
            "--batch-id", batch_id,
            "--gap-json-path", str(gap_json),
        ])
        assert rc == 0

        # BFOScraper instantiated exactly once and called exactly once.
        assert len(scraper_instances) == 1
        inst = scraper_instances[0]
        assert len(inst.calls) == 1
        urls_passed, output_path_passed = inst.calls[0]

        # All 3 synthetic URLs forwarded in order.
        assert urls_passed == [u[3] for u in synthetic_urls]
        # Output CSV path matches the spec: SCRAPE_BATCH_DIR/<batch>/BestFightOdds_odds.csv
        assert output_path_passed.name == "BestFightOdds_odds.csv"
        assert output_path_passed.parent.name == batch_id

    def test_scrape_writes_manifest_with_required_fields(
        self, driver, tmp_path: Path, monkeypatch,
    ) -> None:
        """--scrape writes MANIFEST.json with all 8 required fields."""
        batch_id = "20260514-164300"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        synthetic_urls = [
            (1, "UFC 200", "2016-07-09",
             "https://www.bestfightodds.com/events/ufc-200-1700"),
        ]
        self._write_synthetic_gap_json(gap_json, batch_id, synthetic_urls)

        monkeypatch.setattr(driver, "SCRAPE_BATCH_DIR", str(tmp_path / "v22-backfill"))
        monkeypatch.setattr(
            driver, "AUDIT_01_SHA_MID_PATH",
            str(tmp_path / "21-XGB-V2-SHA-MID.txt"),
        )

        class FakeBFOScraper:
            def __init__(self, *args, **kwargs):
                pass

            def scrape_event_urls(self, event_urls, output_path):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Write a CSV with 1 event = 2 rows.
                output_path.write_text(
                    "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n"
                    "2016-07-09|jones|cormier,jones,-150,-160,-145\n"
                    "2016-07-09|jones|cormier,cormier,+130,+125,+135\n",
                    encoding="utf-8",
                )
                return output_path

        class FakeScraperClient:
            def __init__(self, *args, **kwargs):
                pass

        import sys
        import types

        fake_scraper_mod = types.ModuleType("ufc_prediction.scraper.bfo_scraper")
        fake_scraper_mod.BFOScraper = FakeBFOScraper
        fake_client_mod = types.ModuleType("ufc_prediction.scraper.client")
        fake_client_mod.ScraperClient = FakeScraperClient
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.bfo_scraper",
                            fake_scraper_mod)
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.client",
                            fake_client_mod)

        rc = driver.main([
            "--scrape",
            "--batch-id", batch_id,
            "--gap-json-path", str(gap_json),
        ])
        assert rc == 0

        manifest_path = (
            Path(driver.SCRAPE_BATCH_DIR) / batch_id / "MANIFEST.json"
        )
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        required_keys = {
            "batch_id",
            "scraped_at",
            "n_urls",
            "n_events",
            "n_rows",
            "scrape_duration_seconds",
            "captcha_hit_count",
            "error_count",
        }
        missing = required_keys - manifest.keys()
        assert not missing, f"MANIFEST.json missing required keys: {missing}"
        # Sanity on values we control.
        assert manifest["batch_id"] == batch_id
        assert manifest["n_urls"] == 1
        assert manifest["n_rows"] == 2  # 2 data rows in the synthetic CSV
        # ISO-8601 timestamp parses.
        from datetime import datetime as _dt
        _dt.fromisoformat(manifest["scraped_at"].replace("Z", "+00:00"))

    def test_scrape_writes_fighters_names_csv(
        self, driver, tmp_path: Path, monkeypatch,
    ) -> None:
        """--scrape writes a fighters_names.csv (header present; rows optional)."""
        batch_id = "20260514-164300"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(
            gap_json, batch_id,
            urls=[(1, "UFC 200", "2016-07-09",
                   "https://www.bestfightodds.com/events/ufc-200-1700")],
        )
        monkeypatch.setattr(driver, "SCRAPE_BATCH_DIR", str(tmp_path / "v22-backfill"))
        monkeypatch.setattr(
            driver, "AUDIT_01_SHA_MID_PATH",
            str(tmp_path / "21-XGB-V2-SHA-MID.txt"),
        )

        class FakeBFOScraper:
            def __init__(self, *args, **kwargs):
                pass

            def scrape_event_urls(self, event_urls, output_path):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n",
                    encoding="utf-8",
                )
                return output_path

        class FakeScraperClient:
            def __init__(self, *args, **kwargs):
                pass

        import sys
        import types

        fake_scraper_mod = types.ModuleType("ufc_prediction.scraper.bfo_scraper")
        fake_scraper_mod.BFOScraper = FakeBFOScraper
        fake_client_mod = types.ModuleType("ufc_prediction.scraper.client")
        fake_client_mod.ScraperClient = FakeScraperClient
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.bfo_scraper",
                            fake_scraper_mod)
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.client",
                            fake_client_mod)

        rc = driver.main([
            "--scrape",
            "--batch-id", batch_id,
            "--gap-json-path", str(gap_json),
        ])
        assert rc == 0

        names_csv = (
            Path(driver.SCRAPE_BATCH_DIR) / batch_id / "fighters_names.csv"
        )
        assert names_csv.exists(), (
            "Plan 21-02 Task 1 step 4 expects fighters_names.csv in batch dir"
        )
        # File must have at least a header row (we don't enforce a fixed
        # schema beyond that — fighter resolution lives in BFOOddsIngester).
        content = names_csv.read_text(encoding="utf-8")
        assert content.strip(), "fighters_names.csv must not be empty"

    def test_scrape_emits_audit_01_mid_sha_file(
        self, driver, tmp_path: Path, monkeypatch,
    ) -> None:
        """--scrape writes 21-XGB-V2-SHA-MID.txt with the current xgb_v2 SHA."""
        batch_id = "20260514-164300"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(
            gap_json, batch_id,
            urls=[(1, "UFC 200", "2016-07-09",
                   "https://www.bestfightodds.com/events/ufc-200-1700")],
        )
        monkeypatch.setattr(driver, "SCRAPE_BATCH_DIR", str(tmp_path / "v22-backfill"))

        # Override the SHA-MID path so we don't clobber the real one.
        sha_mid_path = tmp_path / "21-XGB-V2-SHA-MID.txt"
        monkeypatch.setattr(driver, "AUDIT_01_SHA_MID_PATH", str(sha_mid_path))

        class FakeBFOScraper:
            def __init__(self, *args, **kwargs):
                pass

            def scrape_event_urls(self, event_urls, output_path):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n",
                    encoding="utf-8",
                )
                return output_path

        class FakeScraperClient:
            def __init__(self, *args, **kwargs):
                pass

        import sys
        import types

        fake_scraper_mod = types.ModuleType("ufc_prediction.scraper.bfo_scraper")
        fake_scraper_mod.BFOScraper = FakeBFOScraper
        fake_client_mod = types.ModuleType("ufc_prediction.scraper.client")
        fake_client_mod.ScraperClient = FakeScraperClient
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.bfo_scraper",
                            fake_scraper_mod)
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.client",
                            fake_client_mod)

        rc = driver.main([
            "--scrape",
            "--batch-id", batch_id,
            "--gap-json-path", str(gap_json),
        ])
        assert rc == 0

        assert sha_mid_path.exists(), (
            "AUDIT-01 mid SHA file must be emitted by --scrape (Task 1 step 5)"
        )
        sha_text = sha_mid_path.read_text(encoding="utf-8").strip()
        # SHA-256 hex digest = 64 chars, all hex.
        assert len(sha_text) == 64
        assert all(c in "0123456789abcdef" for c in sha_text)
        # Must match the canonical baseline SHA (xgb_v2 byte-identity).
        baseline = Path(".planning/AUDIT-01-BASELINE-SHA.txt").read_text().strip()
        assert sha_text == baseline, (
            "AUDIT-01 mid SHA must equal baseline (xgb_v2 byte-identity); "
            f"got {sha_text} vs baseline {baseline}"
        )

    def test_scrape_dedups_duplicate_urls_in_url_acquired_list(
        self, driver, tmp_path: Path, monkeypatch,
    ) -> None:
        """--scrape de-dups duplicate URLs in url_acquired_list (regression for 2026-05-15 bug).

        Discovered during Plan 21-02 Task 3 first ingest attempt: the gap
        query returned multiple DB Event rows that resolved to the SAME BFO
        URL (e.g. event_id 461 'UFC Event - 2007-06-12' and event_id 1264
        'UFC Fight Night: Stout vs Fisher' both -> ufc-fight-night-...-206-2412).
        Without dedup, BFOScraper.scrape_event_urls iterated the duplicated
        URLs and emitted N copies of identical rows per fight_id. BFOOddsIngester's
        'expected 2 rows' defense then skipped every fight (rows_upserted=0).

        Regression guard: _load_gap_json_for_scrape must dedup URLs while
        preserving first-seen order.
        """
        batch_id = "20260515-001457"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        # Two distinct event_ids both resolve to the SAME URL — exactly
        # the 2026-05-15 production case.
        synthetic_urls = [
            (461, "UFC Event - 2007-06-12", "2007-06-12",
             "https://www.bestfightodds.com/events/ufc-fight-night-stout-vs-fisher-206"),
            (1264, "UFC Fight Night: Stout vs Fisher", "2007-06-12",
             "https://www.bestfightodds.com/events/ufc-fight-night-stout-vs-fisher-206"),
            (1265, "UFC 72: Victory", "2007-06-16",
             "https://www.bestfightodds.com/events/ufc-72-victory-2"),
        ]
        self._write_synthetic_gap_json(gap_json, batch_id, synthetic_urls)

        monkeypatch.setattr(driver, "SCRAPE_BATCH_DIR", str(tmp_path / "v22-backfill"))
        monkeypatch.setattr(
            driver, "AUDIT_01_SHA_MID_PATH",
            str(tmp_path / "21-XGB-V2-SHA-MID.txt"),
        )

        scraper_instances: list[Any] = []

        class FakeBFOScraper:
            def __init__(self, *args, **kwargs):
                self.calls: list[tuple[list[str], Path]] = []
                scraper_instances.append(self)

            def scrape_event_urls(self, event_urls, output_path):
                self.calls.append((list(event_urls), Path(output_path)))
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n",
                    encoding="utf-8",
                )
                return output_path

        class FakeScraperClient:
            def __init__(self, *args, **kwargs):
                pass

        import sys
        import types

        fake_scraper_mod = types.ModuleType("ufc_prediction.scraper.bfo_scraper")
        fake_scraper_mod.BFOScraper = FakeBFOScraper
        fake_client_mod = types.ModuleType("ufc_prediction.scraper.client")
        fake_client_mod.ScraperClient = FakeScraperClient
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.bfo_scraper",
                            fake_scraper_mod)
        monkeypatch.setitem(sys.modules, "ufc_prediction.scraper.client",
                            fake_client_mod)

        rc = driver.main([
            "--scrape",
            "--batch-id", batch_id,
            "--gap-json-path", str(gap_json),
        ])
        assert rc == 0

        # BFOScraper.scrape_event_urls received DEDUPED URLs.
        inst = scraper_instances[0]
        urls_passed, _ = inst.calls[0]
        assert len(urls_passed) == 2, (
            "Two distinct URLs expected after dedup (3 input entries "
            "collapse to 2 unique URLs); got "
            f"{len(urls_passed)} URLs: {urls_passed}"
        )
        # First-seen order preserved: stout-vs-fisher (first seen) before
        # ufc-72-victory.
        assert urls_passed == [
            "https://www.bestfightodds.com/events/ufc-fight-night-stout-vs-fisher-206",
            "https://www.bestfightodds.com/events/ufc-72-victory-2",
        ]

    def test_load_gap_json_rejects_entries_with_null_url(
        self, driver, tmp_path: Path,
    ) -> None:
        """WR-07 regression: entries with null/empty 'url' field raise RuntimeError.

        Previously a payload like
            {"url_acquired_list": [{"event_id": 1, "url": null}, ...]}
        passed the outer ``if not url_list`` check, then the dedup loop
        silently skipped null entries, leaving ``urls=[]``. The --scrape
        mode would proceed with 0 URLs and exit \"successfully\" while
        scraping nothing.

        The validated path now raises RuntimeError on the FIRST null/empty
        url entry — fail loudly, do not silently consume malformed data.
        """
        batch_id = "20260514-170000"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        # Payload with one null-url entry sandwiched between valid ones.
        payload = {
            "batch_id": batch_id,
            "gap_query": {
                "cutoff_date": "2007-06-01",
                "n_in_scope_events": 3,
                "n_events_with_no_odds": 3,
                "n_events_with_partial_odds": 0,
            },
            "url_resolution": {
                "url_acquired": 3, "event_not_found": 0,
                "disambiguation_fail": 0,
            },
            "estimated_scrape_wall_clock_min": 1,
            "per_year_breakdown": {},
            "url_acquired_list": [
                {"event_id": 1, "event_name": "OK 1",
                 "event_date": "2016-07-09",
                 "url": "https://www.bestfightodds.com/events/ok-1-100"},
                {"event_id": 2, "event_name": "Null URL",
                 "event_date": "2016-07-10",
                 "url": None},
                {"event_id": 3, "event_name": "OK 2",
                 "event_date": "2016-07-11",
                 "url": "https://www.bestfightodds.com/events/ok-2-200"},
            ],
            "next_step": "test fixture",
        }
        gap_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(RuntimeError) as excinfo:
            driver._load_gap_json_for_scrape(gap_json, batch_id)
        # Error message names the missing/empty url field for operator
        # diagnostics.
        assert "url" in str(excinfo.value).lower()

    def test_load_gap_json_rejects_all_null_url_entries(
        self, driver, tmp_path: Path,
    ) -> None:
        """WR-07: payload with every url null raises RuntimeError on first one.

        Documents the fail-loud behavior — no silent "0 URLs scraped =
        success" path.
        """
        batch_id = "20260514-170100"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        payload = {
            "batch_id": batch_id,
            "gap_query": {
                "cutoff_date": "2007-06-01",
                "n_in_scope_events": 2,
                "n_events_with_no_odds": 2,
                "n_events_with_partial_odds": 0,
            },
            "url_resolution": {
                "url_acquired": 0, "event_not_found": 2,
                "disambiguation_fail": 0,
            },
            "estimated_scrape_wall_clock_min": 1,
            "per_year_breakdown": {},
            "url_acquired_list": [
                {"event_id": 1, "url": None},
                {"event_id": 2, "url": ""},
            ],
            "next_step": "test fixture",
        }
        gap_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(RuntimeError):
            driver._load_gap_json_for_scrape(gap_json, batch_id)

    def test_count_unique_events_rejects_malformed_date_prefix(
        self, driver, tmp_path: Path,
    ) -> None:
        """WR-04 regression: _count_unique_events skips non-ISO-date prefixes.

        Previously ``fight_id.split("|", 1)[0]`` accepted ANY non-empty
        string as a date prefix — so a malformed fight_id like
        ``"jones|cormier"`` would silently inflate n_events with key
        ``"jones"``. The validated path now skips malformed prefixes and
        counts only ISO-date-parseable prefixes.
        """
        csv_path = tmp_path / "BestFightOdds_odds.csv"
        # Header + 4 rows: 2 valid (same event date), 2 malformed.
        csv_path.write_text(
            "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n"
            "2016-07-09|jones|cormier,jones,-150,-160,-145\n"
            "2016-07-09|jones|cormier,cormier,+130,+125,+135\n"
            # Malformed: no date prefix
            "jones|cormier|extra,jones,-100,-100,-100\n"
            # Malformed: garbage prefix
            "not-a-date|a|b,a,-100,-100,-100\n",
            encoding="utf-8",
        )
        result = driver._count_unique_events(csv_path)
        assert result == 1, (
            f"WR-04: expected exactly 1 unique event from ISO-date-parseable "
            f"prefix; got {result} (malformed prefixes must be skipped)"
        )

    def test_count_unique_events_accepts_multiple_iso_dates(
        self, driver, tmp_path: Path,
    ) -> None:
        """WR-04: well-formed CSV with N distinct ISO dates → n_events == N."""
        csv_path = tmp_path / "BestFightOdds_odds.csv"
        csv_path.write_text(
            "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n"
            "2016-07-09|a|b,a,-100,-100,-100\n"
            "2016-07-09|a|b,b,+90,+90,+90\n"
            "2017-01-01|c|d,c,-110,-110,-110\n"
            "2017-01-01|c|d,d,+100,+100,+100\n",
            encoding="utf-8",
        )
        assert driver._count_unique_events(csv_path) == 2

    def test_scrape_stats_handler_classifies_captcha_vs_error(
        self, driver, tmp_path: Path, monkeypatch,
    ) -> None:
        """WR-05 regression: _StatsHandler keyword classification end-to-end.

        Verifies the live keyword-grep classifier inside _run_scrape_mode:
        the test emits 4 log messages from a fake BFOScraper.scrape_event_urls
        — 2 captcha-flavored + 2 error-flavored + 1 ambiguous (combined
        "captcha" + "fetch failed") that MUST count as captcha (the elif
        precedence). The resulting MANIFEST.json captcha_hit_count and
        error_count must reflect the intended classification, NOT silently
        drop messages whose phrasing doesn't match.

        This test pins the keyword set ("captcha", "fetch failed",
        "parse failed", "returned none") — if the scraper log shape ever
        drifts to phrasing like "Cloudflare challenge" instead of
        "captcha", this test will fail and the keyword list will need
        coordinated update.
        """
        batch_id = "20260514-160000"
        gap_json = tmp_path / "BFO_V22_GAP.json"
        synthetic_urls = [
            (1, "Event A", "2016-07-09",
             "https://www.bestfightodds.com/events/event-a-1700"),
        ]
        self._write_synthetic_gap_json(gap_json, batch_id, synthetic_urls)

        monkeypatch.setattr(
            driver, "SCRAPE_BATCH_DIR", str(tmp_path / "v22-backfill"),
        )
        monkeypatch.setattr(
            driver, "AUDIT_01_SHA_MID_PATH",
            str(tmp_path / "21-XGB-V2-SHA-MID.txt"),
        )

        # Fake scraper that emits 5 log records via the bfo_scraper logger
        # then writes a header-only CSV.
        class FakeBFOScraper:
            def __init__(self, *args, **kwargs):
                pass

            def scrape_event_urls(self, event_urls, output_path):
                bfo_logger = logging.getLogger(
                    "ufc_prediction.scraper.bfo_scraper",
                )
                # 2 pure captcha hits.
                bfo_logger.warning("BFO CAPTCHA gate hit (id=hfmr8) on %s", "u1")
                bfo_logger.warning("captcha gate detected on %s", "u2")
                # 2 pure error hits (one "fetch failed", one "parse failed").
                bfo_logger.warning("fetch failed for %s after retries", "u3")
                bfo_logger.warning("parse failed for %s: BFOParseError", "u4")
                # 1 ambiguous — contains both "captcha" and "fetch failed"
                # → classified as captcha per the elif precedence (intentional).
                bfo_logger.warning(
                    "fetch failed for %s: CAPTCHA gate intercepted", "u5",
                )
                # 1 "returned none" (additional error keyword).
                bfo_logger.warning("returned none for %s", "u6")
                # Write a header-only CSV so downstream tallies don't crash.
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n",
                    encoding="utf-8",
                )
                return output_path

        class FakeScraperClient:
            def __init__(self, *args, **kwargs):
                pass

        import sys
        import types

        fake_scraper_mod = types.ModuleType("ufc_prediction.scraper.bfo_scraper")
        fake_scraper_mod.BFOScraper = FakeBFOScraper
        fake_client_mod = types.ModuleType("ufc_prediction.scraper.client")
        fake_client_mod.ScraperClient = FakeScraperClient
        monkeypatch.setitem(
            sys.modules,
            "ufc_prediction.scraper.bfo_scraper",
            fake_scraper_mod,
        )
        monkeypatch.setitem(
            sys.modules,
            "ufc_prediction.scraper.client",
            fake_client_mod,
        )

        rc = driver.main([
            "--scrape",
            "--batch-id", batch_id,
            "--gap-json-path", str(gap_json),
        ])
        assert rc == 0

        # Read MANIFEST.json and verify the counter values.
        manifest_path = (
            Path(driver.SCRAPE_BATCH_DIR) / batch_id / "MANIFEST.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 2 pure captcha + 1 ambiguous (captcha takes precedence) = 3.
        assert manifest["captcha_hit_count"] == 3, (
            f"WR-05: expected 3 captcha hits (2 pure + 1 ambiguous); "
            f"got {manifest['captcha_hit_count']}. Manifest: {manifest}"
        )
        # 2 pure errors (fetch failed + parse failed) + 1 returned none
        # = 3. The ambiguous case does NOT count here.
        assert manifest["error_count"] == 3, (
            f"WR-05: expected 3 errors (fetch failed + parse failed + "
            f"returned none); got {manifest['error_count']}. "
            f"Manifest: {manifest}"
        )


# ── 8. --integrity-check mode (Plan 21-02 Task 2 — pre-ingest gate) ────────


class TestIntegrityCheck:
    """Plan 21-02 Task 2: --integrity-check rematch-distinctness gate.

    Per CONTEXT.md D-05 (REVISED): the integrity check runs against the new
    BestFightOdds_odds.csv BEFORE BFOOddsIngester.ingest_all() is invoked.
    The check guards against the 999.4-class regression — any same-pair,
    same-event-date rows with DIFFERENT fight_ids would silently overwrite
    each other on the pk_fight_odds upsert. Halt-and-decide (no auto-fix).

    Synthetic CSVs are written to ``tmp_path``; no canonical fixture files
    are added to tests/fixtures/ (per the prompt's tmp-path-only directive).
    """

    # Header used by every synthetic CSV — matches BFOOddsRow's columns.
    _HEADER = (
        "fight_id,fighter_id,opening,closing_range_min,closing_range_max\n"
    )

    @classmethod
    def _write_csv(cls, path: Path, rows: list[str]) -> Path:
        """Helper: write header + data rows to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cls._HEADER + "".join(rows), encoding="utf-8")
        return path

    # ── 8a. Pure-function tests on _run_integrity_check ────────────────

    def test_integrity_check_passes_on_distinct_dates_rematch(
        self, driver, tmp_path: Path,
    ) -> None:
        """Same pair, two DIFFERENT event dates → clean (legitimate rematch).

        Couture vs Liddell I (2003) and II (2006) would be the canonical
        example; we use synthetic IDs here.
        """
        csv = self._write_csv(
            tmp_path / "rematch_ok.csv",
            [
                # Fight 1 — 2018-04-07 — pair (couture, liddell)
                "2018-04-07|couture|liddell,couture,-150,-160,-145\n",
                "2018-04-07|couture|liddell,liddell,+130,+125,+135\n",
                # Fight 2 — 2018-09-15 — same pair, different date
                "2018-09-15|couture|liddell,couture,-110,-115,-105\n",
                "2018-09-15|couture|liddell,liddell,-110,-115,-105\n",
            ],
        )
        summary = driver._run_integrity_check(csv)
        assert summary["violations"] == []
        assert summary["n_pairs"] == 1
        assert summary["n_rematches"] == 1
        assert summary["n_rows"] == 4

    def test_integrity_check_passes_on_single_fight_pairs(
        self, driver, tmp_path: Path,
    ) -> None:
        """3 pairs, each with one fight (no rematches) → clean summary."""
        csv = self._write_csv(
            tmp_path / "no_rematches.csv",
            [
                "2020-01-01|a|b,a,-150,-160,-145\n",
                "2020-01-01|a|b,b,+130,+125,+135\n",
                "2020-02-01|c|d,c,-200,-210,-195\n",
                "2020-02-01|c|d,d,+170,+165,+175\n",
                "2020-03-01|e|f,e,+100,+95,+105\n",
                "2020-03-01|e|f,f,-110,-115,-105\n",
            ],
        )
        summary = driver._run_integrity_check(csv)
        assert summary["violations"] == []
        assert summary["n_pairs"] == 3
        assert summary["n_rematches"] == 0
        assert summary["n_rows"] == 6

    def test_integrity_check_fails_on_identical_dates_rematch(
        self, driver, tmp_path: Path,
    ) -> None:
        """Same pair, same event_date, DIFFERENT fight_ids → BFOIntegrityError.

        This is the 999.4-class regression: two distinct fight_ids for the
        same pair on the same date would silently overwrite each other on
        the pk_fight_odds upsert. The integrity check halts the pipeline.
        """
        csv = self._write_csv(
            tmp_path / "rematch_violation.csv",
            [
                # Two distinct fight_ids on the SAME date for the SAME pair.
                # Realistic 999.4 trigger: the BFO scraper emitted both
                # fighter-orderings of the composite fight_id (a|b and b|a)
                # for the same bout — both share canonical pair_key
                # ("cormier", "jones") but the fight_id strings differ.
                "2020-06-01|jones|cormier,jones,-150,-160,-145\n",
                "2020-06-01|jones|cormier,cormier,+130,+125,+135\n",
                "2020-06-01|cormier|jones,jones,-110,-115,-105\n",
                "2020-06-01|cormier|jones,cormier,+90,+85,+95\n",
            ],
        )
        with pytest.raises(driver.BFOIntegrityError) as excinfo:
            driver._run_integrity_check(csv)
        assert len(excinfo.value.violations) >= 1
        # The violation entry references the offending pair_key.
        violation = excinfo.value.violations[0]
        # Pair key is canonically sorted: ("cormier", "jones").
        assert "cormier" in str(violation).lower()
        assert "jones" in str(violation).lower()

    def test_integrity_check_handles_empty_csv(
        self, driver, tmp_path: Path,
    ) -> None:
        """Header-only CSV → clean summary with all-zero counts."""
        csv = self._write_csv(tmp_path / "empty.csv", [])
        summary = driver._run_integrity_check(csv)
        assert summary["violations"] == []
        assert summary["n_pairs"] == 0
        assert summary["n_rematches"] == 0
        assert summary["n_rows"] == 0

    def test_integrity_check_raises_FileNotFoundError_for_missing_csv(
        self, driver, tmp_path: Path,
    ) -> None:
        """Non-existent CSV path → FileNotFoundError with clear message."""
        missing = tmp_path / "nope" / "BestFightOdds_odds.csv"
        with pytest.raises(FileNotFoundError) as excinfo:
            driver._run_integrity_check(missing)
        # Message must reference the missing path so the operator can act.
        assert str(missing) in str(excinfo.value)

    # ── 8b. main() routing tests ─────────────────────────────────────────

    def test_main_integrity_check_requires_batch_id(
        self, driver, capsys,
    ) -> None:
        """--integrity-check without --batch-id exits non-zero with stderr msg."""
        rc = driver.main(["--integrity-check"])
        assert rc != 0, (
            "--integrity-check without --batch-id must exit non-zero"
        )
        captured = capsys.readouterr()
        assert "--batch-id" in captured.err

    def test_main_integrity_check_invokes_function_with_correct_path(
        self, driver, monkeypatch,
    ) -> None:
        """main(--integrity-check --batch-id X) calls _run_integrity_check
        with ``Path(SCRAPE_BATCH_DIR) / X / BestFightOdds_odds.csv``.
        """
        called_with: list[Path] = []

        def fake_run_integrity_check(csv_path: Path) -> dict:
            called_with.append(csv_path)
            return {
                "n_pairs": 0, "n_rematches": 0, "n_rows": 0, "violations": [],
            }

        monkeypatch.setattr(
            driver, "_run_integrity_check", fake_run_integrity_check,
        )
        rc = driver.main(["--integrity-check", "--batch-id", "20260514-231731"])
        assert rc == 0
        assert len(called_with) == 1
        expected = (
            Path(driver.SCRAPE_BATCH_DIR)
            / "20260514-231731"
            / "BestFightOdds_odds.csv"
        )
        assert called_with[0] == expected

    def test_main_integrity_check_clean_exits_0(
        self, driver, monkeypatch, capsys,
    ) -> None:
        """Clean integrity summary → main returns 0 + prints summary line."""
        def fake_run(csv_path: Path) -> dict:
            return {
                "n_pairs": 50,
                "n_rematches": 7,
                "n_rows": 200,
                "violations": [],
            }
        monkeypatch.setattr(driver, "_run_integrity_check", fake_run)
        rc = driver.main(["--integrity-check", "--batch-id", "fakebatch"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Integrity check passed" in out
        # Summary numbers surface for the operator.
        assert "7" in out and "50" in out and "200" in out

    def test_main_integrity_check_violation_exits_1(
        self, driver, monkeypatch, capsys,
    ) -> None:
        """BFOIntegrityError raised → main returns 1; stderr lists violations."""
        def fake_run(csv_path: Path) -> dict:
            raise driver.BFOIntegrityError(
                violations=[
                    {
                        "pair_key": ("cormier", "jones"),
                        "event_date": "2020-06-01",
                        "fight_ids": [
                            "2020-06-01|jones|cormier",
                            "2020-06-01|jones|cormier-x",
                        ],
                    },
                ],
            )
        monkeypatch.setattr(driver, "_run_integrity_check", fake_run)
        rc = driver.main(["--integrity-check", "--batch-id", "fakebatch"])
        assert rc == 1
        err = capsys.readouterr().err
        # Operator sees the violating pair in stderr.
        assert "cormier" in err.lower() or "jones" in err.lower()


# ── 9. --emit-coverage mode (Plan 21-02 Task 4 — coverage delta JSON) ──────


class TestEmitCoverage:
    """Plan 21-02 Task 4: --emit-coverage writes BFO_V22_COVERAGE.json.

    The mode reads BFO_V22_GAP.json (pre-backfill state), runs a DB query to
    compute coverage_after (post-2007 training-set fights with non-NaN
    closing_implied_prob), and writes the BFO_V22_COVERAGE.json artifact
    with all 11 top-level keys per Task 4 spec (21-02-PLAN.md lines 167-186).

    All DB work is mocked at the session level — no live DB or HTTP. The
    operator passes pre/post training+test counts via CLI flags
    (--n-train-pre, --n-train-post, --n-test-pre, --n-test-post) so the
    cutoff-date discipline check (BFO-V22-04) can be evaluated without
    requiring the ingester's IngestSummary to be marshalled through argv.
    """

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _write_synthetic_gap_json(
        path: Path,
        batch_id: str,
        n_in_scope: int = 1045,
        n_no_odds: int = 1045,
        n_partial: int = 0,
    ) -> None:
        """Write a minimal BFO_V22_GAP.json (matches Plan 21-01 schema)."""
        payload = {
            "batch_id": batch_id,
            "gap_query": {
                "cutoff_date": "2007-06-01",
                "n_in_scope_events": n_in_scope,
                "n_events_with_no_odds": n_no_odds,
                "n_events_with_partial_odds": n_partial,
            },
            "url_resolution": {
                "url_acquired": 861,
                "event_not_found": 0,
                "disambiguation_fail": 184,
            },
            "estimated_scrape_wall_clock_min": 22,
            "per_year_breakdown": {
                "2007": {"n_events": 24, "url_acquired": 24,
                         "event_not_found": 0, "disambiguation_fail": 0},
            },
            "url_acquired_list": [],
            "next_step": "test fixture",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _make_session_mock(
        coverage_after: float,
        per_year: dict[str, tuple[int, int]] | None = None,
    ) -> MagicMock:
        """Build a MagicMock session whose .execute() returns numerator/
        denominator pairs for the coverage_after query and per-year breakdown.

        Contract:
          - The first call returns the overall (covered, total) tuple.
          - Subsequent calls return per-year (covered, total) tuples in
            the order yielded by per_year.items() (year-string keys).

        ``per_year`` shape: ``{"2007": (covered, total), ...}``.

        WR-06 fix (REVIEW.md): the previous implementation used a single
        ``iter(results)`` closed over by ``_execute``; after the sequence
        was exhausted ALL subsequent calls returned (0, 0). If a test
        invoked ``_emit_coverage_json`` twice with the same session mock,
        the second invocation would silently see zeros for every query.
        The current implementation cycles through ``results`` via modulo
        indexing — each ``_emit_coverage_json`` invocation re-starts from
        the overall (covered, total) tuple, making the mock TRULY reusable
        across multiple driver calls. The default ``per_year`` is padded
        so every year in 2007..2025 has an entry, which means the cycle
        length matches the # of queries one ``_emit_coverage_json`` makes.
        """
        # Default per-year: 7 years (2007-2013) with realistic counts.
        if per_year is None:
            per_year = {
                "2007": (10, 12),
                "2010": (40, 44),
                "2013": (60, 66),
                "2016": (70, 73),
                "2019": (75, 78),
                "2022": (80, 84),
                "2025": (50, 53),
            }
        # Reverse-engineer overall (covered, total) from coverage_after when
        # the caller didn't supply a tuple.
        total = 1000
        covered = int(round(coverage_after * total))

        # Build the .execute().one() return-value sequence.
        # First call: overall (covered, total). Subsequent: per-year tuples
        # in 2007..2025 order, defaulting to (0, 0) for years not in
        # ``per_year`` (the helper skips zero-total years in its output, so
        # padding with zeros is safe).
        results: list[Any] = [(covered, total)]
        for year in range(2007, 2026):
            results.append(per_year.get(str(year), (0, 0)))

        session = MagicMock()
        # Stateful counter — but cycle-aware. Each invocation of
        # ``_emit_coverage_json`` makes len(results) queries; modulo
        # indexing rolls back to the overall (covered, total) tuple at the
        # start of each subsequent driver call, so the mock is reusable.
        # Note: if a test relies on iterator-exhaustion as a signal (e.g.
        # "query count must equal len(results) exactly"), it should assert
        # ``session.execute.call_count == len(results)`` explicitly rather
        # than rely on returning (0, 0) for overruns.
        state = {"idx": 0}

        def _execute(*_a: Any, **_kw: Any) -> Any:
            res = MagicMock()
            res.one.return_value = results[state["idx"] % len(results)]
            state["idx"] += 1
            return res

        session.execute.side_effect = _execute
        return session

    # ── 9a. Pure-function tests on _emit_coverage_json ────────────────────

    def test_emit_coverage_writes_all_required_keys(
        self, driver, tmp_path: Path,
    ) -> None:
        """Output JSON has all 11 top-level keys per Task 4 spec."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="testbatch")
        out = tmp_path / "BFO_V22_COVERAGE.json"
        session = self._make_session_mock(coverage_after=0.96)

        driver._emit_coverage_json(
            batch_id="testbatch",
            ingest_summary={
                "bfo_fights_scanned": 100, "fights_matched": 95,
                "rows_upserted": 190,
            },
            gap_json_path=gap_json,
            coverage_json_path=out,
            session=session,
            n_train_pre=4500, n_train_post=4500,
            n_test_pre=900, n_test_post=900,
        )
        assert out.exists()
        payload = json.loads(out.read_text())
        required = {
            "batch_id",
            "ingested_at",
            "coverage_before",
            "coverage_after",
            "coverage_delta",
            "target_rate_threshold",
            "target_rate_achieved",
            "n_training_fights_pre",
            "n_training_fights_post",
            "n_test_fights_pre",
            "n_test_fights_post",
            "cutoff_date_discipline_preserved",
            "per_year_coverage_breakdown",
            "known_acquisition_gaps",
        }
        missing = required - payload.keys()
        assert not missing, f"missing required keys: {missing}"
        # batch_id round-trips.
        assert payload["batch_id"] == "testbatch"
        # threshold is the locked constant.
        assert payload["target_rate_threshold"] == driver.TARGET_RATE_THRESHOLD

    def test_make_session_mock_supports_repeat_invocations(
        self, driver, tmp_path: Path,
    ) -> None:
        """WR-06 regression: _make_session_mock survives reuse.

        Previously the result iterator was built once at construction; a
        second ``_emit_coverage_json`` call against the same session mock
        would silently see (0, 0) for every query, producing wrong
        coverage numbers and a misleading "passing" test.

        With WR-06's cycle-aware indexing, the mock re-starts from the
        overall (covered, total) tuple at the head of every driver call.
        This test invokes ``_emit_coverage_json`` twice against the same
        session and verifies the second output has the SAME
        ``coverage_after`` as the first (proving the second call did
        NOT consume zeros from an exhausted iterator).
        """
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="testbatch")
        out_a = tmp_path / "BFO_V22_COVERAGE_a.json"
        out_b = tmp_path / "BFO_V22_COVERAGE_b.json"
        session = self._make_session_mock(coverage_after=0.93)

        # First invocation.
        driver._emit_coverage_json(
            batch_id="testbatch",
            ingest_summary={
                "bfo_fights_scanned": 100, "fights_matched": 90,
                "rows_upserted": 180,
            },
            gap_json_path=gap_json,
            coverage_json_path=out_a,
            session=session,
            n_train_pre=4500, n_train_post=4500,
            n_test_pre=900, n_test_post=900,
        )
        # Second invocation (same session mock).
        driver._emit_coverage_json(
            batch_id="testbatch",
            ingest_summary={
                "bfo_fights_scanned": 100, "fights_matched": 90,
                "rows_upserted": 180,
            },
            gap_json_path=gap_json,
            coverage_json_path=out_b,
            session=session,
            n_train_pre=4500, n_train_post=4500,
            n_test_pre=900, n_test_post=900,
        )
        payload_a = json.loads(out_a.read_text())
        payload_b = json.loads(out_b.read_text())
        # Coverage numbers must match — proving the second invocation saw
        # the same query-result sequence as the first (no iterator
        # exhaustion → zeros).
        assert payload_a["coverage_after"] == payload_b["coverage_after"], (
            f"WR-06: second invocation drifted from first. "
            f"a={payload_a['coverage_after']!r}, "
            f"b={payload_b['coverage_after']!r}"
        )
        # And the value is NOT zero (which would indicate the iterator
        # exhausted silently — the exact failure mode WR-06 documents).
        assert payload_b["coverage_after"] > 0.0, (
            "WR-06: second invocation got coverage_after=0.0, suggesting "
            "an exhausted iterator returning (0, 0) for every query."
        )

    def test_emit_coverage_target_rate_achieved_true_when_above_threshold(
        self, driver, tmp_path: Path,
    ) -> None:
        """coverage_after >= 0.96 → target_rate_achieved == True."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="b1")
        out = tmp_path / "BFO_V22_COVERAGE.json"
        session = self._make_session_mock(coverage_after=0.97)

        result = driver._emit_coverage_json(
            batch_id="b1",
            ingest_summary={},
            gap_json_path=gap_json,
            coverage_json_path=out,
            session=session,
            n_train_pre=100, n_train_post=100,
            n_test_pre=20, n_test_post=20,
        )
        assert result["target_rate_achieved"] is True
        assert result["coverage_after"] >= 0.96

    def test_emit_coverage_target_rate_achieved_false_when_below_threshold(
        self, driver, tmp_path: Path,
    ) -> None:
        """coverage_after < 0.96 → target_rate_achieved == False."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="b2")
        out = tmp_path / "BFO_V22_COVERAGE.json"
        session = self._make_session_mock(coverage_after=0.85)

        result = driver._emit_coverage_json(
            batch_id="b2",
            ingest_summary={},
            gap_json_path=gap_json,
            coverage_json_path=out,
            session=session,
            n_train_pre=100, n_train_post=100,
            n_test_pre=20, n_test_post=20,
        )
        assert result["target_rate_achieved"] is False
        assert result["coverage_after"] < 0.96

    def test_emit_coverage_delta_computed_correctly(
        self, driver, tmp_path: Path,
    ) -> None:
        """coverage_delta == coverage_after - coverage_before (float math)."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        # Synthetic gap: 600 in-scope events, 360 with no odds, 0 partial
        # → coverage_before = 1 - (360/600) = 0.40.
        self._write_synthetic_gap_json(
            gap_json, batch_id="b3",
            n_in_scope=600, n_no_odds=360, n_partial=0,
        )
        out = tmp_path / "BFO_V22_COVERAGE.json"
        session = self._make_session_mock(coverage_after=0.92)

        result = driver._emit_coverage_json(
            batch_id="b3",
            ingest_summary={},
            gap_json_path=gap_json,
            coverage_json_path=out,
            session=session,
            n_train_pre=100, n_train_post=100,
            n_test_pre=20, n_test_post=20,
        )
        # Allow ±1e-6 for float-tolerance.
        assert abs(result["coverage_before"] - 0.40) < 1e-6
        assert abs(result["coverage_after"] - 0.92) < 1e-6
        assert abs(result["coverage_delta"] - 0.52) < 1e-6

    def test_emit_coverage_cutoff_discipline_preserved_when_counts_unchanged(
        self, driver, tmp_path: Path,
    ) -> None:
        """n_train_pre == n_train_post AND n_test_pre == n_test_post → True."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="b4")
        out = tmp_path / "BFO_V22_COVERAGE.json"
        session = self._make_session_mock(coverage_after=0.96)

        result = driver._emit_coverage_json(
            batch_id="b4",
            ingest_summary={},
            gap_json_path=gap_json,
            coverage_json_path=out,
            session=session,
            n_train_pre=4500, n_train_post=4500,
            n_test_pre=900, n_test_post=900,
        )
        assert result["cutoff_date_discipline_preserved"] is True

    def test_emit_coverage_cutoff_discipline_violated_when_counts_drift(
        self, driver, tmp_path: Path,
    ) -> None:
        """Train or test count drift → cutoff_date_discipline_preserved=False."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="b5")
        out = tmp_path / "BFO_V22_COVERAGE.json"
        session = self._make_session_mock(coverage_after=0.96)

        # Train count drifted by 1 row.
        result = driver._emit_coverage_json(
            batch_id="b5",
            ingest_summary={},
            gap_json_path=gap_json,
            coverage_json_path=out,
            session=session,
            n_train_pre=4500, n_train_post=4501,
            n_test_pre=900, n_test_post=900,
        )
        assert result["cutoff_date_discipline_preserved"] is False

    def test_emit_coverage_per_year_breakdown_has_expected_year_range(
        self, driver, tmp_path: Path,
    ) -> None:
        """per_year_coverage_breakdown has entries for ≥6 years."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="b6")
        out = tmp_path / "BFO_V22_COVERAGE.json"
        per_year = {
            "2007": (5, 12),
            "2010": (40, 44),
            "2013": (33, 66),
            "2016": (70, 73),
            "2019": (75, 78),
            "2022": (80, 84),
            "2025": (50, 53),
        }
        session = self._make_session_mock(
            coverage_after=0.92, per_year=per_year,
        )

        result = driver._emit_coverage_json(
            batch_id="b6",
            ingest_summary={},
            gap_json_path=gap_json,
            coverage_json_path=out,
            session=session,
            n_train_pre=100, n_train_post=100,
            n_test_pre=20, n_test_post=20,
        )
        breakdown = result["per_year_coverage_breakdown"]
        # ≥6 years per acceptance criterion.
        assert len(breakdown) >= 6
        # 2007 + at least one year in 2010-2022 range present.
        assert "2007" in breakdown
        assert any(
            y in breakdown for y in
            ["2010", "2013", "2016", "2019", "2022"]
        )

    # ── 9b. main() routing tests ─────────────────────────────────────────

    def test_main_emit_coverage_requires_all_count_args(
        self, driver, capsys,
    ) -> None:
        """--emit-coverage --batch-id X without count args exits non-zero."""
        rc = driver.main(["--emit-coverage", "--batch-id", "b7"])
        assert rc != 0, (
            "--emit-coverage without count args must exit non-zero"
        )
        captured = capsys.readouterr()
        # Error mentions one of the required count flags.
        assert any(
            flag in captured.err
            for flag in ["--n-train-pre", "--n-train-post",
                         "--n-test-pre", "--n-test-post"]
        )

    def test_main_emit_coverage_invokes_function_with_correct_args(
        self, driver, monkeypatch, tmp_path: Path,
    ) -> None:
        """main(--emit-coverage ...) calls _emit_coverage_json with parsed args."""
        gap_json = tmp_path / "BFO_V22_GAP.json"
        self._write_synthetic_gap_json(gap_json, batch_id="b8")
        coverage_out = tmp_path / "BFO_V22_COVERAGE.json"

        # Override the COVERAGE_JSON_PATH so we don't pollute the real path.
        monkeypatch.setattr(
            driver, "COVERAGE_JSON_PATH", str(coverage_out),
        )

        # Stub the SessionLocal lazy import.
        import sys
        import types
        fake_db_mod = types.ModuleType("ufc_prediction.db.session")
        fake_db_mod.SessionLocal = lambda: MagicMock()
        monkeypatch.setitem(
            sys.modules, "ufc_prediction.db.session", fake_db_mod,
        )

        called_with: dict[str, Any] = {}

        def fake_emit(**kwargs: Any) -> dict[str, Any]:
            called_with.update(kwargs)
            return {
                "batch_id": kwargs["batch_id"],
                "coverage_before": 0.40,
                "coverage_after": 0.97,
                "coverage_delta": 0.57,
                "target_rate_achieved": True,
                "cutoff_date_discipline_preserved": True,
            }

        monkeypatch.setattr(driver, "_emit_coverage_json", fake_emit)

        rc = driver.main([
            "--emit-coverage",
            "--batch-id", "b8",
            "--gap-json-path", str(gap_json),
            "--n-train-pre", "4500",
            "--n-train-post", "4500",
            "--n-test-pre", "900",
            "--n-test-post", "900",
        ])
        assert rc == 0
        # The stub was called with the parsed CLI args.
        assert called_with["batch_id"] == "b8"
        assert called_with["n_train_pre"] == 4500
        assert called_with["n_train_post"] == 4500
        assert called_with["n_test_pre"] == 900
        assert called_with["n_test_post"] == 900
        assert Path(called_with["gap_json_path"]) == gap_json
