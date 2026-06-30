"""Wave-0 RED tests for ``scripts/audit_bfo_reachability.py``.

Per CONTEXT.md D-07..D-10 + PLAN 20-02 Task 2 spec.

Mocks ``ScraperClient`` via MagicMock per the project pattern established
in ``tests/scraper/test_bfo_scraper.py:21``. Uses ``pytest.importorskip``
so tests SKIP cleanly until ``scripts/audit_bfo_reachability.py`` lands,
then turn GREEN on import.

Design refinements from Task 1 findings (operator-curated CSV):
  - BFO returns HTTP 200 with the GENERIC HOMEPAGE for invalid event
    slugs (``<title>UFC & MMA Odds & Betting Lines | Best Fight Odds</title>``).
    The probe MUST inspect page content to distinguish a reachable event
    page (status="reachable") from a homepage fallback
    (status="homepage_fallback"). HTTP 200 alone is NOT sufficient
    (Pitfall #4 / Task 1 finding).
  - Operator-marked unfindable rows use ``event_url == "EVENT_NOT_FOUND"``
    sentinel (e.g., UFC 67/69/71 pre-BFO-archive, UFC 272 search
    disambiguation failure). These count as ``status="event_not_found"``
    in the per-year totals — no HTTP request issued.
  - The pre-registered post-2007 coverage target is derived MECHANICALLY
    from the per-year reachability table with a floor of 80%:
    ``target_rate = max(0.80, n_reachable_post2007 / n_total_post2007)``
    where post-2007 excludes the 2007 boundary anchor.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def audit_bfo():
    """Lazy-import the audit script as a module.

    Mirrors ``tests/scraper/test_bfo_scraper.py:28-36`` ``importorskip``
    pattern so Wave-0 RED can run before
    ``scripts/audit_bfo_reachability.py`` lands.
    """
    return pytest.importorskip("scripts.audit_bfo_reachability")


# ── Test 1: locked constants present and correct ─────────────────────────


class TestLockedConstants:
    """AF-style invariants — drift halts CI before any live HTTP probe."""

    def test_locked_constants_present_and_correct(self, audit_bfo) -> None:
        """All D-07 / D-08 module-top constants match locked values."""
        assert audit_bfo.YEAR_ANCHORS == (2007, 2010, 2013, 2016, 2019, 2022)
        assert audit_bfo.EVENTS_PER_YEAR == 5
        assert audit_bfo.EARLY_STOP_CONSECUTIVE_FAILURES == 5
        assert audit_bfo.TOTAL_PROBE_BUDGET == 30
        assert audit_bfo.RANDOM_STATE == 42
        assert audit_bfo.BFO_TIMEOUT_SECONDS == 5.0
        assert audit_bfo.BFO_DELAY_SECONDS == 1.2
        assert audit_bfo.BFO_MAX_RETRIES == 3
        assert audit_bfo.POST_2007_TARGET_FLOOR == 0.80
        # CSV path canonicalized to the operator-curated targets file
        assert str(audit_bfo.TARGETS_CSV) == "data/bfo_event_probe_targets.csv"


# ── Test 2: AF startup asserts fire on drift ─────────────────────────────


class TestAFStartupAsserts:
    """Drift in any locked constant must halt main() before any live HTTP."""

    def test_af_startup_asserts_fire_on_drift(
        self,
        audit_bfo,
        monkeypatch,
        capsys,
    ) -> None:
        """Mutating a locked constant at runtime triggers a fatal exit.

        Mirrors the ``scripts/spike_noise_floor.py`` AF assert pattern —
        startup checks compare each module-top constant against its
        expected value and ``return 2`` (non-zero exit) on mismatch.
        Tests guard against silent constant drift via patched module attrs.
        """
        # Patch EVENTS_PER_YEAR to a drifted value (4 instead of 5).
        monkeypatch.setattr(audit_bfo, "EVENTS_PER_YEAR", 4)
        # Call the AF-assert function directly. The audit script exposes
        # a helper named ``_assert_locked_constants`` for unit-testability
        # so this test does NOT trigger any HTTP work.
        rc = audit_bfo._assert_locked_constants()
        assert rc == 2, f"_assert_locked_constants should return non-zero on drift, got {rc}"
        captured = capsys.readouterr()
        assert "FATAL" in captured.err or "FATAL" in captured.out, (
            "AF assert failure must surface a FATAL marker on stderr/stdout"
        )


# ── Test 3: HTTP 200 + real event title → status="reachable" ─────────────


class TestProbeClassifyReachable:
    """HTTP 200 + a real event page title or fight-row content → reachable."""

    def test_classify_reachable_real_event_title(self, audit_bfo) -> None:
        """A response with an event-page title (not the generic homepage)
        and fight-row structure is classified as ``reachable``."""
        client = MagicMock()
        # Synthetic event-page HTML with a distinguishing title that is
        # NOT the BFO homepage title.
        client.get.return_value = (
            "<html><head>"
            "<title>UFC 77: Hostile Territory Odds | Best Fight Odds</title>"
            "</head><body>"
            '<div class="event-header">UFC 77</div>'
            '<table class="odds-table"><tr class="fight-row">'
            "<td>Anderson Silva</td><td>-300</td>"
            "<td>Rich Franklin</td><td>+250</td>"
            "</tr></table>"
            "</body></html>"
        )
        row = audit_bfo._probe(
            client,
            url="https://www.bestfightodds.com/events/ufc-77-rich-franklin-vs-anderson-silva-2-237",
            anchor_year=2007,
            event_label="UFC 77 Silva-Franklin II",
        )
        assert row["status"] == "reachable", (
            f"event page with real title + fight rows should classify "
            f"as reachable, got status={row['status']!r}"
        )
        assert row["captcha_hit"] is False
        assert row["anchor_year"] == 2007


# ── Test 4: HTTP 200 + generic homepage title → status="homepage_fallback" ─


class TestProbeClassifyHomepageFallback:
    """HTTP 200 + BFO homepage <title> means the URL ID is invalid.

    Task 1 finding: BFO returns HTTP 200 with the generic homepage body
    (title ``UFC & MMA Odds & Betting Lines | Best Fight Odds``) for
    invalid event slugs. Naive status-code probes mis-classify these as
    reachable. The probe MUST inspect the title to discriminate.
    """

    def test_classify_homepage_fallback_title(self, audit_bfo) -> None:
        """HTTP 200 + generic homepage title → ``status="homepage_fallback"``."""
        client = MagicMock()
        # The actual BFO homepage title observed during Task 1 curation.
        # Note the literal ``&amp;`` entity in the served HTML.
        client.get.return_value = (
            "<html><head>"
            "<title>UFC &amp; MMA Odds &amp; Betting Lines | Best Fight Odds</title>"
            "</head><body><p>Welcome to BestFightOdds</p></body></html>"
        )
        row = audit_bfo._probe(
            client,
            url="https://www.bestfightodds.com/events/fake-nonexistent-9999",
            anchor_year=2007,
            event_label="bogus",
        )
        assert row["status"] == "homepage_fallback", (
            f"generic homepage <title> must NOT count as reachable; got status={row['status']!r}"
        )
        assert row["captcha_hit"] is False


# ── Test 5: HTTP 200 + Cloudflare captcha (id="hfmr8") → "captcha" ───────


class TestProbeClassifyCaptcha:
    """Pitfall #4: Cloudflare gate returns HTTP 200 with challenge body."""

    def test_classify_captcha_hit(self, audit_bfo) -> None:
        """Mirrors ``test_bfo_scraper.py::test_parse_fighter_page_detects_captcha``.

        ``_check_captcha`` from ``bfo_scraper.py:191`` raises
        ``BFOParseError`` on the ``id="hfmr8"`` element; the probe
        wrapper catches that and records ``status="captcha"`` with
        ``captcha_hit=True``.
        """
        client = MagicMock()
        client.get.return_value = (
            "<html><body>"
            '<div id="hfmr8">Verify you are human by completing the action below.</div>'
            "</body></html>"
        )
        row = audit_bfo._probe(
            client,
            url="https://www.bestfightodds.com/events/ufc-77-rich-franklin-vs-anderson-silva-2-237",
            anchor_year=2007,
            event_label="UFC 77 Silva-Franklin II",
        )
        assert row["status"] == "captcha"
        assert row["captcha_hit"] is True


# ── Test 6: CSV EVENT_NOT_FOUND sentinel → no HTTP request issued ────────


class TestEventNotFoundSentinel:
    """Operator-marked unfindable rows skip the HTTP layer entirely.

    Task 1 produced 4 ``EVENT_NOT_FOUND`` rows: UFC 67/69/71 (pre-BFO-archive)
    and UFC 272 (BFO search disambiguation failure). These are recorded as
    ``status="event_not_found"`` and count toward per-year totals but never
    issue an HTTP request.
    """

    def test_classify_event_not_found_no_http(self, audit_bfo) -> None:
        """``event_url == "EVENT_NOT_FOUND"`` → no HTTP call, status sentinel set."""
        client = MagicMock()
        row = audit_bfo._probe(
            client,
            url="EVENT_NOT_FOUND",
            anchor_year=2007,
            event_label="UFC 67: All or Nothing (2007-02-03)",
        )
        assert row["status"] == "event_not_found"
        assert row["captcha_hit"] is False
        # Critical invariant: no HTTP request is issued for sentinel rows.
        client.get.assert_not_called()


# ── Test 7: early-stop after 5 consecutive non-reachable results ─────────


class TestEarlyStop:
    """D-07: stop probing a year-anchor after 5 consecutive non-reachable rows.

    Combines all non-reachable statuses (event_not_found, homepage_fallback,
    not_found, captcha, error) — any 5 in a row triggers the break.
    """

    def test_early_stop_after_5_consecutive_failures(self, audit_bfo) -> None:
        """``_probe_year`` halts after 5 consecutive non-reachable rows."""
        client = MagicMock()
        # All probes return the generic homepage → all classify as
        # homepage_fallback. With 10 candidate URLs and an early-stop
        # threshold of 5, the inner loop should record exactly 5 rows.
        client.get.return_value = (
            "<html><head>"
            "<title>UFC &amp; MMA Odds &amp; Betting Lines | Best Fight Odds</title>"
            "</head><body></body></html>"
        )
        candidates = [
            (
                f"https://www.bestfightodds.com/events/fake-{i}",
                f"Fake Event {i}",
            )
            for i in range(10)
        ]
        rows = audit_bfo._probe_year(client, year=2007, candidates=candidates)
        # Exactly 5 probes (the early-stop budget) — the remaining 5
        # candidates are skipped without issuing HTTP.
        assert len(rows) == 5, (
            f"early-stop on 5 consecutive non-reachable rows: expected len=5, got {len(rows)}"
        )
        # client.get was called at most 5 times (sentinel rows don't
        # call HTTP; all 10 candidates are real URLs here).
        assert client.get.call_count == 5


# ── Test 8: post-2007 target derivation with 80% floor ───────────────────


class TestPost2007TargetDerivation:
    """The pre-registered post-2007 coverage target floors at 80%.

    Phase 21 BFO backfill consumes ``target_rate`` from this audit's MD
    output. The target must be at least 0.80 (operator-pre-registered)
    — if measured reachability exceeds 80%, the target follows measured;
    otherwise the floor governs.
    """

    def test_post_2007_target_derivation_floor_80pct(self, audit_bfo) -> None:
        """Synthetic per-year counts → ``target_rate >= 0.80``."""
        # Construct a per-year rollup where post-2007 reachability is
        # high (e.g., 22 reachable / 25 probed across 2010-2022) AND a
        # case where post-2007 reachability is low (e.g., 10 / 25) —
        # both must yield ``target_rate >= 0.80``.

        # Case A: HIGH reachability — 22/25 = 0.88 → target = 0.88.
        per_year_high: dict[int, list[dict]] = {
            2007: [{"status": "reachable"}] * 2 + [{"status": "event_not_found"}] * 3,
            2010: [{"status": "reachable"}] * 5,
            2013: [{"status": "reachable"}] * 5,
            2016: [{"status": "reachable"}] * 4 + [{"status": "homepage_fallback"}],
            2019: [{"status": "reachable"}] * 5,
            2022: [
                {"status": "reachable"},
                {"status": "reachable"},
                {"status": "reachable"},
                {"status": "event_not_found"},
                {"status": "reachable"},
            ],
        }
        target_high, _ = audit_bfo._derive_post2007_target(per_year_high)
        assert target_high >= 0.80, (
            f"22/25 = 0.88 measured post-2007 → target should be ≥ 0.80, got {target_high}"
        )

        # Case B: LOW reachability — 10/25 = 0.40 → target FLOORS at 0.80.
        per_year_low: dict[int, list[dict]] = {
            2007: [{"status": "event_not_found"}] * 5,
            2010: [{"status": "reachable"}] * 2 + [{"status": "homepage_fallback"}] * 3,
            2013: [{"status": "reachable"}] * 2 + [{"status": "homepage_fallback"}] * 3,
            2016: [{"status": "reachable"}] * 2 + [{"status": "homepage_fallback"}] * 3,
            2019: [{"status": "reachable"}] * 2 + [{"status": "homepage_fallback"}] * 3,
            2022: [{"status": "reachable"}] * 2 + [{"status": "homepage_fallback"}] * 3,
        }
        target_low, _ = audit_bfo._derive_post2007_target(per_year_low)
        assert target_low >= 0.80, (
            f"10/25 = 0.40 measured post-2007 → target must FLOOR at 0.80, got {target_low}"
        )
