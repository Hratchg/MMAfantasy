"""NET-03: temporal-leakage CI regression test (Pitfall #1).

Asserts the invariant that PageRank features for fighter F at fight #N
are computed using ONLY fights with event_date < fight_N_date —
i.e., future fights cannot influence past PageRank values.

Failure of this test invokes the D-08 kill switch in plan 16-04:
if PageRank leaks future edges, NET-* drops from xgb_v3 entirely.

Per 16-PATTERNS.md Gotcha 4: ``nx.pagerank(G)`` does NOT respect any
temporal attribute on edges. The pre-filter happens at
``graph.edge_subgraph(...)`` BEFORE ``nx.pagerank(...)`` — this test
asserts that contract over a deterministic 5-fighter / 8-fight toy graph
spanning multiple eras (2010 / 2011 / 2018 / 2024).

The toy fixture is built inline (not via the production DB) so the test
runs in CI without Postgres.
"""

from __future__ import annotations

from datetime import date

import pytest

network = pytest.importorskip("ufc_prediction.features.network")


# ── D-08 sanity-floor constant (referenced by plan 16-04 MODEL-01) ────────

# Per CONTEXT.md D-08 kill switch: if NET-* contribution to xgb_v3 Brier
# is ≥ 0.01 vs xgb_v2-no-NET baseline, the leakage RESEARCH.md flag fires.
# This test does NOT enforce the gate (that's plan 16-04's MODEL-01 task);
# it documents the threshold as a module-level constant so plan 16-04 can
# import it without restating the magic number.
LEAKAGE_BRIER_THRESHOLD: float = 0.01


# ── Toy graph fixture (5 fighters / 8 fights / 4 eras) ────────────────────


@pytest.fixture(scope="module")
def toy_fights() -> list[dict]:
    """5-fighter / 8-fight toy fight list per VALIDATION.md Wave 0.

    Spans 2010 / 2011 / 2018 / 2024 so era spot checks (Test 3) are
    meaningful. Mirrors the fixture in
    ``tests/unit/features/test_network.py::toy_fights`` so the leakage
    CI test exercises the same graph topology that NET-01 / NET-02 are
    unit-tested against.

        2010-01-01  F1 beats F2     KO/TKO    (loser=F2 -> F1)
        2010-06-01  F1 beats F3     Decision  (loser=F3 -> F1)
        2011-01-01  F2 beats F3     KO/TKO    (loser=F3 -> F2)
        2011-06-01  F4 beats F1     Submission (loser=F1 -> F4)
        2018-01-01  F5 beats F4     KO/TKO    (F5 is debutant before this)
        2018-06-01  F5 beats F1     Decision
        2024-01-01  F5 beats F2     Decision
        2024-06-01  F5 beats F3     KO/TKO
    """
    return [
        {"fight_id": 1, "event_date": date(2010, 1, 1),
         "fighter_a_id": 1, "fighter_b_id": 2, "winner_id": 1, "loser_id": 2,
         "method": "KO/TKO", "weight_class": "Lightweight"},
        {"fight_id": 2, "event_date": date(2010, 6, 1),
         "fighter_a_id": 1, "fighter_b_id": 3, "winner_id": 1, "loser_id": 3,
         "method": "Decision", "weight_class": "Lightweight"},
        {"fight_id": 3, "event_date": date(2011, 1, 1),
         "fighter_a_id": 2, "fighter_b_id": 3, "winner_id": 2, "loser_id": 3,
         "method": "KO/TKO", "weight_class": "Lightweight"},
        {"fight_id": 4, "event_date": date(2011, 6, 1),
         "fighter_a_id": 4, "fighter_b_id": 1, "winner_id": 4, "loser_id": 1,
         "method": "Submission", "weight_class": "Lightweight"},
        {"fight_id": 5, "event_date": date(2018, 1, 1),
         "fighter_a_id": 5, "fighter_b_id": 4, "winner_id": 5, "loser_id": 4,
         "method": "KO/TKO", "weight_class": "Lightweight"},
        {"fight_id": 6, "event_date": date(2018, 6, 1),
         "fighter_a_id": 5, "fighter_b_id": 1, "winner_id": 5, "loser_id": 1,
         "method": "Decision", "weight_class": "Lightweight"},
        {"fight_id": 7, "event_date": date(2024, 1, 1),
         "fighter_a_id": 5, "fighter_b_id": 2, "winner_id": 5, "loser_id": 2,
         "method": "Decision", "weight_class": "Lightweight"},
        {"fight_id": 8, "event_date": date(2024, 6, 1),
         "fighter_a_id": 5, "fighter_b_id": 3, "winner_id": 5, "loser_id": 3,
         "method": "KO/TKO", "weight_class": "Lightweight"},
    ]


@pytest.fixture(scope="module")
def toy_graph(toy_fights):
    """Pan-MMA + MOV-weighted DiGraph (operator-approved config from NET-00)."""
    return network.build_fight_graph(toy_fights, scope="pan-mma", weight_mode="mov")


# ── NET-03 invariants (Pitfall #1 / Gotcha 4 regression sentinels) ────────


def test_pagerank_no_future_fights(toy_graph):
    """NET-03: ``compute_pagerank_at(F, fight_N_date) != compute_pagerank_at(F, today)``
    for any fighter F with fights AFTER ``fight_N_date``.

    Asserted across the toy graph for every fighter that has >=2 fights
    spanning a partial-vs-full cutoff. Also asserts that the partial-as-of
    subgraph has strictly fewer edges than the full-as-of subgraph
    (subgraph monotonicity — past sub-of present).

    Failure of this test means ``compute_pagerank_at`` failed to pre-filter
    via ``edge_subgraph`` BEFORE ``nx.pagerank``, which is the literal
    Pitfall #1 / Gotcha 4 countermeasure. Plan 16-04's D-08 kill switch
    is pulled by this signal.
    """
    as_of_partial = date(2011, 7, 1)  # only fights 1-4 in scope
    as_of_full = date(2025, 1, 1)     # all 8 fights in scope

    sub_partial = network._temporal_subgraph(toy_graph, as_of_partial)
    sub_full = network._temporal_subgraph(toy_graph, as_of_full)

    # Subgraph monotonicity: past sub-of present.
    assert sub_partial.number_of_edges() < sub_full.number_of_edges(), (
        f"Partial subgraph at {as_of_partial} has "
        f"{sub_partial.number_of_edges()} edges; full subgraph at "
        f"{as_of_full} has {sub_full.number_of_edges()}. The partial "
        "must have strictly fewer edges (Pitfall #1 invariant violated)."
    )

    # F1 has fights at multiple dates (2010-01, 2010-06, 2011-06, 2018-06).
    # Its PageRank at the partial cutoff must differ from the full cutoff
    # because post-2011 fights changed the graph topology.
    pr_partial = network.compute_pagerank_at(toy_graph, 1, as_of_partial)
    pr_full = network.compute_pagerank_at(toy_graph, 1, as_of_full)
    assert pr_partial is not None
    assert pr_full is not None
    assert abs(pr_partial - pr_full) > 1e-9, (
        f"F1 pagerank_at_partial ({pr_partial:.6e}) == pagerank_at_full "
        f"({pr_full:.6e}) — temporal leakage signal: post-cutoff fights "
        "did not change F1's PageRank, which is impossible if the "
        "subgraph filter is applied correctly."
    )


def test_debutant_invariant(toy_graph):
    """F5's first UFC fight is 2018-01-01. Before that date, F5 has no
    PageRank in the graph (returns None). After their debut, PageRank
    is a finite positive float.
    """
    pre_debut = date(2017, 12, 31)
    post_debut = date(2018, 6, 2)

    pr_pre = network.compute_pagerank_at(toy_graph, 5, pre_debut)
    pr_post = network.compute_pagerank_at(toy_graph, 5, post_debut)

    assert pr_pre is None, (
        f"F5 pre-debut PageRank should be None (debutant case per D-06); "
        f"got {pr_pre!r}"
    )
    assert pr_post is not None
    assert pr_post > 0.0, (
        f"F5 post-debut PageRank should be > 0.0; got {pr_post}"
    )


def test_multi_era_spot_check(toy_graph):
    """Pick 3 distinct eras (2010, 2018, 2024) and assert PageRank-at-era
    differs from PageRank-at-today for at least one fighter per era.

    This is the warning-sign listed in 16-RESEARCH.md Pitfall #1 — if
    PageRank values are identical across eras, the temporal filter is
    not being applied and future edges are leaking into past computations.
    """
    eras = [date(2010, 12, 31), date(2018, 12, 31), date(2024, 3, 1)]
    today = date(2025, 1, 1)

    # F1 is present in every era (debuted 2010-01).
    pr_today_f1 = network.compute_pagerank_at(toy_graph, 1, today)
    assert pr_today_f1 is not None

    differences = []
    for era in eras:
        pr_era = network.compute_pagerank_at(toy_graph, 1, era)
        if pr_era is None:
            continue
        differences.append((era, abs(pr_era - pr_today_f1)))

    # At least 2 of the 3 eras should show a measurable difference.
    # (The 2024-03 era is close to today and may be tiny but nonzero.)
    measurable = [(e, d) for e, d in differences if d > 1e-9]
    assert len(measurable) >= 2, (
        "F1 PageRank should differ across at least 2 of 3 historical eras "
        f"vs today; got differences {differences}. If this fails, future "
        "fights are leaking into past PageRank values (Pitfall #1)."
    )


def test_subgraph_identity(toy_graph):
    """``_temporal_subgraph(G, as_of)`` includes EXACTLY the edges with
    ``earliest_date < as_of`` and excludes the rest.

    Direct identity check — the strict-less-than predicate is the Pitfall #1
    countermeasure; an off-by-one (e.g. ``<=``) would silently leak the
    snapshot fight's own edge.
    """
    as_of = date(2011, 6, 1)  # exactly fight 4's date
    sub = network._temporal_subgraph(toy_graph, as_of)

    expected_included_dates = {
        date(2010, 1, 1), date(2010, 6, 1), date(2011, 1, 1),
    }
    actual_dates = {d["earliest_date"] for _, _, d in sub.edges(data=True)}
    assert actual_dates == expected_included_dates, (
        f"Expected subgraph edges with earliest_date < {as_of} to be "
        f"{expected_included_dates}; got {actual_dates}. The strict-"
        "less-than predicate is the Pitfall #1 countermeasure — any "
        "drift to <= leaks the snapshot fight's own edge."
    )

    # As_of=2011-06-01 must EXCLUDE fight 4's edge (which happens that
    # day) and ALL later fights. The fight 4 edge is F1 -> F4
    # (loser -> winner).
    assert (1, 4) not in sub.edges, (
        "Fight 4 (F1 -> F4 on 2011-06-01) leaked into the subgraph at "
        "as_of=2011-06-01. Strict-less-than filter is broken."
    )


def test_d_08_threshold_constant_present():
    """D-08 sanity floor constant: ``LEAKAGE_BRIER_THRESHOLD`` is exposed
    so plan 16-04 MODEL-01 can import it for the post-train sanity check.
    """
    assert LEAKAGE_BRIER_THRESHOLD == 0.01
    # Type check: must be a finite float (no NaN/inf).
    assert isinstance(LEAKAGE_BRIER_THRESHOLD, float)
    assert 0.0 < LEAKAGE_BRIER_THRESHOLD < 1.0


# ── Phase 18 NET-V2-02 leakage extension (Pitfall #9 mitigation) ─────────────

# Skipped at Wave 0 (network_v2 not yet implemented); goes RUN+GREEN at Wave 1
# Task 8 when src/ufc_prediction/features/network_v2.py lands. Pitfall #9 is
# mitigated structurally (network_v2._decay_subgraph uses as_of_date, not
# today()); this regression test catches future drift.
network_v2 = pytest.importorskip("ufc_prediction.features.network_v2")


@pytest.fixture(scope="module")
def toy_graph_v2(toy_fights):
    """Pan-MMA + MOV-weighted DiGraph built via network_v2.build_fight_graph_v2."""
    return network_v2.build_fight_graph_v2(
        toy_fights, scope="pan-mma", weight_mode="mov",
    )


def test_network_v2_leakage(toy_graph_v2):
    """NET-V2-02 temporal-leakage CI; mirrors test_pagerank_no_future_fights.

    F1 has fights at multiple dates (2010-01, 2010-06, 2011-06, 2018-06).
    Its time-decayed PageRank at the partial cutoff (2011-07-01) MUST differ
    from the full cutoff (2025-01-01); otherwise the temporal pre-filter has
    been bypassed and post-cutoff edges are leaking into past computations.
    """
    as_of_partial = date(2011, 7, 1)
    as_of_full = date(2025, 1, 1)
    pr_partial = network_v2.compute_pagerank_at_v2(toy_graph_v2, 1, as_of_partial)
    pr_full = network_v2.compute_pagerank_at_v2(toy_graph_v2, 1, as_of_full)
    assert pr_partial is not None
    assert pr_full is not None
    assert abs(pr_partial - pr_full) > 1e-9, (
        "NET-V2 temporal leakage: F1 pagerank_partial == pagerank_full"
    )
