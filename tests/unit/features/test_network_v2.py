"""Phase 18 NET-V2-02 Wave-0 RED — time-decayed PageRank graph-build correctness.

Per D-02(P18): edge weight = mov_multiplier * (DECAY_BASE ** days_since_event)
where DECAY_BASE = 0.98 (Lazova & Basnarkov 2015; arXiv:1503.01331).
Reference half-life ≈ 34.3 days.

Pitfall A (RESEARCH.md): `_decay_subgraph` MUST construct a NEW `nx.DiGraph()`
per call — NOT use `nx.edge_subgraph` (which returns a view). View mutation
leaks decayed weights back into the parent graph and corrupts subsequent
queries. `TestNoParentMutation` enforces this structurally.

These tests RED on import (Wave 0) — `ufc_prediction.features.network_v2` does
not exist yet. Goes GREEN at Wave 1 Task 8.
"""
from __future__ import annotations

from datetime import date

import networkx as nx
import pytest

from ufc_prediction.features.network_v2 import (
    DECAY_BASE,
    _decay_subgraph,
    build_fight_graph_v2,
    compute_pagerank_at_v2,
    compute_2hop_sos_at_v2,
)


def _fights() -> list[dict]:
    """Two fights, 100 days apart."""
    return [
        {"fight_id": 1, "event_date": date(2020, 1, 1),
         "fighter_a_id": 1, "fighter_b_id": 2,
         "winner_id": 1, "loser_id": 2, "method": "U-DEC",
         "weight_class": "Lightweight"},
        {"fight_id": 2, "event_date": date(2020, 4, 10),  # 100 days later
         "fighter_a_id": 1, "fighter_b_id": 3,
         "winner_id": 1, "loser_id": 3, "method": "KO/TKO",
         "weight_class": "Lightweight"},
    ]


class TestEdgeWeightDecay:
    """NET-V2-02 edge-weight decay math (5 tests)."""

    def test_decay_zero_days_returns_full_mov(self):
        """as_of_date == event_date + 1 → days=1; weight = mov * 0.98^1."""
        g = build_fight_graph_v2(_fights())
        sub = _decay_subgraph(g, date(2020, 1, 2))  # 1 day post fight1
        assert (2, 1) in sub.edges
        # MOV for U-DEC = 1.0; 0.98^1 = 0.98
        assert pytest.approx(sub.edges[2, 1]["weight"], abs=1e-9) == 0.98

    def test_decay_100_days(self):
        """Edge from fight1 (100 days before as_of_date) decays by 0.98^100."""
        g = build_fight_graph_v2(_fights())
        sub = _decay_subgraph(g, date(2020, 4, 10))
        # 100 days, U-DEC mov=1.0
        expected_w = 1.0 * (DECAY_BASE ** 100)  # ~0.1326
        assert pytest.approx(sub.edges[2, 1]["weight"], abs=1e-9) == expected_w

    def test_decay_filters_future_edges(self):
        """as_of_date == event_date1 — fight1 NOT included (strict <)."""
        g = build_fight_graph_v2(_fights())
        sub = _decay_subgraph(g, date(2020, 1, 1))
        assert (2, 1) not in sub.edges

    def test_mov_multiplier_composes_with_decay(self):
        """KO/TKO mov=1.2; weight at days=1 = 1.2 * 0.98."""
        g = build_fight_graph_v2(_fights())
        sub = _decay_subgraph(g, date(2020, 4, 11))
        # 1 day post fight2 (KO/TKO, mov=1.2)
        assert pytest.approx(sub.edges[3, 1]["weight"], abs=1e-9) == 1.2 * 0.98

    def test_decay_does_not_mutate_parent(self):
        """Pitfall A: parent graph weight is left as the original mov_weight."""
        g = build_fight_graph_v2(_fights())
        original_mov_2_to_1 = g.edges[2, 1]["mov_weight"]
        original_w_2_to_1 = g.edges[2, 1]["weight"]
        _ = _decay_subgraph(g, date(2020, 6, 1))
        assert g.edges[2, 1]["mov_weight"] == original_mov_2_to_1
        assert g.edges[2, 1]["weight"] == original_w_2_to_1


class TestPageRankAtV2:
    """NET-V2-02 PageRank-at + 2-hop SoS edge cases (3 tests)."""

    def test_debutant_returns_none(self):
        g = build_fight_graph_v2(_fights())
        # Fighter 99 doesn't appear in any fight → not in subgraph → None
        assert compute_pagerank_at_v2(g, 99, date(2020, 6, 1)) is None

    def test_two_hop_sos_for_debutant_returns_none(self):
        g = build_fight_graph_v2(_fights())
        assert compute_2hop_sos_at_v2(g, 99, date(2020, 6, 1)) is None

    def test_pagerank_value_in_zero_one_range(self):
        """Real fighter present in subgraph → PageRank float in (0, 1)."""
        g = build_fight_graph_v2(_fights())
        pr = compute_pagerank_at_v2(g, 1, date(2020, 6, 1))
        assert pr is not None
        assert isinstance(pr, float)
        assert 0.0 < pr < 1.0


class TestNoParentMutation:
    """Pitfall A: explicit class for VALIDATION.md mapping (2 tests).

    `_decay_subgraph` MUST construct a new nx.DiGraph per call, NEVER use
    nx.edge_subgraph (which returns a VIEW). View mutation leaks decayed
    weights back into the parent graph and corrupts subsequent queries.
    """

    def test_decay_does_not_mutate_parent(self):
        """Same assertion as TestEdgeWeightDecay; explicit duplicate for
        VALIDATION.md mapping (T3 maps to TestNoParentMutation by name).
        """
        g = build_fight_graph_v2(_fights())
        original_mov = g.edges[2, 1]["mov_weight"]
        original_w = g.edges[2, 1]["weight"]
        _ = _decay_subgraph(g, date(2020, 6, 1))
        assert g.edges[2, 1]["mov_weight"] == original_mov
        assert g.edges[2, 1]["weight"] == original_w

    def test_two_consecutive_calls_with_different_dates_isolated(self):
        """Calling PageRank at date_A, then date_B (different — would mutate
        weights via view), then date_A again must return identical results.

        If `_decay_subgraph` returned a VIEW, the second call would mutate
        the parent's `weight` attribute and the third call would compute
        against corrupted weights. New-DiGraph-per-call structurally rules
        this out.
        """
        g = build_fight_graph_v2(_fights())
        date_a = date(2020, 5, 1)
        date_b = date(2020, 4, 15)

        pr_first = compute_pagerank_at_v2(g, 1, date_a)
        # This call would corrupt parent weights if _decay_subgraph used a view
        _ = compute_pagerank_at_v2(g, 1, date_b)
        pr_third = compute_pagerank_at_v2(g, 1, date_a)

        assert pr_first is not None
        assert pr_third is not None
        assert pr_first == pytest.approx(pr_third, abs=1e-12)
