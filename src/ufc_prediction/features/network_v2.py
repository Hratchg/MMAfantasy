"""Time-decayed opponent-network features (NET-V2 variant per D-02(P18)).

Mirrors network.py's API surface but multiplies surviving edge weights by
``0.98 ** days_since_event_at_as_of_date`` BEFORE nx.pagerank runs.

Reference: Lazova & Basnarkov 2015 (arXiv:1503.01331),
"PageRank Approach to Ranking National Football Teams" — ``0.98^days``
half-life ~34.3 days.

Pitfall #9 mitigation (Phase 18 carry-forward): days are computed against
the per-snapshot ``as_of_date`` (the prediction-time date), NEVER against
``today()`` or ``now()``. The ``as_of_date`` parameter is required at every
function entry; calls without it raise TypeError. NET-03 temporal-leakage
regression test (LEAKAGE_BRIER_THRESHOLD = 0.01) MUST pass on network_v2.py.

Pitfall A (RESEARCH.md): ``_decay_subgraph`` constructs a NEW
``nx.DiGraph()`` per call — NEVER the networkx view-returning
subgraph helper. View mutation would leak decayed weights back into
the parent graph and corrupt subsequent queries. Enforced structurally
(no view-helper call in this module) and via
``tests/unit/features/test_network_v2.py::TestNoParentMutation``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import networkx as nx

# DRY: import MOV multipliers + helpers from network.py (D-09(P15) discipline:
# the MOV constants are operator-approved at NET-00 spike; no need to fork).
# Open Q1 RESOLVED — Plan 18-01: import constants/helpers from network.py and
# duplicate ONLY graph-build + decay-subgraph + PageRank + 2-hop-SoS logic.
from ufc_prediction.features.network import (
    APPROVED_SCOPE,
    APPROVED_WEIGHT,
    PAGERANK_ALPHA,
    _compute_edge_weight,
)

# Locked per D-02(P18) — Lazova & Basnarkov 2015 ~34.3-day half-life.
DECAY_BASE: float = 0.98


# ── Graph construction (Phase 18 — edge attribute split) ──────────────────


def build_fight_graph_v2(
    fights: list[dict[str, Any]],
    *,
    scope: str = APPROVED_SCOPE,
    weight_mode: str = APPROVED_WEIGHT,
) -> nx.DiGraph:
    """Same as network.build_fight_graph — base weight stored as ``mov_weight``.

    Edge attribute split (Phase 18 ONLY divergence from network.py):
      * ``mov_weight``      — base MOV multiplier (constant across queries)
      * ``earliest_date``   — earliest occurrence date (rematch-aware)
      * ``weight``          — placeholder; OVERWRITTEN by
                              ``_decay_subgraph(graph, as_of_date)`` per query

    Why: PageRank reads the ``weight`` attribute. We can't pre-decay at
    graph-build time because the decay is per-query (per as_of_date).
    """
    G: nx.DiGraph = nx.DiGraph()
    for f in fights:
        if scope == "pan-mma":
            u: Any = f["loser_id"]
            v: Any = f["winner_id"]
        elif scope == "per-division":
            wc = f.get("weight_class") or "_unknown"
            u = (wc, f["loser_id"])
            v = (wc, f["winner_id"])
        else:
            raise ValueError(f"unknown scope: {scope!r}")
        mov = _compute_edge_weight(f.get("method"), weight_mode)
        d = f["event_date"]
        if G.has_edge(u, v):
            G[u][v]["mov_weight"] += mov
            G[u][v]["earliest_date"] = min(G[u][v]["earliest_date"], d)
        else:
            G.add_edge(u, v, mov_weight=mov, earliest_date=d, weight=mov)
    return G


# ── Time-decayed temporal subgraph (Pitfall A view-mutation guard) ────────


def _decay_subgraph(graph: nx.DiGraph, as_of_date: date) -> nx.DiGraph:
    """Filter to edges with earliest_date < as_of_date AND apply 0.98^days decay.

    Returns a NEW DiGraph (NOT a view) because we mutate the 'weight'
    attribute per-query — view-mutation would leak into the parent
    (Pitfall A; see TestNoParentMutation).
    """
    sub: nx.DiGraph = nx.DiGraph()
    for u, v, attrs in graph.edges(data=True):
        if attrs["earliest_date"] >= as_of_date:
            continue
        days = (as_of_date - attrs["earliest_date"]).days
        # Pitfall #9 mitigation: as_of_date is the prediction-time date; days
        # is therefore non-negative (we filtered earliest_date < as_of_date).
        decayed = attrs["mov_weight"] * (DECAY_BASE**days)
        sub.add_edge(u, v, weight=decayed)
    return sub


# ── Pure-function feature computers ────────────────────────────────────────


def compute_pagerank_at_v2(
    graph: nx.DiGraph,
    fighter_id: Any,
    as_of_date: date,
    *,
    alpha: float = PAGERANK_ALPHA,
) -> float | None:
    """Time-decayed PageRank for one fighter as-of as_of_date.

    Returns None for debutants (caller maps to NaN + is_debutant_in_graph=1.0
    per D-06(P16) carry-forward).
    """
    sub = _decay_subgraph(graph, as_of_date)
    if fighter_id not in sub.nodes:
        return None
    pr = nx.pagerank(sub, alpha=alpha, weight="weight")
    value = pr.get(fighter_id)
    return None if value is None else float(value)


def compute_2hop_sos_at_v2(
    graph: nx.DiGraph,
    fighter_id: Any,
    as_of_date: date,
    *,
    alpha: float = PAGERANK_ALPHA,
) -> float | None:
    """2-hop SoS via time-decayed PageRank — mean of in-neighbors' PageRanks.

    Same edge cases as network.compute_2hop_sos_at:
      * Debutant (fighter not in subgraph) -> None
      * Fighter in subgraph but with zero in-neighbors before ``as_of_date``
        (only appears as a loser, no wins yet) -> None
      * Fighter with >=1 in-neighbor -> arithmetic mean of those PageRanks
    """
    sub = _decay_subgraph(graph, as_of_date)
    if fighter_id not in sub.nodes:
        return None
    in_neighbors = list(sub.predecessors(fighter_id))
    if not in_neighbors:
        return None
    pr = nx.pagerank(sub, alpha=alpha, weight="weight")
    return float(sum(pr.get(n, 0.0) for n in in_neighbors) / len(in_neighbors))


# ── 4th-pass integration helper (train/predict parity per Pitfall #12) ────


def apply_network_features_v2(
    raw_results: list[dict[str, Any]],
    fights: list[dict[str, Any]],
    *,
    scope: str = APPROVED_SCOPE,
    weight_mode: str = APPROVED_WEIGHT,
) -> None:
    """4th-pass mutator — same shape as network.apply_network_features.

    Adds ``pagerank``, ``sos_2hop``, ``is_debutant_in_graph`` keys to each
    raw_results[i]['features']. Train/predict parity is preserved by sharing
    these key names with network.py — the diff helper
    ``compute_network_diff_features`` works for both (Open Q2 RESOLVED:
    no ``_v2`` column suffix; variant identity tracked in meta JSON
    ``base_model_kind`` field).
    """
    if not raw_results:
        return
    decisive = [
        f for f in fights if f.get("winner_id") is not None and f.get("loser_id") is not None
    ]
    graph = build_fight_graph_v2(decisive, scope=scope, weight_mode=weight_mode)
    nan = float("nan")
    for row in raw_results:
        fid = row["fighter_id"]
        # ``compute.py`` uses ``as_of_date`` (line 137); some test/spike code
        # uses ``fight_date``. Accept either.
        as_of = row.get("as_of_date") or row.get("fight_date")
        if as_of is None:
            row["features"]["pagerank"] = nan
            row["features"]["sos_2hop"] = nan
            row["features"]["is_debutant_in_graph"] = 1.0
            continue
        if scope == "per-division":
            wc = row.get("weight_class")
            if wc is None:
                row["features"]["pagerank"] = nan
                row["features"]["sos_2hop"] = nan
                row["features"]["is_debutant_in_graph"] = 1.0
                continue
            node: Any = (wc, fid)
        else:
            node = fid

        pr = compute_pagerank_at_v2(graph, node, as_of)
        sos = compute_2hop_sos_at_v2(graph, node, as_of)

        # Per D-06(P16) / Pattern D: debutant => NaN PageRank + flag=1.0
        if pr is None:
            row["features"]["pagerank"] = nan
            row["features"]["is_debutant_in_graph"] = 1.0
        else:
            row["features"]["pagerank"] = float(pr)
            row["features"]["is_debutant_in_graph"] = 0.0
        row["features"]["sos_2hop"] = float(sos) if sos is not None else nan
