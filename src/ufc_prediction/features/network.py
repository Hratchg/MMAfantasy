"""Opponent-network features: PageRank + 2-hop strength of schedule.

Operator-approved configuration (per NET-00 spike, gsd-checkpoint resolved
2026-05-02 with `approve pan-mma-mov`):

  - **Graph scope:** ``pan-mma`` (D-05) — single directed graph spanning all
    UFC fights regardless of weight class. Captures cross-division skill
    transfer (Cormier 205→HW; Costa 185→LHW; McGregor 145↔155).
  - **Edge weighting:** ``mov`` (D-07) — edge weights derived from
    ``EloConfig`` MOV multipliers (KO/TKO 1.2; Submission 1.1; Decision 1.0;
    Split 0.5; DQ 0.8). Encodes finish quality.
  - **Debutant handling:** (D-06) fighters with no incoming edges before
    ``as_of_date`` get ``pagerank = NaN`` and ``is_debutant_in_graph = 1.0``.
    XGBoost handles NaN natively (D-04(P14)); the model learns the
    debutant-specific split rather than treating debutants as pickem.

Per Pitfall #1 / Gotcha 4 countermeasure: pre-filter via
``graph.edge_subgraph(...)`` BEFORE ``nx.pagerank(...)``. ``nx.pagerank``
does NOT respect any temporal attribute on edges; the temporal subgraph
must be constructed first. NET-03 makes this a permanent CI regression
test (``tests/regression/test_temporal_leakage.py``).

Per CONTEXT.md ``<deferred>``: 2-hop SoS aggregation = mean of in-neighbors'
PageRank values. Sum and weighted-by-recency variants are v2.1+ scope.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import networkx as nx

from ufc_prediction.elo.config import EloConfig

logger = logging.getLogger(__name__)


# ── Module-level constants (operator-approved per NET-00 gsd-checkpoint) ──

# Operator decision: `approve pan-mma-mov` — per
# .planning/phases/16-foundation-opponent-network-live-odds/16-03-NET-00-SPIKE-RESULTS.md
APPROVED_SCOPE: str = "pan-mma"
APPROVED_WEIGHT: str = "mov"

# NetworkX default + standard sports-PageRank value (matches spike).
PAGERANK_ALPHA: float = 0.85


# ── Edge-weight helper ─────────────────────────────────────────────────────


def _compute_edge_weight(method: str | None, weight_mode: str) -> float:
    """Map a fight method string to its edge weight per ``weight_mode``.

    Mirrors ``scripts/spike_pagerank_design.py::method_to_mov_weight`` so the
    spike's empirical conclusions transfer directly to NET-01.

    For ``weight_mode = "binary"``, every edge has weight 1.0.
    For ``weight_mode = "mov"``, weights come from the live ``EloConfig``
    MOV multipliers (verified at spike time per Logged Assumption A4):

        KO/TKO       -> mov_ko_tko       (1.2)
        Submission   -> mov_submission   (1.1)
        U-DEC        -> mov_unanimous    (1.0)
        Decision     -> mov_unanimous    (1.0; generic decision = unanimous)
        S-DEC, M-DEC -> mov_split        (0.5)
        DQ           -> mov_dq           (0.8)
        Other / CNC  -> 1.0              (neutral)
    """
    if weight_mode == "binary":
        return 1.0
    if weight_mode != "mov":
        raise ValueError(f"unknown weight_mode: {weight_mode!r}")

    cfg = EloConfig()
    if method is None:
        return cfg.mov_unanimous
    m = method.upper()
    if m in ("KO/TKO", "KO", "TKO"):
        return cfg.mov_ko_tko
    if m in ("SUB", "SUBMISSION"):
        return cfg.mov_submission
    if m == "U-DEC":
        return cfg.mov_unanimous
    if m in ("S-DEC", "M-DEC"):
        return cfg.mov_split
    if m == "DECISION":
        return cfg.mov_unanimous
    if m == "DQ":
        return cfg.mov_dq
    return 1.0  # Other / CNC / Overturned — neutral


# ── Graph construction ─────────────────────────────────────────────────────


def build_fight_graph(
    fights: list[dict[str, Any]],
    *,
    scope: str = APPROVED_SCOPE,
    weight_mode: str = APPROVED_WEIGHT,
) -> nx.DiGraph:
    """Build a directed fight graph (loser -> winner; PageRank "votes for who beat me").

    Args:
        fights: List of fight dicts. Each must contain at minimum:
            ``fight_id``, ``event_date`` (a ``datetime.date``),
            ``winner_id``, ``loser_id``, ``method``, and (when
            ``scope == "per-division"``) ``weight_class``.
        scope: ``"pan-mma"`` (operator default) or ``"per-division"``.
        weight_mode: ``"mov"`` (operator default) or ``"binary"``.

    Returns:
        ``nx.DiGraph`` with edges loser -> winner. Each edge carries
        ``weight`` and ``earliest_date`` (a ``datetime.date``).

    Rematches accumulate weight on the same edge (matches the spike's
    "evidence aggregates" interpretation); ``earliest_date`` keeps the
    earliest occurrence so the temporal subgraph filter admits the edge
    as soon as either rematch is in scope.
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

        w = _compute_edge_weight(f.get("method"), weight_mode)
        evt_date = f["event_date"]

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
            G[u][v]["earliest_date"] = min(G[u][v]["earliest_date"], evt_date)
        else:
            G.add_edge(u, v, weight=w, earliest_date=evt_date)

    return G


# ── Temporal subgraph (Pitfall #1 / Gotcha 4 countermeasure) ──────────────


def _temporal_subgraph(graph: nx.DiGraph, as_of_date: date) -> nx.DiGraph:
    """Filter graph to edges with ``earliest_date < as_of_date``.

    This is the Pitfall #1 / Gotcha 4 countermeasure. ``compute_pagerank_at``
    and ``compute_2hop_sos_at`` both call this so they share a single
    consistent temporal-filter definition (Test 6 in
    ``tests/unit/features/test_network.py::TestTwoHopSoS``).
    """
    return graph.edge_subgraph(
        (u, v) for u, v, d in graph.edges(data=True) if d["earliest_date"] < as_of_date
    )


# ── Pure-function feature computers ────────────────────────────────────────


def compute_pagerank_at(
    graph: nx.DiGraph,
    fighter_id: Any,
    as_of_date: date,
    *,
    alpha: float = PAGERANK_ALPHA,
) -> float | None:
    """PageRank for one fighter, computed against the temporal subgraph.

    Returns ``None`` for debutants (caller maps to NaN +
    ``is_debutant_in_graph=1.0`` per D-06). The pre-filter via
    ``edge_subgraph`` happens BEFORE ``nx.pagerank`` per Gotcha 4.
    """
    sub = _temporal_subgraph(graph, as_of_date)
    if fighter_id not in sub.nodes:
        return None  # debutant — caller maps to NaN
    pr = nx.pagerank(sub, alpha=alpha, weight="weight")
    value = pr.get(fighter_id)
    return None if value is None else float(value)


def compute_2hop_sos_at(
    graph: nx.DiGraph,
    fighter_id: Any,
    as_of_date: date,
    *,
    alpha: float = PAGERANK_ALPHA,
) -> float | None:
    """2-hop strength of schedule: arithmetic mean of in-neighbors' PageRank.

    Per Claude's Discretion (CONTEXT.md ``<decisions>``): aggregation = mean
    (NOT sum or weighted-by-recency). Documented choice for v2.0; v2.1 may
    explore alternatives per CONTEXT.md ``<deferred>`` "2-hop SoS aggregation
    alternatives".

    Why mean and not sum?
      - Sum scales with fight count, conflating "fought many opponents" with
        "fought strong opponents". A 10-fight career vs a 3-fight career
        would always score higher even if the 3-fight career fought
        champion-tier opposition.
      - Mean gives the average opponent strength independent of volume,
        which is the signal we actually want the model to see.
      - A separate UFC fight-count feature (``ufc_fight_count_diff``)
        already encodes the volume signal independently.

    Edge cases (NET-02 plan §<behavior>):
      - Debutant (fighter not in subgraph) → ``None``
      - Fighter in subgraph but with zero in-neighbors before ``as_of_date``
        (only appears as a loser, no wins yet) → ``None``
      - Fighter with ≥1 in-neighbor → arithmetic mean of those PageRanks

    Uses the same ``_temporal_subgraph`` filter as ``compute_pagerank_at``
    so the two functions cannot diverge on Pitfall #1 countermeasures
    (Test 6 in ``TestTwoHopSoS::test_temporal_filter_consistency_with_pagerank``).
    """
    sub = _temporal_subgraph(graph, as_of_date)
    if fighter_id not in sub.nodes:
        return None  # debutant case — caller maps to NaN

    in_neighbors = list(sub.predecessors(fighter_id))
    if not in_neighbors:
        return None  # fighter present as edge tail only — no wins yet

    pr = nx.pagerank(sub, alpha=alpha, weight="weight")
    return float(sum(pr.get(n, 0.0) for n in in_neighbors) / len(in_neighbors))


# ── Differential helper (train/predict parity per Pitfall #12) ──────────


def compute_network_diff_features(
    feats_a: dict[str, float],
    feats_b: dict[str, float],
) -> dict[str, float]:
    """Compute the 3 NET-* ``_diff`` columns from per-fighter feature dicts.

    Used by BOTH ``ml/feature_matrix.py`` (train-time) and
    ``ml/inference_features.py`` (predict-time) so the differential math
    is byte-identical across the two paths (Pitfall #12 train/predict
    parity safeguard).

    Args:
        feats_a, feats_b: dicts with at minimum keys ``pagerank``,
            ``sos_2hop``, ``is_debutant_in_graph``. Values may be NaN
            (debutant case per D-06).

    Returns:
        dict with keys ``pagerank_diff``, ``sos_2hop_diff``,
        ``is_debutant_in_graph_diff``. NaN propagates per Pattern D
        (NEVER 0.0 for missing-data sentinels).

    Note on ``is_debutant_in_graph_diff``: the per-fighter flag is in
    ``{0.0, 1.0}``; the differential lands in ``{-1.0, 0.0, 1.0}`` and
    encodes "is exactly one side a debutant?". XGBoost can split on this.
    """
    pr_a = feats_a.get("pagerank", float("nan"))
    pr_b = feats_b.get("pagerank", float("nan"))
    sos_a = feats_a.get("sos_2hop", float("nan"))
    sos_b = feats_b.get("sos_2hop", float("nan"))
    deb_a = feats_a.get("is_debutant_in_graph", float("nan"))
    deb_b = feats_b.get("is_debutant_in_graph", float("nan"))

    return {
        "pagerank_diff": pr_a - pr_b,
        "sos_2hop_diff": sos_a - sos_b,
        "is_debutant_in_graph_diff": deb_a - deb_b,
    }


# ── 4th-pass integration helper ────────────────────────────────────────────


def apply_network_features(
    raw_results: list[dict[str, Any]],
    fights: list[dict[str, Any]],
    *,
    scope: str = APPROVED_SCOPE,
    weight_mode: str = APPROVED_WEIGHT,
) -> None:
    """4th-pass mutator. Adds ``pagerank``, ``sos_2hop``,
    ``is_debutant_in_graph`` keys to each ``raw_results[i]['features']``.

    Per FEATURES.md F1/F2: cached as part of ``ComputedFeature.features``
    JSONB. Per ARCHITECTURE.md: no new DB table in Phase 16.

    Args:
        raw_results: list of feature snapshot dicts emitted by Pass 1-3
            of ``FeatureComputer.compute_all``. Each row has at minimum
            keys ``fighter_id``, ``as_of_date`` (a ``datetime.date``),
            and ``features`` (dict). Mutated in-place.
        fights: list of fight dicts in the same shape ``build_fight_graph``
            consumes.
        scope, weight_mode: optional overrides — default to the operator-
            approved ``pan-mma`` + ``mov`` per the NET-00 gsd-checkpoint.

    Per Pitfall #1: the GRAPH is built once per call, but the SUBGRAPH is
    computed per ``(fighter_id, as_of_date)`` snapshot inside
    ``compute_pagerank_at`` / ``compute_2hop_sos_at``. The per-call build is
    fine; the per-snapshot subgraph is the leakage countermeasure.
    """
    if not raw_results:
        return

    # Filter ``fights`` to entries with a winner_id/loser_id pair (defensive
    # against draws / NC; the spike used only "decisive" fights for the same
    # reason).
    decisive = [
        f for f in fights if f.get("winner_id") is not None and f.get("loser_id") is not None
    ]
    graph = build_fight_graph(decisive, scope=scope, weight_mode=weight_mode)

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

        # For per-division scope, the graph keys are namespaced tuples.
        if scope == "per-division":
            # The 4th pass doesn't have weight_class on the row by default
            # — mark debutant if not present.
            wc = row.get("weight_class")
            if wc is None:
                row["features"]["pagerank"] = nan
                row["features"]["sos_2hop"] = nan
                row["features"]["is_debutant_in_graph"] = 1.0
                continue
            node: Any = (wc, fid)
        else:
            node = fid

        pr = compute_pagerank_at(graph, node, as_of)
        sos = compute_2hop_sos_at(graph, node, as_of)

        # Per D-06 / Pattern D: debutant => NaN PageRank + flag=1.0
        if pr is None:
            row["features"]["pagerank"] = nan
            row["features"]["is_debutant_in_graph"] = 1.0
        else:
            row["features"]["pagerank"] = float(pr)
            row["features"]["is_debutant_in_graph"] = 0.0
        row["features"]["sos_2hop"] = float(sos) if sos is not None else nan
