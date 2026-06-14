"""NET-01 / NET-02 unit tests: PageRank as-of-date + 2-hop SoS pure functions.

Implements the Wave 0 stub per Plan 16-03 Tasks 2-3. The operator-approved
graph configuration from the NET-00 spike is `pan-mma-mov` (D-05 + D-07
operator priors validated). This test suite covers:

    NET-01 (Task 2):
      - Test 1: PageRank shape (sum to 1.0; valid float in (0, 1))
      - Test 2: Debutant returns None (caller maps to NaN + flag=1.0)
      - Test 3: Pre-filter via edge_subgraph BEFORE pagerank (Pitfall #1
                / Gotcha 4 invariant — future edges never influence)
      - Test 4: MOV-weighted edges (KO weight > DEC weight; PageRank
                reflects this asymmetry)
      - Test 5: 4th-pass integration — apply_network_features mutates
                raw_results entries with pagerank, sos_2hop,
                is_debutant_in_graph keys
      - Test 6: FEATURE_COLUMNS append-only (Gotcha 9 — xgb_v2 byte-
                identity preserved)
      - Test 7: feature_matrix.py / inference_features.py train-predict
                parity — the 3 _diff columns are populated identically

    NET-02 (Task 3):
      - TestTwoHopSoS::test_arithmetic_mean
      - TestTwoHopSoS::test_single_neighbor
      - TestTwoHopSoS::test_no_in_neighbors_returns_none
      - TestTwoHopSoS::test_temporal_filter_applied
      - TestTwoHopSoS::test_high_pagerank_neighbor
      - TestTwoHopSoS::test_temporal_filter_consistency_with_pagerank

The 5-fighter / 8-fight toy graph fixture is shared with
``tests/regression/test_temporal_leakage.py`` so the leakage CI test
exercises the same shape NET-01 / NET-02 are unit-tested against.
"""

from __future__ import annotations

from datetime import date

import pytest

network = pytest.importorskip("ufc_prediction.features.network")
nx = pytest.importorskip("networkx")


# ── Toy graph fixture ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def toy_fights() -> list[dict]:
    """5-fighter / 8-fight toy fight list (chronologically ordered).

    Per VALIDATION.md Wave 0 contract:
      - At least one debutant (no fights before some as_of_date)
      - At least one fighter with fights both before AND after a cutoff
      - At least 3 fights spanning ≥ 1 year (multi-era spot check)

    Convention: each row is a dict shaped to mirror what the production
    fight loader emits, plus the additional `winner_id` / `loser_id` /
    `method` keys the network module needs for graph construction.

    Concrete graph (post-build, pan-MMA scope, MOV weights):
        2010-01-01: F1 beats F2  (KO/TKO,    weight=1.2)   loser=F2 -> F1
        2010-06-01: F1 beats F3  (Decision,  weight=1.0)   loser=F3 -> F1
        2011-01-01: F2 beats F3  (KO/TKO,    weight=1.2)   loser=F3 -> F2
        2011-06-01: F4 beats F1  (Submission,weight=1.1)   loser=F1 -> F4
        2018-01-01: F5 beats F4  (KO/TKO,    weight=1.2)   loser=F4 -> F5
                                  (F5 is a debutant before this date)
        2018-06-01: F5 beats F1  (Decision,  weight=1.0)   loser=F1 -> F5
        2024-01-01: F5 beats F2  (Decision,  weight=1.0)   loser=F2 -> F5
        2024-06-01: F5 beats F3  (KO/TKO,    weight=1.2)   loser=F3 -> F5
    """
    return [
        {"fight_id": 1,  "event_date": date(2010, 1, 1),
         "fighter_a_id": 1, "fighter_b_id": 2, "winner_id": 1, "loser_id": 2,
         "method": "KO/TKO", "weight_class": "Lightweight"},
        {"fight_id": 2,  "event_date": date(2010, 6, 1),
         "fighter_a_id": 1, "fighter_b_id": 3, "winner_id": 1, "loser_id": 3,
         "method": "Decision", "weight_class": "Lightweight"},
        {"fight_id": 3,  "event_date": date(2011, 1, 1),
         "fighter_a_id": 2, "fighter_b_id": 3, "winner_id": 2, "loser_id": 3,
         "method": "KO/TKO", "weight_class": "Lightweight"},
        {"fight_id": 4,  "event_date": date(2011, 6, 1),
         "fighter_a_id": 4, "fighter_b_id": 1, "winner_id": 4, "loser_id": 1,
         "method": "Submission", "weight_class": "Lightweight"},
        {"fight_id": 5,  "event_date": date(2018, 1, 1),
         "fighter_a_id": 5, "fighter_b_id": 4, "winner_id": 5, "loser_id": 4,
         "method": "KO/TKO", "weight_class": "Lightweight"},
        {"fight_id": 6,  "event_date": date(2018, 6, 1),
         "fighter_a_id": 5, "fighter_b_id": 1, "winner_id": 5, "loser_id": 1,
         "method": "Decision", "weight_class": "Lightweight"},
        {"fight_id": 7,  "event_date": date(2024, 1, 1),
         "fighter_a_id": 5, "fighter_b_id": 2, "winner_id": 5, "loser_id": 2,
         "method": "Decision", "weight_class": "Lightweight"},
        {"fight_id": 8,  "event_date": date(2024, 6, 1),
         "fighter_a_id": 5, "fighter_b_id": 3, "winner_id": 5, "loser_id": 3,
         "method": "KO/TKO", "weight_class": "Lightweight"},
    ]


@pytest.fixture(scope="module")
def toy_graph(toy_fights):
    """Pan-MMA + MOV-weighted DiGraph from toy_fights (operator-approved config)."""
    return network.build_fight_graph(toy_fights, scope="pan-mma", weight_mode="mov")


# ── NET-01: PageRank pure-function tests ───────────────────────────────────


class TestPageRankAsOfDate:
    """NET-01 unit coverage for compute_pagerank_at."""

    def test_pagerank_shape_sums_to_one(self, toy_graph):
        """Test 1 — PageRank invariant: per-node values sum to 1.0."""
        # Use as_of_date AFTER all fights, so the full graph is in scope.
        as_of = date(2025, 1, 1)
        sub = network._temporal_subgraph(toy_graph, as_of)
        # Compute the full PR dict and verify the sum invariant.
        pr_all = nx.pagerank(sub, alpha=0.85, weight="weight")
        assert sum(pr_all.values()) == pytest.approx(1.0)
        # Each per-fighter value is in (0, 1).
        for fighter_id in (1, 2, 3, 4, 5):
            pr = network.compute_pagerank_at(toy_graph, fighter_id, as_of)
            assert pr is not None
            assert 0.0 < pr < 1.0

    def test_debutant_returns_none(self, toy_graph):
        """Test 2 — debutant invariant: no in-edges before as_of -> None.

        F5's first fight is 2018-01-01. Before that date, F5 has no edges
        in the graph and compute_pagerank_at must return None (caller
        maps to NaN + is_debutant_in_graph=1.0 per D-06).
        """
        # Before F5's debut: no edges yet for F5.
        as_of = date(2017, 12, 31)
        result = network.compute_pagerank_at(toy_graph, 5, as_of)
        assert result is None
        # After F5's debut: should be a real value.
        as_of = date(2025, 1, 1)
        result = network.compute_pagerank_at(toy_graph, 5, as_of)
        assert result is not None
        assert result > 0.0

    def test_pitfall_1_pre_filter_excludes_future_edges(self, toy_graph):
        """Test 3 — Pitfall #1 invariant: edges with event_date >= as_of
        are excluded from the PageRank computation (Gotcha 4).

        At as_of = 2011-07-01, only the first 4 fights are in scope
        (the 4 fights from 2018+ and 2024+ are excluded). Verify that
        the temporal subgraph has fewer edges than the full graph.
        """
        as_of_partial = date(2011, 7, 1)
        as_of_today = date(2025, 1, 1)

        sub_partial = network._temporal_subgraph(toy_graph, as_of_partial)
        sub_today = network._temporal_subgraph(toy_graph, as_of_today)

        # Subgraph monotonicity — past subgraph ⊂ present subgraph.
        assert sub_partial.number_of_edges() < sub_today.number_of_edges()

        # F1's PageRank-at-partial differs from PageRank-at-today
        # (F1 had post-2011 fights that change the graph topology).
        pr_partial = network.compute_pagerank_at(toy_graph, 1, as_of_partial)
        pr_today = network.compute_pagerank_at(toy_graph, 1, as_of_today)
        assert pr_partial is not None and pr_today is not None
        assert abs(pr_partial - pr_today) > 1e-9

    def test_mov_weighted_edges_reflect_method(self, toy_fights):
        """Test 4 — MOV-weighted edges: KO/TKO has higher weight than DEC.

        Asserts that the edge weight assigned by build_fight_graph matches
        the EloConfig MOV ratios for KO/TKO (1.2) vs Decision (1.0) vs
        Submission (1.1). Also asserts that PageRank values reflect the
        weighting (binary vs MOV-weighted graphs produce different ranks
        for the same fighter).
        """
        # Build both MOV and binary graphs from the same fight list.
        g_mov = network.build_fight_graph(toy_fights, scope="pan-mma", weight_mode="mov")
        g_bin = network.build_fight_graph(toy_fights, scope="pan-mma", weight_mode="binary")

        # Pull edge weights for the 3 distinct methods we use.
        # F1 -> F4 came from a Submission win (fight 4) — weight should be 1.1.
        # F2 -> F1 came from a KO/TKO win (fight 1) — weight should be 1.2.
        # F3 -> F1 came from a Decision win (fight 2) — weight should be 1.0.
        # NB: edges are loser → winner.
        assert g_mov[2][1]["weight"] == pytest.approx(1.2)  # KO/TKO
        assert g_mov[3][1]["weight"] == pytest.approx(1.0)  # Decision
        assert g_mov[1][4]["weight"] == pytest.approx(1.1)  # Submission

        # In binary graph, every edge has weight 1.0.
        assert g_bin[2][1]["weight"] == pytest.approx(1.0)
        assert g_bin[3][1]["weight"] == pytest.approx(1.0)
        assert g_bin[1][4]["weight"] == pytest.approx(1.0)

        # PageRank values differ between MOV and binary configurations
        # for at least one fighter — confirms the weight is consumed.
        as_of = date(2025, 1, 1)
        pr_mov_f1 = network.compute_pagerank_at(g_mov, 1, as_of)
        pr_bin_f1 = network.compute_pagerank_at(g_bin, 1, as_of)
        assert pr_mov_f1 is not None and pr_bin_f1 is not None
        assert abs(pr_mov_f1 - pr_bin_f1) > 1e-9

    def test_apply_network_features_mutates_raw_results(self, toy_fights):
        """Test 5 — 4th-pass integration: apply_network_features writes
        pagerank / sos_2hop / is_debutant_in_graph keys into each row's
        features dict.

        Per Pattern Map: apply_network_features is the entry point invoked
        from FeatureComputer.compute_all as Pass 4.
        """
        # Build raw_results structure mimicking what compute_all emits
        # after Pass 1-3. Each row corresponds to one (fighter_id, fight_id)
        # snapshot at fight_date.
        raw_results = []
        for f in toy_fights:
            for fid in (f["fighter_a_id"], f["fighter_b_id"]):
                raw_results.append({
                    "fighter_id": fid,
                    "fight_id": f["fight_id"],
                    "as_of_date": f["event_date"],
                    "feature_set_version": "v3",
                    "features": {},
                })

        network.apply_network_features(raw_results, toy_fights)

        for row in raw_results:
            assert "pagerank" in row["features"]
            assert "sos_2hop" in row["features"]
            assert "is_debutant_in_graph" in row["features"]
            # is_debutant_in_graph is always 0.0 or 1.0
            flag = row["features"]["is_debutant_in_graph"]
            assert flag in (0.0, 1.0)
            # If debutant, pagerank/sos must be NaN; else floats.
            pr = row["features"]["pagerank"]
            sos = row["features"]["sos_2hop"]
            if flag == 1.0:
                assert pr != pr  # NaN
                # sos can also be NaN (debutant) or NaN (no neighbors)
                assert sos != sos
            else:
                assert isinstance(pr, float) and not (pr != pr)
                # sos can be NaN if no in-neighbors but pagerank is defined,
                # OR a float — either is acceptable.

    def test_feature_columns_append_only(self):
        """Test 6 — FEATURE_COLUMNS append-only invariant (Gotcha 9).

        The 3 NET-* columns must sit at the END of FEATURE_COLUMNS, in
        canonical order, AFTER the existing odds_elo_divergence entry.
        Inserting NET-* features in the middle would break xgb_v2's
        positional column order and is forbidden per Gotcha 9.
        """
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        # Last 3 entries in canonical order.
        assert FEATURE_COLUMNS[-3:] == [
            "pagerank_diff",
            "sos_2hop_diff",
            "is_debutant_in_graph_diff",
        ]
        # Length is 75 (was 72 + 3 NET-* additions).
        assert len(FEATURE_COLUMNS) == 75
        # The previous-last entry is preserved (odds_elo_divergence).
        assert FEATURE_COLUMNS[-4] == "odds_elo_divergence"

    def test_train_predict_parity(self):
        """Test 7 — feature_matrix.py / inference_features.py parity
        (Pitfall #12 train/predict drift safeguard).

        For the same fighter inputs, the 3 _diff columns must be
        identical in the train-time matrix and the predict-time vector.
        We assert by directly invoking the per-feature differential
        helper used by both modules (the math is shared via the same
        keys in raw row['features']).
        """
        # Synthetic per-fighter NET-* features (mirrors what
        # ComputedFeature.features JSONB would store).
        feats_a = {"pagerank": 0.30, "sos_2hop": 0.20, "is_debutant_in_graph": 0.0}
        feats_b = {"pagerank": 0.10, "sos_2hop": 0.05, "is_debutant_in_graph": 0.0}

        # Train-time differential (mirrors feature_matrix.py:560-625 pattern).
        train_pagerank_diff = feats_a["pagerank"] - feats_b["pagerank"]
        train_sos_diff = feats_a["sos_2hop"] - feats_b["sos_2hop"]
        train_debutant_diff = (
            feats_a["is_debutant_in_graph"] - feats_b["is_debutant_in_graph"]
        )

        # Predict-time differential (mirrors inference_features.py).
        # network.compute_network_diff_features encapsulates the math both
        # modules call to ensure parity.
        diffs = network.compute_network_diff_features(feats_a, feats_b)
        assert diffs["pagerank_diff"] == pytest.approx(train_pagerank_diff)
        assert diffs["sos_2hop_diff"] == pytest.approx(train_sos_diff)
        assert diffs["is_debutant_in_graph_diff"] == pytest.approx(train_debutant_diff)

        # Debutant on one side: pagerank_diff is NaN (Pattern D — never 0.0).
        feats_a_debutant = {
            "pagerank": float("nan"),
            "sos_2hop": float("nan"),
            "is_debutant_in_graph": 1.0,
        }
        diffs_d = network.compute_network_diff_features(feats_a_debutant, feats_b)
        assert diffs_d["pagerank_diff"] != diffs_d["pagerank_diff"]  # NaN
        assert diffs_d["sos_2hop_diff"] != diffs_d["sos_2hop_diff"]  # NaN
        # is_debutant_in_graph_diff: 1.0 - 0.0 = 1.0 (still a meaningful float).
        assert diffs_d["is_debutant_in_graph_diff"] == pytest.approx(1.0)


# ── NET-02: 2-hop strength of schedule ─────────────────────────────────────


class TestTwoHopSoS:
    """NET-02 dedicated coverage for compute_2hop_sos_at."""

    def test_arithmetic_mean(self, toy_graph):
        """Test 1 — SoS is the mean of in-neighbors' PageRank values.

        F5 at as_of=today has 4 in-neighbors (F4, F1, F2, F3) — all losers
        of F5. SoS == mean of their PageRank values at the same as_of.
        """
        as_of = date(2025, 1, 1)
        sub = network._temporal_subgraph(toy_graph, as_of)
        in_neighbors = list(sub.predecessors(5))
        assert sorted(in_neighbors) == [1, 2, 3, 4]

        pr = nx.pagerank(sub, alpha=0.85, weight="weight")
        expected_mean = sum(pr[n] for n in in_neighbors) / len(in_neighbors)
        sos = network.compute_2hop_sos_at(toy_graph, 5, as_of)
        assert sos == pytest.approx(expected_mean)

    def test_single_neighbor(self, toy_graph):
        """Test 2 — single in-neighbor: SoS == that neighbor's PageRank."""
        # At as_of=2010-12-31, F1 has 2 in-neighbors (F2 from fight 1,
        # F3 from fight 2). Use a tighter cutoff: F1 with one neighbor.
        # Specifically, at as_of=2010-06-02 (only fights 1 and 2 are in scope),
        # F1's in-neighbors are F2 and F3. To get a single-neighbor case,
        # use F4's snapshot AFTER the F4-beats-F1 fight: F4's in-neighbor
        # is F1 (only one, since F4 hasn't lost yet at that point).
        as_of = date(2011, 6, 2)  # right after fight 4
        sub = network._temporal_subgraph(toy_graph, as_of)
        in_neighbors = list(sub.predecessors(4))
        assert in_neighbors == [1]
        pr = nx.pagerank(sub, alpha=0.85, weight="weight")
        sos = network.compute_2hop_sos_at(toy_graph, 4, as_of)
        assert sos == pytest.approx(pr[1])

    def test_no_in_neighbors_returns_none(self, toy_graph):
        """Test 3 — fighter present in graph but with zero in-neighbors
        before as_of_date returns None (debutant case, caller maps to NaN).

        At as_of=2010-06-02, only fights 1 and 2 are in scope. F1 has 2
        in-neighbors (won fights 1 and 2). F2/F3 have NO in-neighbors yet
        (they only appear as edge tails). compute_2hop_sos_at(F2, ...)
        must return None.
        """
        as_of = date(2010, 6, 2)
        sub = network._temporal_subgraph(toy_graph, as_of)
        assert 2 in sub.nodes  # F2 lost to F1, so it's an edge tail
        assert list(sub.predecessors(2)) == []  # but F2 has no wins yet
        result = network.compute_2hop_sos_at(toy_graph, 2, as_of)
        assert result is None

    def test_temporal_filter_applied(self, toy_graph):
        """Test 4 — temporal filter: in-neighbors with edges only AFTER
        as_of_date are NOT counted in the SoS aggregation.

        F5 has 4 in-neighbors at full as_of. At as_of=2018-06-02, F5's
        in-neighbors should be only F4 (fight 5) and F1 (fight 6) — the
        2024 wins (fights 7, 8) must be excluded.
        """
        as_of = date(2018, 6, 2)
        sub = network._temporal_subgraph(toy_graph, as_of)
        in_neighbors = sorted(sub.predecessors(5))
        assert in_neighbors == [1, 4]  # F2 and F3 not yet beaten

        # SoS is the mean of just F1 + F4 PageRanks at this as_of.
        pr = nx.pagerank(sub, alpha=0.85, weight="weight")
        expected = (pr[1] + pr[4]) / 2.0
        actual = network.compute_2hop_sos_at(toy_graph, 5, as_of)
        assert actual == pytest.approx(expected)

    def test_high_pagerank_neighbor(self, toy_fights):
        """Test 5 — fighters with high-PageRank in-neighbors have higher SoS.

        Build two contrasting toy graphs:
          (i)  F_strong beats F_GOAT (F_GOAT has many wins)
          (ii) F_strong beats F_TomatoCan (F_TomatoCan has no wins)

        Then SoS(F_strong | i) > SoS(F_strong | ii).
        """
        # Graph (i): F_strong (fid=99) beats F_GOAT (fid=5, who's the
        # GOAT in our toy_fights). At as_of=2025-01-01, F5 has high PR.
        graph_i_fights = toy_fights + [
            {"fight_id": 99, "event_date": date(2024, 12, 1),
             "fighter_a_id": 99, "fighter_b_id": 5,
             "winner_id": 99, "loser_id": 5,
             "method": "Decision", "weight_class": "Lightweight"},
        ]
        g_i = network.build_fight_graph(graph_i_fights, scope="pan-mma",
                                        weight_mode="mov")
        sos_i = network.compute_2hop_sos_at(g_i, 99, date(2025, 1, 1))

        # Graph (ii): F_strong beats F_TomatoCan (fid=99 beats fid=98 who
        # has no wins). The toy_fights graph never references F98.
        graph_ii_fights = toy_fights + [
            {"fight_id": 100, "event_date": date(2024, 12, 1),
             "fighter_a_id": 99, "fighter_b_id": 98,
             "winner_id": 99, "loser_id": 98,
             "method": "Decision", "weight_class": "Lightweight"},
            # Also add a 101 fight where F98 loses to someone else so
            # F98 IS in the graph (otherwise compute_pagerank skips entirely).
            {"fight_id": 101, "event_date": date(2024, 11, 1),
             "fighter_a_id": 97, "fighter_b_id": 98,
             "winner_id": 97, "loser_id": 98,
             "method": "Decision", "weight_class": "Lightweight"},
        ]
        g_ii = network.build_fight_graph(graph_ii_fights, scope="pan-mma",
                                         weight_mode="mov")
        sos_ii = network.compute_2hop_sos_at(g_ii, 99, date(2025, 1, 1))

        assert sos_i is not None and sos_ii is not None
        assert sos_i > sos_ii, (
            f"SoS_with_GOAT_neighbor ({sos_i:.6f}) should exceed "
            f"SoS_with_tomato_can_neighbor ({sos_ii:.6f})"
        )

    def test_temporal_filter_consistency_with_pagerank(self, toy_graph):
        """Test 6 — compute_2hop_sos_at and compute_pagerank_at use the
        SAME edge_subgraph filter. This guarantees no divergence between
        Pitfall #1 countermeasures across the two functions.
        """
        as_of = date(2018, 6, 2)
        sub_pr = network._temporal_subgraph(toy_graph, as_of)
        # Both functions should reference _temporal_subgraph internally.
        # Verify by checking that the subgraph that pagerank operates on
        # at this as_of is the same shape both functions see.
        for fighter_id in (1, 2, 3, 4, 5):
            if fighter_id not in sub_pr.nodes:
                # Debutant case both functions handle identically.
                assert network.compute_pagerank_at(
                    toy_graph, fighter_id, as_of) is None
                assert network.compute_2hop_sos_at(
                    toy_graph, fighter_id, as_of) is None
                continue
            pr_val = network.compute_pagerank_at(toy_graph, fighter_id, as_of)
            sos_val = network.compute_2hop_sos_at(toy_graph, fighter_id, as_of)
            # If the fighter has zero in-neighbors at this as_of, pagerank
            # is still defined (a node with no incoming edges still gets
            # the dangling teleport mass) but sos is None.
            in_count = sub_pr.in_degree(fighter_id)
            assert pr_val is not None
            if in_count == 0:
                assert sos_val is None
            else:
                assert sos_val is not None
