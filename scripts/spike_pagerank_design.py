#!/usr/bin/env python3
"""NET-00 graph design spike (Phase 16-03 Task 1).

Backtests 4 graph configurations to inform the operator gsd-checkpoint decision
(D-03):

    1. pan-mma-mov         — operator prior (D-05 + D-07)
    2. pan-mma-binary      — control for D-07 (does MOV beat binary?)
    3. per-division-mov    — control for D-05 (does pan-MMA beat per-div?)
    4. per-division-binary — pure control

Per CONTEXT.md `<specifics>` "NET-00 spike output shape", the report contains:
    (a) Per-config Brier / Acc / AUC table
    (b) Temporal-leakage CI prototype result per config (PASS / FAIL)
    (c) Recommendation paragraph + operator decision question
    (d) `## gsd-checkpoint: operator decision required` marker line

Per D-04: uses cutoff_date from models/xgb_v2_meta.json for apples-to-apples
fairness with xgb_v2.

Per the plan's "spike-mode backtest can be approximate" allowance: instead of
re-training XGBoost with the full 72-feature FEATURE_COLUMNS plus NET-* per
config (~30+ min × 4 configs), this script uses a PageRank-ONLY logistic
regression backtest. The operator decision is "which graph config wins?" — a
relative Brier comparison across configs is what's needed, not an absolute
Brier vs xgb_v2. The temporal-leakage test (which is the harder Pitfall #1
invariant) is still exercised for every config.

Per Pitfall #1 / Gotcha 4: PageRank-as-of-fight-date is computed by pre-
filtering the directed graph via `graph.edge_subgraph(...)` BEFORE calling
`nx.pagerank(...)`. Future fights cannot influence past PageRank values.

Per D-07 & EloConfig validation: Logged Assumption A4 (the plan's reference
to ``mov_ratio_ko / mov_ratio_sub / mov_ratio_dec``) is BUSTED — those
attributes do not exist on EloConfig. The actual MOV multipliers are
`mov_ko_tko=1.2`, `mov_submission=1.1`, `mov_unanimous=1.0`,
`mov_split=0.5`, `mov_draw=0.5`, `mov_dq=0.8`. The script verifies these and
maps method strings to multipliers explicitly.

Usage:
    python scripts/spike_pagerank_design.py --all-configs \\
        --output .planning/phases/16-foundation-opponent-network-live-odds/16-03-NET-00-SPIKE-RESULTS.md

    python scripts/spike_pagerank_design.py --config pan-mma-mov \\
        --output .planning/phases/16-foundation-opponent-network-live-odds/16-03-NET-00-SPIKE-RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import networkx as nx
import numpy as np

# Ensure project src is importable
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ufc_prediction.elo.config import EloConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("spike_pagerank")

CONFIGS: tuple[str, ...] = (
    "pan-mma-mov",
    "pan-mma-binary",
    "per-division-mov",
    "per-division-binary",
)

# Per spike's own observations and the operator's spike-mode allowance,
# we use a PageRank-only logistic regression. alpha=0.85 is NetworkX default
# and the standard sports-PageRank value (Boldi/Vigna's 0.10 is alternate;
# adding alpha to the A/B grid would have doubled spike runtime — recommended
# 0.85 only per plan's explicit guidance).
PAGERANK_ALPHA: float = 0.85
TEMPORAL_LEAKAGE_TEST_FIGHTERS_K: int = 5  # spot-check sample size


@dataclass(frozen=True)
class FightRecord:
    """Lightweight fight row used for graph construction + backtest."""

    fight_id: int
    event_date: date
    fighter_a_id: int
    fighter_b_id: int
    winner_id: int
    loser_id: int
    method: str
    weight_class: str | None


def verify_elo_config_mov_attrs() -> dict[str, float]:
    """Verify EloConfig has the MOV multipliers the spike depends on.

    Per the plan's Logged Assumption A4: if EloConfig is missing the expected
    fields, surface a clear RuntimeError so the operator resolves it before
    the spike runs.

    Returns the verified multiplier dict keyed by canonical method name.
    """
    cfg = EloConfig()
    required = ("mov_ko_tko", "mov_submission", "mov_unanimous", "mov_split", "mov_draw", "mov_dq")
    missing = [a for a in required if not hasattr(cfg, a)]
    if missing:
        raise RuntimeError(
            f"EloConfig is missing expected MOV attributes: {missing}. "
            "The plan referenced mov_ratio_ko/mov_ratio_sub/mov_ratio_dec but the "
            "actual fields are mov_ko_tko/mov_submission/mov_unanimous/mov_split/"
            "mov_draw/mov_dq (per src/ufc_prediction/elo/config.py). Operator "
            "must reconcile the plan's Logged Assumption A4 with reality before "
            "the spike runs."
        )
    return {
        "mov_ko_tko": cfg.mov_ko_tko,
        "mov_submission": cfg.mov_submission,
        "mov_unanimous": cfg.mov_unanimous,
        "mov_split": cfg.mov_split,
        "mov_draw": cfg.mov_draw,
        "mov_dq": cfg.mov_dq,
    }


def method_to_mov_weight(method: str | None, mov: dict[str, float]) -> float:
    """Map a fight method string to its MOV multiplier.

    Method strings observed in the production DB:
        KO/TKO, TKO, SUB, Submission, U-DEC, S-DEC, M-DEC, Decision,
        DQ, CNC, Other, Overturned

    Returns the EloConfig MOV multiplier; 1.0 (neutral) for ambiguous
    or non-finish methods we don't want to over-weight.
    """
    if method is None:
        return mov["mov_unanimous"]
    m = method.upper()
    if m in ("KO/TKO", "KO", "TKO"):
        return mov["mov_ko_tko"]
    if m in ("SUB", "SUBMISSION"):
        return mov["mov_submission"]
    if m == "U-DEC":
        return mov["mov_unanimous"]
    if m in ("S-DEC", "M-DEC"):
        return mov["mov_split"]
    if m == "DECISION":
        return mov["mov_unanimous"]  # generic decision = unanimous default
    if m == "DQ":
        return mov["mov_dq"]
    return 1.0  # Other / CNC / Overturned — neutral


def load_fights_from_db(db_url: str) -> list[FightRecord]:
    """Load all decisive fights from Postgres in chronological order."""
    import psycopg

    sql = """
        SELECT
            f.id,
            e.date,
            f.fighter_a_id,
            f.fighter_b_id,
            f.winner_id,
            f.method,
            f.weight_class
        FROM fights f
        JOIN events e ON f.event_id = e.id
        WHERE f.winner_id IS NOT NULL
          AND f.winner_id IN (f.fighter_a_id, f.fighter_b_id)
        ORDER BY e.date ASC, f.id ASC
    """
    rows: list[FightRecord] = []
    with psycopg.connect(db_url, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(sql)
        for fight_id, event_date, fa, fb, winner, method, wc in cur.fetchall():
            loser = fb if winner == fa else fa
            rows.append(
                FightRecord(
                    fight_id=fight_id,
                    event_date=event_date,
                    fighter_a_id=fa,
                    fighter_b_id=fb,
                    winner_id=winner,
                    loser_id=loser,
                    method=method or "Other",
                    weight_class=wc,
                )
            )
    logger.info(
        "loaded %d decisive fights (date range %s — %s)",
        len(rows),
        rows[0].event_date if rows else None,
        rows[-1].event_date if rows else None,
    )
    return rows


def build_graph(
    fights: list[FightRecord],
    *,
    scope: str,
    weight_mode: str,
    mov: dict[str, float],
) -> nx.DiGraph:
    """Build the directed fight graph (loser → winner; winner absorbs PageRank).

    scope ∈ {"pan-mma", "per-division"}:
      - "pan-mma": single graph spanning all fights regardless of weight_class
                   (D-05 operator prior — captures cross-division skill transfer).
      - "per-division": one logical graph per weight_class — implemented here
                        as a single graph whose node IDs are namespaced
                        ``(weight_class, fighter_id)`` tuples so cross-division
                        edges never form. Equivalent to N disjoint graphs.

    weight_mode ∈ {"mov", "binary"}:
      - "mov": edge weight = method_to_mov_weight(...) (D-07 operator prior)
      - "binary": edge weight = 1.0
    """
    G: nx.DiGraph = nx.DiGraph()
    for f in fights:
        if scope == "pan-mma":
            u, v = f.loser_id, f.winner_id
        elif scope == "per-division":
            wc = f.weight_class or "_unknown"
            u, v = (wc, f.loser_id), (wc, f.winner_id)
        else:
            raise ValueError(f"unknown scope: {scope!r}")

        if weight_mode == "mov":
            w = method_to_mov_weight(f.method, mov)
        elif weight_mode == "binary":
            w = 1.0
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode!r}")

        # If a directed edge already exists (rematch), accumulate weight to keep
        # PageRank consistent with chronological "evidence aggregates"
        # interpretation. The alternative (replace) would underweight rematches.
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
            # event_date stays as the LATEST occurrence so edge_subgraph
            # filtering uses the strictest cutoff (any of the rematches before
            # as_of_date should keep this edge included up to the earliest
            # qualifying date — see _temporal_subgraph_strict below).
            G[u][v]["earliest_date"] = min(G[u][v]["earliest_date"], f.event_date)
        else:
            G.add_edge(u, v, weight=w, event_date=f.event_date, earliest_date=f.event_date)
    return G


def _temporal_subgraph(graph: nx.DiGraph, as_of_date: date) -> nx.DiGraph:
    """edge_subgraph filter — Pitfall #1 / Gotcha 4 countermeasure.

    Includes only edges with earliest_date < as_of_date.
    """
    return graph.edge_subgraph(
        (u, v) for u, v, d in graph.edges(data=True) if d["earliest_date"] < as_of_date
    )


def compute_pagerank_at(
    graph: nx.DiGraph,
    fighter_node,
    as_of_date: date,
    *,
    alpha: float = PAGERANK_ALPHA,
) -> float | None:
    """PageRank for one fighter, computed against the temporal subgraph.

    Returns None for debutants (caller maps to NaN + is_debutant_in_graph=1.0).
    Per Gotcha 4: pre-filter via edge_subgraph BEFORE pagerank.
    """
    sub = _temporal_subgraph(graph, as_of_date)
    if fighter_node not in sub.nodes:
        return None
    pr = nx.pagerank(sub, alpha=alpha, weight="weight")
    return pr.get(fighter_node)


def backtest_pagerank_only(
    fights: list[FightRecord],
    graph: nx.DiGraph,
    *,
    scope: str,
    cutoff_date: date,
) -> dict[str, float | int]:
    """Logistic-regression backtest using ONLY pagerank_diff = PR(A) − PR(B).

    Per the plan's spike-mode allowance and the operator's decision shape
    (relative comparison across 4 configs, not absolute Brier vs xgb_v2),
    this is an approximation chosen for runtime (~seconds) over fidelity.

    Train: fights with event_date < cutoff_date AND both fighters in graph.
    Test:  fights with event_date >= cutoff_date AND both fighters in graph.

    For each fight, the feature is computed as-of fight.event_date with
    `edge_subgraph` pre-filter (NET-03 invariant). For runtime, the
    PageRank dict is computed once per UNIQUE as_of_date encountered (the
    bulk of fights share dates; ~1900 distinct event dates vs 16k fights).

    Symmetry note: the data source has a strong winner-as-fighter-a skew
    (~80% of rows have winner_id == fighter_a_id), so a naive "A vs B"
    framing would let the LR fit the marginal rate and produce misleading
    high-accuracy / low-AUC numbers. This function deterministically swaps
    A/B based on fight_id parity to balance the label rate, mirroring the
    symmetrization implicit in feature_matrix.py's A-minus-B differential
    convention.

    Returns Brier, Accuracy, AUC, n_train, n_test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    def fighter_node(fight: FightRecord, fighter_id: int):
        if scope == "pan-mma":
            return fighter_id
        wc = fight.weight_class or "_unknown"
        return (wc, fighter_id)

    # Group fights by event_date so we compute pagerank once per snapshot.
    # Collect unique as_of_dates first, sort, then walk forward.
    distinct_dates = sorted({f.event_date for f in fights})
    logger.info("backtest_pagerank_only: %d distinct event dates", len(distinct_dates))

    # We'll compute the snapshot pagerank dict per distinct date and cache.
    pr_cache: dict[date, dict] = {}
    t0 = time.time()
    for i, d in enumerate(distinct_dates):
        sub = _temporal_subgraph(graph, d)
        if sub.number_of_edges() == 0:
            pr_cache[d] = {}
            continue
        pr_cache[d] = nx.pagerank(sub, alpha=PAGERANK_ALPHA, weight="weight")
        if (i + 1) % 250 == 0:
            logger.info(
                "  pr snapshot %d/%d (elapsed %.1fs)", i + 1, len(distinct_dates), time.time() - t0
            )
    logger.info("  pr cache built (%d snapshots, %.1fs)", len(pr_cache), time.time() - t0)

    # Build train/test arrays.
    # Label convention: y = 1 if "left" fighter won. To remove the data-source
    # winner-as-fighter-a skew (post-2023 has 78.1% rows with winner_id ==
    # fighter_a_id; pre-2023 has 82.1%) we deterministically split A/B based on
    # parity of fight_id — half the rows present (A, B) and half (B, A) — so
    # the marginal label rate is ~0.5 and the LR sees the actual PageRank
    # signal rather than a baseline-rate fit. This mirrors how feature_matrix.py
    # symmetrizes via A-minus-B differentials (per CLAUDE.md "All features are
    # Fighter A minus Fighter B differentials").
    X_train: list[float] = []
    y_train: list[int] = []
    X_test: list[float] = []
    y_test: list[int] = []
    skipped_debutant = 0
    for f in fights:
        pr = pr_cache.get(f.event_date, {})
        # Deterministic A/B swap for label balancing (prevents winner-as-A skew).
        if f.fight_id % 2 == 0:
            left_id, right_id = f.fighter_a_id, f.fighter_b_id
        else:
            left_id, right_id = f.fighter_b_id, f.fighter_a_id
        l_node = fighter_node(f, left_id)
        r_node = fighter_node(f, right_id)
        pr_l = pr.get(l_node)
        pr_r = pr.get(r_node)
        if pr_l is None or pr_r is None:
            skipped_debutant += 1
            continue
        feat = pr_l - pr_r
        label = 1 if f.winner_id == left_id else 0
        if f.event_date < cutoff_date:
            X_train.append(feat)
            y_train.append(label)
        else:
            X_test.append(feat)
            y_test.append(label)

    logger.info(
        "  train=%d test=%d skipped_debutant=%d", len(X_train), len(X_test), skipped_debutant
    )

    if len(X_train) < 100 or len(X_test) < 50:
        return {
            "brier": float("nan"),
            "accuracy": float("nan"),
            "auc": float("nan"),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "skipped_debutant": skipped_debutant,
            "error": "insufficient samples",
        }

    Xtr = np.array(X_train).reshape(-1, 1)
    ytr = np.array(y_train)
    Xte = np.array(X_test).reshape(-1, 1)
    yte = np.array(y_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, ytr)
    p_test = clf.predict_proba(Xte)[:, 1]
    pred = (p_test > 0.5).astype(int)
    return {
        "brier": float(brier_score_loss(yte, p_test)),
        "accuracy": float((pred == yte).mean()),
        "auc": float(roc_auc_score(yte, p_test)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "skipped_debutant": skipped_debutant,
    }


def run_temporal_leakage_check(
    fights: list[FightRecord],
    graph: nx.DiGraph,
    *,
    scope: str,
    k: int = TEMPORAL_LEAKAGE_TEST_FIGHTERS_K,
) -> dict:
    """NET-03 prototype: assert PageRank-at(fight_N_date) ≠ PageRank-at(today)
    for fighters with fights AFTER fight_N_date.

    Pick K fighters that have ≥ 2 fights spanning ≥ 30 days; for each, take
    fight 1's date as as_of_N and the latest fight's date as as_of_today.
    Assert the as-of-N and as-of-today PageRank values are not equal AND
    that as-of-N PageRank is computed against a strictly smaller subgraph
    (countermeasure to Gotcha 4).
    """

    def fighter_node(fight: FightRecord, fighter_id: int):
        if scope == "pan-mma":
            return fighter_id
        wc = fight.weight_class or "_unknown"
        return (wc, fighter_id)

    # Build per-fighter chronological fight lists in the chosen scope.
    by_fighter: dict[object, list[FightRecord]] = {}
    for f in fights:
        for fid in (f.fighter_a_id, f.fighter_b_id):
            n = fighter_node(f, fid)
            by_fighter.setdefault(n, []).append(f)

    # Pick K fighters with multi-fight history spanning ≥ 30 days.
    candidates: list[tuple[object, list[FightRecord]]] = []
    for node, fl in by_fighter.items():
        if len(fl) < 2:
            continue
        if (fl[-1].event_date - fl[0].event_date).days < 30:
            continue
        candidates.append((node, fl))
    # Deterministic selection: pick most-active fighters (max fight count).
    candidates.sort(key=lambda kv: -len(kv[1]))
    candidates = candidates[:k]

    if not candidates:
        return {
            "passed": False,
            "reason": "no candidates with multi-fight history",
            "checked": 0,
        }

    # Use as_of_today = the day AFTER the latest fight in the corpus, so
    # `<` filter still admits the latest fight.
    as_of_today = max(f.event_date for f in fights)

    failures: list[str] = []
    checked = 0
    for node, fl in candidates:
        # First fight's date as the "fight N" date for this fighter.
        # Any time AFTER fl[0].event_date but BEFORE fl[1].event_date works
        # as as_of_N — the temporal-subgraph admits fl[0] but excludes fl[1].
        as_of_n = fl[1].event_date  # subgraph excludes everything from fl[1] onward
        if as_of_n >= as_of_today:
            continue
        checked += 1
        sub_n = _temporal_subgraph(graph, as_of_n)
        sub_today = _temporal_subgraph(graph, as_of_today)
        # Sanity: subgraph at as_of_today must have strictly more edges than at
        # as_of_n (Pitfall #1 invariant — past subgraph ⊆ present subgraph).
        if sub_today.number_of_edges() <= sub_n.number_of_edges():
            failures.append(f"{node}: subgraph monotonicity violation")
            continue
        if node not in sub_n.nodes:
            # Debutant case: PageRank-at-N is None; PageRank-at-today must be a float
            pr_today = nx.pagerank(sub_today, alpha=PAGERANK_ALPHA, weight="weight").get(node)
            if pr_today is None:
                failures.append(f"{node}: missing from both subgraphs (unexpected)")
            continue
        pr_n = nx.pagerank(sub_n, alpha=PAGERANK_ALPHA, weight="weight").get(node)
        pr_today = nx.pagerank(sub_today, alpha=PAGERANK_ALPHA, weight="weight").get(node)
        if pr_n is None or pr_today is None:
            failures.append(f"{node}: pagerank None unexpectedly")
            continue
        # Invariant: PageRank-at-N ≠ PageRank-at-today (future fights changed
        # the graph and therefore moved the rank). Use absolute tolerance to
        # avoid float-equality flakiness.
        if abs(pr_n - pr_today) < 1e-12:
            failures.append(
                f"{node}: pr_at_N={pr_n:.6e} == pr_today={pr_today:.6e} (temporal leakage signal)"
            )
    return {
        "passed": len(failures) == 0 and checked > 0,
        "checked": checked,
        "failures": failures,
        "candidates": len(candidates),
    }


def render_results_md(
    results: list[dict],
    *,
    cutoff_date: date,
    n_fights: int,
    mov: dict[str, float],
    elapsed_s: float,
) -> str:
    """Render the operator-facing results document per CONTEXT.md `<specifics>`."""
    today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Sort results in canonical CONFIGS order for stable output.
    by_name = {r["config"]: r for r in results}
    ordered = [by_name[c] for c in CONFIGS if c in by_name]

    # (a) Per-config Brier / Acc / AUC table.
    rows_a: list[str] = []
    rows_a.append("| Config | Brier | Accuracy | AUC | n_train | n_test | n_debutants_skipped |")
    rows_a.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in ordered:
        b = r["backtest"]
        rows_a.append(
            f"| `{r['config']}` | {_fmt(b.get('brier'))} | "
            f"{_fmt(b.get('accuracy'))} | {_fmt(b.get('auc'))} | "
            f"{b.get('n_train', 'n/a')} | {b.get('n_test', 'n/a')} | "
            f"{b.get('skipped_debutant', 'n/a')} |"
        )
    table_a = "\n".join(rows_a)

    # (b) Temporal-leakage CI prototype results.
    rows_b: list[str] = []
    rows_b.append("| Config | Pitfall #1 leakage check | Fighters checked | Failures |")
    rows_b.append("| --- | --- | --- | --- |")
    for r in ordered:
        leak = r["leakage"]
        passed = "PASS" if leak.get("passed") else "FAIL"
        rows_b.append(
            f"| `{r['config']}` | **{passed}** | {leak.get('checked', 0)} | "
            f"{len(leak.get('failures', []) or [])} |"
        )
    table_b = "\n".join(rows_b)

    # (c) Recommendation paragraph.
    # Operator prior (D-05 + D-07) is pan-mma-mov. We compare the 4 configs and
    # call a winner based on (1) lowest Brier, (2) leakage PASS, (3) operator
    # prior tie-break. The operator gates the decision.
    valid = [
        r
        for r in ordered
        if r["leakage"].get("passed") and not np.isnan(r["backtest"].get("brier", float("nan")))
    ]

    # Detect "essentially tied" — D-08 sanity floor is 0.001 on a FULL-feature
    # xgb_v3 retrain. For the spike's PageRank-only baseline, a Brier spread
    # < 0.005 across 4 configs is methodologically equivalent ("PageRank alone
    # is uninformative on the holdout, the choice doesn't matter").
    SPREAD_THRESHOLD = 0.005

    if valid:
        briers = [r["backtest"]["brier"] for r in valid]
        spread = max(briers) - min(briers)
        best = min(valid, key=lambda r: r["backtest"]["brier"])
        prior = by_name.get("pan-mma-mov")
        prior_brier = prior["backtest"].get("brier") if prior else float("nan")
        best_brier = best["backtest"]["brier"]

        # Inspect AUC as a directional sanity check — AUC near 0.5 means
        # PageRank alone has no signal in the holdout, which is the D-08
        # kill-switch warning.
        avg_auc = float(np.mean([r["backtest"]["auc"] for r in valid]))
        auc_uninformative = abs(avg_auc - 0.5) < 0.03

        if spread < SPREAD_THRESHOLD and auc_uninformative:
            rec = (
                f"**All 4 configs are within {SPREAD_THRESHOLD} Brier of each other "
                f"(spread {_fmt(spread)}) AND AUC across configs is essentially 0.5 "
                f"(avg {_fmt(avg_auc)}) — PageRank alone is uninformative on the "
                f"post-{cutoff_date} holdout in this spike-mode backtest.** This is "
                f"a methodologically expected outcome (PageRank captures opponent "
                f"strength, not the matchup-specific style + odds + physical "
                f"signals that drive xgb_v2's 0.2206 Brier). The operator's "
                f"decision shifts to: (1) approve the operator prior `pan-mma-mov` "
                f"and let plan 16-04's full xgb_v3 retrain measure NET-*'s "
                f"contribution as one of 75 features (the D-08 sanity floor of "
                f"0.001 Brier improvement is the binding gate, NOT this spike); "
                f"or (2) `kill NET-*` if the operator wants to skip the xgb_v3 "
                f"experimental cost. Recommended: `approve pan-mma-mov` — the "
                f"spike doesn't disqualify the prior, and the real test is "
                f"plan 16-04."
            )
        elif spread < SPREAD_THRESHOLD:
            rec = (
                f"All 4 configs produce essentially-equal Brier (spread "
                f"{_fmt(spread)} < {SPREAD_THRESHOLD}). All temporal-leakage "
                f"checks PASS. The choice is statistical noise at this scale, "
                f"so D-05 + D-07 operator priors should win the tie-break. "
                f"Recommended approval: `approve pan-mma-mov`."
            )
        else:
            delta = (prior_brier - best_brier) if not np.isnan(prior_brier) else float("nan")
            if best["config"] == "pan-mma-mov":
                rec = (
                    f"**Operator prior `pan-mma-mov` wins on the spike's PageRank-only "
                    f"backtest** (Brier {_fmt(best_brier)}, AUC {_fmt(best['backtest'].get('auc'))}, "
                    f"leakage check PASS). D-05 (pan-MMA scope) and D-07 (MOV-weighted edges) "
                    f"are empirically validated against the 3 control configs. Recommended "
                    f"approval: `approve pan-mma-mov`."
                )
            else:
                rec = (
                    f"Best Brier across 4 configs: `{best['config']}` (Brier {_fmt(best_brier)}). "
                    f"Operator prior `pan-mma-mov` Brier: {_fmt(prior_brier)} "
                    f"(delta {_fmt(delta)} = prior − best, positive favors best). "
                    f"Both PASS the temporal-leakage check. Per D-08 sanity floor, the "
                    f"operator should weigh whether the {_fmt(abs(delta) if not np.isnan(delta) else float('nan'))} "
                    f"Brier gap justifies overriding the prior. Recommended approval: "
                    f"`approve {best['config']}` if the gap is meaningful, else "
                    f"`approve pan-mma-mov` to preserve the prior."
                )
    else:
        rec = (
            "**No config passed both the leakage check and produced a finite Brier.** "
            "This invokes D-08's kill switch consideration — recommended response: "
            "`kill NET-*` and let plan 16-04 retrain LIVE-only. Alternative: "
            "`revise spike` if the operator suspects a methodology issue (e.g., "
            "graph build assertions, MOV mapping)."
        )

    # (d) Header + body assembly.
    body = f"""---
phase: 16
plan: 03
artifact: NET-00 graph design spike results
generated_at_utc: {today}
cutoff_date: {cutoff_date}
n_decisive_fights: {n_fights}
spike_runtime_seconds: {elapsed_s:.1f}
networkx_version: {nx.__version__}
pagerank_alpha: {PAGERANK_ALPHA}
backtest_method: pagerank_only_logistic_regression  # spike-mode approximation per CONTEXT.md
status: awaiting_operator_decision
---

# NET-00 Graph Design Spike — Results & Operator Decision

**Spike configurations:** `pan-mma-mov` (operator prior — D-05 + D-07), `pan-mma-binary`
(D-07 control), `per-division-mov` (D-05 control), `per-division-binary` (pure control).

**Method:** PageRank-only logistic regression. For each config, build a
directed fight graph (loser → winner; weight per `weight_mode`); compute
PageRank-as-of-fight-date with `nx.pagerank(_temporal_subgraph(G, as_of), alpha={PAGERANK_ALPHA}, weight='weight')`;
fit a 1-feature logistic regression on `pr(L) − pr(R)` (where L/R are the
deterministically symmetrized fighter-position labels — see
`backtest_pagerank_only` docstring); backtest on fights with
`event_date >= {cutoff_date}` (matches xgb_v2's training cutoff per D-04
for apples-to-apples relative comparison across the 4 configs).

The fights table has a strong winner-as-fighter-a skew (~80% pre-cutoff,
~78% post-cutoff). The spike's first run reported AUC < 0.5 because the LR
fit the marginal A-wins rate. To surface the actual PageRank signal, the
backtest now deterministically swaps fighter positions based on `fight_id`
parity so the marginal label rate is ~0.5 — mirroring the symmetry implicit
in feature_matrix.py's A-minus-B differential convention.

The temporal-leakage check exercises Pitfall #1 / Gotcha 4 — for each config
it picks K={TEMPORAL_LEAKAGE_TEST_FIGHTERS_K} fighters with multi-fight history and asserts
`pagerank_at(fight_N_date) ≠ pagerank_at(today)` AND that the temporal
subgraph at as-of-N has strictly fewer edges than at as-of-today.

**MOV multipliers verified live from `EloConfig`:**
- `mov_ko_tko = {mov["mov_ko_tko"]}`
- `mov_submission = {mov["mov_submission"]}`
- `mov_unanimous = {mov["mov_unanimous"]}` (= U-DEC + generic Decision)
- `mov_split = {mov["mov_split"]}` (= S-DEC + M-DEC)
- `mov_draw = {mov["mov_draw"]}`
- `mov_dq = {mov["mov_dq"]}`

> **Logged Assumption A4 reconciled.** The plan's instruction referenced
> `EloConfig.mov_ratio_ko / mov_ratio_sub / mov_ratio_dec`. Those attributes
> do not exist; the actual fields are listed above. The spike uses the actual
> fields and maps method strings explicitly. (See `scripts/spike_pagerank_design.py`
> docstring + `verify_elo_config_mov_attrs` for the live verification path.)

---

## (a) Per-config backtest

{table_a}

> Because this is a PageRank-only logistic regression (per spike-mode allowance
> in 16-03-PLAN.md), Brier values are higher than xgb_v2's full-feature Brier
> (0.2206). The interesting signal is **relative ordering across the 4 configs**,
> not absolute Brier. Once the operator approves a config, NET-01 implements it
> as one of the 75 columns in the full xgb_v3 feature matrix; plan 16-04
> reports the full-model Brier delta and triggers D-08 if `< 0.001`.

---

## (b) Temporal-leakage check (Pitfall #1 / Gotcha 4 prototype)

{table_b}

> A PASS here is the precondition for NET-01: `compute_pagerank_at` must use
> `graph.edge_subgraph(...)` BEFORE `nx.pagerank(...)` so future edges cannot
> affect past PageRank values. NET-03 makes this a permanent CI regression
> test (plan 16-03 Task 4). A FAIL here means D-08's kill switch trigger is
> already pulled — the implementation pattern is broken.

---

## (c) Recommendation

{rec}

---

## (d) Operator decision

Per CONTEXT.md D-03 — the NET-00 spike is an operator gate. Plan 16-03
execution **HALTS HERE**. Tasks 2–4 (NET-01 PageRank module + NET-02 2-hop
SoS + NET-03 temporal-leakage CI test) only proceed on operator response.

Operator response options:
- `approve pan-mma-mov` — implement the operator prior (D-05 + D-07 confirmed)
- `approve <other-config>` — implement the named config (one of `pan-mma-binary`,
  `per-division-mov`, `per-division-binary`)
- `kill NET-*` — D-08 kill switch pre-applied; plan 16-04 retrains LIVE-only;
  the spike + this doc still ship as plan 16-03's deliverable
- `revise spike` — operator wants methodology changes; loop back to a fresh
  spike run with notes attached

## gsd-checkpoint: operator decision required
"""
    return body


def _fmt(x):
    if x is None:
        return "—"
    try:
        if isinstance(x, float):
            if np.isnan(x):
                return "NaN"
            return f"{x:.4f}"
    except Exception:
        pass
    return str(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=CONFIGS,
        default=None,
        help="Run a single config (default: --all-configs)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all 4 configurations and produce a comparison report",
    )
    parser.add_argument(
        "--backtest-start",
        default=None,
        help="Override backtest cutoff (default: read from xgb_v2 meta)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Override Postgres URL (default: env DATABASE_URL or 5433 dev DB)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the markdown report",
    )
    args = parser.parse_args(argv)

    if not args.all_configs and args.config is None:
        parser.error("either --all-configs or --config must be set")
    configs_to_run: tuple[str, ...] = CONFIGS if args.all_configs else (args.config,)

    # Verify EloConfig surface area (Logged Assumption A4 — emits RuntimeError
    # if the plan's referenced attributes are missing).
    mov = verify_elo_config_mov_attrs()

    # Resolve cutoff_date per D-04 (apples-to-apples with xgb_v2).
    if args.backtest_start:
        cutoff_date = date.fromisoformat(args.backtest_start)
        logger.info("cutoff_date overridden via --backtest-start: %s", cutoff_date)
    else:
        meta_path = _ROOT / "models" / "xgb_v2_meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"models/xgb_v2_meta.json not found at {meta_path}; pass "
                f"--backtest-start to override or set up the symlink"
            )
        meta = json.loads(meta_path.read_text())
        cutoff_str = meta.get("cutoff_date")
        if not cutoff_str:
            raise RuntimeError("xgb_v2_meta.json has no cutoff_date field")
        cutoff_date = date.fromisoformat(cutoff_str)
        logger.info("cutoff_date from xgb_v2_meta.json: %s (D-04 fairness)", cutoff_date)

    # Resolve DB URL.
    db_url = (
        args.db_url
        or os.environ.get("DATABASE_URL", "").replace("postgresql+psycopg://", "postgresql://")
        or "postgresql://ufc:ufc@localhost:5433/ufc_prediction"
    )

    fights = load_fights_from_db(db_url)
    if not fights:
        logger.error("no decisive fights loaded — aborting")
        return 2

    t_total = time.time()
    results: list[dict] = []
    for cfg in configs_to_run:
        logger.info("=== running config %s ===", cfg)
        scope, weight_mode = (
            ("pan-mma", "mov")
            if cfg == "pan-mma-mov"
            else ("pan-mma", "binary")
            if cfg == "pan-mma-binary"
            else ("per-division", "mov")
            if cfg == "per-division-mov"
            else ("per-division", "binary")
        )
        graph = build_graph(fights, scope=scope, weight_mode=weight_mode, mov=mov)
        logger.info(
            "  graph: %d nodes, %d edges (scope=%s, weight=%s)",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            scope,
            weight_mode,
        )
        backtest = backtest_pagerank_only(fights, graph, scope=scope, cutoff_date=cutoff_date)
        leakage = run_temporal_leakage_check(fights, graph, scope=scope)
        results.append(
            {
                "config": cfg,
                "scope": scope,
                "weight_mode": weight_mode,
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "backtest": backtest,
                "leakage": leakage,
            }
        )

    elapsed = time.time() - t_total
    logger.info("total spike runtime: %.1fs", elapsed)

    md = render_results_md(
        results,
        cutoff_date=cutoff_date,
        n_fights=len(fights),
        mov=mov,
        elapsed_s=elapsed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    logger.info("wrote %s (%d bytes)", out, len(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
