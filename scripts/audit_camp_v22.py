#!/usr/bin/env python
"""Phase 20 / CAMP-V22-00 — Sherdog Association field coverage audit.

Operator-driven entry point for the 1-day CAMP audit spike. Mechanically
gates v2.2 CAMP-V22-01..03 scope (Phase 22 + 23) on a 200-fighter stratified
random sample drawn from the 3,375-Sherdog-fighter subset. Emits
``CAMP_00_AUDIT.json`` with a ``scope_recommendation ∈ {full, reduced, drop}``
derived MECHANICALLY from pre-registered D-06 thresholds — no post-hoc
operator interpretation (Pitfall #7 / D-18 carry-forward).

Locked decisions (CONTEXT.md):
  - D-01: 200 fighters, stratified by division × is_active; random_state=42
  - D-02: sibling helper ``parse_association_from_html`` (NOT a mutation
    of ``parse_sherdog_fighter_page``); raw + normalized persisted side-by-side
  - D-03: cache-first fetch policy (``data/sherdog_html_cache/``)
  - D-05: 12-key JSON output schema (11 required D-05 keys + risk_flags)
  - D-06: scope_recommendation tiers (presence/parseable/top30 = 0.30/0.25/0.60
    minimum; 0.70/0.60/0.70 full); mechanical derivation, no slop
  - D-13: Wave-0 RED unit tests cover thresholds + sampling determinism +
    JSON schema BEFORE any live HTTP fetch

Anti-features (BANNED):
  - AF-1: hyperparameter retuning. This is an audit, not a model retrain;
    no ML touches.
  - AF-2: feature engineering. No new feature columns produced.
  - AF-3: model file mutation. ``models/xgb_v2.joblib`` is byte-identical
    (read-only); AUDIT-01 chain verifies SHA at phase end.

Pitfalls mitigated:
  - Pitfall #1 (selector-vs-live-HTML mismatch): operator-saved fixture +
    saved-fixture test in TestParseAssociationFromHtml lands BEFORE this
    script runs against live Sherdog HTML
  - Pitfall #2 (stratification derivation): is_active derived from
    Fight.event_date >= today - timedelta(days=730); rule recorded in
    stratification_breakdown.derivation_rule
  - Pitfall #5 (denominator semantics): present_rate / parseable_rate use
    n_fetched as denominator (NOT sample_size), so fetch failures don't
    artificially deflate the rates
  - Pitfall #7 (post-hoc threshold renegotiation): all 6 thresholds are
    module-top constants asserted at startup; drift halts before any DB
    or HTTP work

D-09(P15) carry-forward: audit never persists models to disk. ``models/
xgb_v2.joblib`` is read-only; AUDIT-01 chain checkpoint runs in Plan 02
Task 4 via ``bash scripts/audit_xgb_v2_sha.sh``.

Usage:
    uv run python scripts/audit_camp_v22.py \\
        --output-path .planning/phases/20-audit-spikes-camp-00-bfo-00/CAMP_00_AUDIT.json

    # Cache-only mode (no fresh fetches; for re-running off pre-warmed cache):
    uv run python scripts/audit_camp_v22.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ── Locked constants (D-01..D-06, D-13) ──────────────────────────────────

SAMPLE_SIZE: int = 200                       # D-01
RANDOM_STATE: int = 42                       # D-01 (v2.x convention)
ACTIVE_WINDOW_DAYS: int = 730                # A2 derivation rule (2 years)

PRESENCE_THRESHOLD_MIN: float = 0.30         # D-06 minimum tier
PRESENCE_THRESHOLD_FULL: float = 0.70        # D-06 full tier
PARSEABLE_THRESHOLD_MIN: float = 0.25        # D-06 minimum tier
PARSEABLE_THRESHOLD_FULL: float = 0.60       # D-06 full tier
TOP30_THRESHOLD_MIN: float = 0.60            # D-06 minimum tier
TOP30_THRESHOLD_FULL: float = 0.70           # D-06 full tier

CACHE_DIR: Path = Path("data/sherdog_html_cache")
ALIAS_SUFFIXES_TO_STRIP: tuple[str, ...] = (" (USA)", " MMA", " Academy", " Gym")
OUTPUT_PATH_DEFAULT: Path = Path(
    ".planning/phases/20-audit-spikes-camp-00-bfo-00/CAMP_00_AUDIT.json"
)

AUDIT_VERSION: str = "v22.00"
HIGH_FAILURE_RATE_THRESHOLD: float = 0.10    # Pitfall #5 risk flag


logger = logging.getLogger(__name__)


# ── Public helpers (exported for tests; underscore-prefixed but stable) ──


def _derive_scope_recommendation(
    present_rate: float,
    parseable_rate: float,
    top30_rate: float,
) -> str:
    """Mechanical D-06 derivation. No operator interpretation.

    Returns one of ``{"full", "reduced", "drop"}``:
      - ``full`` iff all three rates ≥ FULL tier
      - ``reduced`` iff all three rates ≥ MIN tier AND not ``full``
      - ``drop`` otherwise (any rate below its MIN tier)
    """
    if (
        present_rate >= PRESENCE_THRESHOLD_FULL
        and parseable_rate >= PARSEABLE_THRESHOLD_FULL
        and top30_rate >= TOP30_THRESHOLD_FULL
    ):
        return "full"
    if (
        present_rate >= PRESENCE_THRESHOLD_MIN
        and parseable_rate >= PARSEABLE_THRESHOLD_MIN
        and top30_rate >= TOP30_THRESHOLD_MIN
    ):
        return "reduced"
    return "drop"


def _stratified_sample(df, n: int, seed: int):
    """Stratified random sample by ``df["strat_key"]`` of size ``n``.

    Uses ``sklearn.model_selection.train_test_split(stratify=...)`` when
    every stratum has ≥2 members; falls back to per-stratum proportional
    sampling on ``ValueError`` (degenerate single-member strata) per
    RESEARCH.md Pattern 3 caveat.

    Determinism guarantee: same (df, n, seed) returns IDENTICAL fighter_id
    ordering on repeat calls — RANDOM_STATE=42 is the v2.x convention.

    Args:
        df: pandas.DataFrame with at least 'fighter_id' and 'strat_key' columns.
        n: Sample size (≤ len(df)).
        seed: Random seed (RANDOM_STATE=42 for canonical audit).

    Returns:
        pandas.DataFrame of length ``n``, with index reset.
    """
    import pandas as pd  # noqa: PLC0415 — lazy import; keeps test cost low
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    if n >= len(df):
        return df.reset_index(drop=True)

    try:
        sample, _ = train_test_split(
            df,
            train_size=n,
            random_state=seed,
            stratify=df["strat_key"],
        )
        # Sort by index to make output deterministic across sklearn versions
        # (train_test_split shuffles internally; identical seed → identical
        # row set, but row ORDER can drift across sklearn minor versions).
        return sample.sort_index().reset_index(drop=True)
    except ValueError:
        # Degenerate stratum (single-member class) — fall back to per-group
        # proportional sampling per RESEARCH.md Pattern 3 caveat.
        logger.warning(
            "[audit-camp] train_test_split degenerate stratum; "
            "falling back to groupby.sample proportional allocation."
        )
        ratio = n / len(df)
        sampled = (
            df.groupby("strat_key", group_keys=False)
              .apply(
                  lambda g: g.sample(
                      n=max(1, int(round(len(g) * ratio))),
                      random_state=seed,
                  ),
                  include_groups=True,
              )
              .sort_index()
              .reset_index(drop=True)
        )
        # Trim or pad to exactly n if rounding diverged.
        if len(sampled) > n:
            return sampled.iloc[:n].reset_index(drop=True)
        return sampled


def _normalize_association(raw: str | None) -> str | None:
    """Strip ALIAS_SUFFIXES_TO_STRIP suffixes, lowercase, trim.

    Returns ``None`` for ``None`` / empty / whitespace-only input. Suffix
    matching is case-insensitive and right-anchored. Multiple suffixes are
    NOT stripped iteratively (one pass; rare edge case ' (USA) MMA' would
    strip ' MMA' first, leaving ' (USA)' for the next audit run — accepted
    per D-02 simplicity).
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    # Case-insensitive right-anchored suffix removal.
    pattern = (
        r"("
        + "|".join(re.escape(s) for s in ALIAS_SUFFIXES_TO_STRIP)
        + r")$"
    )
    normalized = re.sub(pattern, "", stripped, flags=re.IGNORECASE).strip()
    return normalized.lower() if normalized else None


def _load_or_fetch_html(client, sherdog_url: str) -> str | None:
    """Cache-first fetch per D-03.

    Returns the HTML body (str) or None on fetch failure. Defensive cache_key
    per Security V5: slug component restricted to ``[A-Za-z0-9-]`` to block
    path-traversal-shaped URLs.
    """
    if not sherdog_url:
        return None
    slug = sherdog_url.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9-]+", slug):
        logger.warning(
            "[audit-camp] cache_key rejected (slug=%r); skipping URL.", slug,
        )
        return None
    cache_path = CACHE_DIR / f"{slug}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    # Cache miss — fetch via _safe_fetch_sherdog sentinel-tuple wrapper.
    from ufc_prediction.scraper.sherdog import _safe_fetch_sherdog  # noqa: PLC0415
    _, html, exc = _safe_fetch_sherdog(client, sherdog_url)
    if exc is not None or html is None:
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
    except OSError:
        # Cache-write failure is non-fatal — the fetch already succeeded.
        logger.warning("[audit-camp] cache write failed for %s", slug)
    return html


def _load_fighters_with_strata(session):
    """Load fighters joined with their latest fight + event date.

    Derives:
      - last_weight_class: weight_class of the most recent Fight per fighter
      - is_active: latest Fight.event.date >= today - timedelta(days=730)
      - strat_key: f"{last_weight_class}_{is_active}"

    Returns:
        pandas.DataFrame columns: fighter_id, sherdog_url, last_weight_class,
        is_active, strat_key.
    """
    import pandas as pd  # noqa: PLC0415
    from sqlalchemy import or_, select  # noqa: PLC0415

    from ufc_prediction.models.event import Event  # noqa: PLC0415
    from ufc_prediction.models.fight import Fight  # noqa: PLC0415
    from ufc_prediction.models.fighter import Fighter  # noqa: PLC0415

    today = date.today()
    cutoff = today - timedelta(days=ACTIVE_WINDOW_DAYS)

    # Pull (fighter_id, sherdog_url, weight_class, event_date) for every
    # fight involving each fighter (as either A or B). Use SQL groupby in
    # pandas (cheaper than two DB queries with windows on >100k rows).
    stmt = (
        select(
            Fighter.id.label("fighter_id"),
            Fighter.sherdog_url.label("sherdog_url"),
            Fight.weight_class.label("weight_class"),
            Event.date.label("event_date"),
        )
        .join(
            Fight,
            or_(
                Fight.fighter_a_id == Fighter.id,
                Fight.fighter_b_id == Fighter.id,
            ),
        )
        .join(Event, Fight.event_id == Event.id)
        .where(Fighter.sherdog_url.is_not(None))
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(
            columns=[
                "fighter_id", "sherdog_url",
                "last_weight_class", "is_active", "strat_key",
            ]
        )
    df = pd.DataFrame(rows, columns=[
        "fighter_id", "sherdog_url", "weight_class", "event_date",
    ])
    # Reduce per-fighter: latest fight wins for last_weight_class + is_active.
    df = df.sort_values(["fighter_id", "event_date"], ascending=[True, False])
    latest = (
        df.drop_duplicates(subset=["fighter_id"], keep="first")
          .copy()
          .reset_index(drop=True)
    )
    latest["last_weight_class"] = latest["weight_class"].fillna("unknown")
    latest["is_active"] = (latest["event_date"] >= cutoff).astype(bool)
    latest["strat_key"] = (
        latest["last_weight_class"].astype(str)
        + "_"
        + latest["is_active"].map({True: "active", False: "retired"})
    )
    return latest[[
        "fighter_id", "sherdog_url",
        "last_weight_class", "is_active", "strat_key",
    ]]


def _compute_top30(
    results: list[dict],
) -> tuple[list[dict], float]:
    """Group sampled fighters by normalized association; rank top 30.

    Returns ``(per_camp_top30, top30_coverage_rate)`` where
    ``top30_coverage_rate = sum(fights_at_top30_camps) / sum(all_fights_in_sample)``.

    Each per_camp entry: ``{"normalized_name", "fight_count", "alias_variations"}``.
    ``fight_count`` is the number of SAMPLED fighters whose latest camp is
    this camp (Sherdog only records the current/latest camp per fighter; we
    approximate fight count as fighter count × 1 for the audit). A future
    Phase 22 full-corpus scrape would join Fight rows for true coverage.
    """
    from collections import defaultdict  # noqa: PLC0415

    bucket: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"fight_count": 0, "alias_variations": set()}
    )
    total = 0
    for r in results:
        norm = r.get("normalized_association")
        raw = r.get("raw_association")
        if not norm:
            continue
        bucket[norm]["fight_count"] += 1
        if raw and raw != norm:
            bucket[norm]["alias_variations"].add(raw)
        total += 1

    ranked = sorted(
        bucket.items(),
        key=lambda kv: (-kv[1]["fight_count"], kv[0]),
    )
    top30 = ranked[:30]
    per_camp_top30 = [
        {
            "normalized_name": name,
            "fight_count": data["fight_count"],
            "alias_variations": sorted(data["alias_variations"]),
        }
        for name, data in top30
    ]
    top30_count = sum(c["fight_count"] for c in per_camp_top30)
    top30_coverage_rate = round(top30_count / total, 4) if total > 0 else 0.0
    return per_camp_top30, top30_coverage_rate


def _write_audit_json(
    output_path: Path,
    *,
    sample_size: int,
    n_fetched: int,
    n_fetch_failures: int,
    association_present_rate: float,
    parseable_rate: float,
    top30_coverage_rate: float,
    scope_recommendation: str,
    per_camp_top30: list[dict],
    unparseable_examples: list[dict],
    stratification_breakdown: dict,
    risk_flags: list[str] | None = None,
) -> None:
    """Emit the D-05 schema JSON artifact.

    All 11 D-05 required keys plus ``risk_flags`` are written. The
    ``audit_version`` and ``sample_random_state`` keys are locked at module
    level (AUDIT_VERSION / RANDOM_STATE constants) — callers do NOT supply
    them.
    """
    payload: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "sample_size": sample_size,
        "sample_random_state": RANDOM_STATE,
        "n_fetched": n_fetched,
        "n_fetch_failures": n_fetch_failures,
        "association_present_rate": round(association_present_rate, 4),
        "parseable_rate": round(parseable_rate, 4),
        "top30_coverage_rate": round(top30_coverage_rate, 4),
        "scope_recommendation": scope_recommendation,
        "per_camp_top30": per_camp_top30,
        "unparseable_examples": unparseable_examples,
        "stratification_breakdown": stratification_breakdown,
        "risk_flags": list(risk_flags or []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ── argparse + main() ────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 20 / CAMP-V22-00 — Sherdog Association field audit "
            "(200-fighter stratified sample)."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(OUTPUT_PATH_DEFAULT),
        help="Output path for CAMP_00_AUDIT.json (default: per D-05).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip live fetches; use cache-only mode (cache misses → fetch_failure).",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    logging.basicConfig(level=logging.INFO, format="[audit-camp] %(message)s")

    spike_started = datetime.now(timezone.utc)

    # ── AF startup assertions (D-13: thresholds frozen BEFORE live HTTP) ──
    if SAMPLE_SIZE != 200:
        print(
            f"[audit-camp] FATAL: SAMPLE_SIZE drift: expected 200, got {SAMPLE_SIZE}",
            file=sys.stderr,
        )
        return 2
    if RANDOM_STATE != 42:
        print(
            f"[audit-camp] FATAL: RANDOM_STATE drift: expected 42, got {RANDOM_STATE}",
            file=sys.stderr,
        )
        return 2
    expected_thresholds = {
        "PRESENCE_THRESHOLD_MIN": (PRESENCE_THRESHOLD_MIN, 0.30),
        "PRESENCE_THRESHOLD_FULL": (PRESENCE_THRESHOLD_FULL, 0.70),
        "PARSEABLE_THRESHOLD_MIN": (PARSEABLE_THRESHOLD_MIN, 0.25),
        "PARSEABLE_THRESHOLD_FULL": (PARSEABLE_THRESHOLD_FULL, 0.60),
        "TOP30_THRESHOLD_MIN": (TOP30_THRESHOLD_MIN, 0.60),
        "TOP30_THRESHOLD_FULL": (TOP30_THRESHOLD_FULL, 0.70),
    }
    for name, (got, expected) in expected_thresholds.items():
        if got != expected:
            print(
                f"[audit-camp] FATAL: {name} drift: expected {expected}, got {got}",
                file=sys.stderr,
            )
            return 2
    print(
        "[audit-camp] AF OK: SAMPLE_SIZE=200, RANDOM_STATE=42, thresholds locked."
    )

    # ── Load fighters + strata from DB ────────────────────────────────────
    from ufc_prediction.db.session import SessionLocal  # noqa: PLC0415
    from ufc_prediction.scraper.client import ScraperClient  # noqa: PLC0415
    from ufc_prediction.scraper.sherdog import (  # noqa: PLC0415
        parse_association_from_html,
    )

    print("[audit-camp] Loading fighters with strata derivation from DB...")
    t0 = time.time()
    session = SessionLocal()
    try:
        fighters_df = _load_fighters_with_strata(session)
    finally:
        session.close()
    print(
        f"[audit-camp] Loaded {len(fighters_df)} fighters in "
        f"{time.time() - t0:.1f}s"
    )
    if len(fighters_df) == 0:
        print(
            "[audit-camp] FATAL: no fighters with sherdog_url in DB. "
            "Cannot audit empty corpus.",
            file=sys.stderr,
        )
        return 2

    sample_df = _stratified_sample(
        fighters_df, n=SAMPLE_SIZE, seed=RANDOM_STATE,
    )
    print(f"[audit-camp] Sampled {len(sample_df)} fighters (seed={RANDOM_STATE}).")

    # ── Per-fighter fetch + parse loop ────────────────────────────────────
    client = ScraperClient(delay=1.2, max_retries=3, timeout=30.0)
    results: list[dict] = []
    unparseable_examples: list[dict] = []
    n_fetched = 0
    n_fetch_failures = 0
    n_present = 0
    n_parseable = 0

    print(f"[audit-camp] Begin fetch loop (dry_run={args.dry_run})...")
    for idx, row in sample_df.iterrows():
        url = row.get("sherdog_url")
        if not url:
            n_fetch_failures += 1
            continue
        slug = url.rsplit("/", 1)[-1] if url else ""
        cache_path = CACHE_DIR / f"{slug}.html"
        if args.dry_run and not cache_path.exists():
            # dry-run policy: cache miss → fetch failure (don't go to network).
            n_fetch_failures += 1
            continue
        html = _load_or_fetch_html(client, url)
        if html is None:
            n_fetch_failures += 1
            continue
        n_fetched += 1
        raw_assoc = parse_association_from_html(html)
        norm_assoc = _normalize_association(raw_assoc)
        if raw_assoc is not None:
            n_present += 1
        if norm_assoc is not None:
            n_parseable += 1
        elif raw_assoc is not None:
            # Present but unparseable (post-normalization) — diagnostic seed
            # for Phase 22 alias-normalization table (D-02 / D-05).
            unparseable_examples.append({
                "sherdog_id": slug,
                "raw_field_value": raw_assoc,
                "why": "post-normalization empty",
            })
        results.append({
            "fighter_id": int(row["fighter_id"]),
            "raw_association": raw_assoc,
            "normalized_association": norm_assoc,
        })

    # ── Aggregate rates (Pitfall #5: denom = n_fetched, NOT sample_size) ──
    present_rate = n_present / n_fetched if n_fetched > 0 else 0.0
    parseable_rate = n_parseable / n_fetched if n_fetched > 0 else 0.0
    per_camp_top30, top30_rate = _compute_top30(results)

    scope_recommendation = _derive_scope_recommendation(
        present_rate, parseable_rate, top30_rate,
    )

    # ── Stratification breakdown ──────────────────────────────────────────
    strata_counts: dict[str, int] = {}
    for k in sample_df["strat_key"].tolist():
        strata_counts[k] = strata_counts.get(k, 0) + 1
    stratification_breakdown = {
        "derivation_rule": (
            f"is_active = latest_fight_event_date >= today - "
            f"{ACTIVE_WINDOW_DAYS}d"
        ),
        "strata": strata_counts,
    }

    # ── Risk flags (Pitfall #5) ───────────────────────────────────────────
    risk_flags: list[str] = []
    if n_fetch_failures > HIGH_FAILURE_RATE_THRESHOLD * SAMPLE_SIZE:
        risk_flags.append("high_fetch_failure_rate")
    if present_rate < 0.20 and n_fetch_failures < 0.05 * SAMPLE_SIZE:
        # Pitfall #1 signal: probably a selector mismatch, not a real
        # coverage gap.
        risk_flags.append("possible_selector_mismatch")

    # ── Emit artifact ─────────────────────────────────────────────────────
    _write_audit_json(
        output_path,
        sample_size=SAMPLE_SIZE,
        n_fetched=n_fetched,
        n_fetch_failures=n_fetch_failures,
        association_present_rate=present_rate,
        parseable_rate=parseable_rate,
        top30_coverage_rate=top30_rate,
        scope_recommendation=scope_recommendation,
        per_camp_top30=per_camp_top30,
        unparseable_examples=unparseable_examples,
        stratification_breakdown=stratification_breakdown,
        risk_flags=risk_flags,
    )
    spike_finished = datetime.now(timezone.utc)

    # ── Final 5-line stdout summary (CONTEXT.md Discretion) ───────────────
    print(
        f"\n[audit-camp] Audit complete. Wrote:\n"
        f"  {output_path}\n"
        f"[audit-camp] n_fetched / n_fetch_failures = {n_fetched} / "
        f"{n_fetch_failures}\n"
        f"[audit-camp] association_present_rate = {present_rate:.3f}\n"
        f"[audit-camp] parseable_rate           = {parseable_rate:.3f}\n"
        f"[audit-camp] top30_coverage_rate      = {top30_rate:.3f}\n"
        f"[audit-camp] scope_recommendation     = {scope_recommendation}\n"
        f"[audit-camp] duration                 = "
        f"{(spike_finished - spike_started).total_seconds():.0f}s"
    )
    if risk_flags:
        print(f"[audit-camp] risk_flags = {risk_flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
