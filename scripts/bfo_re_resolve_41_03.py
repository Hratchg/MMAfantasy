"""Plan 41-03 Task 2: Full re-resolve backfill driver.

Re-runs BFOOddsIngester.ingest_all() against the existing data/bfo/ CSVs
with the patched canonical-database_id-first matcher logic (Plan 41-02).
Captures IngestSummary canonical-vs-fuzzy split + AUDIT-01 MID/END SHAs.

UPSERT semantics ensure existing-correct rows stay byte-identical;
only anomaly-affected rows update.

Output: tee'd transcript to 41-BACKFILL-LOG.txt
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Paths
LOG_PATH = Path(".planning/phases/41-bfo-disambiguation-anomaly-resolution/41-BACKFILL-LOG.txt")
XGB_V2 = Path("models/xgb_v2.joblib")
META_V2 = Path("models/meta/meta_v2.joblib")
DATA_FOLDER = Path("data/bfo")

CANONICAL_XGB_V2_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
CANONICAL_META_V2_SHA = "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class TeeLogger:
    """Capture all writes to BOTH stdout AND the log file."""

    def __init__(self, log_path: Path):
        self._fh = log_path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        sys.stdout.write(msg)
        sys.stdout.flush()
        self._fh.write(msg)
        self._fh.flush()

    def line(self, msg: str = "") -> None:
        self.write(msg + "\n")

    def close(self) -> None:
        self._fh.close()


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(LOG_PATH)

    start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    # ── Header: start timestamp + command + START SHAs ───────────────────
    tee.line("=" * 78)
    tee.line("Plan 41-03 Task 2 — BFO Full Re-Resolve Backfill")
    tee.line("=" * 78)
    tee.line(f"started_iso={start_iso}")
    tee.line(f"command=python /tmp/run_41_03_backfill.py")
    tee.line(f"data_folder={DATA_FOLDER}")
    tee.line(f"mode=re-ingest existing CSV through patched BFOOddsIngester (Plan 41-02 fix)")
    tee.line(f"year_range=2010-2026 (operator decision: full re-resolve, all years)")
    tee.line(f"UPSERT_semantics=existing-correct rows untouched; anomaly rows update")
    tee.line("")

    # AUDIT-01 START SHAs
    xgb_start_sha = sha256_of(XGB_V2)
    meta_start_sha = sha256_of(META_V2)
    tee.line(f"AUDIT-01 START xgb_v2 SHA = {xgb_start_sha}")
    tee.line(f"AUDIT-01 START meta_v2 SHA = {meta_start_sha}")
    if xgb_start_sha != CANONICAL_XGB_V2_SHA:
        tee.line(f"FATAL: xgb_v2 SHA does NOT match canonical anchor")
        tee.close()
        return 1
    if meta_start_sha != CANONICAL_META_V2_SHA:
        tee.line(f"FATAL: meta_v2 SHA does NOT match canonical anchor")
        tee.close()
        return 1
    tee.line("AUDIT-01 START: PASS (both SHAs byte-identical to PROJECT.md canonical anchors)")
    tee.line("")

    # ── Pre-ingest coverage probe (overall + anomaly years) ──────────────
    from sqlalchemy import Integer, and_, case, func
    from sqlalchemy.orm import aliased
    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fight_odds import FightOdds

    def per_year_coverage(session):
        odds_a = aliased(FightOdds)
        odds_b = aliased(FightOdds)
        year_expr = func.cast(func.extract("year", Event.date), Integer)
        populated_expr = case(
            (and_(odds_a.closing_implied_prob.isnot(None),
                  odds_b.closing_implied_prob.isnot(None)), 1),
            else_=0,
        )
        rows = (
            session.query(year_expr.label("yr"),
                          func.count(Fight.id).label("total"),
                          func.coalesce(func.sum(populated_expr), 0).label("populated"))
            .join(Event, Fight.event_id == Event.id)
            .outerjoin(odds_a, and_(odds_a.fight_id == Fight.id, odds_a.fighter_id == Fight.fighter_a_id))
            .outerjoin(odds_b, and_(odds_b.fight_id == Fight.id, odds_b.fighter_id == Fight.fighter_b_id))
            .filter(year_expr.in_(list(range(2010, 2027))))
            .group_by(year_expr).order_by(year_expr.asc()).all()
        )
        return {int(yr): (int(t), int(p)) for yr, t, p in rows}

    s = SessionLocal()
    try:
        pre_per_year = per_year_coverage(s)
    finally:
        s.close()
    pre_total = sum(t for t, _ in pre_per_year.values())
    pre_pop = sum(p for _, p in pre_per_year.values())
    tee.line(f"Pre-ingest coverage probe:")
    tee.line(f"  overall: {pre_pop}/{pre_total} = {round(100.0*pre_pop/pre_total, 4)}%")
    for yr in (2011, 2013, 2019, 2020):
        t, p = pre_per_year.get(yr, (0, 0))
        tee.line(f"  {yr} (ANOMALY): {p}/{t} = {round(100.0*p/t if t else 0, 4)}%")
    tee.line("")

    # ── Configure Python logging to capture warnings into the tee log ─────
    class TeeLogHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tee.line(f"[log:{record.levelname}] {msg}")
            except Exception:
                pass

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    log_handler = TeeLogHandler()
    log_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root.addHandler(log_handler)

    # ── Run BFOOddsIngester.ingest_all() against existing CSV ────────────
    from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

    tee.line("Starting BFOOddsIngester.ingest_all() ...")
    tee.line(f"  data_folder = {DATA_FOLDER}")
    tee.line(f"  patched code = src/ufc_prediction/scraper/bfo_ingest.py (Plan 41-02 fix)")
    tee.line("")

    ingest_t0 = time.monotonic()
    session = SessionLocal()
    error_count = 0
    summary = None
    try:
        ingester = BFOOddsIngester(session, data_folder=DATA_FOLDER)
        summary = ingester.ingest_all()
    except Exception as exc:
        error_count += 1
        tee.line(f"FATAL: ingest_all() raised {type(exc).__name__}: {exc}")
        session.rollback()
        session.close()
        root.removeHandler(log_handler)
        tee.line("")
        tee.line(f"end_iso={datetime.now(timezone.utc).isoformat()}")
        tee.close()
        return 1
    finally:
        if summary is not None:
            session.close()
    ingest_duration = time.monotonic() - ingest_t0

    root.removeHandler(log_handler)

    # ── IngestSummary report ─────────────────────────────────────────────
    tee.line("")
    tee.line("─" * 78)
    tee.line("IngestSummary (post-ingest counters):")
    tee.line("─" * 78)
    tee.line(f"  bfo_fights_scanned         = {summary.bfo_fights_scanned}")
    tee.line(f"  fights_matched             = {summary.fights_matched}")
    tee.line(f"  fighters_matched           = {summary.fighters_matched}")
    tee.line(f"  fighters_matched_canonical = {summary.fighters_matched_canonical}  (Plan 41-02 canonical-path)")
    tee.line(f"  fighters_matched_fuzzy     = {summary.fighters_matched_fuzzy}      (fuzzy fallback)")
    tee.line(f"  fighters_unmatched         = {summary.fighters_unmatched}")
    tee.line(f"  rows_upserted              = {summary.rows_upserted}")
    tee.line(f"  rows_skipped_bad_range     = {summary.rows_skipped_bad_range}")
    tee.line(f"  unmatched_bfo_names count  = {len(summary.unmatched_bfo_names)}")
    tee.line(f"  coverage_pct (bfo-scan)    = {round(100.0*summary.coverage_pct, 4)}%")
    tee.line(f"  ingest_duration_s          = {round(ingest_duration, 2)}")
    tee.line("")

    # Compute canonical-vs-fuzzy share
    matched = summary.fighters_matched
    if matched > 0:
        canon_pct = round(100.0 * summary.fighters_matched_canonical / matched, 2)
        fuzzy_pct = round(100.0 * summary.fighters_matched_fuzzy / matched, 2)
        tee.line(f"Canonical-path share: {canon_pct}%  |  Fuzzy-fallback share: {fuzzy_pct}%")
    tee.line("")

    # Top-20 unmatched samples for forensic trail
    if summary.unmatched_bfo_names:
        sample_unmatched = summary.unmatched_bfo_names[:20]
        tee.line(f"Sample of first {len(sample_unmatched)} unmatched BFO names (forensic trail):")
        for nm in sample_unmatched:
            tee.line(f"  - {nm!r}")
        tee.line("")

    # ── Post-ingest coverage probe (delta vs pre) ─────────────────────────
    s = SessionLocal()
    try:
        post_per_year = per_year_coverage(s)
    finally:
        s.close()
    post_total = sum(t for t, _ in post_per_year.values())
    post_pop = sum(p for _, p in post_per_year.values())
    tee.line("Per-year populated-count delta (pre → post):")
    tee.line("Year | Total | Pre-populated | Post-populated | Delta | Coverage pre% → post% | Delta pp | Anomaly?")
    tee.line("-----+-------+---------------+----------------+-------+-----------------------+----------+---------")
    anomaly_set = {2011, 2013, 2019, 2020}
    for yr in sorted(set(pre_per_year) | set(post_per_year)):
        t_pre, p_pre = pre_per_year.get(yr, (0, 0))
        t_post, p_post = post_per_year.get(yr, (0, 0))
        # totals stay constant (Event table didn't change); we'll show t_post
        pre_pct = round(100.0*p_pre/t_pre, 2) if t_pre else 0.0
        post_pct = round(100.0*p_post/t_post, 2) if t_post else 0.0
        delta = p_post - p_pre
        delta_pp = round(post_pct - pre_pct, 2)
        flag = "ANOMALY" if yr in anomaly_set else ""
        tee.line(f"{yr} | {t_post:>5} | {p_pre:>13} | {p_post:>14} | {delta:>+5} | {pre_pct:>7.2f}% → {post_pct:>6.2f}% | {delta_pp:>+7.2f} | {flag}")
    overall_pre_pct = round(100.0*pre_pop/pre_total, 2) if pre_total else 0.0
    overall_post_pct = round(100.0*post_pop/post_total, 2) if post_total else 0.0
    tee.line(f"OVERALL: {pre_pop}/{pre_total} ({overall_pre_pct}%) → {post_pop}/{post_total} ({overall_post_pct}%)  delta_pp={round(overall_post_pct-overall_pre_pct,2):+.2f}")
    tee.line("")

    # ── AUDIT-01 END SHAs ─────────────────────────────────────────────────
    xgb_end_sha = sha256_of(XGB_V2)
    meta_end_sha = sha256_of(META_V2)
    tee.line(f"AUDIT-01 END xgb_v2 SHA  = {xgb_end_sha}")
    tee.line(f"AUDIT-01 END meta_v2 SHA = {meta_end_sha}")
    halt = False
    if xgb_end_sha != CANONICAL_XGB_V2_SHA:
        tee.line("HALT: xgb_v2 SHA drifted during ingest!")
        halt = True
    if meta_end_sha != CANONICAL_META_V2_SHA:
        tee.line("HALT: meta_v2 SHA drifted during ingest!")
        halt = True
    if not halt:
        tee.line("AUDIT-01 END: PASS (xgb_v2 + meta_v2 byte-identical to START)")
    tee.line("")

    # ── Write mid-checkpoint SHA files for plan artifact spec ─────────────
    Path(".planning/phases/41-bfo-disambiguation-anomaly-resolution/41-XGB-V2-SHA-PHASE-41-MID.txt").write_text(xgb_end_sha + "\n")
    Path(".planning/phases/41-bfo-disambiguation-anomaly-resolution/41-META-V2-SHA-PHASE-41-MID.txt").write_text(meta_end_sha + "\n")

    # ── Footer ────────────────────────────────────────────────────────────
    end_iso = datetime.now(timezone.utc).isoformat()
    total_duration = time.monotonic() - t0
    tee.line("=" * 78)
    tee.line("FOOTER")
    tee.line("=" * 78)
    tee.line(f"started_iso={start_iso}")
    tee.line(f"end_iso={end_iso}")
    tee.line(f"total_duration_s={round(total_duration, 2)}")
    tee.line(f"rows_scanned={summary.bfo_fights_scanned}")
    tee.line(f"rows_updated={summary.rows_upserted}  (UPSERT — includes existing-correct no-op rewrites)")
    tee.line(f"rows_unchanged={2*summary.bfo_fights_scanned - summary.rows_upserted}  (BFO fights NOT resolved + ingested)")
    tee.line(f"error_count={error_count}")
    tee.line(f"xgb_v2_sha_at_end={xgb_end_sha}")
    tee.line(f"meta_v2_sha_at_end={meta_end_sha}")
    tee.line(f"audit_01_verdict={'HALT' if halt else 'PASS'}")
    tee.line(f"backfill_status=complete")
    tee.line("")
    tee.close()
    return 1 if halt else 0


if __name__ == "__main__":
    sys.exit(main())
