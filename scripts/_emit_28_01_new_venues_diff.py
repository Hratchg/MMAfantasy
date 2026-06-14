"""Phase 28 Plan 28-01 Task 1 — emit operator-review diff of new venues.

One-off operator-analysis script. Reads:
- data/venues.csv (current 52 rows)
- events table (DISTINCT location strings + event counts)

Writes:
- .planning/phases/28-referee-venue-ingestion-pipeline/28-NEW-VENUES-DIFF.md

NO writes to production code; NO writes to data/venues.csv (that's Task 3).
NO Nominatim calls (network-free).

Banned imports per Pitfall #1 / Phase 22 Finding 11: nothing under ufc_prediction.ml.*.
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.models.event import Event


VENUES_CSV = Path("data/venues.csv")
OUT_PATH = Path(
    ".planning/phases/28-referee-venue-ingestion-pipeline/28-NEW-VENUES-DIFF.md"
)


def main() -> int:
    # Load existing CSV
    if not VENUES_CSV.exists():
        print(f"FATAL: {VENUES_CSV} missing", file=sys.stderr)
        return 1
    with open(VENUES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
    existing_names = {row["name"] for row in existing_rows}

    # Query DB
    session = SessionLocal()
    try:
        db_rows = (
            session.query(Event.location, func.count(Event.id).label("n"))
            .filter(Event.location.isnot(None))
            .group_by(Event.location)
            .order_by(func.count(Event.id).desc())
            .all()
        )
    finally:
        session.close()
    db_rows = [(loc, n) for loc, n in db_rows]  # materialize

    total_distinct = len(db_rows)
    total_in_csv = len(existing_names)
    missing = [(loc, n) for loc, n in db_rows if loc not in existing_names]
    total_missing = len(missing)
    sum_n_missing = sum(n for _, n in missing)
    total_n_all = sum(n for _, n in db_rows)
    pct_missing = (sum_n_missing / total_n_all * 100.0) if total_n_all else 0.0

    expected_calls = total_missing
    expected_wallclock_min_raw = expected_calls * 1.2 / 60.0

    ts = datetime.now(timezone.utc).isoformat()

    top30 = missing[:30]

    lines: list[str] = []
    lines.append("# Phase 28 — New Venues Diff (Pre-Sweep Operator Review)")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(
        "**Scope:** Distinct events.location strings present in DB but absent from "
        "data/venues.csv."
    )
    lines.append(
        "**D-09 justification:** Operator-confirmed Nominatim sweep gate (CONTEXT "
        "D-09). Supersedes D-06 \"no Nominatim at backfill time\" because D-06 "
        "guards the per-event scrape loop, not a one-off pre-scrape sweep."
    )
    lines.append(
        "**Phase 22 analog:** Same operation as scripts/backfill_venue_geocodes.py, "
        "just on the complete distinct-set instead of operator-curated subset."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Distinct locations in DB | {total_distinct} |")
    lines.append(f"| Currently in venues.csv | {total_in_csv} |")
    lines.append(f"| Missing (to-be-swept) | {total_missing} |")
    lines.append(
        f"| Event-count of missing | {sum_n_missing} ({pct_missing:.1f}% of corpus) |"
    )
    lines.append(f"| Event-count of all | {total_n_all} |")
    lines.append(f"| Expected Nominatim calls | {expected_calls} |")
    lines.append(
        f"| Expected wall-clock | ~{expected_wallclock_min_raw:.1f}min raw + "
        "back-off margin |"
    )
    lines.append("")
    lines.append("## Top 30 Missing Venues (by event count)")
    lines.append("")
    lines.append("| n_events | location |")
    lines.append("|---:|---|")
    for loc, n in top30:
        # Escape pipe characters in location (rare but safe)
        safe_loc = loc.replace("|", "\\|")
        lines.append(f"| {n} | {safe_loc} |")
    lines.append("")
    lines.append("## Full Missing List (markdown table)")
    lines.append("")
    lines.append(
        f"All {total_missing} missing locations (operator can scan / sort). "
        "Sorted by n_events DESC."
    )
    lines.append("")
    lines.append("| n_events | location |")
    lines.append("|---:|---|")
    for loc, n in missing:
        safe_loc = loc.replace("|", "\\|")
        lines.append(f"| {n} | {safe_loc} |")
    lines.append("")
    lines.append("## Full Missing List (CSV block)")
    lines.append("")
    lines.append("(Paste into spreadsheet)")
    lines.append("")
    lines.append("```")
    lines.append("n_events,location")
    for loc, n in missing:
        # CSV-quote location
        loc_q = loc.replace('"', '""')
        lines.append(f'{n},"{loc_q}"')
    lines.append("```")
    lines.append("")
    lines.append("## Operator Decision Gate")
    lines.append("")
    lines.append("Before Task 2 (Nominatim sweep) runs, operator MUST confirm:")
    lines.append(
        '- [ ] D-09 sweep is approved (set NOMINATIM_USER_AGENT = '
        '"ufc-fight-prediction-v23-venues-sweep")'
    )
    lines.append(
        "- [ ] Rate limit 1.2s + back-off discipline preserved (Phase 22 WR-04/WR-05 "
        "carry-forward)"
    )
    lines.append(
        "- [ ] xgb_v2.joblib SHA PREFLIGHT verified (28-XGB-V2-SHA-PREFLIGHT.txt "
        "committed)"
    )
    lines.append("")
    lines.append(
        "After operator confirms in checkpoint Task 2, Task 3 commits the expanded "
        "venues.csv."
    )
    lines.append("")

    body = "\n".join(lines)

    # Atomic write (WR-05 discipline)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, OUT_PATH)

    print(
        f"[diff] wrote {OUT_PATH} — distinct={total_distinct}, in_csv={total_in_csv}, "
        f"missing={total_missing}, missing_event_pct={pct_missing:.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
