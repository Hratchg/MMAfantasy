"""Read-only recon for cross-source Event/Fight dedup (Quick 260520-g76).

Investigation only — emits markdown table fragments matching the six sections
of `.planning/phases/28-referee-venue-ingestion-pipeline/28-DEDUP-RECON.md`.

CONSTRAINTS:
  - No UPDATE / DELETE / ALTER / DROP / INSERT — verify with:
        ! grep -Eq '\\b(UPDATE|DELETE|ALTER|DROP|INSERT)\\b' scripts/recon_dedup.py
  - Reuses `SessionLocal` from `ufc_prediction.db.session`; does NOT create a
    fresh SQLAlchemy engine.
  - `session.rollback()` at end as defense-in-depth (nothing is written, but
    explicit).
  - No new dependencies — uses stdlib `re` + simple Jaccard for fuzzy.

KEY DATA-SHAPE FINDINGS (drove the script's structure; documented in the
recon doc):
  1. Event-name fuzzy matching is a no-op: Kaggle event names are placeholder
     `"UFC Event - YYYY-MM-DD"` strings; ufcstats names are descriptive
     (`"UFC Fight Night: Burns vs. Malott"`). Token-overlap is structurally
     zero. Date is the only reliable event-pair signal — and 697/697 distinct
     Kaggle dates align to ufcstats dates (100% same-date counterpart).
  2. Fighter_id pair-matching FAILS across sources: same person has different
     `fighters.id` per source (e.g., "Rashad Evans" is id 916 in
     kaggle-rajeevw and id 4507 in ufcstats). Pair-matching MUST use
     normalized fighter NAME, not fighter_id.
  3. Per-source `fights.source` always agrees with parent `events.source`
     (0 mismatch rows).

Usage:
    PYTHONPATH=src python scripts/recon_dedup.py > /tmp/recon_output.md
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import date as date_type
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from ufc_prediction.db.session import SessionLocal


# ─────────────────────────────────────────────────────────────────
# Section 1 — Per-source fight counts
# ─────────────────────────────────────────────────────────────────


def section_1_per_source_counts(session: Session) -> dict[str, Any]:
    """Total fights grouped by events.source, with and without the
    `winner_id IS NOT NULL` filter that `load_fight_records()` applies today.

    Also reports the count of rows where `fights.source != events.source`
    (anomaly callout per plan).
    """
    rows = session.execute(
        text("""
        SELECT e.source,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE f.winner_id IS NOT NULL) AS with_winner
        FROM fights f
        JOIN events e ON f.event_id = e.id
        GROUP BY e.source
        ORDER BY e.source
    """)
    ).all()
    per_source = [(r[0], r[1], r[2]) for r in rows]

    mismatch_count = session.execute(
        text("""
        SELECT COUNT(*) FROM fights f
        JOIN events e ON f.event_id = e.id
        WHERE f.source != e.source
    """)
    ).scalar_one()

    # Distinct fights by (date, normalized-name-pair-canonical-order) —
    # the proxy for "true unique fight count" after cross-source dedup.
    distinct_unfiltered = session.execute(
        text("""
        SELECT COUNT(DISTINCT (
            e.date,
            LEAST(LOWER(TRIM(fa.name)), LOWER(TRIM(fb.name))),
            GREATEST(LOWER(TRIM(fa.name)), LOWER(TRIM(fb.name)))
        ))
        FROM fights f
        JOIN events e ON f.event_id = e.id
        JOIN fighters fa ON f.fighter_a_id = fa.id
        JOIN fighters fb ON f.fighter_b_id = fb.id
    """)
    ).scalar_one()
    distinct_with_winner = session.execute(
        text("""
        SELECT COUNT(DISTINCT (
            e.date,
            LEAST(LOWER(TRIM(fa.name)), LOWER(TRIM(fb.name))),
            GREATEST(LOWER(TRIM(fa.name)), LOWER(TRIM(fb.name)))
        ))
        FROM fights f
        JOIN events e ON f.event_id = e.id
        JOIN fighters fa ON f.fighter_a_id = fa.id
        JOIN fighters fb ON f.fighter_b_id = fb.id
        WHERE f.winner_id IS NOT NULL
    """)
    ).scalar_one()

    total_unfiltered = sum(r[1] for r in per_source)
    total_with_winner = sum(r[2] for r in per_source)

    return {
        "per_source": per_source,
        "mismatch_count": mismatch_count,
        "distinct_unfiltered": distinct_unfiltered,
        "distinct_with_winner": distinct_with_winner,
        "total_unfiltered": total_unfiltered,
        "total_with_winner": total_with_winner,
    }


# ─────────────────────────────────────────────────────────────────
# Section 2 — Per-Kaggle-event match status (date-based; fuzzy is no-op)
# ─────────────────────────────────────────────────────────────────


def section_2_event_match_buckets(session: Session) -> dict[str, Any]:
    """For each Kaggle event, bucket by whether a same-date ufcstats event
    exists.

    Fuzzy-name match is a structural no-op in this DB: Kaggle event names are
    placeholders ("UFC Event - YYYY-MM-DD"). We still emit the token-overlap
    Jaccard for transparency, but it is uniformly zero against descriptive
    ufcstats names. Therefore the practical buckets reduce to:
      - "exact_date_match" — at least one ufcstats event on same date
      - "no_date_match"    — no ufcstats event on same date (Path-B risk set)

    Returns matched_pairs for downstream Sections 3 + 5.
    """
    # Exact-date matches: (kaggle_event_id, ufcstats_event_id) pairs.
    matched_rows = session.execute(
        text("""
        SELECT ke.id AS k_eid, ke.name AS k_name, ke.source AS k_src,
               ke.date AS d,
               ue.id AS u_eid, ue.name AS u_name
        FROM events ke
        JOIN events ue ON ue.date = ke.date AND ue.source = 'ufcstats'
        WHERE ke.source LIKE 'kaggle%'
        ORDER BY ke.date, ke.source, ke.id, ue.id
    """)
    ).all()

    no_match_rows = session.execute(
        text("""
        SELECT id, name, source, date FROM events ke
        WHERE ke.source LIKE 'kaggle%'
          AND NOT EXISTS (
              SELECT 1 FROM events ue WHERE ue.source = 'ufcstats' AND ue.date = ke.date
          )
        ORDER BY ke.date
    """)
    ).all()

    # Bucket counts by Kaggle source.
    matched_kaggle_event_ids: set[int] = {r[0] for r in matched_rows}
    matched_pairs: list[tuple[int, int]] = [(r[0], r[4]) for r in matched_rows]
    # All Kaggle events grouped by source for bucket-count denominator.
    all_kaggle = session.execute(
        text("""
        SELECT id, source FROM events WHERE source LIKE 'kaggle%'
    """)
    ).all()
    by_src_total: Counter = Counter(r[1] for r in all_kaggle)
    by_src_matched: Counter = Counter()
    for keid in matched_kaggle_event_ids:
        # find source via per-row lookup
        pass
    src_of_keid: dict[int, str] = {r[0]: r[1] for r in all_kaggle}
    by_src_matched = Counter(src_of_keid[k] for k in matched_kaggle_event_ids)
    by_src_no_match = Counter(r[2] for r in no_match_rows)

    # Multi-ufcstats-per-date sanity: how many Kaggle events have >1 ufcstats
    # counterpart on the same date (drives Section 3 fan-out)?
    multi_match_dates = session.execute(
        text("""
        SELECT ke.date, COUNT(DISTINCT ue.id) AS n_u
        FROM events ke
        JOIN events ue ON ue.date = ke.date AND ue.source = 'ufcstats'
        WHERE ke.source LIKE 'kaggle%'
        GROUP BY ke.date
        HAVING COUNT(DISTINCT ue.id) > 1
        ORDER BY ke.date
    """)
    ).all()

    # Sample 10 of each bucket (matched + no-match) for the recon doc.
    matched_sample = matched_rows[:10]
    no_match_sample = no_match_rows[:10]

    return {
        "matched_pairs": matched_pairs,
        "matched_kaggle_event_ids": matched_kaggle_event_ids,
        "kaggle_only_event_ids": [r[0] for r in no_match_rows],
        "by_src_total": dict(by_src_total),
        "by_src_matched": dict(by_src_matched),
        "by_src_no_match": dict(by_src_no_match),
        "matched_sample": matched_sample,
        "no_match_sample": no_match_sample,
        "multi_match_dates": multi_match_dates,
        # No fuzzy bucket separate from exact_date — name-fuzzy is a no-op
        # here. We emit an empty list so section_5 callers can still iterate.
        "fuzzy_pairs": [],
    }


# ─────────────────────────────────────────────────────────────────
# Section 3 — Per-fight pair matching
# ─────────────────────────────────────────────────────────────────


def section_3_fight_pair_match(
    session: Session, *, matched_pairs: list[tuple[int, int]]
) -> dict[str, Any]:
    """For Kaggle fights on dates that have a ufcstats counterpart, match
    individual fights by normalized fighter NAME pair (swap-tolerant).

    Important caveat documented in-doc: fighter_id is NOT consistent across
    sources (e.g., "Rashad Evans" = id 916 in kaggle-rajeevw, 4507 in
    ufcstats), so the canonical pair-key is NAME-based (lowercased, trimmed).
    """
    # Bucket counts using a single SQL pass over Kaggle fights joined to
    # ufcstats fights on (same date, normalized name-pair, swap-tolerant).
    rows = session.execute(
        text("""
        WITH kf AS (
          SELECT f.id AS fid, e.source AS src, e.date AS d,
                 LOWER(TRIM(fa.name)) AS a, LOWER(TRIM(fb.name)) AS b,
                 fa.name AS aname, fb.name AS bname
          FROM fights f
          JOIN events e ON f.event_id = e.id
          JOIN fighters fa ON f.fighter_a_id = fa.id
          JOIN fighters fb ON f.fighter_b_id = fb.id
          WHERE e.source LIKE 'kaggle%'
        ),
        uf AS (
          SELECT f.id AS fid, e.date AS d,
                 LOWER(TRIM(fa.name)) AS a, LOWER(TRIM(fb.name)) AS b
          FROM fights f
          JOIN events e ON f.event_id = e.id
          JOIN fighters fa ON f.fighter_a_id = fa.id
          JOIN fighters fb ON f.fighter_b_id = fb.id
          WHERE e.source = 'ufcstats'
        )
        SELECT kf.src,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE uf.fid IS NOT NULL AND kf.a = uf.a AND kf.b = uf.b) AS exact,
               COUNT(*) FILTER (WHERE uf.fid IS NOT NULL AND kf.a = uf.b AND kf.b = uf.a) AS swapped
        FROM kf
        LEFT JOIN uf ON uf.d = kf.d AND (
          (uf.a = kf.a AND uf.b = kf.b) OR (uf.a = kf.b AND uf.b = kf.a)
        )
        GROUP BY kf.src
        ORDER BY kf.src
    """)
    ).all()

    bucket_counts = []
    for src, total, exact, swapped in rows:
        no_match = total - exact - swapped
        bucket_counts.append(
            {
                "kaggle_source": src,
                "total": total,
                "exact_pair": exact,
                "swapped_pair": swapped,
                "no_pair_match": no_match,
            }
        )

    # 10-row sample of no-pair-match Kaggle fights (these are the alias-
    # mismatch cases — same fight exists in ufcstats but under a different
    # canonical fighter name).
    no_match_sample = session.execute(
        text("""
        WITH kf AS (
          SELECT f.id AS fid, e.source AS src, e.date AS d,
                 LOWER(TRIM(fa.name)) AS a, LOWER(TRIM(fb.name)) AS b,
                 fa.name AS aname, fb.name AS bname
          FROM fights f
          JOIN events e ON f.event_id = e.id
          JOIN fighters fa ON f.fighter_a_id = fa.id
          JOIN fighters fb ON f.fighter_b_id = fb.id
          WHERE e.source LIKE 'kaggle%'
        ),
        uf AS (
          SELECT f.id AS fid, e.date AS d,
                 LOWER(TRIM(fa.name)) AS a, LOWER(TRIM(fb.name)) AS b
          FROM fights f
          JOIN events e ON f.event_id = e.id
          JOIN fighters fa ON f.fighter_a_id = fa.id
          JOIN fighters fb ON f.fighter_b_id = fb.id
          WHERE e.source = 'ufcstats'
        )
        SELECT kf.d AS date, kf.src AS kaggle_source, kf.aname, kf.bname
        FROM kf
        LEFT JOIN uf ON uf.d = kf.d AND (
          (uf.a = kf.a AND uf.b = kf.b) OR (uf.a = kf.b AND uf.b = kf.a)
        )
        WHERE uf.fid IS NULL
        ORDER BY kf.d
        LIMIT 10
    """)
    ).all()

    return {
        "bucket_counts": bucket_counts,
        "no_match_sample": no_match_sample,
        "match_key": "lower(trim(fighter_name)) pair, swap-tolerant, on same date",
        "caveat": (
            "fighter_id is NOT shared across sources — each ingestion path created "
            "separate `fighters` rows per name; pair-match MUST be name-based, not id-based"
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Section 4 — Kaggle-only fights (Path-B risk set)
# ─────────────────────────────────────────────────────────────────


def section_4_kaggle_only_fights(
    session: Session, *, kaggle_only_event_ids: list[int]
) -> dict[str, Any]:
    """Fights from Kaggle events with NO same-date ufcstats counterpart.
    These are the fights that would be DROPPED if Path B (filter at load,
    `Event.source == 'ufcstats'`) is taken.

    If `kaggle_only_event_ids` is empty (which is the case in current DB:
    100% of Kaggle events have same-date ufcstats counterparts), the function
    returns count=0 with an empty histogram + sample.
    """
    if not kaggle_only_event_ids:
        return {
            "count": 0,
            "year_histogram": {},
            "sample": [],
            "note": (
                "Empty set: every Kaggle event has at least one ufcstats event "
                "on the same date. Path B would drop ZERO Kaggle-only events at "
                "the event level — the only risk is per-fight name-mismatches "
                "(see Section 3 no_pair_match buckets)."
            ),
        }

    rows = session.execute(
        text("""
        SELECT f.id, e.date, e.source, fa.name, fb.name
        FROM fights f
        JOIN events e ON f.event_id = e.id
        JOIN fighters fa ON f.fighter_a_id = fa.id
        JOIN fighters fb ON f.fighter_b_id = fb.id
        WHERE f.event_id = ANY(:ids)
        ORDER BY e.date
    """),
        {"ids": kaggle_only_event_ids},
    ).all()

    year_hist: Counter = Counter()
    for _fid, d, _src, _a, _b in rows:
        year_hist[d.year] += 1
    sample = rows[:20]

    return {
        "count": len(rows),
        "year_histogram": dict(sorted(year_hist.items())),
        "sample": sample,
        "note": "",
    }


# ─────────────────────────────────────────────────────────────────
# Section 5 — Fuzzy-match audit (N/A — name-fuzzy is structurally no-op)
# ─────────────────────────────────────────────────────────────────


def section_5_fuzzy_audit_sample(session: Session, *, fuzzy_pairs: list) -> dict[str, Any]:
    """Inspect fuzzy-matched event pairs.

    In this DB, fuzzy name match is a no-op (Kaggle event names are
    placeholder strings without descriptive content). Section 2's "exact-
    date" match is the only signal, so this section reduces to a side-by-
    side display of 20 same-date event-name pairs — useful for the operator
    to eyeball that the date-based join is sound.

    We pre-seed a token-overlap ratio so the operator can see exactly how
    little signal the name field carries.
    """
    import re

    sample_rows = session.execute(
        text("""
        SELECT ke.date, ke.source, ke.name AS kaggle_name, ue.name AS ufcstats_name
        FROM events ke
        JOIN events ue ON ue.date = ke.date AND ue.source = 'ufcstats'
        WHERE ke.source LIKE 'kaggle%'
        ORDER BY ke.date DESC
        LIMIT 20
    """)
    ).all()

    def _tokens(s: str) -> set[str]:
        s = s.lower().replace("ufc ", "").replace("fight night", "")
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return {t for t in s.split() if t and not t.isdigit()}

    audit: list[dict] = []
    for d, src, kname, uname in sample_rows:
        kt, ut = _tokens(kname), _tokens(uname)
        jaccard = (len(kt & ut) / len(kt | ut)) if (kt | ut) else 0.0
        # Manual label: in this DB the kaggle name is always placeholder
        # so the only signal is the date — label "reliable (date-only)".
        label = "reliable (date-only)"
        reasoning = (
            "Kaggle name is placeholder format `UFC Event - YYYY-MM-DD`; "
            "matched purely by date. Token overlap is structurally near-zero."
        )
        audit.append(
            {
                "date": d,
                "kaggle_source": src,
                "kaggle_name": kname,
                "ufcstats_name": uname,
                "jaccard": round(jaccard, 3),
                "label": label,
                "reasoning": reasoning,
            }
        )

    # Confidence distribution summary
    label_counts = Counter(row["label"] for row in audit)

    return {
        "audit": audit,
        "label_counts": dict(label_counts),
        "note": (
            "Fuzzy name match is a structural no-op against this DB: Kaggle "
            "event-name field is a placeholder string. All 20 sampled "
            "matches are date-only, labeled 'reliable (date-only)'."
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Markdown emission
# ─────────────────────────────────────────────────────────────────


def emit_markdown(r1: dict, r2: dict, r3: dict, r4: dict, r5: dict) -> str:
    out: list[str] = []
    out.append("# Section 1 — Per-Source Fight Counts\n")
    out.append("**Table 1a: Fights grouped by `events.source` — unfiltered**\n")
    out.append("| source | total_fights | with_winner_id |")
    out.append("| --- | ---: | ---: |")
    for src, total, with_w in r1["per_source"]:
        out.append(f"| {src} | {total:,} | {with_w:,} |")
    out.append(
        f"| **TOTAL** | **{r1['total_unfiltered']:,}** | **{r1['total_with_winner']:,}** |\n"
    )

    out.append("**Distinct fights by `(date, lowercased-name-pair-canonical-order)`:**")
    out.append(f"- Unfiltered: **{r1['distinct_unfiltered']:,}**")
    out.append(f"- With winner_id: **{r1['distinct_with_winner']:,}**")
    factor = (
        (r1["total_with_winner"] / r1["distinct_with_winner"])
        if r1["distinct_with_winner"]
        else 0.0
    )
    out.append(
        f"- Inflation factor (with winner): {r1['total_with_winner']:,} / {r1['distinct_with_winner']:,} = **{factor:.2f}×**\n"
    )

    out.append(
        f"**Anomaly check:** rows where `fights.source != events.source`: **{r1['mismatch_count']}**\n"
    )

    out.append("# Section 2 — Per-Kaggle-Event Match Status\n")
    out.append("Match keys: same `date`. Name-fuzzy is a structural no-op (Section 5).\n")
    out.append("| kaggle_source | total | matched (same-date ufcstats event) | no-date-match |")
    out.append("| --- | ---: | ---: | ---: |")
    for src, total in r2["by_src_total"].items():
        m = r2["by_src_matched"].get(src, 0)
        nm = r2["by_src_no_match"].get(src, 0)
        out.append(f"| {src} | {total:,} | {m:,} | {nm:,} |")
    out.append("")
    out.append(
        f"Multi-ufcstats-per-date dates (Kaggle event has >1 ufcstats counterpart): **{len(r2['multi_match_dates'])}**"
    )
    for d, n in r2["multi_match_dates"]:
        out.append(f"  - {d.isoformat()}: {n} ufcstats events")
    out.append("")

    out.append("**Sample of matched event pairs (10 rows, date-sorted):**\n")
    out.append("| date | kaggle_source | kaggle_name | ufcstats_name |")
    out.append("| --- | --- | --- | --- |")
    for k_eid, k_name, k_src, d, u_eid, u_name in r2["matched_sample"]:
        out.append(f"| {d.isoformat()} | {k_src} | {k_name} | {u_name} |")
    out.append("")
    if r2["no_match_sample"]:
        out.append("**Sample of no-date-match Kaggle events (10 rows):**\n")
        out.append("| date | kaggle_source | kaggle_name |")
        out.append("| --- | --- | --- |")
        for eid, name, src, d in r2["no_match_sample"]:
            out.append(f"| {d.isoformat()} | {src} | {name} |")
        out.append("")
    else:
        out.append(
            "**No-date-match bucket: empty.** Every Kaggle event has at least one ufcstats event on its date.\n"
        )

    out.append("# Section 3 — Per-Fight Pair Matching For Matched Events\n")
    out.append(
        "Match key: `(LOWER(TRIM(fighter_a.name)), LOWER(TRIM(fighter_b.name)))` pair, swap-tolerant, on same date.\n"
    )
    out.append(f"**Caveat:** {r3['caveat']}\n")
    out.append(
        "| kaggle_source | total_kaggle_fights | exact pair | swapped pair | no_pair_match |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: |")
    grand_total = 0
    grand_matched = 0
    grand_no_match = 0
    for b in r3["bucket_counts"]:
        matched = b["exact_pair"] + b["swapped_pair"]
        pct_matched = (100.0 * matched / b["total"]) if b["total"] else 0.0
        out.append(
            f"| {b['kaggle_source']} | {b['total']:,} | {b['exact_pair']:,} | "
            f"{b['swapped_pair']:,} | {b['no_pair_match']:,} ({100 - pct_matched:.1f}%) |"
        )
        grand_total += b["total"]
        grand_matched += matched
        grand_no_match += b["no_pair_match"]
    out.append(
        f"| **TOTAL** | **{grand_total:,}** | "
        f"**{sum(b['exact_pair'] for b in r3['bucket_counts']):,}** | "
        f"**{sum(b['swapped_pair'] for b in r3['bucket_counts']):,}** | "
        f"**{grand_no_match:,}** |"
    )
    overall_match_pct = (100.0 * grand_matched / grand_total) if grand_total else 0.0
    out.append(
        f"\n**Match rate (exact + swapped):** {grand_matched:,} / {grand_total:,} = **{overall_match_pct:.1f}%**\n"
    )

    if r3["no_match_sample"]:
        out.append(
            "**10-row sample of no-pair-match Kaggle fights (alias-mismatch candidates):**\n"
        )
        out.append("| date | kaggle_source | kaggle_a_name | kaggle_b_name |")
        out.append("| --- | --- | --- | --- |")
        for d, src, aname, bname in r3["no_match_sample"]:
            out.append(f"| {d.isoformat()} | {src} | {aname} | {bname} |")
        out.append("")

    out.append("# Section 4 — Kaggle-Only Fights (Path-B Risk Set)\n")
    out.append(
        "Fights from Kaggle events whose date has NO ufcstats counterpart. These are the fights that Path B (filter `Event.source == 'ufcstats'`) would drop.\n"
    )
    out.append(f"**Count:** {r4['count']:,}\n")
    if r4["note"]:
        out.append(f"_{r4['note']}_\n")
    if r4["year_histogram"]:
        out.append("**Year-bucket histogram:**\n")
        out.append("| year | count |")
        out.append("| --- | ---: |")
        for y, c in r4["year_histogram"].items():
            out.append(f"| {y} | {c:,} |")
        out.append("")
    if r4["sample"]:
        out.append("**20-row sample:**\n")
        out.append("| date | source | fighter_a | fighter_b |")
        out.append("| --- | --- | --- | --- |")
        for fid, d, src, a, b in r4["sample"]:
            out.append(f"| {d.isoformat()} | {src} | {a} | {b} |")
        out.append("")

    out.append("# Section 5 — Fuzzy-Match Audit (20-row Sample)\n")
    out.append(f"_{r5['note']}_\n")
    out.append(
        "| date | kaggle_source | kaggle_name | ufcstats_name | token_jaccard | label | reasoning |"
    )
    out.append("| --- | --- | --- | --- | ---: | --- | --- |")
    for row in r5["audit"]:
        out.append(
            f"| {row['date'].isoformat()} | {row['kaggle_source']} | {row['kaggle_name']} | "
            f"{row['ufcstats_name']} | {row['jaccard']:.3f} | {row['label']} | {row['reasoning']} |"
        )
    out.append("")
    out.append("**Confidence distribution:**")
    for label, n in r5["label_counts"].items():
        out.append(f"- {label}: {n}")
    out.append("")

    return "\n".join(out)


def main() -> int:
    with SessionLocal() as session:
        try:
            r1 = section_1_per_source_counts(session)
            r2 = section_2_event_match_buckets(session)
            r3 = section_3_fight_pair_match(session, matched_pairs=r2["matched_pairs"])
            r4 = section_4_kaggle_only_fights(
                session, kaggle_only_event_ids=r2["kaggle_only_event_ids"]
            )
            r5 = section_5_fuzzy_audit_sample(session, fuzzy_pairs=r2["fuzzy_pairs"])
            sys.stdout.write(emit_markdown(r1, r2, r3, r4, r5))
            sys.stdout.write("\n")
        finally:
            # Defense-in-depth — script is read-only but be explicit.
            session.rollback()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
