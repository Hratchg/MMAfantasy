#!/usr/bin/env python
"""Recompute fight_odds implied probabilities from the stored raw moneylines.

Corrected-corpus re-ingest (2026-07-01): PR #12 fixed the closing-consensus math
(probability-space `devig_closing_range` instead of the ML-midpoint that
corrupted near-even/straddle-zero fights) but only NEW ingests use it — the
historical `fight_odds.closing_implied_prob` / `opening_implied_prob` were
written with the OLD formula, and xgb_v2 was trained on those.

The raw American moneylines (opening_ml, closing_range_min/max_ml) are preserved
per D-03, so we recompute the implied probs in place from them — no re-scrape.
This makes the whole corpus internally consistent with the corrected math, which
is the prerequisite for a gate-validated retrain.

Idempotent. Reports how much each field moved (the skew magnitude). Invalid
moneylines (|ml| < 100) degrade to NULL per the ingest guard — never fabricated.

Usage:
    uv run python scripts/recompute_fight_odds_implied_probs.py [--apply]
Without --apply it is a DRY RUN (reports the shift, writes nothing).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from sqlalchemy import select, update

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.scraper.bfo_math import (
    InvalidMoneylineError,
    devig_closing_range,
    devig_proportional,
)


def _recompute_pair(a: FightOdds, b: FightOdds) -> dict[int, dict[str, float | None]]:
    """Return {fighter_id: {opening_implied_prob, closing_implied_prob}} recomputed
    from raw MLs with the corrected math. Missing/invalid → None (never fabricated)."""
    out = {
        a.fighter_id: {"opening_implied_prob": None, "closing_implied_prob": None},
        b.fighter_id: {"opening_implied_prob": None, "closing_implied_prob": None},
    }
    if a.opening_ml is not None and b.opening_ml is not None:
        try:
            pa, pb = devig_proportional(a.opening_ml, b.opening_ml)
            out[a.fighter_id]["opening_implied_prob"] = pa
            out[b.fighter_id]["opening_implied_prob"] = pb
        except InvalidMoneylineError:
            pass
    if (
        a.closing_range_min_ml is not None
        and a.closing_range_max_ml is not None
        and b.closing_range_min_ml is not None
        and b.closing_range_max_ml is not None
    ):
        try:
            pa, pb = devig_closing_range(
                a.closing_range_min_ml,
                a.closing_range_max_ml,
                b.closing_range_min_ml,
                b.closing_range_max_ml,
            )
            out[a.fighter_id]["closing_implied_prob"] = pa
            out[b.fighter_id]["closing_implied_prob"] = pb
        except InvalidMoneylineError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write updates (default: dry run)")
    args = ap.parse_args()

    session = SessionLocal()
    try:
        rows = list(session.execute(select(FightOdds)).scalars().all())
        by_fight: dict[int, list[FightOdds]] = defaultdict(list)
        for r in rows:
            by_fight[r.fight_id].append(r)

        n_rows_changed = 0
        n_close_changed = 0
        close_deltas: list[float] = []
        max_delta = 0.0
        pending: list[tuple[int, int, float | None, float | None]] = []  # fight,fighter,op,cl

        for fid, pair in by_fight.items():
            if len(pair) != 2:
                continue  # 2-rows-per-fight invariant; skip malformed
            a, b = pair
            new = _recompute_pair(a, b)
            for row in (a, b):
                nv = new[row.fighter_id]
                op_old, cl_old = row.opening_implied_prob, row.closing_implied_prob
                op_new, cl_new = nv["opening_implied_prob"], nv["closing_implied_prob"]
                if cl_old is not None and cl_new is not None:
                    d = abs(cl_new - cl_old)
                    if d > 1e-9:
                        n_close_changed += 1
                        close_deltas.append(d)
                        max_delta = max(max_delta, d)
                changed = (op_old != op_new) or (cl_old != cl_new)
                if changed:
                    n_rows_changed += 1
                    pending.append((row.fight_id, row.fighter_id, op_new, cl_new))

        close_deltas.sort()
        n = len(close_deltas)
        mean_d = sum(close_deltas) / n if n else 0.0
        p50 = close_deltas[n // 2] if n else 0.0
        p99 = close_deltas[int(n * 0.99)] if n else 0.0
        big = sum(1 for d in close_deltas if d > 0.05)
        print(
            f"rows total={len(rows)}  rows_changed={n_rows_changed}  closing_probs_changed={n_close_changed}"
        )
        print(
            f"closing_implied_prob |Δ|: mean={mean_d:.4f} p50={p50:.4f} p99={p99:.4f} "
            f"max={max_delta:.4f}  (|Δ|>0.05: {big} rows — the near-even corrections)"
        )

        if not args.apply:
            print("DRY RUN — no writes. Re-run with --apply to persist.")
            return 0

        for fid, fighter_id, op_new, cl_new in pending:
            session.execute(
                update(FightOdds)
                .where(FightOdds.fight_id == fid, FightOdds.fighter_id == fighter_id)
                .values(opening_implied_prob=op_new, closing_implied_prob=cl_new)
            )
        session.commit()
        print(f"APPLIED — updated {len(pending)} rows.")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
