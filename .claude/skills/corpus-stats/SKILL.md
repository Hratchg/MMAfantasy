---
name: corpus-stats
description: Report MMAfantasy Postgres corpus stats — fight/fighter/odds/event counts, event-source distribution + 1.95x inflation factor, odds coverage (new vs old fights), and latest events. Use when checking corpus state before/after ingest, scrape, elo/features recompute, or a retrain. Runs read-only queries via the mmafantasy-db-1 container (host has no psql).
---

# corpus-stats

Run the bundled query script and summarize the corpus state.

```bash
bash "$CLAUDE_PROJECT_DIR/.claude/skills/corpus-stats/corpus_stats.sh"
```

(If `$CLAUDE_PROJECT_DIR` is unset, use the repo root path to `.claude/skills/corpus-stats/corpus_stats.sh`.)

It prints: overall counts; event `source` distribution + the all-source-vs-ufcstats **inflation factor** (frozen `xgb_v2` trained on the inflated corpus, current pipeline dedups to `source='ufcstats'`); odds coverage split by new (≥2026-05-01) vs old fights; and the latest 8 events.

After running, give the user a 3–5 line summary: total fights/odds, the inflation factor, whether the recent slice is odds-complete, and the newest event date (how current the corpus is). Flag anything anomalous (e.g. assembled dedup row count far from ~8.5k, or new fights missing odds).
