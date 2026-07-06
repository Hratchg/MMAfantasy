# UFC Corpus v3.0 Seed — Provenance

**Artifact:** `data/seed/ufc_corpus_v30.dump`
**Generated:** 2026-07-06 (re-baselined — `RETRAIN-V31-01` / `SEED-REBASE-01`)
**Phase:** 88 — HANDOFF-V30-02 DB Dump Packaging (regenerated on the promoted substrate)

> **Re-baseline note (2026-07-06):** this dump was regenerated on the corrected,
> promoted substrate that produced the re-baselined `xgb_v2` (sha256
> `0b0b40…fecd`): the deduplicated `ufcstats` corpus current to 2026-06-27,
> corrected implied-probability odds, BFO odds for the new fights, and
> Sherdog-seeded debutant Elo. A fresh `ufc db seed` from this dump followed by
> `elo compute` / `features compute` / `predict train` reproduces the promoted
> model. The prior v3.0 dump (postgres:16, sha `8661a327…`) reflected the
> pre-promotion corpus.

## Source

| Field | Value |
|-------|-------|
| Container | `mmafantasy-db-1` |
| Image | `postgres:18` |
| Host port | `5433` |
| Database | `ufc_prediction` |
| User | `ufc` |
| Postgres server version | `PostgreSQL 18.4 (Debian 18.4-1.pgdg13+1)` |
| Raw DB size | 116 MB |

## Dump command

```bash
docker exec mmafantasy-db-1 pg_dump \
  --format=custom -Z 9 \
  --no-owner --no-privileges \
  -U ufc -d ufc_prediction \
  > data/seed/ufc_corpus_v30.dump
```

## Sizes

| Field | Value |
|-------|-------|
| Compressed dump size | 11,125,458 bytes (10.61 MB) |
| SHA256 | `72d13c23d699f08d7928a358071e2cad22c24699ac64d4e99f605596562a9dd6` |

## Hosting route

Committed in repo (≤ 30 MB threshold per D-B1).

The compressed dump (10.61 MB) sits well under the 30 MB cutoff, so the
binary lives directly in the repo alongside this provenance file and the
SHA256 sidecar. No external GitHub Release hosting required for v3.0.

## Tables included (12)

Per-table exact row counts in the dump (`SELECT COUNT(*)`). These are exact
counts (not `pg_stat_user_tables` planner estimates) and match the goldens in
`tests/integration/test_db_seed.py`:

```
      relname      |   count
-------------------+------------
 elo_snapshots     |      90642
 round_stats       |      69684
 computed_features |      28816
 fight_odds        |      25812
 fights            |      17011
 fighters          |       6846
 events            |       1881
 fighter_aliases   |        399
 venues            |        174
 referees          |         39
 alembic_version   |          1
 model_runs        |          0
(12 rows)
```

All 12 user tables included per D-A3. `model_runs` is empty by design
(filled in by downstream model-training workflows; not part of the seed
corpus). `alembic_version` carries the migration head stamp so a fresh
restore lands on the same schema revision as the source DB.

## Verify integrity

```bash
cd data/seed
shasum -a 256 -c ufc_corpus_v30.dump.sha256
# Expect: ufc_corpus_v30.dump: OK
```

## Restore

Canonical pg_restore invocation (Plan 88-02's `ufc db seed` wraps this):

```bash
pg_restore \
  --no-owner --no-privileges \
  --clean --if-exists \
  --dbname=$DATABASE_URL \
  data/seed/ufc_corpus_v30.dump
```

See `uv run ufc db seed --help` (Plan 88-02) or `docs/INSTALL.md` step 4
(Plan 88-03) for the operator-facing path.
