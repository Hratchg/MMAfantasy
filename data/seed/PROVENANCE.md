# UFC Corpus v3.0 Seed — Provenance

**Artifact:** `data/seed/ufc_corpus_v30.dump`
**Generated:** 2026-06-10T05:14:32Z
**Phase:** 88 — HANDOFF-V30-02 DB Dump Packaging

## Source

| Field | Value |
|-------|-------|
| Container | `ufc-fight-prediction-db-1` |
| Image | `postgres:16` |
| Host port | `5433` |
| Database | `ufc_prediction` |
| User | `ufc` |
| Postgres server version | `PostgreSQL 16.13 (Debian 16.13-1.pgdg13+1)` |
| Raw DB size | 85 MB |

## Dump command

```bash
docker exec ufc-fight-prediction-db-1 pg_dump \
  --format=custom -Z 9 \
  -U ufc -d ufc_prediction \
  > data/seed/ufc_corpus_v30.dump
```

## Sizes

| Field | Value |
|-------|-------|
| Compressed dump size | 10,926,354 bytes (10.42 MB) |
| SHA256 | `8661a327659d48890636fa8781f8947253abcf9c132a73b05c68872b94c79cdb` |

## Hosting route

Committed in repo (≤ 30 MB threshold per D-B1).

The compressed dump (10.42 MB) sits well under the 30 MB cutoff, so the
binary lives directly in the repo alongside this provenance file and the
SHA256 sidecar. No external GitHub Release hosting required for v3.0.

## Tables included (12)

Per-table exact row counts in the dump (`SELECT COUNT(*)`). NOTE: earlier
versions of this file reported `n_live_tup` from `pg_stat_user_tables`, which is
a planner *estimate* refreshed only on VACUUM/ANALYZE — it undercounted
`round_stats` by 74 rows (68886 vs the true 68960). The values below are exact
and match the goldens in `tests/integration/test_db_seed.py`:

```
      relname      |   count
-------------------+------------
 elo_snapshots     |      89988
 round_stats       |      68960
 computed_features |      28624
 fight_odds        |      25632
 fights            |      16902
 fighters          |       6820
 events            |       1872
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
