# UFC Corpus v3.0 Seed — Provenance

**Artifact:** `data/seed/ufc_corpus_v30.dump`
**Generated:** 2026-10-01 (`SEED-REBASE-02`: regenerated on the D6-B substrate with the `debutant_seed_inputs` table; previously 2026-09-25 and 2026-07-06 `RETRAIN-V31-01` / `SEED-REBASE-01`)
**Phase:** 88 — HANDOFF-V30-02 DB Dump Packaging (regenerated on the promoted substrate)

> **Regeneration note (2026-10-01, `SEED-REBASE-02`):** regenerated on the
> substrate that produced the D6-B `xgb_v2` (sha256 `760307…5677a`). The dump
> now carries a 13th table, `debutant_seed_inputs` (migration `9fc47fdd75b6`).
> It holds the Sherdog pre-UFC records behind debutant Elo seeding: 1,673 rows,
> made up of local 1,374, regional 178, major 112 and none 9. So a fresh
> `ufc db seed` followed by `elo compute` works without the untracked
> `data/sherdog/pre_ufc_records.csv`. Substrate changes since 2026-09-25:
> - ufcstats round-stat orientation repair;
> - Kaggle-twin odds backfill (`fight_odds` 25,812 → 26,302);
> - no-contest fights skipped by Elo (`elo_snapshots` 90,642 → 90,090);
> - as-of league-mean shrinkage;
> - ufcstats title-fight flags (489 title fights);
> - the merged "Bruno Silva" record split in two (`fighters` 6,846 → 6,847;
>   `computed_features` 28,816 → 28,815).
>
> The source DB is postgres:18. The dump was bridged through postgres:16
> exactly as below, and a `pg_restore 16` round trip into a fresh postgres:16
> reproduced every count in the table list.
>
> **Regeneration note (2026-09-25):** this dump was regenerated on the same
> corpus after two substrate fixes: `features compute` no longer leaks the
> current fight into the opponent-adjusted (`opp_adj_*`) rates, and the domain
> Elo computer keeps per-domain bookkeeping so grappling ratings now regress
> after inactivity and transfer between divisions like striking does (and both
> domains shrink on their own fight count). Overall Elo rows are byte-identical
> to the 2026-07-06 dump; 12,853 grappling `elo_before`/`elo_after` values,
> the domain `elo_after_shrinkage` values, the four `opp_adj_*` feature keys and
> 2,917 `style_tag` values changed. Row counts are unchanged. The frozen
> `xgb_v2` (sha256 `0b0b40…fecd`) was NOT re-promoted on this substrate.
>
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
| Host port | `5433` (any host port works; the dump is taken from inside the container) |
| Database | `ufc_prediction` |
| User | `ufc` |
| Postgres server version | `PostgreSQL 18.4 (Debian 18.4-1.pgdg13+1)` |
| Raw DB size | 116 MB |

## Dump command

The promoted corpus lives in a **postgres:18** dev DB, but CI seeds via the
runner's **pg_restore 16** (`.github/workflows/ci.yml` uses `postgres:16-alpine`).
A pg_dump-18 custom archive (format v1.16) is **not** readable by pg_restore 16
(`unsupported version (1.16)`), so the shipped dump is produced in **PG16 custom
format**: the PG18 data is bridged through a throwaway postgres:16 container, then
dumped with pg_dump 16.

```bash
# 1. bridge the promoted PG18 corpus into a throwaway postgres:16 container
docker run -d --rm --name pg16 -e POSTGRES_USER=ufc -e POSTGRES_PASSWORD=ufc \
  -e POSTGRES_DB=ufc_prediction postgres:16-alpine
docker exec mmafantasy-db-1 pg_dump --format=plain --no-owner --no-privileges \
  -U ufc -d ufc_prediction | docker exec -i pg16 psql -q -U ufc -d ufc_prediction
# 2. re-dump in PG16 custom format (archive readable by pg_restore 16)
docker exec pg16 pg_dump --format=custom -Z 9 --no-owner --no-privileges \
  -U ufc -d ufc_prediction > data/seed/ufc_corpus_v30.dump
```

Verified: `pg_restore 16` restores the resulting dump into a fresh postgres:16
with the exact per-table counts below.

## Sizes

| Field | Value |
|-------|-------|
| Compressed dump size | 11,612,692 bytes (11.07 MB) |
| SHA256 | `d695404fb38120448043803356d26afda134c99ce32951d3adc0ec513da6b12b` |

## Hosting route

Committed in repo (≤ 30 MB threshold per D-B1).

The compressed dump (11.07 MB) sits well under the 30 MB cutoff, so the
binary lives directly in the repo alongside this provenance file and the
SHA256 sidecar. No external GitHub Release hosting required for v3.0.

## Tables included (13)

Per-table exact row counts in the dump (`SELECT COUNT(*)`). These are exact
counts (not `pg_stat_user_tables` planner estimates) and match the goldens in
`tests/integration/test_db_seed.py`:

```
       relname        |   count
----------------------+------------
 elo_snapshots        |      90090
 round_stats          |      69684
 computed_features    |      28815
 fight_odds           |      26302
 fights               |      17011
 fighters             |       6847
 events               |       1881
 debutant_seed_inputs |       1673
 fighter_aliases      |        399
 venues               |        174
 referees             |         39
 alembic_version      |          1
 model_runs           |          0
(13 rows)
```

All 13 user tables included per D-A3. `model_runs` is empty by design
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
