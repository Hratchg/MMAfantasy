#!/usr/bin/env bash
# Dump the local UFC Postgres DB to a gzipped pg_dump file + sha256 companion.
#
# Produces: $OUT (default: ufc_prediction-v1.0.sql.gz) and $OUT.sha256.
#
# The dump is produced with --no-owner --no-privileges so it restores cleanly
# into Supabase (owner "postgres") as well as the local "ufc" user. Do NOT
# call `alembic upgrade head` after a restore — the dump carries the schema.
#
# Env overrides:
#   UFC_DB_CONTAINER   (default: ufc-fight-prediction-db-1)
#   UFC_DB_NAME        (default: ufc_prediction)
#   UFC_DB_USER        (default: ufc)
#   UFC_DUMP_VERSION   (default: v1.0)
#   UFC_DUMP_OUT       (default: ufc_prediction-${VERSION}.sql.gz)

set -euo pipefail

CONTAINER="${UFC_DB_CONTAINER:-ufc-fight-prediction-db-1}"
DB_NAME="${UFC_DB_NAME:-ufc_prediction}"
DB_USER="${UFC_DB_USER:-ufc}"
VERSION="${UFC_DUMP_VERSION:-v1.0}"
OUT="${UFC_DUMP_OUT:-ufc_prediction-${VERSION}.sql.gz}"

if ! docker ps --format '{{.Names}}' | grep -Fx "$CONTAINER" >/dev/null; then
    echo "DB container $CONTAINER is not running. Start it with: docker compose up -d db" >&2
    exit 1
fi

echo "Dumping $DB_NAME from $CONTAINER → $OUT ..." >&2

docker exec "$CONTAINER" pg_dump -U "$DB_USER" --no-owner --no-privileges "$DB_NAME" \
    | gzip > "$OUT"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT" > "$OUT.sha256"
else
    shasum -a 256 "$OUT" > "$OUT.sha256"
fi

du -h "$OUT"
head -n 1 "$OUT.sha256"
