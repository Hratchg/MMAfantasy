#!/usr/bin/env bash
# Restore the UFC Postgres DB from the published GitHub Release asset.
#
# Downloads ufc_prediction-${VERSION}.sql.gz + .sha256 from release
# ${RELEASE_TAG} on Hratchg/ufc-fight-prediction (unless already on disk),
# verifies the checksum, optionally prompts before overwriting a populated DB,
# drops + recreates the public schema, and pipes the dump into Postgres via
# `docker exec -i`.
#
# NOTE: This does NOT run `alembic upgrade head` afterwards — the dump IS the
# schema. Running alembic would cause drift / duplicate-table errors.
#
# Env overrides:
#   UFC_DB_CONTAINER    (default: ufc-fight-prediction-db-1)
#   UFC_DB_NAME         (default: ufc_prediction)
#   UFC_DB_USER         (default: ufc)
#   UFC_DUMP_VERSION    (default: v1.0)
#   UFC_RELEASE_TAG     (default: v1.0-data)
#   UFC_RESTORE_FORCE   (default: 0; set 1 to skip the overwrite prompt)

set -euo pipefail

CONTAINER="${UFC_DB_CONTAINER:-ufc-fight-prediction-db-1}"
DB_NAME="${UFC_DB_NAME:-ufc_prediction}"
DB_USER="${UFC_DB_USER:-ufc}"
VERSION="${UFC_DUMP_VERSION:-v1.0}"
RELEASE_TAG="${UFC_RELEASE_TAG:-v1.0-data}"
ASSET="ufc_prediction-${VERSION}.sql.gz"
FORCE="${UFC_RESTORE_FORCE:-0}"

if ! docker ps --format '{{.Names}}' | grep -Fx "$CONTAINER" >/dev/null; then
    echo "DB container $CONTAINER is not running. Start it with: docker compose up -d db" >&2
    exit 1
fi

# 1. Fetch the asset + checksum if either is missing.
if [ ! -f "$ASSET" ] || [ ! -f "$ASSET.sha256" ]; then
    echo "Downloading $ASSET from release $RELEASE_TAG ..." >&2
    gh release download "$RELEASE_TAG" \
        --repo Hratchg/ufc-fight-prediction \
        --pattern "$ASSET" \
        --pattern "$ASSET.sha256" \
        --clobber
fi

# 2. Verify checksum.
echo "Verifying checksum ..." >&2
if command -v sha256sum >/dev/null 2>&1; then
    if ! sha256sum -c "$ASSET.sha256"; then
        rm -f "$ASSET" "$ASSET.sha256"
        echo "Checksum mismatch — deleted local copy. Re-run to re-download." >&2
        exit 1
    fi
else
    if ! shasum -a 256 -c "$ASSET.sha256"; then
        rm -f "$ASSET" "$ASSET.sha256"
        echo "Checksum mismatch — deleted local copy. Re-run to re-download." >&2
        exit 1
    fi
fi

# 3. Detect existing data and prompt unless FORCE=1.
EXISTING=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null || echo 0)
EXISTING="${EXISTING//[[:space:]]/}"
EXISTING="${EXISTING:-0}"

if [ "$EXISTING" -gt 0 ] && [ "$FORCE" != "1" ]; then
    echo "WARNING: $DB_NAME already contains $EXISTING public tables. Continuing will DROP and recreate them." >&2
    read -r -p "Type 'yes' to overwrite: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted." >&2
        exit 2
    fi
fi

# 4. Clean slate the public schema.
echo "Dropping and recreating public schema in $DB_NAME ..." >&2
docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" \
    -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT ALL ON SCHEMA public TO $DB_USER; GRANT ALL ON SCHEMA public TO public;"

# 5. Pipe the dump in.
echo "Restoring $ASSET → $DB_NAME ..." >&2
gunzip -c "$ASSET" \
    | docker exec -i "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" \
        --single-transaction -v ON_ERROR_STOP=1 \
        >/dev/null

# 6. Smoke check.
FIGHTS=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM fights")
FIGHTS="${FIGHTS//[[:space:]]/}"
if [ -z "$FIGHTS" ] || [ "$FIGHTS" = "0" ]; then
    echo "Restore smoke check FAILED: fights table is empty." >&2
    exit 1
fi
echo "Restored $FIGHTS fights into $DB_NAME." >&2

echo "Restore complete. Next: \`ufc fighter rankings Lightweight\` should show Islam Makhachev at #1."
