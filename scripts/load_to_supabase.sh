#!/usr/bin/env bash
# Load the published UFC Postgres dump into a Supabase project.
#
# Reuses the release asset cached locally (from a previous restore_db.sh run)
# or re-downloads it. Verifies the checksum, drops + recreates the remote
# public schema, and pipes the dump into Supabase over SSL.
#
# Requires: SUPABASE_DB_URL (the Postgres connection URI from the Supabase
# Dashboard → Project Settings → Database → Connection string). Supabase
# requires sslmode=require; this script appends it if missing. The script
# also strips a leading "+psycopg" dialect suffix if the user copied from the
# app's DATABASE_URL, since psql doesn't understand SQLAlchemy dialect URLs.
#
# Env overrides:
#   UFC_DUMP_VERSION    (default: v1.0)
#   UFC_RELEASE_TAG     (default: v1.0-data)
#   UFC_RESTORE_FORCE   (default: 0; set 1 to skip the overwrite prompt)

set -euo pipefail

if [ -z "${SUPABASE_DB_URL:-}" ]; then
    cat >&2 <<'USAGE'
Usage: SUPABASE_DB_URL='postgresql://postgres:PASS@db.xxx.supabase.co:5432/postgres?sslmode=require' bash scripts/load_to_supabase.sh

Get the connection string from:
  Supabase Dashboard → Project Settings → Database → Connection string (URI mode)
USAGE
    exit 1
fi

# Strip SQLAlchemy dialect suffix if present (+psycopg is unrecognized by psql).
PSQL_URL="${SUPABASE_DB_URL/+psycopg/}"

# Ensure sslmode=require is present (Supabase rejects insecure connections).
if [[ "$PSQL_URL" != *"sslmode="* ]]; then
    if [[ "$PSQL_URL" == *"?"* ]]; then
        PSQL_URL="${PSQL_URL}&sslmode=require"
    else
        PSQL_URL="${PSQL_URL}?sslmode=require"
    fi
fi

VERSION="${UFC_DUMP_VERSION:-v1.0}"
RELEASE_TAG="${UFC_RELEASE_TAG:-v1.0-data}"
ASSET="ufc_prediction-${VERSION}.sql.gz"
FORCE="${UFC_RESTORE_FORCE:-0}"

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

# 3. Sanity probe the remote.
if ! psql "$PSQL_URL" -tAc "SELECT 1" >/dev/null 2>&1; then
    echo "Cannot reach Supabase — check SUPABASE_DB_URL and network." >&2
    exit 1
fi

# 4. Detect existing public tables and prompt unless FORCE=1.
EXISTING=$(psql "$PSQL_URL" -tAc \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null || echo 0)
EXISTING="${EXISTING//[[:space:]]/}"
EXISTING="${EXISTING:-0}"

if [ "$EXISTING" -gt 0 ] && [ "$FORCE" != "1" ]; then
    echo "WARNING: Supabase DB already contains $EXISTING public tables. Continuing will DROP and recreate them." >&2
    read -r -p "Type 'yes' to overwrite: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted." >&2
        exit 2
    fi
fi

# 5. Clean slate the remote public schema. Supabase's owner is "postgres".
echo "Dropping and recreating public schema on Supabase ..." >&2
psql "$PSQL_URL" -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public; GRANT ALL ON SCHEMA public TO postgres; GRANT ALL ON SCHEMA public TO public;" >/dev/null

# 6. Pipe the dump in.
echo "Loading $ASSET → Supabase ..." >&2
gunzip -c "$ASSET" \
    | psql "$PSQL_URL" --single-transaction -v ON_ERROR_STOP=1 \
        >/dev/null

# 7. Smoke check.
FIGHTS=$(psql "$PSQL_URL" -tAc "SELECT COUNT(*) FROM fights")
FIGHTS="${FIGHTS//[[:space:]]/}"
if [ -z "$FIGHTS" ] || [ "$FIGHTS" = "0" ]; then
    echo "Load smoke check FAILED: fights table is empty on Supabase." >&2
    exit 1
fi
echo "Loaded $FIGHTS fights into Supabase." >&2

cat >&2 <<'DONE'

Next step: point the app at Supabase. See SUPABASE.md § 4 — set
  DATABASE_URL='postgresql+psycopg://postgres:PASS@db.xxx.supabase.co:5432/postgres?sslmode=require'
(note the +psycopg dialect prefix; the app's SQLAlchemy layer requires it even
though this script stripped it for psql).
DONE
