#!/usr/bin/env bash
# Phase 53 API-V26-03 — TypeScript SDK codegen driver.
#
# Reads the canonical OpenAPI artifact from
# src/ufc_prediction/contracts/openapi.v1.X.Y.json and emits TypeScript
# types via openapi-typescript@^7.
#
# Output: clients/typescript/src/v1/predict.ts
#
# Run from the repo root:
#   ./clients/typescript/codegen.sh
# Or via npm:
#   cd clients/typescript && npm run codegen
#
# CI integration (deferred to v2.6.1): a GitHub Actions workflow at
# .github/workflows/sdk-codegen.yml will run this script and then
# `git diff --exit-code` against the committed predict.ts — drift fails
# the PR.

set -euo pipefail

# Resolve REPO_ROOT relative to this script's location so the codegen
# is portable regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source artifact — pinned to v1.3.0 as of Phase 70 (v2.6.1 API-V261-03).
# Phase 69 (API-V261-01) shipped openapi.v1.3.0.json as the live source-of-truth
# replacing v1.1.0. v1.3.0 success-path bytes are byte-identical to v1.2.0
# (locked by test_forward_compat_v1_3_0.py); v1.3.0 additionally captures the
# RFC 7807 ProblemDetails opt-in error response and the full live FastAPI app
# surface (fighters/rankings/matchup/history routes mounted on create_app()).
SOURCE_OPENAPI="${REPO_ROOT}/src/ufc_prediction/contracts/openapi.v1.3.0.json"
OUTPUT_TS="${SCRIPT_DIR}/src/v1/predict.ts"

if [[ ! -f "${SOURCE_OPENAPI}" ]]; then
    echo "ERROR: source OpenAPI artifact not found at ${SOURCE_OPENAPI}" >&2
    exit 1
fi

# Use npx with explicit version pin so the script is reproducible
# regardless of the developer's local install state.
mkdir -p "$(dirname "${OUTPUT_TS}")"

# Two paths:
#   1. `openapi-typescript` available locally via npm install (preferred)
#   2. `npx --yes openapi-typescript@^7` for one-shot generation
#
# The CI workflow (v2.6.1) installs once via `npm ci` and uses path 1.
# Developers running ad-hoc generation can use either.

if command -v openapi-typescript >/dev/null 2>&1; then
    BIN="openapi-typescript"
elif command -v npx >/dev/null 2>&1; then
    BIN="npx --yes openapi-typescript@^7"
else
    echo "ERROR: neither openapi-typescript nor npx is available. " >&2
    echo "Install Node + npm, then run: npm install -g openapi-typescript" >&2
    exit 1
fi

echo "Codegen source: ${SOURCE_OPENAPI}"
echo "Codegen output: ${OUTPUT_TS}"
echo "Codegen bin:    ${BIN}"

${BIN} "${SOURCE_OPENAPI}" --output "${OUTPUT_TS}"

echo "Codegen complete. Diff committed types with:"
echo "  git diff ${OUTPUT_TS}"
