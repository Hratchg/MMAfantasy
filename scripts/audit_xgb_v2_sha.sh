#!/usr/bin/env bash
# audit_xgb_v2_sha.sh — verify xgb_v2.joblib byte-identity across all phases (reads canonical baseline).
# Run at start (16-01 task 0), mid (16-03 final task), and end (16-04 Task 9).
# Per CONTEXT.md D-09(P15): xgb_v2.joblib MUST remain byte-identical so the
# rollback path stays valid; any HOUSE-* edit that changes its SHA is a regression.
set -euo pipefail

BASELINE_FILE=".planning/AUDIT-01-BASELINE-SHA.txt"
EXPECTED_SHA="$(cat "$BASELINE_FILE")"
ACTUAL_SHA="$(shasum -a 256 models/xgb_v2.joblib | awk '{print $1}')"

if [[ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]]; then
  echo "FAIL: xgb_v2.joblib SHA changed!"
  echo "Expected: $EXPECTED_SHA"
  echo "Actual:   $ACTUAL_SHA"
  exit 1
fi
echo "OK: xgb_v2.joblib byte-identity preserved ($ACTUAL_SHA)"
