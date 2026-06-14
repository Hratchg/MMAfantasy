#!/usr/bin/env bash
# scripts/prune_stale_worktrees.sh — Phase 38 HYGIENE-V24-04
#
# Remove .claude/worktrees/agent-* directories that are NOT in the active
# `git worktree list`. Idempotent + safe to re-run. One-time invoked in
# Phase 38; preserved for future hygiene drift cleanup.
#
# Usage:  bash scripts/prune_stale_worktrees.sh
#         (run from repo root; script discovers root via `git rev-parse`)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

WORKTREES_DIR="${REPO_ROOT}/.claude/worktrees"

if [[ ! -d "${WORKTREES_DIR}" ]]; then
  echo "[prune] ${WORKTREES_DIR} does not exist; nothing to do."
  exit 0
fi

# Build a set of active worktree absolute paths.
# Use a portable while-read loop (avoids `mapfile`, which is bash 4+ only;
# macOS ships system bash 3.2 by default).
ACTIVE=()
while IFS= read -r line; do
  ACTIVE+=("${line}")
done < <(git worktree list --porcelain | awk '/^worktree /{print $2}')

is_active() {
  local candidate="$1"
  local active
  # Guard for empty ACTIVE: under `set -u`, expanding "${ACTIVE[@]}" on an
  # empty array errors in bash 3.2. Skip the loop entirely if empty.
  if [[ "${#ACTIVE[@]}" -eq 0 ]]; then
    return 1
  fi
  for active in "${ACTIVE[@]}"; do
    if [[ "${active}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

removed_count=0
skipped_count=0

shopt -s nullglob
for dir in "${WORKTREES_DIR}"/agent-*; do
  [[ -d "${dir}" ]] || continue
  abs="$(cd "${dir}" 2>/dev/null && pwd -P || echo "${dir}")"
  if is_active "${abs}"; then
    echo "[prune] keep   ${abs}  (active worktree)"
    skipped_count=$((skipped_count + 1))
  else
    echo "[prune] remove ${abs}  (stale)"
    rm -rf "${dir}"
    removed_count=$((removed_count + 1))
  fi
done
shopt -u nullglob

echo "[prune] running 'git worktree prune' to purge orphaned admin metadata"
git worktree prune

echo "[prune] done — removed=${removed_count} skipped=${skipped_count}"
