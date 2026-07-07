#!/usr/bin/env python3
"""PreToolUse guard: block Edit/Write on AUDIT-01 protected files at EDIT time
(the pre-commit guard only fires at commit time — this catches it earlier).

Reads the Claude Code hook JSON on stdin, extracts tool_input.file_path, and
exits 2 (blocking) if that path is in scripts/check_audit01_protected_files.py's
PROTECTED_FILES set. Bypass deliberately with AUDIT01_OVERRIDE=1.

Fail-open: any parsing/loading error exits 0 (never blocks on a hook bug).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _protected_set(repo: Path) -> frozenset[str]:
    script = repo / "scripts" / "check_audit01_protected_files.py"
    spec = importlib.util.spec_from_file_location("_audit01", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.PROTECTED_FILES


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    fp = (data.get("tool_input") or {}).get("file_path", "")
    if not fp:
        return 0
    if os.environ.get("AUDIT01_OVERRIDE"):
        return 0

    repo = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
    try:
        rel = str(Path(fp).resolve().relative_to(repo.resolve()))
    except ValueError:
        return 0  # outside the repo — not our concern

    try:
        protected = _protected_set(repo)
    except Exception:
        return 0  # fail open on any loader error

    if rel in protected:
        sys.stderr.write(
            f"BLOCKED (AUDIT-01): {rel} is a frozen/protected artifact "
            "(models, spike scripts, predictor/feature_matrix/persistence/train, "
            "predictor schema). Editing it breaks byte-identity / D-03 locks. "
            "If this change is a deliberate, operator-approved promotion, re-run "
            "with AUDIT01_OVERRIDE=1 in the environment.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
