#!/usr/bin/env python3
"""PostToolUse hook: auto-format + auto-fix Python files after Edit/Write, so
edits conform to the repo's ruff pre-commit gate (ruff format --check + ruff
check) without manual reformat churn.

Reads the Claude Code hook JSON on stdin, extracts tool_input.file_path, and
runs `uv run ruff format` + `uv run ruff check --fix` on it if it's a .py file
under the repo. Fail-open and non-blocking (PostToolUse errors never block).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    fp = (data.get("tool_input") or {}).get("file_path", "")
    if not fp or not fp.endswith(".py"):
        return 0

    repo = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
    try:
        Path(fp).resolve().relative_to(repo.resolve())
    except ValueError:
        return 0  # outside repo

    for cmd in (["uv", "run", "ruff", "format", fp], ["uv", "run", "ruff", "check", "--fix", fp]):
        try:
            subprocess.run(cmd, cwd=str(repo), capture_output=True, timeout=60, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 0  # ruff/uv unavailable or slow — never block
    return 0


if __name__ == "__main__":
    sys.exit(main())
