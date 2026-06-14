"""Phase 50 DX-V26-04 + Phase 72 METH-V261-01 — AUDIT-01 protected-file
pre-commit guard (with SOFT-PROTECT extension for refit-baseline siblings).

Blocks `git commit` when any staged file matches the AUDIT-01 HARD-protected
set documented in CONTRIBUTING.md L58-78. Bypass requires:

  AUDIT01_OVERRIDE=1 git commit ...

The override env var is intentionally verbose so the operator decision
appears in `env` audit trails and shell history. CONTRIBUTING.md's PR
checklist also requires the override be cited in the PR description.

Phase 72 SOFT-PROTECT extension (per gate_methodology_v2.6.md §7.3):
Refit-baseline siblings matching ``models/meta/meta_v2_refit_v*.joblib``
(and ``*_meta.json``) are SOFT-protected — edits are allowed when the
operator opts in via either:

  (1) the literal token ``refit-driver-re-emit`` appearing in the commit
      message (read from .git/COMMIT_EDITMSG at hook invocation), OR
  (2) the env var ``GSD_REFIT_REEMIT=1`` (matches the AUDIT01_OVERRIDE
      pattern for deterministic operator opt-in).

Both off → block. This catches accidental modifications while allowing the
intentional re-emission triggered by corpus growth (METH-V27-01 trigger).

Exit codes:
  0  no protected files staged (or override / soft-protect opt-in set)
  1  protected files staged without override — commit blocked
  2  invocation error (e.g., no files passed)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Single source of truth for the AUDIT-01 HARD-protected set. Mirrors
# CONTRIBUTING.md L62-74 line-by-line; the README + this script are the
# only places this set is enumerated.
PROTECTED_FILES: frozenset[str] = frozenset(
    {
        "models/xgb_v2.joblib",
        "models/meta/meta_v2.joblib",
        "models/meta/meta_v2_dedup.joblib",
        "src/ufc_prediction/ml/predictor.py",
        "src/ufc_prediction/ml/feature_matrix.py",
        "src/ufc_prediction/ml/persistence.py",
        "src/ufc_prediction/ml/train.py",
        "scripts/spike_noise_floor_v22.py",
        "scripts/spike_noise_floor_v23.py",
        "scripts/train_meta_v22.py",
        "src/ufc_prediction/contracts/predictor.schema.v1.0.0.json",
    }
)

OVERRIDE_ENV_VAR: str = "AUDIT01_OVERRIDE"

# ── Phase 72 METH-V261-01 SOFT-PROTECT extension ─────────────────────────

# Regex matching refit-baseline sibling filenames. Both the joblib and the
# sidecar JSON are covered.
SOFT_PROTECTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^models/meta/meta_v2_refit_v[^/]+\.joblib$"),
    re.compile(r"^models/meta/meta_v2_refit_v[^/]+_meta\.json$"),
)

# Literal token operators must include in their commit message to authorize
# a refit-sibling re-emit. Case-sensitive by design — accidental commit
# messages won't trigger.
SOFT_PROTECT_TOKEN: str = "refit-driver-re-emit"

# Env-var fallback (matches AUDIT01_OVERRIDE pattern; deterministic opt-in).
SOFT_PROTECT_ENV_VAR: str = "GSD_REFIT_REEMIT"


def find_violations(staged_paths: list[str]) -> list[str]:
    """Return staged paths that fall inside the HARD-protected set."""
    # Normalize via Path to handle ./ prefixes + alternate separators.
    return sorted(
        {str(Path(p)) for p in staged_paths if str(Path(p)) in PROTECTED_FILES}
    )


def find_soft_violations(staged_paths: list[str]) -> list[str]:
    """Return staged paths matching the SOFT-protect patterns.

    These are NOT in the HARD-protected set — they are refit-baseline
    siblings that the GATE-RECALIB workflow may legitimately overwrite.
    """
    matches: set[str] = set()
    for p in staged_paths:
        normalized = str(Path(p))
        for pattern in SOFT_PROTECTED_PATTERNS:
            if pattern.match(normalized):
                matches.add(normalized)
                break
    return sorted(matches)


def _read_commit_message() -> str:
    """Return the contents of .git/COMMIT_EDITMSG, or empty string if absent.

    pre-commit hooks run BEFORE git finalizes the message; the file holds
    the in-progress message (either operator-supplied via `-m` or the
    template). We scan for SOFT_PROTECT_TOKEN as a literal substring.
    """
    # Resolve .git dir. Most repos keep COMMIT_EDITMSG at .git/COMMIT_EDITMSG;
    # worktrees expose it under .git/worktrees/<name>/COMMIT_EDITMSG. We try
    # the conventional path first.
    candidates: list[Path] = []
    git_dir_env = os.environ.get("GIT_DIR")
    if git_dir_env:
        candidates.append(Path(git_dir_env) / "COMMIT_EDITMSG")
    candidates.append(Path(".git") / "COMMIT_EDITMSG")
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return ""


def soft_protect_opt_in_active() -> bool:
    """Return True iff the operator has opted into refit-sibling re-emit.

    Two equally-authoritative signals:
      1. SOFT_PROTECT_ENV_VAR == "1" (deterministic, env-trail captured)
      2. SOFT_PROTECT_TOKEN appears in .git/COMMIT_EDITMSG
    """
    if os.environ.get(SOFT_PROTECT_ENV_VAR) == "1":
        return True
    msg = _read_commit_message()
    return SOFT_PROTECT_TOKEN in msg


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        # pre-commit invokes us with file paths as positional args; an
        # empty arg list typically means no matching files staged.
        return 0

    # ── HARD-PROTECT path (existing behaviour, unchanged) ─────────────────
    violations = find_violations(args)
    if violations:
        if os.environ.get(OVERRIDE_ENV_VAR) == "1":
            sys.stderr.write(
                f"AUDIT-01 protected files staged with {OVERRIDE_ENV_VAR}=1 "
                f"override — proceeding:\n"
            )
            for v in violations:
                sys.stderr.write(f"  - {v}\n")
            sys.stderr.write(
                "Cite the override in the PR description per "
                "CONTRIBUTING.md L228-229 checklist item.\n"
            )
            # Fall through to SOFT-PROTECT check below — both gates may apply
            # in the same commit (rare but supported).
        else:
            sys.stderr.write(
                "AUDIT-01 protected files staged — commit BLOCKED.\n\n"
                "These files are byte-identity locked (CONTRIBUTING.md "
                "L58-78):\n"
            )
            for v in violations:
                sys.stderr.write(f"  - {v}\n")
            sys.stderr.write(
                "\nIf this change is INTENTIONAL and operator-approved, "
                "re-run with:\n"
                f"  {OVERRIDE_ENV_VAR}=1 git commit ...\n\n"
                "Do NOT use `--no-verify` to bypass this hook "
                "(CONTRIBUTING.md L78).\n",
            )
            return 1

    # ── SOFT-PROTECT path (Phase 72 METH-V261-01 extension) ───────────────
    soft_violations = find_soft_violations(args)
    if soft_violations:
        if soft_protect_opt_in_active():
            sys.stderr.write(
                "AUDIT-01 SOFT-protected refit siblings staged with operator "
                "opt-in — proceeding with warning:\n"
            )
            for v in soft_violations:
                sys.stderr.write(f"  - {v}\n")
            sys.stderr.write(
                "These artifacts are managed by the refit-baseline driver "
                "(scripts/refit_meta_v22_v2.6.py per gate_methodology_v2.6.md "
                "§7.2). Re-emission requires a corpus-growth trigger "
                "(METH-V27-01 re-eligibility).\n"
            )
            return 0
        sys.stderr.write(
            "AUDIT-01 SOFT-protected refit siblings staged — commit BLOCKED.\n\n"
            "These artifacts are managed by the refit-baseline driver "
            "(scripts/refit_meta_v22_v2.6.py per gate_methodology_v2.6.md "
            "§7.2). Accidental modifications are caught here to preserve the "
            "audit trail:\n"
        )
        for v in soft_violations:
            sys.stderr.write(f"  - {v}\n")
        sys.stderr.write(
            "\nIf this is an INTENTIONAL re-emission (e.g., corpus-growth "
            "trigger fired), use ONE of:\n"
            f"  (1) include the literal token '{SOFT_PROTECT_TOKEN}' in your "
            f"commit message, OR\n"
            f"  (2) re-run with: {SOFT_PROTECT_ENV_VAR}=1 git commit ...\n\n"
            "Do NOT use `--no-verify` to bypass this hook.\n",
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
