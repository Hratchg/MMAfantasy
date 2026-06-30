"""Phase 58 FEAT-V26-02 — TRAVEL gate-verifier invocation harness.

DEPRECATED (Phase 64 — v2.6.1, 2026-06-04).
====================================================

This wrapper is a BACK-COMPAT SHIM. New code, automation, and operator
scripts MUST migrate to the canonical Phase 63 CLI surface::

    ufc gate verify \\
        --candidate models/meta/meta_v22_travel.joblib \\
        --substrate-parquet data/intermediate/travel_substrate_v261.parquet \\
        --out results/travel_promotion_gate_v261.json

The legacy ``--substrate-parquet`` invocation path here now delegates
1:1 to ``ufc gate verify`` via ``subprocess.run`` (the Phase 58 rc=2
"v2.6.1 follow-on" placeholder has been removed in Phase 64). Existing
flag surface (``--candidate``, ``--canonical``, ``--substrate-parquet``,
``--strategy``, ``--out``) is preserved so Phase 58-era shell aliases
and automation continue to function.

The legacy NO-``--substrate-parquet`` invocation path (which emits the
Phase 58 PROVISIONAL Path B writeup at ``results/travel_promotion_gate_v26.{md,json}``)
is preserved unchanged for v2.6 audit-trail continuity. The provisional
writeup is now informational only — the canonical v2.6.1 verdict lives
at ``results/travel_promotion_gate_v261.json`` (Phase 64 Plan 64-04).

See ``.planning/phases/64-travel-feat-v261-01-verifier-run/64-CONTEXT.md``
D-03a for the back-compat-shim contract and the migration plan.

Re-gates ``models/meta/meta_v22_travel.joblib`` (Phase 42 sibling) through
the GATE-V26-02 substrate-drift–safe verifier (Phase 55) per the Phase 54
methodology spec. In Phase 58 (v2.6), this emitted
``results/travel_promotion_gate_v26.{md,json}`` per the Phase 45
verdict-emission precedent. In Phase 64 (v2.6.1), it delegates to
``ufc gate verify`` for the real-substrate path.

Usage:
  # Modern (recommended — direct CLI):
  ufc gate verify \\
      --candidate models/meta/meta_v22_travel.joblib \\
      --substrate-parquet <path> \\
      --out results/travel_promotion_gate_v261.json

  # Back-compat (shim — delegates to ``ufc gate verify``):
  python scripts/run_travel_gate_v26.py \\
      --candidate models/meta/meta_v22_travel.joblib \\
      --canonical models/meta/meta_v2.joblib \\
      --substrate-parquet <path>

Without ``--substrate-parquet``, emits the Phase 58 provisional Path B
writeup (unchanged audit-trail behavior).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_MD = PROJECT_ROOT / "results" / "travel_promotion_gate_v26.md"
RESULTS_JSON = PROJECT_ROOT / "results" / "travel_promotion_gate_v26.json"

# Phase 64 D-04 canonical default output path for the v2.6.1 verifier-run
# verdict sidecar (distinct from the Phase 58 v2.6 audit artifact).
DEFAULT_V261_OUT = PROJECT_ROOT / "results" / "travel_promotion_gate_v261.json"

# Deprecation banner — printed to stderr on every shim invocation so
# operators see the migration pointer in their terminal.
DEPRECATION_BANNER = (
    "DEPRECATION: scripts/run_travel_gate_v26.py is a back-compat shim "
    "as of Phase 64 (v2.6.1). New code should invoke `ufc gate verify` "
    "directly. See .planning/phases/64-travel-feat-v261-01-verifier-run/"
    "64-CONTEXT.md D-03a."
)

PROVISIONAL_PATH_B_WRITEUP = """\
# TRAVEL Promotion Gate — v2.6 Verdict (Provisional Path B)

**Authored:** Phase 58 FEAT-V26-02
**Date:** 2026-06-03
**Status:** PROVISIONAL Path B based on Phase 42 + Phase 45 evidence
**v2.6.1 follow-on:** actual GATE-V26-02 verifier run against Phase 42
substrate parquet may revise this disposition.

## Provisional verdict

**Path B — TRAVEL stays out; formally retired pending v2.6.1 confirmation.**

## Rationale

Phase 42 originally reported a +0.249 Brier delta against META-V22 on
widened slices. Plan 45-01 re-verification on training-time OOF found
**~67-83% of the delta evaporated** when the OOF-source-divergence
artifact was stripped (the baseline OOF was generated from a different
XGBoost serialization than the candidate's OOF — same substrate-drift
pattern as the Phase 45 meta_v3 confound, manifested at the
feature-engineering layer rather than the scaler layer).

The Phase 42 substrate-drift pattern is structurally equivalent to the
Phase 45 confound documented in
`.planning/gate_methodology_v2.6.md` §1: persisted artifact + shifted-
distribution substrate → inflated baseline metrics → apparent lift that
is dominantly measurement artifact.

The Phase 55 GATE-V26-02 verifier, when invoked with `methodology=
refit_baseline` against the Phase 42 substrate, is expected to produce
`verdict="confound_block"` — the meta-gate auto-detects the OOF-source-
divergence pattern via the raw-vs-aligned delta disagreement >
confound_threshold (0.05) check.

## v2.6 disposition

- `models/meta/meta_v22_travel.joblib` stays as Phase 42 advisory-only
  sibling (NOT promoted; NOT loaded by predictor.py)
- AUDIT-01 invariant: canonical `meta_v2.joblib` SHA `77076d3b…9196`
  UNCHANGED; `xgb_v2.joblib` SHA `6e7641…0099` UNCHANGED
- Phase 26 D-10 rename to `meta_v2_travel.joblib` NOT applied
- Spec §7 sibling-artifact discipline preserved: TRAVEL stays in the
  advisory tier alongside `meta_v3_candidate.joblib` (Phase 48)

## v2.6.1 path forward

1. Substrate-snapshot loader ships (Phase 56 backlog item)
2. `scripts/run_travel_gate_v26.py --substrate-parquet <path>` runs the
   actual verifier against the Phase 42 substrate
3. Verdict written to `results/travel_promotion_gate_v26.json`
4. Two outcomes:
   - **Confirmation of Path B** → `meta_v22_travel.joblib` archived to
     `.planning/phases/58/archive/`; v2.7+ backlog row added for
     potential re-evaluation under corpus growth
   - **Surprise Path A** → operator decision on promotion rename to
     `meta_v2_travel.joblib` per Phase 26 D-10 convention (would require
     AUDIT-01 chain MID anchor; see §7 of methodology spec)

## References

- `results/travel_oof_verification_v25.md` — Phase 45 Plan 45-01
  OOF-source-divergence evidence
- `.planning/milestones/v2.5-phases/42-travel-composition/` — Phase 42
  archive (advisory-only sibling lineage)
- `.planning/gate_methodology_v2.6.md` §1, §6 — substrate-drift confound
  failure mode + verifier contract
- `src/ufc_prediction/ml/gate_verifier.py` — Phase 55 verifier
"""

PROVISIONAL_PATH_B_VERDICT_JSON: dict[str, object] = {
    "phase": "58-feat-v26-02",
    "candidate": "models/meta/meta_v22_travel.joblib",
    "canonical": "models/meta/meta_v2.joblib",
    "verdict": "path_b_provisional",
    "rationale": (
        "Provisional Path B based on Phase 42 + Phase 45 evidence "
        "(OOF-source-divergence artifact pattern; substrate-drift confound "
        "suspected). v2.6.1 actual verifier-run against Phase 42 substrate "
        "parquet may revise."
    ),
    "methodology": "refit_baseline (planned for v2.6.1 actual run)",
    "verifier_version": "v2.6.0",
    "confound_threshold": 0.05,
    "audit_01_invariant": {
        "xgb_v2_sha": "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099",
        "meta_v2_sha": "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196",
        "status": "UNCHANGED",
    },
    "v2_6_1_followon_needed": True,
    "references": [
        "results/travel_oof_verification_v25.md",
        ".planning/gate_methodology_v2.6.md",
        "src/ufc_prediction/ml/gate_verifier.py",
    ],
}


def emit_provisional_path_b(
    md_path: Path = RESULTS_MD,
    json_path: Path = RESULTS_JSON,
) -> tuple[Path, Path]:
    """Emit the Phase 58 PROVISIONAL Path B audit-trail pair (unchanged in Phase 64).

    Preserved verbatim from Phase 58 — this is the v2.6 audit artifact and
    is NOT what the v2.6.1 verifier-run produces. See Phase 64 Plan 64-04
    for the canonical v2.6.1 verdict pathway.

    Phase 64 CR-01 guard: refuse to overwrite the committed Phase 58
    artifacts (D-04/D-05 invariant). If either destination file already
    exists, raise ``RuntimeError`` and point operators at the v2.6.1
    substrate-snapshot path. The provisional emission was a v2.6 shipping
    crutch; v2.6.1 ships the real verifier via ``ufc gate verify
    --substrate-parquet`` and the shim's delegation path.
    """
    if md_path.exists() or json_path.exists():
        raise RuntimeError(
            f"emit_provisional_path_b: refusing to overwrite committed "
            f"Phase 58 audit artifacts (D-04/D-05 invariant):\n"
            f"  {md_path}\n  {json_path}\n"
            f"These files are PROTECTED v2.6 PROVISIONAL audit trail and "
            f"must not be modified. Invoke with --substrate-parquet to "
            f"run the actual v2.6.1 verifier instead, e.g.:\n"
            f"  python scripts/run_travel_gate_v26.py \\\n"
            f"      --candidate models/meta/meta_v22_travel.joblib \\\n"
            f"      --substrate-parquet data/intermediate/travel_substrate_v261.parquet"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(PROVISIONAL_PATH_B_WRITEUP, encoding="utf-8")
    json_path.write_text(
        json.dumps(PROVISIONAL_PATH_B_VERDICT_JSON, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return md_path, json_path


def delegate_to_ufc_gate_verify(
    candidate: Path,
    canonical: Path,
    substrate_parquet: Path,
    strategy: str,
    out: Path,
) -> int:
    """Phase 64 D-03a — delegate to ``ufc gate verify`` via subprocess.

    Replaces the Phase 58 ``run_actual_verifier`` rc=2 placeholder. Builds
    the canonical Phase 63 CLI argv and shells out to ``ufc gate verify``;
    propagates its exit code 1:1. Process isolation keeps the shim's
    behavior identical to what a direct ``ufc gate verify`` invocation
    would produce (no in-process Typer interaction surface to maintain).

    Args:
        candidate: Path to candidate Pipeline (.joblib). Required.
        canonical: Path to canonical Pipeline (.joblib). Defaults to
            ``models/meta/meta_v2.joblib`` per Phase 63 CLI default.
        substrate_parquet: Path to Phase 63 substrate-snapshot parquet.
            Required for the delegation path (the no-substrate path is
            handled by ``emit_provisional_path_b`` upstream).
        strategy: ``refit_baseline`` (default) or ``dual_test_set``;
            passed through 1:1 to ``ufc gate verify --strategy``.
        out: JSON sidecar output path. Defaults to ``DEFAULT_V261_OUT``
            (``results/travel_promotion_gate_v261.json``).

    Returns:
        ``subprocess.run().returncode`` from the delegated invocation.
        0 = verifier wrote verdict file successfully (including
        ``confound_block`` outcomes — these are clean verdicts, not
        crashes). Non-zero = verifier failed (load error, etc.) per
        Phase 63 CLI exit-code contract.
    """
    argv = [
        "ufc",
        "gate",
        "verify",
        "--candidate",
        str(candidate),
        "--canonical",
        str(canonical),
        "--substrate-parquet",
        str(substrate_parquet),
        "--strategy",
        str(strategy),
        "--out",
        str(out),
    ]
    sys.stderr.write(f"Delegating to: {' '.join(argv)}\n")
    # check=False so we propagate the verifier's exit code verbatim
    # rather than raising CalledProcessError on non-zero. The Phase 63
    # CLI uses exit 1 for clean operator errors (load failures); we
    # want those to surface as exit 1 from this shim too.
    #
    # CR-02 guard: ``subprocess.run`` raises ``FileNotFoundError`` (not
    # ``CalledProcessError``) when the ``ufc`` entry-point is not on
    # ``PATH``, and ``check=False`` does NOT suppress it. Catch the
    # exception explicitly and emit a clean operator-facing message;
    # propagate as rc=127 (the conventional "command not found" exit
    # code per POSIX / bash) so callers can distinguish a missing-tool
    # failure from a verifier-internal failure (rc=1).
    try:
        result = subprocess.run(argv, check=False)
    except FileNotFoundError:
        sys.stderr.write(
            "ERROR: `ufc` command not found on PATH. The Phase 64 "
            "delegation shim cannot reach `ufc gate verify`. Ensure the "
            "ufc_prediction package is installed in the active environment "
            "(e.g., `pip install -e .` or `uv pip install -e .`) and that "
            "the venv's `bin/` directory is on PATH.\n"
        )
        return 127
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default="models/meta/meta_v22_travel.joblib",
        type=Path,
        help="Path to candidate Pipeline (.joblib).",
    )
    parser.add_argument(
        "--canonical",
        default="models/meta/meta_v2.joblib",
        type=Path,
        help=(
            "Path to canonical Pipeline (.joblib). Default matches the "
            "Phase 63 `ufc gate verify` default."
        ),
    )
    parser.add_argument(
        "--substrate-parquet",
        default=None,
        type=Path,
        help=(
            "Path to Phase 63 substrate-snapshot parquet. When provided, "
            "Phase 64 delegates to `ufc gate verify` (back-compat shim). "
            "When omitted, the legacy Phase 58 PROVISIONAL Path B writeup "
            "is emitted at results/travel_promotion_gate_v26.{md,json}."
        ),
    )
    parser.add_argument(
        "--strategy",
        default="refit_baseline",
        type=str,
        help=(
            "Substrate alignment strategy: refit_baseline (default) or "
            "dual_test_set. Passed through to `ufc gate verify --strategy`."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help=(
            f"JSON sidecar output path (delegation path only). Default: "
            f"{DEFAULT_V261_OUT.relative_to(PROJECT_ROOT)} per Phase 64 D-04."
        ),
    )
    args = parser.parse_args(argv)

    # Always emit the deprecation banner — operators should see it on
    # every invocation. Cheap (one stderr write) and load-bearing for
    # the migration plan.
    sys.stderr.write(DEPRECATION_BANNER + "\n")

    if args.substrate_parquet is None:
        # Legacy Phase 58 path. The CR-01 guard inside
        # ``emit_provisional_path_b`` refuses to overwrite the committed
        # D-04/D-05 protected Phase 58 artifacts; surface its error as a
        # clean operator-facing message + non-zero exit instead of an
        # unhandled traceback.
        try:
            md, js = emit_provisional_path_b()
        except RuntimeError as e:
            sys.stderr.write(f"ERROR: {e}\n")
            return 1
        sys.stdout.write(f"Provisional Path B writeup emitted:\n  {md}\n  {js}\n")
        return 0

    # Phase 64 delegation path. Resolve default --out if not provided.
    out_path = args.out if args.out is not None else DEFAULT_V261_OUT
    return delegate_to_ufc_gate_verify(
        candidate=args.candidate,
        canonical=args.canonical,
        substrate_parquet=args.substrate_parquet,
        strategy=args.strategy,
        out=out_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
