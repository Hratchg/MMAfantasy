"""Plan 29-02 integration test: assert Plan 28-04 T3 harness now runs.

Plan 28-04 Task 3 (``scripts/rerun_v22_meta_spike_on_deduplicated_corpus.py``)
was DEFERRED-PENDING-PHASE-29 because the v2.2 symmetric NaN-drop collapsed the
eval slice to 1 surviving row on the deduplicated corpus, causing a
``PolynomialFeatures: 0 samples`` ValueError inside MetaLearnerLogistic.

Plan 29-02 ships the per-feature NaN handling refactor that unblocks T3. This
test asserts:
  1. The T3 harness now exits with returncode 0 on the current state.
  2. ``28-04-METRIC-INTEGRITY.md`` frontmatter ``verdict:`` field is no longer
     DEFERRED-shaped (i.e., ∈ {PRESERVED, REDUCED, MARGIN_LOST}).

Plan 29-03 owns the CANONICAL T3 re-run + the verdict-driven amend of
28-04-METRIC-INTEGRITY.md. This test only verifies the harness now COMPLETES —
the specific verdict (PRESERVED vs REDUCED vs MARGIN_LOST) does NOT gate this
test and is captured for Plan 29-03's branching logic.

The test skips gracefully if the harness file is missing (defense in depth).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS_PATH = Path("scripts/rerun_v22_meta_spike_on_deduplicated_corpus.py")
REPORT_PATH = Path(".planning/phases/28-referee-venue-ingestion-pipeline/28-04-METRIC-INTEGRITY.md")
NON_DEFERRED_VERDICTS = frozenset({"PRESERVED", "REDUCED", "MARGIN_LOST"})


@pytest.fixture(scope="module")
def t3_harness_result():
    """Run the T3 harness once per session; share returncode + report for assertions.

    Skips the entire module if the harness file is missing (defense in depth).
    """
    if not HARNESS_PATH.exists():
        pytest.skip(f"T3 harness {HARNESS_PATH} missing — nothing to verify")
    cmd = [sys.executable, str(HARNESS_PATH)]
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@pytest.mark.regression
def test_t3_harness_runs_to_completion(t3_harness_result):
    """Harness must exit cleanly (returncode 0) on the post-Plan-29-02 state.

    A non-zero returncode here means the slice-collapse fix in Plan 29-02 did
    NOT propagate to the T3 harness — investigate train_meta_v22.py + the
    new apply_nan_drop_policy wiring before declaring Plan 29-02 done.
    """
    rc = t3_harness_result["returncode"]
    assert rc == 0, (
        f"T3 harness returncode={rc}; expected 0.\n"
        f"stdout tail:\n{t3_harness_result['stdout'][-2000:]}\n"
        f"stderr tail:\n{t3_harness_result['stderr'][-2000:]}"
    )


@pytest.mark.regression
def test_t3_verdict_is_not_deferred(t3_harness_result):
    """28-04-METRIC-INTEGRITY.md frontmatter `verdict:` must no longer be DEFERRED.

    Any of {PRESERVED, REDUCED, MARGIN_LOST} unblocks Phase 28. The specific
    verdict is captured for Plan 29-03's branching logic and does NOT gate
    this test.
    """
    # The harness must have written the report; if not, fail loudly.
    assert REPORT_PATH.exists(), (
        f"T3 report {REPORT_PATH} missing — harness did not emit it.\n"
        f"stdout tail:\n{t3_harness_result['stdout'][-2000:]}"
    )
    text = REPORT_PATH.read_text(encoding="utf-8")
    match = re.search(r"^verdict:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    assert match, f"could not parse `verdict:` from {REPORT_PATH} frontmatter"
    verdict = match.group(1).strip()
    assert verdict in NON_DEFERRED_VERDICTS, (
        f"T3 verdict still DEFERRED-shaped: got {verdict!r}; "
        f"expected ∈ {sorted(NON_DEFERRED_VERDICTS)}. "
        f"Plan 29-02 did not unblock T3 — investigate."
    )
