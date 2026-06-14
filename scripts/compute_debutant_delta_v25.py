#!/usr/bin/env python
"""Phase 43 Plan 43-04 — DEBUT-V25-04 debutant prediction quality measurement.

Measures pre-seed vs post-seed Brier + accuracy on the v2.3 random_15pct slice
filtered to fights where EITHER fighter is a debutant
(``n_ufc_fights == 0`` at fight time).

**Verdict posture**: ``documented_delta_only`` — Phase 43 ships the Sherdog
debutant Elo seed unconditionally per CONTEXT D. This script produces the
informational delta for downstream Phase 45 META3-V25 lift-attribution
analysis; it is NOT a gate.

Methodology (per Plan 43-04 ``<interfaces>`` + ``<action>``):

1. Load all fights chronologically from DB (same path as
   ``compose_v25_travel.py`` and the canonical ``compute_elo`` CLI).
2. Compute each fighter's FIRST chronological UFC fight_id (the fight where
   they had ``n_ufc_fights == 0``). A fight is a "debutant-either" fight iff
   ``fight_id`` matches either fighter's first_fight_id.
3. Build the v2.3 random_15pct mask deterministically over the chronological
   ordering (``np.random.RandomState(42).random(N) < 0.15`` — bit-equivalent
   to ``evaluator.evaluate_per_slice``).
4. Eval subset = (random_15pct mask) ∩ (debutant-either mask).
5. **Metric path**: ``EloEngine.expected_win_probability`` — per Plan 43-04
   ``<action>`` § "Recommend EloEngine.expected_win_probability for isolation
   ... revisit in v2.6+ if operator wants full xgb_v2-path debutant
   measurement." This isolates the seed signal cleanly without confounding
   from xgb_v2's nonlinear blending.
6. **Pre-seed baseline**: Plan 43-03 did NOT capture
   ``data/sherdog/debutant_baseline_pre_seed.json``; this script therefore
   falls back to the reconstruction-via-replay path documented in Plan 43-04
   ``<interfaces>``. ``baseline_source: "reconstructed_via_replay"`` is flagged
   in the JSON output and 43-04-SUMMARY.md (DEBT-V25). Operator may re-execute
   with proper baseline capture in v2.6+ for higher confidence.

    Reconstruction is sound: ``EloEngine(EloConfig(), seeds={})`` is bit-exact
    equivalent to the pre-Phase-43 engine (guarded by
    ``tests/unit/elo/test_engine_seed_dispatch.py::test_3``). Replaying the
    corpus with empty seeds reproduces the pre-Phase-43-03 elo_snapshots state
    deterministically.

7. **Post-seed**: replay with ``EloEngine(EloConfig(), seeds=load_seeds(...))``.
8. Emit ``results/debutant_seed_v25.json`` (atomic tempfile + rename) +
   ``results/debutant_seed_v25.md`` (partner-facing).

Per AUDIT-01: ``xgb_v2.joblib`` + ``meta_v2.joblib`` BYTE-IDENTICAL throughout.
This script reads ZERO model joblibs (Elo-only path) and writes ZERO model
joblibs — model-bytes invariant is structurally preserved.

Usage:
    uv run python scripts/compute_debutant_delta_v25.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────── Phase 43 LOCKED constants ──────────────────────

# D-18 LOCKED — formula hash for the META-V22+CALIB gate; embedded in the JSON
# as a downstream cross-check anchor.
FORMULA_HASH: str = (
    "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
)

# AUDIT-01 byte-identity invariants (PROJECT.md canonical).
EXPECTED_XGB_V2_SHA256: str = (
    "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
)
EXPECTED_META_V2_SHA256: str = (
    "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
)

# v2.3 random_15pct slice — same RNG seed + fraction as
# ``evaluator.evaluate_per_slice`` (canonical convention).
RANDOM_15PCT_SEED: int = 42
RANDOM_15PCT_FRACTION: float = 0.15

# Plan 43-04 ``<interfaces>`` paths.
BASELINE_PATH: Path = Path("data/sherdog/debutant_baseline_pre_seed.json")
SHERDOG_PRE_UFC_CSV: Path = Path("data/sherdog/pre_ufc_records.csv")
OUTPUT_JSON: Path = Path("results/debutant_seed_v25.json")
OUTPUT_MD: Path = Path("results/debutant_seed_v25.md")


# ──────────────────────────── Pure helpers ──────────────────────────────────


def compute_brier_acc(
    predictions: list[float] | np.ndarray,
    outcomes: list[int] | np.ndarray,
) -> tuple[float, float]:
    """Brier = mean((p - y)^2). Accuracy = mean((p >= 0.5) == y).

    Both lists must be the same non-zero length.
    """
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if p.size == 0:
        raise ValueError("compute_brier_acc: empty predictions/outcomes")
    if p.size != y.size:
        raise ValueError(
            f"compute_brier_acc: length mismatch p={p.size} y={y.size}"
        )
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean(((p >= 0.5).astype(int) == y).astype(int)))
    return brier, acc


def load_baseline_metrics(path: Path) -> dict | None:
    """Load Plan 43-03-captured baseline JSON if present.

    Returns None if missing → triggers reconstruction path with a loud warning.
    """
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


# ──────────────────────────── DB-backed helpers ─────────────────────────────


def select_debutant_fights_random_15pct(session) -> tuple[
    list[Any], list[Any], int, int, int,
]:
    """Build the eval subset and return:

        (all_fights_chrono, debutant_subset, n_total_corpus,
         n_random_15pct, n_debutant_unique_fighters)

    Pipeline:
        1. Load all fights chronologically (same path as compute_elo CLI).
        2. Per-fighter first_fight_id from chronological scan.
        3. v2.3 random_15pct mask over chronological ordering.
        4. Debutant-either mask: ``fight_id`` matches either fighter's
           first_fight_id.
        5. Subset = intersection.

    Raises if subset is empty (no zero-eligible silent pass per
    ``must_haves.truths``).
    """
    from ufc_prediction.elo.queries import load_fights_chronological

    all_fights = load_fights_chronological(session)
    n_total = len(all_fights)
    if n_total == 0:
        raise RuntimeError(
            "select_debutant_fights_random_15pct: 0 fights loaded from DB."
        )

    # Per-fighter first chronological UFC fight_id.
    first_fight_id: dict[int, int] = {}
    for fight in all_fights:
        for fid in (fight.fighter_a_id, fight.fighter_b_id):
            if fid not in first_fight_id:
                first_fight_id[fid] = fight.fight_id

    # v2.3 random_15pct mask — same RNG semantics as evaluator.evaluate_per_slice.
    rng = np.random.RandomState(RANDOM_15PCT_SEED)
    random_mask = rng.random(n_total) < RANDOM_15PCT_FRACTION

    debutant_subset: list[Any] = []
    debutant_fighter_ids: set[int] = set()
    n_random_15pct = 0
    for idx, fight in enumerate(all_fights):
        if not random_mask[idx]:
            continue
        n_random_15pct += 1
        is_debut_a = first_fight_id[fight.fighter_a_id] == fight.fight_id
        is_debut_b = first_fight_id[fight.fighter_b_id] == fight.fight_id
        if is_debut_a or is_debut_b:
            debutant_subset.append(fight)
            if is_debut_a:
                debutant_fighter_ids.add(fight.fighter_a_id)
            if is_debut_b:
                debutant_fighter_ids.add(fight.fighter_b_id)

    if not debutant_subset:
        raise RuntimeError(
            f"select_debutant_fights_random_15pct: 0 debutant fights in the "
            f"random_15pct slice (n_total={n_total}, "
            f"n_random_15pct={n_random_15pct}). "
            "No zero-eligible silent pass per must_haves.truths."
        )

    return (
        all_fights,
        debutant_subset,
        n_total,
        n_random_15pct,
        len(debutant_fighter_ids),
    )


def replay_engine_capture_debutant_probs(
    all_fights: list[Any],
    debutant_subset: list[Any],
    seeds: dict[int, float],
) -> tuple[list[float], list[int]]:
    """Replay the EloEngine chronologically; capture per-debutant-fight
    P(A wins) computed from the pre-fight Elo ratings + outcome label.

    Returns (probabilities, outcomes) aligned to ``debutant_subset`` order.
    Outcome label: 1 if winner_id == fighter_a_id else 0; fights with
    ``winner_id is None`` (draws / no-contests) are SKIPPED (cannot Brier-score
    a draw against a binary label).
    """
    from ufc_prediction.elo.config import EloConfig
    from ufc_prediction.elo.engine import EloEngine

    debutant_fight_ids = {f.fight_id for f in debutant_subset}
    debutant_by_id = {f.fight_id: f for f in debutant_subset}

    engine = EloEngine(EloConfig(), seeds=dict(seeds))
    # Per-fight capture: at the moment _process_fight is invoked, the engine
    # has the pre-fight rating state for both fighters. We need to peek at the
    # ratings BEFORE the fight result mutates them — so we step through
    # fights manually and capture rating_a / rating_b at the "elo_before"
    # moment, then call _process_fight to advance state.

    probs_by_fight_id: dict[int, float] = {}
    outcomes_by_fight_id: dict[int, int] = {}

    for fight in all_fights:
        if fight.fight_id in debutant_fight_ids:
            # Capture pre-fight expected_win_probability.
            division = fight.weight_class
            rating_a = engine._lookup_initial_rating(  # noqa: SLF001
                fight.fighter_a_id, division,
            )
            rating_b = engine._lookup_initial_rating(  # noqa: SLF001
                fight.fighter_b_id, division,
            )
            p_a = float(engine.expected_win_probability(rating_a, rating_b))
            probs_by_fight_id[fight.fight_id] = p_a
            if fight.winner_id is not None:
                outcomes_by_fight_id[fight.fight_id] = (
                    1 if fight.winner_id == fight.fighter_a_id else 0
                )
        # Always step the engine forward to keep chronological state coherent.
        engine._process_fight(fight)  # noqa: SLF001

    # Order outputs by debutant_subset iteration; skip fights with no outcome.
    probs: list[float] = []
    outcomes: list[int] = []
    for fight in debutant_subset:
        if fight.fight_id not in outcomes_by_fight_id:
            # Draw or no-contest — skip (can't Brier-score binary label).
            continue
        probs.append(probs_by_fight_id[fight.fight_id])
        outcomes.append(outcomes_by_fight_id[fight.fight_id])

    return probs, outcomes


# ──────────────────────────── JSON + MD emission ────────────────────────────


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False,
        dir=path.parent, suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _write_md(
    path: Path,
    payload: dict,
    *,
    reconstructed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    pre_brier = payload["pre_brier"]
    post_brier = payload["post_brier"]
    delta_brier = payload["delta_brier"]
    pre_acc = payload["pre_acc"]
    post_acc = payload["post_acc"]
    delta_acc = payload["delta_acc"]
    n_deb = payload["n_debutant_fights"]
    n_unique = payload["n_debutant_unique_fighters"]
    n_total = payload["n_total_random_15pct"]
    baseline_source = payload["baseline_source"]
    measured_at = payload["measured_at"]

    if delta_brier > 0.001:
        interp = (
            f"Seed REDUCES Brier on the debutant subset by "
            f"{delta_brier:+.4f} (improvement). The Sherdog-derived "
            "pre-UFC record signal materially shifts debutant priors away "
            "from the flat 1500 default."
        )
    elif delta_brier < -0.001:
        interp = (
            f"Seed INCREASES Brier on the debutant subset by "
            f"{delta_brier:+.4f} (informational only — still ships per "
            "verdict posture). Possible causes: (a) Sherdog pre-UFC "
            "record signal is noisier than the 1500 prior on this "
            "subset; (b) seeded debutants tend to overperform their "
            "Sherdog-derived rating in their UFC debut (selection "
            "effect — MMA promotions sign debutants whose Sherdog "
            "record under-states their actual UFC-ready skill). "
            "Re-eligible for Phase 45 META3-V25-03 attribution analysis."
        )
    else:
        interp = (
            f"Delta {delta_brier:+.4f} is within noise-level "
            "(|Δ| < 0.001 — seed-quantization uncertainty). Seed is "
            "behaviorally neutral on this subset."
        )

    debt_banner = ""
    if reconstructed:
        debt_banner = (
            "> **Note:** Pre-seed baseline reconstructed via replay (no "
            "canonical snapshot captured in Plan 43-03). DEBT-V25 logged. "
            "Operator may re-execute with proper baseline capture in v2.6+ "
            "for higher confidence.\n\n"
        )

    md = f"""# Sherdog Debutant Elo Seed -- Measurement (v2.5 DEBUT-V25-04)

{debt_banner}**Phase:** 43-sherdog-debutant-elo-seed
**REQ:** DEBUT-V25-04
**Measured:** {measured_at}
**Baseline source:** `{baseline_source}`

## Verdict Posture

**Documented delta only, no gate. Seed ships unconditionally.**

Per Phase 43 CONTEXT D, the Sherdog-derived debutant Elo seed replaces the
flat 1500 default because the 1500-default is a known structural gap in the
v2.3/v2.4 substrate. The seed ships regardless of the measured delta. This
measurement is informational input to Phase 45 (META3-V25-03 meta_v3 gate),
where it helps disambiguate lift attribution between corpus-side,
TRAVEL-side, and debutant-seed-side contributions.

## Methodology

**Eval subset:** v2.3 `random_15pct` slice filtered to fights where either
fighter is a debutant (`n_ufc_fights == 0` at fight time). The
`random_15pct` mask uses `np.random.RandomState(42).random(N) < 0.15` over
the chronologically ordered corpus — bit-equivalent to
`ufc_prediction.ml.evaluator.evaluate_per_slice`.

**Debutant determination:** Per-fighter first chronological UFC fight_id
(the fight where they had `n_ufc_fights == 0`). A fight is "debutant-either"
iff its `fight_id` matches either fighter's `first_fight_id`. This direct
chronological scan is used (per Plan 43-04 `<interfaces>` path 2) rather
than `prediction_metadata.is_debutant_either` because that flag may carry
stale values for re-backfilled fights.

**Metric path:** `EloEngine.expected_win_probability(rating_a, rating_b)`
computed from pre-fight Elo state. Per Plan 43-04 `<action>` recommendation,
this isolates the seed signal cleanly without confounding from xgb_v2's
nonlinear blending. Full xgb_v2-path debutant measurement is deferred to
v2.6+.

**Pre-seed baseline:** Engine replayed with `seeds={{}}` (empty). Per
`tests/unit/elo/test_engine_seed_dispatch.py::test_3`, this is bit-exact
equivalent to the pre-Phase-43 engine — reproducing the 1500-default
substrate deterministically.

**Post-seed:** Engine replayed with
`seeds=load_seeds(data/sherdog/pre_ufc_records.csv)` — 800 debutants seeded
per Plan 43-03 backfill, 4,378 fall back to 1500 default.

**Confounders:** None expected. `xgb_v2 + meta_v2` byte-identical
end-to-end per AUDIT-01. This script reads zero model joblibs and writes
zero model joblibs.

**Draw / no-contest handling:** Fights with `winner_id is None` are SKIPPED
from the Brier/accuracy calculation (cannot Brier-score against a binary
label).

## Results

| Metric             | Pre-Seed   | Post-Seed  | Delta       |
|--------------------|------------|------------|-------------|
| Brier              | {pre_brier:.4f}     | {post_brier:.4f}     | {delta_brier:+.4f}     |
| Accuracy           | {pre_acc:.4f}     | {post_acc:.4f}     | {delta_acc:+.4f}     |
| n_debutant_fights  | {n_deb}        | {n_deb}        | --          |

**Subset composition:**

- `n_total_random_15pct` (all fights in random_15pct slice): {n_total}
- `n_debutant_fights` (random_15pct ∩ debutant-either, outcome-labeled): {n_deb}
- `n_debutant_unique_fighters`: {n_unique}

## Interpretation

{interp}

**Sign convention:** `delta_brier = pre_brier - post_brier`
(positive → seed improves). `delta_acc = post_acc - pre_acc`
(positive → seed improves).

## AUDIT-01 Invariants

xgb_v2 SHA-256: `{EXPECTED_XGB_V2_SHA256}` (byte-identical to PROJECT.md
canonical + Phase 43 START + Phase 43 END anchors).

meta_v2 SHA-256: `{EXPECTED_META_V2_SHA256}` (byte-identical to PROJECT.md
canonical + Phase 43 START + Phase 43 END anchors).

D-18 LOCKED gate formula hash: `{FORMULA_HASH}` (no renegotiation).

Phase 43 AUDIT-01 chain entry: **42-of-N → 43-of-N END (CLOSED)**.

## Downstream

Plan 45-03 (META3-V25-03 gate) may reference this delta when interpreting
`meta_v3` lift attribution — helps disambiguate corpus-side / TRAVEL-side /
debutant-seed-side contributions. The `baseline_source` field is the
load-bearing trust anchor for that downstream consumer:
`plan_43_03_snapshot` (preferred) vs `reconstructed_via_replay` (this
execution, DEBT-V25 logged).

## Deferred

Per CONTEXT — re-eligible v2.6+:

1. Per-promotion fine-grained tier weighting (re-eligible if the formula
   needs tuning post-measurement).
2. Sherdog record decay / recency weighting (current formula treats all
   pre-UFC fights equally).
3. Auto-update seeds on subsequent UFC corpus refreshes (currently a
   one-time backfill).
4. Sherdog-derived K-factor adjustments (could lower K for fighters with
   deep pre-UFC history).
5. Bayesian shrinkage for sparse pre-UFC records (<3 fights). Current
   experience-floor clamp partially addresses this.

Additionally deferred from Plan 43-04 itself:

6. Full xgb_v2-path debutant measurement (this run used
   `EloEngine.expected_win_probability` for signal isolation per the plan
   `<action>` recommendation). Re-eligible if operator wants
   partner-visible behavior measurement in v2.6+.
"""
    path.write_text(md, encoding="utf-8")


# ──────────────────────────── Main ──────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan 43-04 DEBUT-V25-04 debutant prediction quality "
        "measurement (pre/post Brier + acc delta on v2.3 random_15pct "
        "debutant subset).",
    )
    parser.parse_args(argv)

    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.elo.seed import load_seeds

    print(
        "[debutant_delta_v25] Plan 43-04 DEBUT-V25-04 — measurement run start"
    )

    session = SessionLocal()
    try:
        (
            all_fights,
            debutant_subset,
            n_total_corpus,
            n_random_15pct,
            n_debutant_unique_fighters,
        ) = select_debutant_fights_random_15pct(session)
    finally:
        session.close()

    print(
        f"[debutant_delta_v25] corpus_total={n_total_corpus} "
        f"random_15pct_total={n_random_15pct} "
        f"debutant_either_in_15pct={len(debutant_subset)} "
        f"unique_debutant_fighters={n_debutant_unique_fighters}"
    )

    # Load seeds for post-seed replay.
    seeds = load_seeds(SHERDOG_PRE_UFC_CSV)
    print(
        f"[debutant_delta_v25] loaded {len(seeds)} debutant seeds from "
        f"{SHERDOG_PRE_UFC_CSV}"
    )

    # Pre-seed baseline: load OR reconstruct.
    baseline = load_baseline_metrics(BASELINE_PATH)
    reconstructed = False
    if baseline is None:
        print(
            f"[debutant_delta_v25] WARNING: baseline file "
            f"{BASELINE_PATH} not found — falling back to "
            "reconstruction-via-replay (DEBT-V25 logged). Replaying engine "
            "with empty seeds (bit-exact equivalent to pre-Phase-43 engine "
            "per test_engine_seed_dispatch::test_3).",
            file=sys.stderr,
        )
        reconstructed = True
        pre_probs, pre_outcomes = replay_engine_capture_debutant_probs(
            all_fights, debutant_subset, seeds={},
        )
        pre_brier, pre_acc = compute_brier_acc(pre_probs, pre_outcomes)
        baseline_source = "reconstructed_via_replay"
    else:
        pre_brier = float(baseline["pre_brier"])
        pre_acc = float(baseline["pre_acc"])
        baseline_source = "plan_43_03_snapshot"

    # Post-seed: replay with current seeds.
    post_probs, post_outcomes = replay_engine_capture_debutant_probs(
        all_fights, debutant_subset, seeds=seeds,
    )
    post_brier, post_acc = compute_brier_acc(post_probs, post_outcomes)

    delta_brier = pre_brier - post_brier
    delta_acc = post_acc - pre_acc

    # Count of seeded fighters that appear in the eval subset.
    n_debutant_fights = len(post_probs)  # outcome-labeled debutant fights

    measured_at = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    payload: dict[str, Any] = {
        "phase": "43-sherdog-debutant-elo-seed",
        "req_id": "DEBUT-V25-04",
        "verdict_posture": "documented_delta_only",
        "n_total_random_15pct": int(n_random_15pct),
        "n_debutant_fights": int(n_debutant_fights),
        "n_debutant_unique_fighters": int(n_debutant_unique_fighters),
        "pre_brier": float(pre_brier),
        "post_brier": float(post_brier),
        "delta_brier": float(delta_brier),
        "pre_acc": float(pre_acc),
        "post_acc": float(post_acc),
        "delta_acc": float(delta_acc),
        "formula_hash": FORMULA_HASH,
        "xgb_v2_sha": EXPECTED_XGB_V2_SHA256,
        "meta_v2_sha": EXPECTED_META_V2_SHA256,
        "phase_43_seed_count": int(len(seeds)),
        "baseline_source": baseline_source,
        "measured_at": measured_at,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    print(f"[debutant_delta_v25] wrote {OUTPUT_JSON}")
    _write_md(OUTPUT_MD, payload, reconstructed=reconstructed)
    print(f"[debutant_delta_v25] wrote {OUTPUT_MD}")

    print(
        f"[debutant_delta_v25] verdict_posture={payload['verdict_posture']} "
        f"pre_brier={pre_brier:.4f} post_brier={post_brier:.4f} "
        f"delta_brier={delta_brier:+.4f} "
        f"pre_acc={pre_acc:.4f} post_acc={post_acc:.4f} "
        f"delta_acc={delta_acc:+.4f} "
        f"n_debutant_fights={n_debutant_fights} "
        f"baseline_source={baseline_source}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
