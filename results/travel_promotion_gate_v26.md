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
