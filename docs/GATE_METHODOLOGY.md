# Gate Methodology — Partner-Facing Reference

**Status:** v2.6 (2026-06-03). Methodology spec is operator-approved.
**Audience:** partners + new contributors + future-developer onboarding.
**Canonical internal spec:** [`.planning/gate_methodology_v2.6.md`](../.planning/gate_methodology_v2.6.md) — this doc extends and simplifies it.

This document covers (a) the Phase 45 substrate-drift case study, (b) the v2.6 redesigned methodology, (c) the role of the gate verifier in preventing recurrence, (d) the GATE-RECALIB-PERIODIC runbook, (e) v2.7+ implications for the META-V26 retrain candidate.

For the executable verifier API: `from ufc_prediction.ml.gate_verifier import verify_candidate_vs_canonical`. For the v2.6 contract: `.planning/gate_contract_v2.6.json`.

---

## 1. The Phase 45 Substrate-Drift Case Study

In v2.5, the project's `meta_v3.joblib` candidate (a META-V22 retrain) reported a +0.28-0.33 Brier delta against the canonical `meta_v2.joblib` baseline on the v2.5-close substrate. Mechanically, the existing gate verifier reported floor PASS / hurdle PASS on all three slices — verdict `path_a`.

**The verdict was wrong.** The baseline measurement was structurally invalid. The verifier had loaded the persisted `meta_v2.joblib` sklearn `Pipeline` (`StandardScaler` + `LogisticRegression`) — both fit on the Phase 26 substrate (pre-Phase-41 BFO disambiguation cleanup, pre-Phase-43 seeded Elo, pre-Phase-44 corpus growth). Applied to the v2.5 substrate (a different feature distribution), the frozen `StandardScaler` emitted out-of-distribution Level-1 vectors. The downstream frozen `LogisticRegression` then miscalibrated on this OOD input, **inflating the baseline Brier from ~0.21 (Phase 26 in-sample) to ~0.38-0.43 (v2.5 OOD).**

The candidate `meta_v3.joblib` was fit end-to-end on the v2.5 substrate — no scaler drift; its Brier ~0.10 was real. The "+0.28-0.33 Brier delta" was therefore dominantly a **baseline-scaler-OOD artifact**, not real model lift.

The operator declared Path B (no promotion) on 2026-06-03 via a corrigendum layered above the raw verifier verdict. The raw numbers were preserved verbatim in `results/meta_v3_gate_verdict_v25.json` for audit-trail transparency. Full Path B writeup: [`results/meta_v3_spike_findings_v25.md`](../results/meta_v3_spike_findings_v25.md).

This case was not unique. Phase 42 TRAVEL recomposition reported a +0.249 Brier delta on widened slices; Plan 45-01 re-verification found **~67-83% of the delta evaporated** when an OOF-source-divergence artifact was stripped (same pattern, different layer). Two distinct instances in a single milestone confirms the failure mode is **structural**, not coincidental.

---

## 2. The v2.6 Redesigned Methodology

The v2.6 redesign establishes two distinct gate layers:

| Layer | Question | Locked? |
|---|---|---|
| **Meta-gate** | Is the measurement structurally valid? | v2.6: **methodology (a) refit-baseline ALWAYS** |
| **Formula-gate** (D-18) | Given a valid measurement, does it clear the locked thresholds? | LOCKED at Phase 20 BFO_ARCHIVE_REACHABILITY anchor; preserved verbatim |

The meta-gate operates **above** the D-18 LOCKED formula-gate. D-18 prevents post-measurement renegotiation of the threshold formula. D-18 does NOT prevent recognition of a structurally invalid measurement.

### Methodology (a) — refit-baseline (v2.6 standard)

Operator-preferred (2026-06-03). Strip the canonical Pipeline of its frozen `StandardScaler` + `LogisticRegression`. Refit a fresh `StandardScaler` + `LogisticRegression` on the META-V22 architecture against the **current substrate's distribution**. The refit baseline is then directly comparable to the candidate (both fit on the same substrate; no OOD scaler response).

**Cost:** minutes (Level-1 substrate is small).
**AUDIT-01 impact:** canonical `meta_v2.joblib` STAYS byte-identical; the refit ships as a SIBLING artifact `meta_v2_refit_v2.6.joblib`.

Two fallback methodologies are documented in [`.planning/gate_methodology_v2.6.md` §3](../.planning/gate_methodology_v2.6.md):
- **(b) Dual test-set comparison** — sanity check; not v2.6 standard.
- **(c) Shadow-traffic A/B** — production-deployment concern; v3.x scope.

---

## 3. The Verifier — Preventing Recurrence Automatically

The v2.6 verifier lives at `src/ufc_prediction/ml/gate_verifier.py` (Phase 55). Its contract:

```python
from ufc_prediction.ml.gate_verifier import verify_candidate_vs_canonical

verdict = verify_candidate_vs_canonical(
    candidate=Path("models/meta/your_candidate.joblib"),
    canonical=Path("models/meta/meta_v2.joblib"),
    eval_slices={...},  # per-slice EvalSlice instances
    substrate_align_strategy="refit_baseline",  # default; v2.6 standard
)
```

`verdict` is a `SubstrateDriftSafeGateVerdict` dataclass with three possible states:

| Verdict | Meaning |
|---|---|
| `path_a_promote` | Clean PASS on substrate-aligned numbers |
| `path_b_reject` | Formula-gate failed on valid measurement |
| `confound_block` | **Meta-gate fired**; substrate-drift confound detected; NEVER promote |

The verdict's `to_dict()` produces a JSON-serializable form for the `gate_verdict_v26_<candidate>.json` artifact pattern. Raw + substrate-aligned numbers are both preserved — audit-trail discipline mirrors the Phase 45 corrigendum, but no operator override is needed because the verifier surfaces the confound automatically.

**Phase 45 regression test:** the verifier is required to produce `confound_block` for the Phase 45 meta_v3 candidate. Synthetic equivalent ships in Phase 55 unit tests; the integration test against the actual Phase 45 substrate parquet is a v2.6.1 follow-on.

---

## 4. GATE-RECALIB-PERIODIC Runbook

Per-slice thresholds (`brier_max`, `accuracy_min`) are pinned to the substrate at contract-derivation time. As the corpus grows, the substrate distribution shifts — eventually, the pinned thresholds become stale.

The CLI: `ufc gate recalib --feature-set v2.6` (Phase 56 GATE-V26-03).

### When to run

When the active corpus has grown **>10% above** the contract's `n_training_fights` baseline. The 10% threshold is per the methodology spec; tune via `--threshold N` if needed.

### Default mode: --dry-run

```bash
$ ufc gate recalib --feature-set v2.6
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field                        ┃ Value                                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ contract version             │ v2.6                                         │
│ derived_at                   │ 2026-06-03                                   │
│ formula_hash (D-18 LOCKED)   │ 7d221b4ac21e550c3341db32c2bcec0d…            │
│ methodology_version          │ v2.6                                         │
│ substrate_alignment_strategy │ refit_baseline                               │
│ confound_threshold           │ 0.0500                                       │
│ corpus-growth threshold      │ ≥10.0% above baseline triggers re-derivation │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

The dry-run reports what the contract currently looks like + would-be threshold delta WITHOUT writing a new contract.

### --apply mode (v2.6.1 follow-on)

`--apply` is the operator-action mode that writes a new `.planning/gate_contract_v2.6.json`. v2.6 ships the CLI scaffold; the actual threshold re-derivation logic is gated on the corpus-growth trigger firing (deferred to v2.6.1 per Phase 56 scope).

### D-18 LOCKED preserved

`formula_hash` does NOT change between contract versions. The recalibration re-runs only the substrate measurement; the formula stays fixed. If `formula_hash` ever changes, that's a methodology-spec-level change requiring its own operator-approval cycle.

---

## 5. v2.7+ Implications for META-V26

The v2.7+ backlog row 23 (META-V26 retrain candidate) becomes eligible when **any** of:

1. **Corpus growth ≥10%** — RESIL-INGEST-V26 (alt-events Path A) ships from Phase 44 RESIL-V25-04 escalation, OR UFCStats anti-bot resolution adds materially.
2. **GATE-V26-02 verifier ships clean** — the Phase 55 verifier auto-detects the Phase 45 confound on the actual substrate parquet (v2.6.1 integration test).
3. **Architectural escalation operator-approved** — NN base learner spike OR xgb_v3 hyperparameter re-tune (v3.x milestone scope).

When v2.7 opens with META-V26 in scope, the gate verifier built per this methodology is the eligibility judge. No manual operator-override corrigenda needed — the verdict comes from the verifier directly.

See [`results/meta_v3_spike_findings_v25.md` §5-§6](../results/meta_v3_spike_findings_v25.md) for the original Phase 45 enumeration of re-entry criteria; [`.planning/gate_methodology_v2.6.md` §8](../.planning/gate_methodology_v2.6.md) for the consolidated v2.6 lock.

---

## 6. References

- `.planning/gate_methodology_v2.6.md` — internal spec (10 H2 sections; this doc is a partner-facing extension)
- `.planning/gate_contract_v2.6.json` — v2.6 contract (per_slice + methodology block)
- `src/ufc_prediction/ml/gate_verifier.py` — Phase 55 verifier implementation
- `src/ufc_prediction/ml/gate_contract.py` — `load_gate_contract(version="v2.6")` dispatcher
- `results/meta_v3_spike_findings_v25.md` — Phase 45 Path B writeup (operator-grade documentation)
- `results/meta_v3_gate_verdict_v25.md` — Phase 45 verdict with operator corrigendum (raw numbers preserved)
- `results/travel_oof_verification_v25.md` — Phase 42 TRAVEL OOF-source-divergence evaporation evidence
- `CONTRIBUTING.md` § AUDIT-01 protected files — byte-identity discipline for canonical artifacts
