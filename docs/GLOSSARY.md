# Glossary

Terms used across the codebase, planning artifacts, and partner contracts. Each entry: one-sentence definition + where it lives.

## Modeling fundamentals

### Brier score
Mean squared error between predicted probability and observed binary outcome. Range [0, 0.25] for well-calibrated binary models (0 = perfect, 0.25 = always-50% baseline). Lower is better. Production reference: META-V22 random_15pct Brier ≈ 0.14; xgb_v2 random_15pct Brier ≈ 0.18.

### Accuracy
Fraction of fights where `argmax(predicted_prob) == observed_winner`. Range [0, 1]. Less informative than Brier for sports models (Brier rewards calibration; accuracy is binary). Production reference: ~78% lower bound across slices.

### OOF (Out-Of-Fold) probability
A prediction for a row that was held out during training. xgb_v2's `xgb_oof_prob` column is the prediction xgb_v2 makes for each meta-train row when that row was excluded from the xgb_v2 training fold. Used to feed xgb_v2's signal into the meta-learner without train-on-test leakage.

### Calibration / CalibratedClassifierCV
Maps raw model scores → calibrated probabilities so that `P(model says 70%) ≈ 70% of those actually win`. Sigmoid (Platt scaling) and isotonic are the two common methods. The "CALIB" step in v2.3 composition uses isotonic.

### ECE (Expected Calibration Error)
Calibration metric: average gap between predicted probability and empirical accuracy across probability bins. Lower is better. Secondary metric in the v2.3 gate spike (not gated on).

## Elo system (your "core differentiator")

### Multi-domain Elo
Three separate Elo ratings per fighter: `elo_overall` (skill-overall), `elo_striking` (striking exchanges + finish rates), `elo_grappling` (takedowns + control time + submissions). Updated chronologically after each fight using actual performance signal, not just W/L.

### Elo K-factor decay
Initial K=40 for a fighter's first 5 UFC fights, then K=20 thereafter. Allows new fighters to find their level quickly without making veterans bounce around on a single bad night.

### Elo append-only invariant
A fighter's Elo at fight N is immutable once computed. Future fights extend the chain; past fights are never recomputed. This is the temporal-integrity guarantee for Elo-based features.

## Composition + promotion (v2.3 Phase 32 vocabulary)

### Forward-stepwise composition
META → CALIB → REF → TRAVEL ordered chain. Each step's candidate must beat the prior step by ≥0.003 Brier on every eval slice OR the step is rejected and the next step composes on the prior survivor. Implements D-13(v2.0) "every additive feature must justify itself."

### Triple-gate (D-03 of Phase 32)
A meta_v3 candidate is promoted to `models/meta/meta_v3.joblib` only if **all three** conditions hold:
1. **Gate clearance** — beats `gate_contract_v2.3.json` per-slice thresholds (`brier_max` and `accuracy_min`) on all 3 slices
2. **Total margin** — beats META-V22 baseline Brier by ≥0.003 on all 3 slices
3. **Per-step hurdle** — every composition step in the chain clears ≥0.003 Brier improvement over the prior step

If any leg fails: Path B or C materializes, no promotion. v2.3 materialized Path B (legs 1+2 PASS, leg 3 FAIL).

### Path A / Path B / Path C / Path D
Pre-templated outcome paths per D-09 carry-forward. Each phase pre-registers its possible outcomes BEFORE measurement runs, so the analysis isn't post-hoc. v2.3 examples: Phase 31 gate Path A (substrate clears 0.70 floor) vs Path D (HALT-AND-DECIDE); Phase 32 composition Path A (meta_v3 promoted) vs Path B (partial) vs Path C (tried and didn't help).

### "Tried and didn't help"
Phase 32 STEPWISE-V23-03 outcome path. Distinguishes v2.3's empirical finding ("REF/TRAVEL tested on populated substrate, didn't add value") from v2.2's deferred finding ("data wasn't there yet"). Substrate-blocking vs feature-engineering-blocking are different failure modes; v2.3 names the difference.

## Gate contract numerics

### Gate contract
Versioned JSON file (`.planning/gate_contract.json` for v2.1, `gate_contract_v2.2.json`, `gate_contract_v2.3.json`) emitted by the noise-floor spike. Defines per-slice promotion thresholds for candidate models. The same `formula_hash` produces all three; only the per-slice numerics differ across versions.

### Formula hash (D-18 binding)
SHA-256 of the canonical formula source string: `gate_brier_max = round(median_brier - 1 * max(seed_std_brier, bootstrap_ci_half_brier), 4)` + analogous for accuracy. Value: `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a`. LOCKED across v2.1 → v2.2 → v2.3. "No post-measurement renegotiation."

### Operator floor (D-05 of v2.2 / D-01 of v2.3 Phase 31)
`accuracy_min = max(formula_output, 0.70)` applied per-slice at spike time. Pre-committed to PROJECT.md `D-{N}` row BEFORE the spike runs. If the substrate fills and the formula output still misses 0.70, the spike emits a HALT-AND-DECIDE artifact instead of relaxing the floor.

### Eval slices
Three time-based + random-sampled holdout windows for evaluating per-slice generalization:
- `most_recent_12mo` — fights in the most recent 12 months (459 fights post Phase 29 NaN-drop)
- `most_recent_24mo` — fights in the most recent 24 months (967 fights post-drop)
- `random_15pct` — 15% random sample across the full timeline (160 fights — under the v2.3 D-06 ≥500 floor; operator-accepted deviation)

### Seed std + bootstrap CI half (variance fields)
The gate contract's variance fields:
- `seed_std_brier` — empirical std of Brier across N seeds
- `bootstrap_ci_half_brier` — BCa 68% bootstrap CI half-width
- `std_brier_used = max(seed_std_brier, bootstrap_ci_half_brier)` — conservative max plugged into the gate formula

v2.2 produced collapsed-to-zero variance (LR closed-form → identical tuples across seeds); v2.3 Phase 30 fixed this with bootstrap row-resampling so N seeds produce N DIFFERENT metrics.

## Audit + governance

### AUDIT-01 (chain)
A discipline that preserves the byte-identity of `models/xgb_v2.joblib` (SHA-256 `6e7641…ba099`) across all releases since v2.1. Each phase contributes "chain leaves" — MID + END SHA artifacts at `.planning/phases/*/...-XGB-V2-SHA-*.txt` — proving the model file was not edited during that phase. Chain leaf count at v2.3 close: **32 of N**.

### Pre-commit hook (AUDIT-01 enforcement)
Active hook on `feature_matrix.py / persistence.py / predictor.py / train.py / models/xgb_v2.joblib`. Any edit to these files fails the commit unless deliberately approved by the operator (typically as part of a planned phase that documents the change in PROJECT.md as a `D-{N}` decision row).

### D-{N} decision row
Numbered entries in PROJECT.md Key Decisions table, each one a binding governance decision (e.g., D-13(v2.0) ≥0.003 Brier hurdle; D-18 no post-measurement renegotiation; D-24(v2.3, GATE) Path A pre-commit 0.70 floor). Cross-referenced from CONTEXT.md and SUMMARY.md across phases.

### LIVE-03 delta
The 1-line edit to `src/ufc_prediction/ml/predictor.py` that lets the inference path discover the latest promoted meta-model version (`get_latest_meta_version` helper, pattern from Plan 26-04). v2.3 Phase 32 Plan 32-03 *skipped* this delta because Path B materialized — no meta_v3 to dispatch to.

## PARTNER schema vocabulary

### v1.0.0 byte-frozen
The original partner contract (`predictor.schema.v1.0.0.json`) shipped at Phase 25 (v2.2). Cannot be modified — Phase 25 forward-compat lock is binding. Future minor bumps (v1.1.0, v1.2.0, …) are additive-only.

### Additive trio (v1.1.0)
Three optional fields added in Phase 32 Plan 32-02:
- `gate_contract_ref: str | null` — path/version of gate contract used
- `model_candidates: list[{name, sha256, phase}] | null` — supported models with byte-identity SHAs
- `phase_chain_audit_sha: str | null` — AUDIT-01 chain leaf SHA

All Optional with None defaults. v1.0.0 partners see byte-identical response shape.

## Data sources

### UFCStats (canonical)
Primary source for UFC fight outcomes + per-round stats. Mostly reliable for fight-level fields; "significant strike" classification is subjective (ringside staff judgment).

### Sherdog
Secondary source for fighter career data — camp affiliations (`Association:` field), nationality, weight class history. Used in v2.3 Phase 29 CAMP audit (top-30 coverage at 45.92%, below 60% threshold → CAMP deferred to v2.4+).

### BFO (BestFightOdds)
Source for closing odds. Used to derive `closing_prob_diff` + `sharp_money_signal` features — the two odds-derived meta-learner inputs that account for most of META-V22's accuracy lift over xgb_v2.

### Substrate (referee + venue)
v2.3 Phase 28 filled `events.referee_id` + `events.venue_id` to 100% coverage. REF features use referee_id; TRAVEL features want venue_id lat/lon/tz but the FeatureMatrixAssembler doesn't compute those primitives yet (v2.4+ backlog).

## Project / planning

### GSD (Get Shit Done)
Planning framework used in `.planning/`. Each milestone has a ROADMAP + REQUIREMENTS; each phase has CONTEXT (locked decisions) → RESEARCH (technical patterns) → PLAN (executable tasks) → SUMMARY (outcome) → VERIFICATION (goal-backward audit). The `.planning/` directory is **history**, not current state — read `src/` for current truth.

### Phase / Plan / Wave / Task
Hierarchy: a **milestone** contains **phases**; a phase contains **plans** (often parallel-eligible); a plan groups **tasks** into **waves** (sequenced for parallel execution by `gsd-executor` subagents).

### CONTEXT.md
Locked decisions from operator at phase open. Read FIRST when reasoning about why a phase's code looks the way it does. Decisions are numbered (`D-01`, `D-02`, …) and binding for the phase.

### RESEARCH.md
Phase-level technical research output: code references with line numbers, design patterns, pitfalls, validation architecture. Consumed by the gsd-planner agent before plans are drafted.

### SUMMARY.md
Phase-close documentation: what shipped, key commits, deviations from plan, AUDIT-01 chain status, outcome path materialization.

### RETROSPECTIVE.md
Milestone-close documentation rolling up all phase summaries + outcome path realization + v{X+1}+ backlog items.
